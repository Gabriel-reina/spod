"""SPOD — Selective POpularity Debiasing. Removes the RM's popularity bias per semantic
category, but exempts the single top-RM-scoring item of each class.

Unified formulation (the four schemes share one operator, only phi and the presence of min change):

    d_i  = beta * std(r_group) * phi(c_i)
    r'_i = r_i                      if i == top1(c_i)   -- the in-class peak always keeps its raw score
    r'_i = r_i - d_i                if SPOD_MIN=0       -- scheme A
    r'_i = min(r_i, r_i - d_i)      if SPOD_MIN=1       -- scheme B / C

    top1(c) = the item with the highest raw RM score among items of category c in the group

Meaning of the min constraint: phi can be positive or negative (positive for popular, negative for rare),
so d can be positive or negative. Without min, a rare item would be lifted above its raw score (scheme A);
with min the lift is clamped, so debiasing may only push popular items down, and no item's final score exceeds
the quality the RM originally assigned it (scheme B/C). Thus "quality does not drop" is a structural guarantee,
not something that relies on tuning beta. Note: if phi is always >= 0, min is identical to plain subtraction,
and A and B produce exactly the same result -- so do not pair the always-positive conventions freq/group with SPOD_MIN=1.

The four conventions for phi (SPOD_POP):
  rmscore  (mu_c - mu_bar) / sigma        z-score of the RM offline category means. A static prior,
                                         aimed directly at the RM's measured overestimation direction. Used by schemes A/B.
  logq     log p_hat(c) + H(p_hat)       online logQ correction (PMI form). p_hat is the policy's own
                                         category exposure distribution, updated in real time over a sliding window. Used by scheme C.
  freq     p(c) / max_c p(c) in (0,1]    training-library category frequency, always positive. Used by the freq comparison convention.
  group    n_c / k                       the class's share within the group, always positive. A DRA-flavored comparison.

Why logq is not just a different normalization: phi = log p_hat + H(p_hat) satisfies
E_{c~p_hat}[phi] = 0, i.e. it is the additive correction along the -grad_{p_hat} KL(p_hat || Unif) direction. In other
words, "reward debiasing by popularity" and "adding an entropy regularizer on the marginal distribution over semantic
categories" are two ways of writing the same thing. And it is negative feedback: whichever class the policy collapses
toward sees its p_hat rise and its penalty automatically increase, without relying on any offline-estimated bias direction.

Why not use multiplicative IPS (w_i = p(c_i)^-alpha): in-group whitening cancels most of the in-group multiplicative
factor; only the additive form genuinely changes the in-group relative ordering.

Why it must be at the reward layer and not the advantage layer: in-group multiplicative reweighting is canceled by
in-group whitening, and an additive constant offset is canceled between numerator and denominator. SPOD is different --
it is a **selective** subtraction, penalizing only the non-top items within the same class, so within the group it is not
a constant offset and whitening cannot cancel it. Moreover judging top-1, computing std, and computing min all need the
**pre-whitening** raw RM scores, which only the reward worker has; the advantage that loss_func receives has already been
whitened and the raw-scale information is lost.

Why the penalty is multiplied by std(r_group): in-group whitening divides out std(r), so the penalty is written in units of
std, which makes beta scale-invariant. After whitening the penalty's magnitude in advantage space is approximately
beta * phi, directly readable.

Experiment arms:
  spodv  SPOD_ENABLE=0                              pure GRPO baseline
  spodo  POP=rmscore MIN=0 EXEMPT=top1 BETA=0       observer: numerically equivalent to v, but logs the full set
  spoda  POP=rmscore MIN=0 EXEMPT=top1              scheme A
  spodb  POP=rmscore MIN=1 EXEMPT=top1              scheme B
  spodc  POP=logq    MIN=1 EXEMPT=top1              scheme C
  ctrl   POP=rmscore MIN=1 EXEMPT=none              no exemption (isolates the contribution of the exemption term)
  ctrl   POP=freq    MIN=0 EXEMPT=top1              frequency convention (freq always-positive comparison)

BETA=0 is observer mode, not off: d = 0*std*phi is always 0, the returned scores are bit-identical to the input, but
category parsing, phi, share, entropy, and detail dumping all still run. If the baseline arm used SPOD_ENABLE=0 the whole
module short-circuits and no category distribution is available -- so there would be no diversity baseline measured with the
same convention as the method arms. To truly turn it off, use SPOD_ENABLE=0.

Online monitoring of peak preservation (P1): spod/top1_preserved -- whether the top-scoring item within the group after
debiasing is still the original one. SPOD only penalizes the non-top items within a class, so the cross-class optimum can
still be swapped out (if it belongs to a very popular class); this metric quantifies exactly that, rather than assuming it holds.

Activation: env SPOD_ENABLE=1 (off by default).
Config: SPOD_BETA(0.5) / SPOD_POP(rmscore|logq|freq|group|uniform) / SPOD_MIN(0|1)
        SPOD_EXEMPT(top1|top2|top3|none) / SPOD_GROUP(8) / SPOD_CATMODE(tag|kw)
        SPOD_PHI_FILE(json) / SPOD_ZW(0|1 whether sigma is weighted by item count)
        SPOD_POP_WINDOW(512) / SPOD_POP_ALPHA(1.0) / SPOD_OUT_DIR / SPOD_TAG
"""

import json
import math
import os
import re

import torch

__all__ = ['spod_enabled', 'spod_base_metrics', 'compute_spod_rewards',
           'spod_category', 'spod_phi_table', 'spod_phi_now', 'spod_phi_src']

# the format of the category label in generated text: something like [C1_sys_kernel] at the start of a line
_TAG = re.compile(r'^\s*\[([A-Za-z0-9_]+)\]')

# keyword fallback: models (especially small ones) often disobey the label format; on parse failure, classify by keyword hit.
# order is priority, stop at first hit -- kept consistent with the rules in build_library/build_dataset.py.
_KW = [
    ('C1_sys_kernel', ('kernel', 'cuda', 'triton', 'fp8', 'bf16', 'memory', 'fused',
                       'compile', 'flash', 'throughput', 'overlap', 'dataloader')),
    ('C2_optimizer', ('muon', 'adam', 'optimizer', 'newton-schulz', 'momentum',
                      'learning rate', 'lr schedule', 'warmup', 'weight decay')),
    ('C3_architecture', ('embedding', 'residual', 'skip connection', 'depth', 'width',
                         'layer', 'norm', 'rmsnorm', 'value embed', 'u-net', 'tie')),
    ('C4_attention', ('attention', 'rope', 'qk', 'softmax', 'window', 'head',
                      'sliding', 'block mask', 'causal')),
    ('C5_loss', ('loss', 'logit', 'softcap', 'label smoothing', 'objective',
                 'cross entropy', 'z-loss')),
]

# ======================== category statistics tables (all the raw material for phi) ========================
# This set of tables changes together with "which RM did the scoring, which library was used"; it is a whole and cannot be mixed:
#   freq     the category distribution of listed items (tier=strong/null) -- the phi for the freq convention, the smoothing prior for logq
#   n        the number of listed items per class -- the weighting convention for sigma when SPOD_ZW=1
#   rm_mean  the per-category mean that this RM assigns to listed items -- the phi for the rmscore convention is its z-score
#
# Selected by SPOD_PHI_SRC, defaulting to and the only built-in one spod2clean (open-source convention). freq can additionally be overridden by SPOD_PHI_FILE.
_PHI_TABLES = {
    # open-source convention. scorer = spod2cleanrm (the same ruler used for that hold-out training and eval);
    # library = data/rm_items_prompted_clean.jsonl (148 items, with items attached to test28 removed),
    # of which 79 are listed. Re-measured by entry/start_spod_rm_phi.py, character-for-character consistent with
    # the score_all_items convention of the RM training script: text = prompt + '\n' + response, padding_side=right,
    # truncation max_len=768, fp32 weights + autocast(bf16), take the logit of the last non-pad token.
    'spod2clean': {
        'freq': {
            'C1_sys_kernel': 0.367,
            'C2_optimizer': 0.215,
            'C3_architecture': 0.241,
            'C4_attention': 0.139,
            'C5_loss': 0.038,
        },
        'n': {
            'C1_sys_kernel': 29, 'C2_optimizer': 17, 'C3_architecture': 19,
            'C4_attention': 11, 'C5_loss': 3,
        },
        'rm_mean': {
            'C1_sys_kernel': 19.130, 'C2_optimizer': 24.261, 'C3_architecture': 19.159,
            'C4_attention': 31.651, 'C5_loss': 26.365,
        },
    },
    # the third public scenario, rustc-perf (Rust compiler build speedup, 6 B-classes). Base Qwen3-1.7B-Base.
    #   scorer = rustcrm-20260910/ckpt_id=rm_final (the same ruler in use by GRPO/eval)
    #   library = rm_items_prompted_rustc.jsonl (1414 train-only, 1067 listed)
    #   direction = same family as speedrun: freq-rm_mean negatively correlated Pearson r=-0.475 / Spearman -0.600;
    #            the rarest B6_infra_tooling has the highest mean (phi=+1.722), the most common B1_backend_codegen
    #            has a lower mean (phi=-0.587) -- the RM popularity bias holds, spread 19.46 / sd 5.96, no small-class noise.
    'rustc': {
        'freq': {
            'B1_backend_codegen': 0.257, 'B2_type_trait_solver': 0.229,
            'B3_query_incremental': 0.098, 'B4_frontend_parse': 0.177,
            'B5_library_std': 0.151, 'B6_infra_tooling': 0.088,
        },
        'n': {
            'B1_backend_codegen': 274, 'B2_type_trait_solver': 244,
            'B3_query_incremental': 105, 'B4_frontend_parse': 189,
            'B5_library_std': 161, 'B6_infra_tooling': 94,
        },
        'rm_mean': {
            'B1_backend_codegen': 9.945, 'B2_type_trait_solver': 15.038,
            'B3_query_incremental': 15.833, 'B4_frontend_parse': 11.895,
            'B5_library_std': 4.230, 'B6_infra_tooling': 23.692,
        },
    },
}

_PHI_SRC = os.environ.get('SPOD_PHI_SRC', '').strip().lower() or 'spod2clean'
if _PHI_SRC not in _PHI_TABLES:
    print(f'[SPOD] SPOD_PHI_SRC={_PHI_SRC!r} not recognized (only {list(_PHI_TABLES)}), falling back to spod2clean',
          flush=True)
    _PHI_SRC = 'spod2clean'

# rm_mean and freq are negatively correlated (r=-0.720): the highest-share C1 has the lowest mean,
# and the second-lowest-share C4 has the highest mean. So "debias by frequency" and "debias by RM overestimation" will
# penalize entirely different classes -- this is the entire point of having the two arms rmscore and freq.
_DEFAULT_FREQ = dict(_PHI_TABLES[_PHI_SRC]['freq'])
_DEFAULT_N = dict(_PHI_TABLES[_PHI_SRC]['n'])
_RM_MEAN = dict(_PHI_TABLES[_PHI_SRC]['rm_mean'])

PRINT_EVERY = 16
_STATE = {
    'calls': 0,
    'phi': None,      # the phi table of the static conventions (rmscore/freq), computed once and cached
    'freq': None,     # the normalized frequency prior, also used as logq's smoothing prior
    'pop': None,      # the online category counts for the logq convention {cat: float}
    'pop_n': 0.0,     # total online count
    'sink': None,     # dump handle
}


def spod_enabled():
    return os.environ.get('SPOD_ENABLE', '0') == '1'


def spod_phi_src():
    """The currently effective phi table name (after fallback correction). Callers use it to verify the table and dataset are a matched set."""
    return _PHI_SRC


def _beta():
    return float(os.environ.get('SPOD_BETA', '0.5'))


def _pop_mode():
    m = os.environ.get('SPOD_POP', 'rmscore').strip().lower()
    return m if m in ('rmscore', 'logq', 'freq', 'group', 'uniform') else 'rmscore'


def _use_min():
    return os.environ.get('SPOD_MIN', '0') == '1'


def _exempt_mode():
    m = os.environ.get('SPOD_EXEMPT', 'top1').strip().lower()
    return m if m in ('top1', 'top2', 'top3', 'none') else 'top1'


def _exempt_k():
    """exemption level k: within each category, keep the top k items by RM score (high to low) unpenalized. none -> 0.
    
    Note the exemption is **within-class**, not within-group. So as k grows, classes with item count <= k are entirely
    exempted, and the penalized items remain only in the most crowded classes -- which is exactly what the top-k ablation tests.
    """
    m = _exempt_mode()
    return 0 if m == 'none' else int(m[3:])


def _group():
    return int(os.environ.get('SPOD_GROUP', '8'))


def _catmode():
    # the [C1_...] tag at the start of a generated line goes through tag; use the kw keyword fallback when the model disobeys the format.
    default = 'tag'
    m = os.environ.get('SPOD_CATMODE', default).strip().lower()
    return m if m in ('tag', 'kw') else default


def _freq_prior():
    """the normalized category frequency prior. SPOD_PHI_FILE can override it (json: {cat: weight})."""
    if _STATE['freq'] is not None:
        return _STATE['freq']
    freq = dict(_DEFAULT_FREQ)
    path = os.environ.get('SPOD_PHI_FILE', '')
    if path and os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and loaded:
                freq = {str(k): float(v) for k, v in loaded.items()}
                print(f'[SPOD] frequency table loaded from {path}, {len(freq)} classes', flush=True)
        except Exception as exc:
            print(f'[SPOD] frequency table load failed, using built-in table: {exc}', flush=True)
    tot = sum(freq.values()) or 1.0
    _STATE['freq'] = {k: v / tot for k, v in freq.items()}
    return _STATE['freq']


def spod_phi_table():
    """the phi table of the static conventions. rmscore can be positive or negative; freq is always positive in (0,1]; uniform is always 1."""
    if _STATE['phi'] is not None:
        return _STATE['phi']
    mode = _pop_mode()
    if mode == 'uniform':
        # ablation arm u: erase all category differences, keeping only the structure of "exempt the in-class top1 + push the rest down equally".
        # The constant 1.0 is not arbitrary: rmscore's phi is a class-equal-weight z-score whose root-mean-square is exactly 1
        # (measured -0.892/+0.764/-0.720/+1.595/-0.747, sum of squares 5.000), so this arm's perturbation magnitude
        # rms(d)=beta*std(r_group) is exactly equal to arm a's -- the only difference left between the two arms is whether this perturbation
        # is distributed by category or shared equally across all non-exempt items; a minus u is the net contribution of the category debiasing information.
        # Must pair with MIN=0 and EXEMPT=top1: when phi is always positive, MIN=1 degenerates into plain subtraction (same prohibition as freq);
        # and if the top1 exemption is also removed, every item in the group subtracts the same constant, and GRPO in-group whitening cancels it entirely,
        # so this arm becomes identical to arm v -- which is exactly why the fourth cell of the 2x2 ablation table need not be run separately.
        _STATE['phi'] = {c: 1.0 for c in _DEFAULT_FREQ}
        print('[SPOD] uniform phi: all classes always +1.000 (ablation arm u, no category difference, '
              'rms aligned with rmscore)', flush=True)
    elif mode == 'freq':
        freq = _freq_prior()
        mx = max(freq.values()) or 1.0
        _STATE['phi'] = {k: v / mx for k, v in freq.items()}
    else:
        # rmscore: z-score of the category means. sigma defaults to class-equal-weight (debiasing is a class-level operation, each class
        # should be weighted equally); SPOD_ZW=1 switches to weighting by item count, which makes C4 stand out a bit more.
        mu = dict(_RM_MEAN)
        cats = list(mu)
        if os.environ.get('SPOD_ZW', '0') == '1':
            w = {c: float(_DEFAULT_N.get(c, 1)) for c in cats}
        else:
            w = {c: 1.0 for c in cats}
        tw = sum(w.values()) or 1.0
        bar = sum(mu[c] * w[c] for c in cats) / tw
        var = sum(w[c] * (mu[c] - bar) ** 2 for c in cats) / tw
        sig = math.sqrt(var) or 1.0
        _STATE['phi'] = {c: (mu[c] - bar) / sig for c in cats}
        print(f'[SPOD] rmscore phi: mu_bar={bar:.3f} sigma={sig:.3f} '
              + ' '.join(f'{c}={v:+.3f}' for c, v in _STATE['phi'].items()), flush=True)
    return _STATE['phi']


def spod_phi_now(counts=None):
    """the current phi for the logq convention: log p_hat(c) + H(p_hat), satisfying E_{p_hat}[phi]=0.

    p_hat uses online sliding-window counts + the frequency prior with add-alpha smoothing, avoiding log blowing up when some class has zero frequency.
    """
    prior = _freq_prior()
    cats = list(prior)
    alpha = float(os.environ.get('SPOD_POP_ALPHA', '1.0'))
    cnt = counts if counts is not None else (_STATE['pop'] or {})
    # smoothing: each class is first padded with alpha * len(cats) * prior(c) pseudo-counts, the prior shape being the frequency table
    pseudo = alpha * len(cats)
    raw = {c: float(cnt.get(c, 0.0)) + pseudo * prior.get(c, 1.0 / len(cats))
           for c in cats}
    tot = sum(raw.values()) or 1.0
    p = {c: raw[c] / tot for c in cats}
    ent = -sum(v * math.log(max(v, 1e-12)) for v in p.values())
    return {c: math.log(max(p[c], 1e-12)) + ent for c in cats}, p, ent


def _bump_pop(cats):
    """count the batch's category hits into the online exposure distribution, decaying proportionally when over-window (exponential sliding window)."""
    pop = _STATE['pop']
    if pop is None:
        pop = {c: 0.0 for c in _DEFAULT_FREQ}
        _STATE['pop'] = pop
    for c in cats:
        if c in pop:
            pop[c] = pop[c] + 1.0
        else:
            pop[c] = pop.get(c, 0.0) + 1.0
    _STATE['pop_n'] = sum(pop.values())
    win = float(os.environ.get('SPOD_POP_WINDOW', '512'))
    if win > 0 and _STATE['pop_n'] > win:
        sc = win / _STATE['pop_n']
        for c in pop:
            pop[c] *= sc
        _STATE['pop_n'] = sum(pop.values())


def spod_category(text, mode=None):
    """returns (category name, whether parsing succeeded). Returns ('UNK', False) on failure."""
    mode = mode or _catmode()
    t = text or ''
    if mode == 'tag':
        m = _TAG.match(t)
        if m:
            tag = m.group(1)
            if tag in _DEFAULT_FREQ:
                return tag, True
    low = t.lower()
    for name, kws in _KW:
        if any(w in low for w in kws):
            return name, True
    return 'UNK', False


def _sink():
    """the directory to dump detail/metrics per step. Just set the env var SPOD_OUT_DIR to any local directory.
    Failure must not kill training; after marking ok=False it only prints to screen thereafter."""
    if _STATE['sink'] is not None:
        return _STATE['sink']
    d = os.environ.get('SPOD_OUT_DIR', '').strip()
    tag = os.environ.get('SPOD_TAG', 'spod').strip() or 'spod'
    s = {'detail': '', 'metrics': '', 'ok': False}
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            # rank/pid go into the file name: the reward worker may be multi-process, and same-named files would truncate each other
            sfx = f"{os.environ.get('RANK', '0')}_{os.getpid()}"
            s['detail'] = os.path.join(d, f'step_detail_{tag}_r{sfx}.jsonl')
            s['metrics'] = os.path.join(d, f'step_metrics_{tag}_r{sfx}.jsonl')
            with open(s['metrics'], 'a', encoding='utf-8') as f:
                f.write(json.dumps({'_open': tag, 'pop': _pop_mode(),
                                    'beta': _beta(), 'min': _use_min(),
                                    'exempt': _exempt_mode()}, ensure_ascii=False) + '\n')
            s['ok'] = True
            print(f"[SPOD] dump -> {s['detail']}", flush=True)
        except Exception as exc:
            print(f'[SPOD] dump directory not writable, skipping dump: {exc}', flush=True)
    _STATE['sink'] = s
    return s


def _write(path, rows):
    if not path:
        return
    try:
        with open(path, 'a', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    except Exception as exc:
        print(f'[SPOD] writing {path} failed: {exc}', flush=True)


def spod_base_metrics():
    """constant key set: inconsistent metric keys across micro-batches / steps would make the driver-side
    reduce_metrics form a jagged array, and np.mean would raise ValueError outright and kill training
    (a real incident observed in practice). So here everything must be listed at once, and every branch returns the same set of keys."""
    m = {
        'spod/enabled': 0.0,
        'spod/exit_code': 0.0,      # 0 normal / 6 exception fallback
        'spod/group_ok': 0.0,       # whether the batch is evenly divisible into groups by SPOD_GROUP
        'spod/beta': 0.0,
        'spod/pop_logq': 0.0,       # 1=logq online convention
        'spod/pop_freq': 0.0,       # 1=freq always-positive convention
        'spod/pop_group': 0.0,      # 1=group in-group share convention
        'spod/use_min': 0.0,        # 1=the min constraint of scheme B/C is in effect
        'spod/exempt_top1': 0.0,    # 1=exempt the in-class top 0=no exemption (uniform comparison)
        'spod/exempt_k': 0.0,       # exemption level k: keep the top k items in the class unpenalized (none=0)
        'spod/n_cat_mean': 0.0,     # the average number of categories hit per group (a direct read of exploration breadth)
        'spod/unk_frac': 0.0,       # the share of category parse failures; too high means the format has not kept up
        'spod/exempt_frac': 0.0,    # the share of exempted items
        'spod/phi_mean': 0.0,
        'spod/phi_pos_frac': 0.0,   # the share of items with phi>0 (pushed down)
        'spod/delta_mean': 0.0,     # how much was deducted on average (reward raw scale, positive=deduction)
        'spod/delta_max': 0.0,
        'spod/min_clip_frac': 0.0,  # the share of items where the min constraint actually took effect (clamping the lift)
        'spod/r_std_before': 0.0,
        'spod/r_std_after': 0.0,
        'spod/std_ratio': 0.0,      # after/before, the actual scaling of beta after whitening
        'spod/rank_change_frac': 0.0,   # the share of items whose in-group ordering changed
        'spod/top1_preserved': 0.0,     # whether the top-scoring item in the group is still the original one (P1 peak preservation)
        'spod/cat_entropy': 0.0,        # the mean in-group category entropy (a direct read of diversity)
        'spod/r_raw_mean': 0.0,         # the mean pre-debias raw RM score -- comparing quality across arms can only use this,
                                        # looking at the post-debias score means "whoever got deducted less has higher quality", which is not comparable
        'spod/r_out_mean': 0.0,         # the post-debias mean (what actually enters advantage)
        'spod/pop_entropy': 0.0,        # the entropy of the online exposure distribution (the adaptive state of the logq convention)
    }
    for c in _DEFAULT_FREQ:
        m[f'spod/phi_{c}'] = 0.0        # each class's current phi; it changes step by step under logq, must be traceable
        m[f'spod/share_{c}'] = 0.0      # each class's share in this batch's rollout (a direct read of the collapse direction)
    return m


@torch.no_grad()
def compute_spod_rewards(rewards, texts, group_size=None):
    """Called on the reward worker side.

    rewards: [B] the pre-whitening raw RM scores (response_level_rewards)
    texts:   a list of generated texts of length B, in the same order as rewards; the k items of the same prompt are
             laid out contiguously in the batch (matching reward_norm's reshape(-1, n_sample))
    returns (new rewards [B], metrics)
    """
    m = spod_base_metrics()
    beta, pop_mode, ex_mode, use_min = _beta(), _pop_mode(), _exempt_mode(), _use_min()
    ex_k = _exempt_k()
    m['spod/beta'] = beta
    m['spod/pop_logq'] = 1.0 if pop_mode == 'logq' else 0.0
    m['spod/pop_freq'] = 1.0 if pop_mode == 'freq' else 0.0
    m['spod/pop_group'] = 1.0 if pop_mode == 'group' else 0.0
    m['spod/use_min'] = 1.0 if use_min else 0.0
    m['spod/exempt_top1'] = 1.0 if ex_mode == 'top1' else 0.0
    m['spod/exempt_k'] = float(_exempt_k())

    r = rewards.detach().float().clone()
    B = r.numel()
    if B == 0 or len(texts) != B:
        return rewards, m
    # beta == 0 is intentionally not early-returned here: that is observer mode, which must run the full stats and dumping (d is always 0, scores unchanged).
    m['spod/enabled'] = 1.0

    k = group_size or _group()
    if k > 1 and B % k == 0:
        m['spod/group_ok'] = 1.0
    else:
        # the reward worker in practice receives only one group per call (see the cdiv_reward record),
        # so when it is not evenly divisible, treat the whole batch as one group rather than silently failing.
        k = B

    # first parse the categories of the whole batch: the logq convention needs the batch distribution to update the online exposure counts,
    # and must update before computing phi, otherwise the first collapse class encountered goes unpenalized that step.
    cats_all = []
    n_unk = 0
    for j in range(B):
        c, ok = spod_category(texts[j])
        if not ok:
            n_unk += 1
        cats_all.append(c)
    if pop_mode == 'logq':
        _bump_pop(cats_all)

    if pop_mode == 'logq':
        phi_tab, p_hat, pop_ent = spod_phi_now()
        m['spod/pop_entropy'] = pop_ent
        phi_default = 0.0   # under logq, UNK is treated as the "distribution mean", i.e. phi=0 unchanged
    else:
        phi_tab = spod_phi_table()
        vals = list(phi_tab.values())
        phi_default = sum(vals) / max(len(vals), 1)   # UNK treated as average popularity
    for c, v in phi_tab.items():
        m[f'spod/phi_{c}'] = float(v)

    out = r.clone()
    n_cat, n_exempt, n_rank_chg, n_grp, n_top1_keep, n_clip, n_pos = 0, 0, 0, 0, 0, 0, 0
    phi_sum, delta_sum, delta_max, ent_sum = 0.0, 0.0, 0.0, 0.0
    share = {c: 0 for c in _DEFAULT_FREQ}
    detail_rows = []
    log_detail = os.environ.get('SPOD_LOG_DETAIL', '1') == '1'

    for st in range(0, B, k):
        seg = slice(st, st + k)
        rg = r[seg]
        kk = rg.numel()
        if kk == 0:
            continue
        n_grp += 1
        std = float(rg.std(unbiased=False)) if kk > 1 else 0.0

        by_cat = {}
        for j in range(kk):
            c = cats_all[st + j]
            by_cat.setdefault(c, []).append(j)
            if c in share:
                share[c] += 1
        n_cat += len(by_cat)
        # in-group category entropy: a direct read of diversity, finer than "how many classes hit" (it can distinguish 4+4 from 7+1)
        for idxs in by_cat.values():
            q = len(idxs) / float(kk)
            ent_sum -= q * math.log(max(q, 1e-12))

        for c, idxs in by_cat.items():
            if pop_mode == 'group':
                # in-group adaptive: penalize in proportion to the class's share within the group (a DRA-flavored comparison arm)
                phi = len(idxs) / float(kk)
            else:
                phi = float(phi_tab.get(c, phi_default))
            # the top ex_k highest-scoring items within the class: SPOD's core -- they are not penalized, so the direction's peak quality is preserved.
            # ex_k=1 is the main scheme; ex_k=2/3 are ablation levels; ex_k=0 (none) exempts none.
            order = sorted(idxs, key=lambda j: -float(rg[j]))
            star = order[0]
            stars = set(order[:ex_k])
            for j in idxs:
                r_raw = float(rg[j])
                is_star = (j in stars)
                if is_star:
                    n_exempt += 1
                    d_eff, clipped = 0.0, False
                else:
                    d = beta * std * phi
                    # min constraint: equivalent to r - max(d, 0), i.e. no item may be lifted above its raw score
                    d_eff = max(d, 0.0) if use_min else d
                    clipped = bool(use_min and d < 0.0)
                    out[st + j] = r_raw - d_eff
                    phi_sum += phi
                    delta_sum += d_eff
                    delta_max = max(delta_max, d_eff)
                    if phi > 0:
                        n_pos += 1
                    if clipped:
                        n_clip += 1
                if log_detail:
                    detail_rows.append({
                        'call': _STATE['calls'], 'grp': st // max(k, 1), 'i': j,
                        'cat': c, 'r_raw': round(r_raw, 4),
                        'top1': int(j == star), 'exempt': int(is_star),
                        'phi': round(float(phi), 4), 'std': round(std, 4),
                        'delta': round(d_eff, 4),
                        'r_out': round(float(out[st + j]), 4),
                        'clip': int(clipped),
                    })

        # in-group ordering change / whether the peak is preserved
        og = out[seg]
        if kk > 1:
            n_rank_chg += int((rg.argsort(descending=True) != og.argsort(descending=True)).sum())
            if int(rg.argmax()) == int(og.argmax()):
                n_top1_keep += 1

    n_pen = max(B - n_exempt, 1)
    m['spod/n_cat_mean'] = n_cat / max(n_grp, 1)
    m['spod/unk_frac'] = n_unk / B
    # all parses failed = SPOD silently ineffective (phi all falls back to default, which under rmscore is 0, nothing moves).
    # The most common cause is the phi table not matching the dataset (the category set is inconsistent with the label system of the generated text).
    # Looking only at delta_mean cannot reveal this (it is also 0), so here we print a line directly.
    if n_unk == B and _STATE['calls'] <= 3:
        print(f'[SPOD] WARNING: all {B} items in this batch failed category parsing (catmode={_catmode()} '
              f'phi_src={_PHI_SRC} category_set={sorted(_DEFAULT_FREQ)}) -- phi all falls back to default, '
              f'SPOD is effectively inactive; first check whether SPOD_PHI_SRC / SPOD_CATMODE is misconfigured', flush=True)
    m['spod/exempt_frac'] = n_exempt / B
    m['spod/phi_mean'] = phi_sum / n_pen
    m['spod/phi_pos_frac'] = n_pos / n_pen
    m['spod/delta_mean'] = delta_sum / n_pen
    m['spod/delta_max'] = delta_max
    m['spod/min_clip_frac'] = n_clip / n_pen
    m['spod/rank_change_frac'] = n_rank_chg / B
    m['spod/top1_preserved'] = n_top1_keep / max(n_grp, 1)
    m['spod/cat_entropy'] = ent_sum / max(n_grp, 1)
    m['spod/r_raw_mean'] = float(r.mean())
    m['spod/r_out_mean'] = float(out.mean())
    for c in share:
        m[f'spod/share_{c}'] = share[c] / B
    sb = float(r.std(unbiased=False)) if B > 1 else 0.0
    sa = float(out.std(unbiased=False)) if B > 1 else 0.0
    m['spod/r_std_before'] = sb
    m['spod/r_std_after'] = sa
    m['spod/std_ratio'] = sa / sb if sb > 1e-8 else 0.0

    _STATE['calls'] += 1
    sk = _sink()
    if sk['ok']:
        _write(sk['metrics'], [dict(m, call=_STATE['calls'], B=B, k=k)])
        if detail_rows:
            _write(sk['detail'], detail_rows)
    if _STATE['calls'] % PRINT_EVERY == 1:
        sh = ' '.join(f"{c.split('_')[0]}={m[f'spod/share_{c}']:.2f}" for c in _DEFAULT_FREQ)
        ph = ' '.join(f"{c.split('_')[0]}={m[f'spod/phi_{c}']:+.2f}" for c in _DEFAULT_FREQ)
        print(f"[SPOD #{_STATE['calls']}] pop={pop_mode} beta={beta} min={int(use_min)} "
              f"exempt={ex_mode} k={k} n_cat={m['spod/n_cat_mean']:.2f} "
              f"ent={m['spod/cat_entropy']:.3f} unk={m['spod/unk_frac']:.2f}\n"
              f"    share  {sh}\n    phi    {ph}\n"
              f"    exempt={m['spod/exempt_frac']:.2f} phi_pos={m['spod/phi_pos_frac']:.2f} "
              f"clip={m['spod/min_clip_frac']:.2f} delta_mean={m['spod/delta_mean']:.4f} "
              f"delta_max={delta_max:.4f} std {sb:.3f}->{sa:.3f} "
              f"rank_chg={m['spod/rank_change_frac']:.2f} "
              f"top1_keep={m['spod/top1_preserved']:.2f}", flush=True)
    return out.to(rewards.dtype), m
