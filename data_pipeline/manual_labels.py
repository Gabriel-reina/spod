#!/usr/bin/env python3
"""
D2 manual review labels (a pure-data module, no logic).

Review method: read the Description of each of the 105 records one by one (read the PR body when needed), and manually assign three fields:
  primary  : primary category. Pick the category of "the change most likely to dominate the benefit in this record"
  allcats  : **all** categories this record touches (multi-label)
  n_change : the number of independent changes packed into the same record (counted from the parallel items in the description / the numbered list in the body)

Why multi-label is mandatory: the keyword rules and my previous round both assumed "one record = one idea = one category".
Reading them one by one showed this is wrong —— many records pack multiple changes into one submission, e.g.
  #20 "Merged QKV weights, long-short attention, attention scale, lower Adam epsilon, batched Muon"
one record contains five changes across C3/C4/C2. On these items single-label is not "labeled inaccurately", it is **undefined by construction**.

Three special flags (not technical categories, must be removed from the idea pool):
  PORT    : carrying over an existing result from another track, not a new idea
  COMBINE : merging two existing records, not a new idea
"""

# id -> (primary, allcats, n_change)
RECORDS = {
    "t1_r2":  ("C2_optimizer",   ["C2_optimizer", "C4_attention"], 2),
    "t1_r3":  ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r4":  ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r5":  ("C3_architecture", ["C3_architecture", "C4_attention"], 4),
    "t1_r6":  ("C1_sys_kernel",  ["C1_sys_kernel", "C2_optimizer"], 1),
    "t1_r7":  ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r8":  ("C3_architecture", ["C3_architecture"], 1),
    "t1_r9":  ("C3_architecture", ["C3_architecture", "C2_optimizer", "C5_loss"], 3),
    "t1_r10": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r11": ("C3_architecture", ["C3_architecture", "C2_optimizer"], 2),
    "t1_r12": ("C4_attention",   ["C4_attention"], 1),
    "t1_r13": ("C4_attention",   ["C4_attention", "C2_optimizer"], 1),
    "t1_r14": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r15": ("C3_architecture", ["C3_architecture", "C1_sys_kernel"], 2),
    "t1_r16": ("C4_attention",   ["C3_architecture", "C4_attention"], 3),
    "t1_r17": ("C3_architecture", ["C3_architecture", "C4_attention"], 3),
    "t1_r18": ("C5_loss",        ["C5_loss"], 1),
    "t1_r19": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss", "C2_optimizer"], 3),
    "t1_r20": ("C4_attention",   ["C3_architecture", "C4_attention", "C2_optimizer"], 5),
    "t1_r21": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r22": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r23": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r24": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r25": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r26": ("C2_optimizer",   ["C1_sys_kernel", "C2_optimizer"], 2),
    "t1_r27": ("C1_sys_kernel",  ["C1_sys_kernel", "C3_architecture"], 2),
    "t1_r28": ("C4_attention",   ["C4_attention", "C3_architecture"], 1),
    "t1_r29": ("C4_attention",   ["C1_sys_kernel", "C4_attention", "C2_optimizer"], 3),
    "t1_r30": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r31": ("C4_attention",   ["C4_attention"], 1),
    "t1_r32": ("C1_sys_kernel",  ["C1_sys_kernel", "C3_architecture"], 3),
    "t1_r33": ("C1_sys_kernel",  ["C1_sys_kernel", "C4_attention"], 2),
    "t1_r34": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r35": ("C3_architecture", ["C3_architecture", "C4_attention", "C2_optimizer"], 3),
    "t1_r36": ("C1_sys_kernel",  ["C1_sys_kernel", "C2_optimizer"], 2),
    "t1_r37": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss"], 1),
    "t1_r38": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r39": ("C2_optimizer",   ["C2_optimizer"], 2),
    "t1_r40": ("C2_optimizer",   ["C2_optimizer", "C3_architecture"], 3),
    "t1_r41": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r42": ("C2_optimizer",   ["C2_optimizer"], 2),
    "t1_r43": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r44": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r45": ("C3_architecture", ["C3_architecture"], 2),
    "t1_r46": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r47": ("C4_attention",   ["C4_attention", "C2_optimizer"], 2),
    "t1_r48": ("C2_optimizer",   ["C2_optimizer", "C1_sys_kernel", "C3_architecture"], 5),
    "t1_r49": ("C4_attention",   ["C4_attention", "C3_architecture", "C1_sys_kernel"], 5),
    "t1_r50": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t1_r51": ("C3_architecture", ["C3_architecture", "C1_sys_kernel"], 2),
    "t1_r52": ("C2_optimizer",   ["C2_optimizer", "C1_sys_kernel"], 4),
    "t1_r53": ("C5_loss",        ["C5_loss", "C3_architecture", "C2_optimizer"], 4),
    "t1_r54": ("C5_loss",        ["C5_loss"], 1),
    "t1_r55": ("C3_architecture", ["C3_architecture"], 2),
    "t1_r56": ("C1_sys_kernel",  ["C1_sys_kernel", "C2_optimizer"], 3),
    "t1_r57": ("C1_sys_kernel",  ["C1_sys_kernel", "C2_optimizer"], 4),
    "t1_r58": ("C4_attention",   ["C4_attention"], 1),
    "t1_r59": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r60": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss"], 1),
    "t1_r61": ("C1_sys_kernel",  ["C1_sys_kernel", "C2_optimizer", "C3_architecture"], 2),
    "t1_r62": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r63": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r64": ("C3_architecture", ["C3_architecture", "C4_attention"], 1),
    "t1_r65": ("C1_sys_kernel",  ["C1_sys_kernel", "C3_architecture"], 1),
    "t1_r66": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r67": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss"], 2),
    "t1_r68": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r69": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r70": ("C3_architecture", ["C3_architecture"], 2),
    "t1_r71": ("C1_sys_kernel",  ["C1_sys_kernel"], 2),
    "t1_r72": ("C2_optimizer",   ["C2_optimizer", "C4_attention"], 2),
    "t1_r73": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r74": ("C1_sys_kernel",  ["C1_sys_kernel", "C3_architecture", "C4_attention"], 3),
    "t1_r75": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss"], 1),
    "t1_r76": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r77": ("C3_architecture", ["C3_architecture", "C1_sys_kernel"], 1),
    "t1_r78": ("C4_attention",   ["C4_attention", "C1_sys_kernel"], 1),
    "t1_r79": ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss"], 1),
    "t1_r80": ("C2_optimizer",   ["C2_optimizer", "C4_attention"], 1),
    "t1_r81": ("C3_architecture", ["C3_architecture"], 1),
    "t1_r82": ("C4_attention",   ["C4_attention", "C3_architecture"], 1),
    "t1_r83": ("C3_architecture", ["C3_architecture"], 2),
    "t1_r84": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r85": ("C3_architecture", ["C3_architecture", "C4_attention"], 2),
    "t1_r86": ("C1_sys_kernel",  ["C1_sys_kernel", "C4_attention"], 1),
    "t1_r87": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t1_r88": ("C5_loss",        ["C5_loss"], 1),
    "t1_r89": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    # ---- track 2 ----
    "t2_r2":  ("PORT",           [], 0),   # bulk-transfer the short track's result to the medium track
    "t2_r3":  ("C2_optimizer",   ["C2_optimizer"], 1),
    "t2_r4":  ("C2_optimizer",   ["C2_optimizer"], 1),
    "t2_r5":  ("C2_optimizer",   ["C2_optimizer"], 1),
    "t2_r6":  ("C2_optimizer",   ["C2_optimizer", "C4_attention", "C1_sys_kernel", "C3_architecture"], 4),
    "t2_r7":  ("C1_sys_kernel",  ["C1_sys_kernel", "C5_loss", "C2_optimizer", "C3_architecture"], 4),
    "t2_r8":  ("C4_attention",   ["C4_attention", "C2_optimizer"], 2),
    "t2_r9":  ("C3_architecture", ["C3_architecture"], 1),
    "t2_r10": ("C3_architecture", ["C3_architecture"], 1),
    "t2_r11": ("C1_sys_kernel",  ["C1_sys_kernel"], 1),
    "t2_r12": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t2_r13": ("C2_optimizer",   ["C2_optimizer"], 1),
    "t2_r14": ("COMBINE",        [], 0),   # "Combine both records 12 & 13", not a new idea
    "t2_r15": ("C3_architecture", ["C3_architecture"], 1),
    "t2_r16": ("C5_loss",        ["C5_loss", "C3_architecture"], 1),
    "t2_r17": ("C1_sys_kernel",  ["C1_sys_kernel", "C4_attention"], 1),
    "t2_r18": ("PORT",           [], 0),   # "Bulk transfer short track features"
}

# ---------------------------------------------------------------------------
# reject pool (85 closed-unmerged PRs) split three ways.
# After reading the title + body of each one, it turned out that treating all 85 as "failed improvement ideas" is wrong.
#   tech  : a genuine technical improvement attempt that was rejected —— this is the hard negative we want
#   noise : docs / CI / cleanup / accidentally-opened PR / hardware port / task switch, unrelated to idea quality
#           keeping them would let the RM learn the trivial task of "telling a README change from a technical change", inflating performance
#   dup   : consecutive patches on the same branch, not an independent idea
# ---------------------------------------------------------------------------
REJECT_NOISE = {
    30, 36, 38, 41, 46, 52, 59, 63, 64, 66, 67, 90, 95, 105, 110, 121,
    189, 192, 195, 202, 222, 226, 228, 236, 237, 242, 262, 289, 319, 320, 329, 356,
}
REJECT_DUP = {165, 225, 227, 229}

# ---------------------------------------------------------------------------
# idea-level positive/negative conflict: the same technical content was both merged as a record (positive/null)
# and had a PR closed (negative). The reason is that an early implementation was rejected and a later, better one was adopted.
# ⇒ reject does not mean "the idea is invalid", only "this submission was not adopted".
# If not handled, the RM would be fed self-contradictory labels.
# Format: pr number -> (the conflicting record id, idea name)
# ---------------------------------------------------------------------------
REJECT_CONTESTED = {
    141: ("t1_r41", "NorMuon"),
    147: ("t1_r43", "Cautious Weight Decay"),
    208: ("t1_r63", "Untie Value Embeddings"),
    219: ("t1_r71", "Sparse bigram gradient comms"),
    255: ("t1_r82", "Exclusive Self Attention (XSA)"),
    260: ("t1_r86", "Exclusive Self Attention (XSA)"),
    86:  ("t2_r4",  "Optimize Newton-Schulz coefficients"),
    16:  ("t1_r4",  "Faster Muon implementation"),
}
