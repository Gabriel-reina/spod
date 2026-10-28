"""Discriminative RM reward worker (the SPOD injection point).

Reuses base_worker.RewardWorker for model loading and scoring (the last-position logit of
TokenClassification), and only fills in the `scores` field required by the RLVR DynamicSamplingScheduler --
the generic RewardWorker returns only token_level_rewards / response_level_rewards, and the missing scores
would raise a KeyError in collect_items_as_batch.

The SPOD injection point is here too. Why it must be in the reward worker and not in loss_func: SPOD needs
to identify "the top-RM-scoring item within a class", to compute the in-group std, and (schemes B/C) to take
the min against the raw score -- all three need the **pre-whitening** raw RM scores, and need to see the full
k items of the same group at once. The reward worker satisfies both conditions (the k items of a group are
laid out contiguously in the batch, matching reward_norm's reshape(-1, n_sample)); loss_func can only see a
single item within a micro-batch, and the advantage has already been in-group whitened.
"""
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import RewardWorker


class SpodRMRewardWorker(RewardWorker):

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        output = RewardWorker.compute_rewards.__wrapped__(self, data)
        metrics = dict(output.meta_info.get("metrics", {})) if output.meta_info else {}

        # SPOD: remove the RM's popularity bias per semantic category, exempting the single top-RM-scoring item of each class.
        #   d_i  = beta * std(r_group) * phi(c_i)
        #   r'_i = r_i (in-class top1) / r_i - d_i (scheme A) / min(r_i, r_i - d_i) (schemes B/C)
        # The RM per-category means have a systematic gap, and it is negatively correlated with training frequency --
        # the RM's bias is not frequency-driven, so the rmscore convention directly uses the measured means as a z-score,
        # and the freq convention is kept as a comparison arm.
        # On any exception, always fall back to a constant key set: inconsistent keys would make the driver-side reduce_metrics
        # form a jagged array, and np.mean would raise ValueError outright and kill training.
        try:
            from roll.utils.spod_reward import spod_enabled
            if spod_enabled():
                from roll.utils.spod_reward import compute_spod_rewards
                spod_txt = self.tokenizer.batch_decode(
                    data.batch["responses"], skip_special_tokens=True)
                new_r, spod_metrics = compute_spod_rewards(
                    output.batch["response_level_rewards"], spod_txt)
                output.batch["response_level_rewards"] = new_r
                metrics.update(spod_metrics)
        except Exception as exc:
            print(f"[SPOD] exception: {exc}", flush=True)
            from roll.utils.spod_reward import spod_base_metrics
            spod_metrics = spod_base_metrics()
            spod_metrics['spod/exit_code'] = 6.0
            metrics.update(spod_metrics)

        # the scheduler side records the sequence score by scores (and uses it for metrics / dynamic sampling).
        # Must clone after SPOD: scores must match the reward actually used for training, otherwise
        # the score_mean in the log is the pre-debias score and will not line up with the advantage.
        output.batch["scores"] = output.batch["response_level_rewards"].clone()
        output.meta_info = dict(output.meta_info or {})
        output.meta_info["metrics"] = metrics
        return output
