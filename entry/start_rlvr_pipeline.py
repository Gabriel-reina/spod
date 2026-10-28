import argparse

from dacite import from_dict, Config
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, open_dict

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    parser.add_argument("--tables", help="optional external table refs, consumed via data_args.", default=None)
    # The upstream model uri is assembled from the version variables at the top of the launch script
    # and passed in, to avoid duplicating version numbers in the yaml.
    # If the launcher splits the user_params token at the first '=', you can escape the '=' inside the
    # uri with '@' on the submit side; this entry restores it.
    parser.add_argument("--pretrain", help="override cfg.pretrain (model uri, '=' escaped as '@')", default=None)
    parser.add_argument("--reward_pretrain", help="override cfg.reward_pretrain (model uri, '=' escaped as '@')", default=None)
    # Multi-seed reproduction channel. seed is hard-coded in the yaml; if you switch seeds by editing the
    # yaml, forgetting to restore it afterwards makes all later submissions silently run with the new seed
    # and breaks reproduction of existing experiments. Default None => no override at all, follow the yaml.
    parser.add_argument("--seed", help="override cfg.seed (for multi-seed reproduction; follows yaml if not passed)", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_gpus_per_node", type=int, default=None)
    parser.add_argument("--validate_only", action="store_true", help="validate config without starting Ray or training")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)
    if args.pretrain:
        cfg.pretrain = args.pretrain.replace("@", "=")
    if args.reward_pretrain:
        cfg.reward_pretrain = args.reward_pretrain.replace("@", "=")  # the ${reward_pretrain} interpolation in rewards.*.model_name_or_path then takes effect
    if args.seed is not None:
        cfg.seed = int(args.seed)  # base_config propagates it to each worker's training_args.seed and the vllm engine

    if args.output_dir:
        import os
        cfg.output_dir = os.path.abspath(args.output_dir)
        cfg.logging_dir = os.path.join(cfg.output_dir, "logs")
        cfg.checkpoint_config.output_dir = os.path.join(cfg.output_dir, "ckpt")
        cfg.tracker_kwargs.log_dir = os.path.join(cfg.output_dir, "tensorboard")
    if args.num_gpus_per_node:
        cfg.num_gpus_per_node = args.num_gpus_per_node

    # Ray workers can outlive a launcher. Send this arm's switches explicitly,
    # rather than inheriting stale environment values from an existing Ray cluster.
    import os
    reward_env = dict(cfg.rewards.spod.get("system_envs", {}))
    reward_env.update({k: v for k, v in os.environ.items() if k.startswith("SPOD_")})
    with open_dict(cfg.rewards.spod):
        cfg.rewards.spod.system_envs = reward_env

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config: RLVRConfig = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    if args.validate_only:
        print("[SPOD_CONFIG_VALID]", ppo_config.seed, ppo_config.max_steps, flush=True)
        return

    init()
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
