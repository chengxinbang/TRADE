"""Run the SAC baseline and SAC+TRADE."""
import argparse
import json
from dataclasses import asdict

from config import SEEDS, MODES, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dmc", "metaworld", "mujoco"], required=True)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--c-min", type=int, default=100, help="TRADE minimum interval in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = [configuration(args.suite, env, mode, seed, args.total_steps,
                             args.c_min, args.eval_episodes)
               for env in args.envs for mode in args.modes for seed in args.seeds]
    if args.dry_run:
        print(json.dumps([asdict(c) for c in configs], indent=2))
        return
    import torch
    from train_worker import train
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    for cfg in configs:
        train(cfg, device, args.output)


if __name__ == "__main__":
    main()
