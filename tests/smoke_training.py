"""Tiny functional runs; budgets here are for code checks, not paper scores."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"SAC_TRADE"))
from config import configuration
from train_worker import train


def main():
    base=ROOT/"tests/.smoke"
    assert base.resolve().is_relative_to(ROOT.resolve())
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sac-",dir=base) as output:
        for suite,env in [("dmc","cartpole_balance"),("metaworld","door-open-v2"),
                          ("mujoco","Hopper-v5")]:
            for mode in ["soft","hard_adaptive"]:
                cfg=configuration(suite,env,mode,0,total_steps=8,c_min=2,eval_episodes=1)
                cfg=replace(cfg,warmup_steps=2,batch_size=2,replay_capacity=32,
                            horizon=4,eval_interval=8)
                path=train(cfg,"cpu",output)
                assert (path/"curves.npz").exists() and (path/"summary.json").exists()
                print("Passed",suite,mode)


if __name__=="__main__": main()
