"""Run both DDQN worker modes briefly in a real Atari environment."""
from pathlib import Path
from queue import Queue
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"DDQN_TRADE"))
import config
from train_worker import run_single_training


def main():
    base=ROOT/"tests/.smoke"
    assert base.resolve().is_relative_to(ROOT.resolve())
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddqn-",dir=base) as output:
        config.BASE_RESULTS_DIR=output
        config.WARMUP_STEPS=4
        config.BATCH_SIZE=2
        config.REPLAY_CAPACITY=32
        config.EVAL_INTERVAL=8
        config.EVAL_EPISODES=1
        config.EVAL_MAX_FRAMES=8
        tasks,results=Queue(),Queue()
        env="BreakoutNoFrameskip-v4"
        for fixed in [True,False]:
            tasks.put((env,0,fixed,config.get_env_paths(env),4,16))
        tasks.put(None)
        run_single_training(tasks,results,None)
        completed=[]
        while not results.empty():
            result=results.get()
            if result is None: continue
            if result.get("status")=="error": raise RuntimeError(result)
            assert list(result["step_records"])==[8,16]
            assert result["total_steps"]==16
            completed.append(result)
        assert len(completed)==2
        print("Passed both DDQN worker modes")


if __name__=="__main__": main()
