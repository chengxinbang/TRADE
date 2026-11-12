# TRADE

Implementation of **Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning** by Hongming Zhang, Xinbang Cheng, Rongye Shi, Stefano V. Albrecht, Fengshuo Bai, Bo Xu, and Martin Müller.

The code is organized by learning algorithm:

```text
TRADE/
├── DDQN_TRADE/    # DDQN and DDQN+TRADE on Atari
├── SAC_TRADE/     # SAC and SAC+TRADE on DMC, Meta-World, and MuJoCo
└── tests/         # Scheduler, target-update, configuration, and environment checks
```

## Target refresh

The scheduler follows the current manuscript's inner–outer target-fitting protocol. It observes the **pre-optimizer fitting loss**, smooths that loss, and computes the score in **environment steps**, not gradient-update counts:

```text
L_m(0) = S_m,0 = max(first_fitting_loss, 1e-8)
S_m,j = 0.99 * S_m,j-1 + 0.01 * max(current_fitting_loss, 1e-8)
g_m(k) = [log(S_m,j) - log(L_m(0))] / (2k)
```

The initial loss is fixed within a target-freezing cycle. After `k >= C_min`, record the minimum score and refresh on a **strict increase** `g_m(k) > g_best`. A refresh copies the complete target module and resets the initial loss, EMA, and minimum. There is no rebound tolerance, maximum interval, or automatic theoretical-constant estimate. DDQN monitors its squared TD fitting loss; SAC monitors the sum of both critic squared Bellman errors. Smoothing only changes the scheduler's input, not the optimizer's objective.

The first fitting cycle starts after random-action warm-up. Every subsequent interval includes the environment interactions since the last target copy, including interactions without a gradient update. Checkpoints preserve the scheduler state and carry the `loss-ema-env-step-v1` protocol identifier.

`C_min` is an empirical monitoring floor. Defaults are 500 environment steps for DDQN+TRADE and 100 for SAC+TRADE, within the manuscript's respective 100–500 and 100–200 ranges. Both are configurable for sensitivity sweeps. The DDQN baseline keeps its separate fixed interval `C = 30,000` environment steps.

## Run

Install PyTorch for your machine, then the requirements of the implementation you use. Python 3.10–3.12 is recommended. Each algorithm folder runs independently; see its README for details.

```bash
cd DDQN_TRADE
pip install -r requirements.txt
python main.py --envs FrostbiteNoFrameskip-v4 --modes ddqn hard_adaptive --seeds 0 1 2 3 4 --adaptive-c-min 500
```

```bash
cd SAC_TRADE
pip install -r requirements.txt
python main.py --suite dmc --envs cheetah_run --modes soft hard_adaptive --seeds 0 1 2 3 4 --c-min 100
```

For Meta-World, install `SAC_TRADE/requirements-metaworld.txt`. This pins an official revision with **v2** tasks; the adapter never substitutes a v3 task. DMC uses native low-dimensional observations, and Meta-World uses 39-dimensional observations, native rewards, and `info['success']` for evaluation. All methods in an algorithm share the same backbone and non-target settings.

## Checks

From the repository root, with both implementations' dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The checks cover the EMA, strict rebound, environment-step clock, reset and checkpoint behavior, frozen/copy/Polyak target updates, paper defaults, and the real Atari/DMC/Meta-World adapters when their optional dependencies are installed. Short end-to-end training checks are provided in `tests/smoke_ddqn.py` and `tests/smoke_training.py`.

Generated results, checkpoints, and caches are excluded from the repository. Plots use every completed run, without selecting seeds by performance. Log the resolved configuration and software versions for each run.

## Citation

```bibtex
@article{zhang2026trade,
  title={Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning},
  author={Zhang, Hongming and Cheng, Xinbang and Shi, Rongye and Albrecht, Stefano V. and Bai, Fengshuo and Xu, Bo and M{\"u}ller, Martin},
  year={2026}
}
```
