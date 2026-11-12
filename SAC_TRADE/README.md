# SAC + TRADE

The actor, double critic, replay buffer, Gaussian policy, and SAC losses are retained from the local SAC-DMC implementation. This folder provides one clean entry point for DMC, Meta-World v2, and continuous-action MuJoCo tasks.

## Install

```bash
pip install -r requirements.txt
# Required only for Meta-World v2:
pip install -r requirements-metaworld.txt
```

The pinned Meta-World commit `cca35cff0ec62f1a18b11440de6b09e2d10a1380` exposes v2 tasks with Gymnasium and MuJoCo 3. A missing v2 task raises an error rather than running a different v3 environment.

## Run

```bash
python main.py --suite dmc --envs cheetah_run humanoid_run --modes soft hard_adaptive --seeds 0 1 2 3 4 --c-min 100
python main.py --suite metaworld --envs door-open-v2 --total-steps 1000000 --modes soft hard_adaptive --seeds 0 1 2 3 4 --c-min 100
python main.py --suite mujoco --envs Hopper-v5 --modes soft hard_adaptive --seeds 0 1 2 3 4
```

`soft` is original SAC and `hard_adaptive` is SAC+TRADE. `--c-min` is the TRADE monitoring floor in environment steps, defaulting to 100. Each command executes the requested runs sequentially; independent commands can use separate GPUs with `--device cuda:N` and separate seed lists. `--dry-run` prints the resolved configurations without starting training.

| Setting | DMC | Meta-World |
|---|---|---|
| Default budget | 1M steps | 1M steps; use `--total-steps` for 1M–3M task budgets |
| Default seeds | 0–4 | 0–4 |
| Replay capacity / batch | 1M / 256 | 1M / 256 |
| Discount / actor, critic, alpha learning rates | 0.99 / 0.0003 | 0.99 / 0.0003 |
| Hidden architecture | Two 256-unit ReLU layers | Two 256-unit ReLU layers |
| Actor / critic / temperature updates | One per interaction after warm-up | One per interaction after warm-up |
| Initial entropy temperature | 0.1, auto-tuned | 1.0, auto-tuned |
| Random-action warm-up | 1K steps | 10K steps |
| Episode horizon | 1,000 steps | 500 steps |
| Evaluation interval | 20K steps | 10K steps |
| Evaluation | 10 episodes, policy mean | 10 episodes, policy mean, episode success |
| Soft target coefficient | 0.005 | 0.005 |
| TRADE | EMA 0.99 on summed critic loss, strict g rebound | Same |
| Default C_min | 100 environment steps | 100 environment steps |

Only the critic-target synchronization mechanism changes between modes. Actor and temperature updates continue at their original cadence. TRADE observes the sum of the two pre-optimizer critic MSEs; the initial loss is held fixed within each cycle, and EMA and minimum score reset on a hard copy. There is no tolerance, patience, maximum interval, or extra diagnostic batch. Time-limit truncations bootstrap; genuine terminals do not. Meta-World retains its native dense training reward and evaluates the maximum `info['success']` within each episode on ML1 test tasks.

## Outputs and plots

Every run writes to `results/SUITE/ENV/MODE/seed_N/`:

- `config.json`: resolved parameters, protocol, and software versions.
- `curves.npz`: evaluation steps/returns/success rates, loss observations, and refresh steps/intervals/scores.
- `summary.json`: completed budget, runtime, target update count, and final evaluation.
- `final.pt`: learner and scheduler state for inspection.

```bash
python plot_utils.py results/metaworld/door-open-v2 --metric success_rates
python plot_utils.py results/dmc/humanoid_run --metric returns
```

Plotting includes all completed runs and shows their mean with one sample standard deviation. No performance-based seed selection is performed. A completed output directory is not overwritten; choose a new `--output` for another configuration. The standalone runner starts fresh and does not implement training resumption.
