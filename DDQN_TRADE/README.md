# DDQN + TRADE

This folder contains the DDQN backbone, Atari preprocessing, parallel run launcher, result plotting, and the paper's loss-EMA target scheduler.

## Install and run

```bash
pip install -r requirements.txt
python main.py --envs FrostbiteNoFrameskip-v4 --modes ddqn hard_adaptive --seeds 0 1 2 3 4 --adaptive-c-min 500
python plot_utils.py --help
```

`--total-steps` overrides the 10M training budget. Other settings are centralized in `config.py`; the configured GPU list is filtered to devices actually available, and CPU execution is supported.

| Setting | Value |
|---|---|
| Training budget | 10,000,000 environment interactions |
| Random seeds | 0–4 |
| Replay / batch | 100,000 transitions / 32 |
| Optimizer / learning rate / discount | RMSprop / 0.00025 / 0.99 |
| Optimization cadence | One update every 4 interactions, after 50K warm-up steps |
| Exploration | Linear 1.0 → 0.1 over 100K interactions |
| Fixed baseline interval | 30,000 environment steps |
| TRADE floor | 500 environment steps by default; configurable |
| Scheduler | Loss EMA 0.99, numerical floor 1e-8, strict score rebound |
| Evaluation | 100 episodes every 100K interactions; epsilon-greedy 0.05 |
| Evaluation episode cap | 18,000 emulator frames |
| No-op starts / sticky actions | Up to 30 / disabled |
| Training / evaluation terminal | Life loss / game over or time limit |
| Frame skip / scheduling unit | 4 emulator frames / one agent decision |

Legacy `NoFrameskip-v4` names resolve to ALE v5 **with internal frame skip 1 and sticky probability 0**, followed by one external four-frame repeat. This prevents double frame skipping. Observations use two-frame maximum pooling, grayscale 84×84 images, and a four-observation stack. Training rewards are sign-clipped; evaluation uses raw game returns.

The 30 default tasks match the manuscript's Atari table. The same squared-TD objective and backbone settings are used in fixed and adaptive modes. `trade_scheduler.py` records intervals on the environment clock even though optimization is performed only every fourth step. Its first cycle starts at the first fitting update after warm-up. The legacy rho-tolerance trigger is replaced by the current manuscript rule, not retained as a hidden option.

Results are saved under `results/ENV/`; checkpoint and plot paths are managed by the existing launcher. New checkpoints retain the online/target networks, optimizer, scheduler, loss history, and refresh history. Incompatible legacy checkpoints are rejected. The DDQN runner rebuilds replay and resets the environment when resuming; a resumed run is therefore a restart from saved learner state, not a bitwise continuation.

## Files

- `main.py`, `train_worker.py`: launch and execute runs.
- `config.py`: task list, budgets, and hyperparameters.
- `ddqn_agent.py`, `networks.py`, `replay_buffer.py`: DDQN learner.
- `atari_wrappers.py`: separate training and evaluation protocols.
- `trade_scheduler.py`: loss EMA, score, reset, and serializable cycle state.
- `plot_utils.py`: plot all available runs.
