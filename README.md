# TRADE

This repository contains the DDQN/Atari implementation accompanying the paper **“Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning.”**

TRADE (Target Refresh via Adaptive Decay Estimation) adapts the target-network refresh interval by monitoring the empirical decay of the value-function fitting loss. The scheduler refreshes the target after the loss decay plateaus, reducing redundant optimization after local fitting saturation while leaving the rest of the learning pipeline unchanged.

## Paper

- [Main paper (PDF)](paper/TRADE_main.pdf)
- [Supplementary material (PDF)](paper/TRADE_appendix.pdf)

## Repository contents

- `main.py`: experiment launcher for fixed-period DDQN and TRADE.
- `config.py`: Atari environments, seeds, training hyperparameters, and hardware settings.
- `ddqn_agent.py`: DDQN agent and adaptive target-refresh logic.
- `train_worker.py`: training, evaluation, checkpointing, and multiprocessing worker code.
- `atari_wrappers.py`: Atari preprocessing and frame stacking.
- `networks.py` and `replay_buffer.py`: network and replay-buffer implementations.
- `plot_utils.py`, `plot_top.py`, and `plot_tau_scores.py`: plotting utilities.
- `extract_tau_scores.py`, `summarize_pkl_folder.py`, and `time_get.py`: result-analysis utilities.
- `sensitivity_adaptive_epsilon_tol.py`: sensitivity experiments for the adaptive threshold.

Training results, replay data, model checkpoints, and other generated artifacts are intentionally excluded from version control.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The code expects Gymnasium/ALE to expose Atari environment IDs such as `BreakoutNoFrameskip-v4`. Make sure that the Atari ROMs are installed and that you have accepted their license before running experiments.

## Usage

Choose the environments, methods, and random seeds on the command line. For example:

```bash
# Fixed target-network update baseline
python main.py --envs BreakoutNoFrameskip-v4 --modes ddqn --seeds 0 1 2

# TRADE adaptive target refresh
python main.py --envs BreakoutNoFrameskip-v4 --modes hard_adaptive --seeds 0 1 2

# Override the minimum adaptive refresh interval
python main.py --envs BreakoutNoFrameskip-v4 --modes hard_adaptive \
  --adaptive-c-min 200 --adaptive-epsilon-tol 1e-3 --seeds 0 1 2
```

Edit `config.py` to select GPUs, change the full environment list, or adjust training and evaluation hyperparameters. By default, generated files are written below `ddqn_adaptive_results/`.

## Notes

- The default configuration is designed for multi-GPU experiments. On a machine with fewer GPUs, update `AVAILABLE_GPUS` in `config.py`.
- Full paper experiments are computationally intensive: the default training budget is 10 million environment steps per run.
- Analysis and plotting scripts consume the `.pkl` result files produced during training; those generated files are not included in this repository.

## Citation

If this code is useful in your research, please cite:

```bibtex
@article{zhang2026trade,
  title   = {Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning},
  author  = {Zhang, Hongming and Cheng, Xinbang and Shi, Rongye and Albrecht, Stefano V. and Bai, Fengshuo and Xu, Bo and M\"{u}ller, Martin},
  year    = {2026}
}
```
