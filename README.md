# TRADE

Official implementation of **TRADE (Target Refresh via Adaptive Decay Estimation)**, the adaptive target-network scheduler introduced in:

> **Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning**<br>
> Hongming Zhang, Xinbang Cheng, Rongye Shi, Stefano V. Albrecht, Fengshuo Bai, Bo Xu, and Martin Müller.

## Overview

Target networks stabilize deep reinforcement learning by keeping the value target fixed while the online value function is optimized. Most algorithms refresh the target with a static rule, such as a fixed-period hard copy or a constant-rate exponential moving average. A static interval creates a stability-efficiency trade-off:

- refreshing too early propagates a value function that has not sufficiently fitted the current TD target;
- refreshing too late spends additional updates fitting a stale target after the local regression has already saturated.

TRADE adapts only **when** the target network is refreshed. It does not change the backbone architecture, Bellman target, optimizer, replay buffer, or exploration rule. The scheduler reuses the value or critic loss already produced by the base learner, so its computational overhead is small.

## Algorithm

TRADE views target-network learning as a bilevel process:

1. **Inner target-fitting phase:** freeze the target parameters and optimize the online value/critic network against the resulting TD targets.
2. **Outer Bellman refresh:** copy the fitted online parameters to the target network and begin a new fitting cycle.

Let `k` be the number of environment interactions since the last target refresh. At the start of a cycle, TRADE records an initial diagnostic fitting loss `L0`. With the target still frozen, it later evaluates `Lk` and computes the empirical log-decay score

```text
g_k = [log(L_k) - log(L_0)] / (2k).
```

A smaller, more negative score means faster fitting. As the loss approaches a noisy plateau, the score rebounds toward zero. TRADE therefore follows this event-triggered rule:

1. Do not refresh before the minimum stability interval `C_min`.
2. After `C_min`, track the best score `g_best` observed in the current cycle.
3. Refresh when `g_k > g_best + tau`.
4. Copy the online parameters to the target network and reset the cycle statistics.

The paper uses `tau = 1e-3` as the default tolerance. The included Atari implementation tracks the monotonic contraction estimate `rho_hat = exp(g_k)` and applies the same minimum-and-rebound logic through `epsilon_tol`.

If `C_min` is not specified, the implementation can estimate a conservative stability floor from the discount factor `gamma` and a prior contraction estimate `rho`:

```text
C_min = ceil(log((1 - gamma) / (1 + gamma)) / log(rho)).
```

## Runtime flow

For every environment interaction, the implementation:

1. collects a transition and stores it in replay memory;
2. samples a mini-batch after the warm-up period;
3. executes the original DDQN, SAC, or REDQ value/critic update;
4. records the fitting loss and updates the TRADE cycle statistics;
5. refreshes the target only when the stability floor and decay-rebound condition are both satisfied;
6. periodically evaluates the policy and saves checkpoints/results.

TRADE was evaluated in the paper with DDQN, SAC, REDQ, MR.Q, and BRO on Atari 2600, DeepMind Control Suite, MuJoCo, and Meta-World tasks.

## Repository layout

```text
TRADE/
├── README.md
└── Atari_DDQN/
    ├── README.md
    ├── requirements.txt
    ├── config.py
    ├── main.py
    ├── train_worker.py
    ├── ddqn_agent.py
    ├── networks.py
    ├── replay_buffer.py
    ├── atari_wrappers.py
    └── plot_utils.py
```

The current public snapshot contains the Atari DDQN implementation. The continuous-control environment profiles below document the SAC and REDQ setups used by the broader TRADE experiments.

## Environment setup

Use an isolated Conda environment. Install the PyTorch build that matches your operating system, GPU driver, and CUDA runtime using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

### Atari DDQN

Python 3.10 is recommended for the included Atari implementation.

```bash
conda create -n trade-atari python=3.10 pip -y
conda activate trade-atari
python -m pip install --upgrade pip setuptools wheel

# Install the appropriate CPU/CUDA build of PyTorch first, then:
pip install -r Atari_DDQN/requirements.txt
```

Gymnasium uses ALE for Atari environments. Installing the Atari and ROM-license extras is required; by doing so, you confirm that you have the right to use the ROMs. See the [Gymnasium Atari documentation](https://gymnasium.farama.org/environments/atari/) for details.

The current configuration uses legacy IDs such as `BreakoutNoFrameskip-v4`. Use a Gymnasium/ALE combination that exposes these IDs; do not silently replace them with newer IDs without validating the preprocessing and frame-skip settings.

Verify the Python dependencies:

```bash
python -c "import torch, gymnasium, cv2, numpy, matplotlib, tqdm; print('Atari dependencies OK')"
```

### SAC with DeepMind Control Suite

The SAC/DMC experiments used Python 3.10 and NumPy 1.26.4.

```bash
conda create -n trade-sac-dmc python=3.10 pip -y
conda activate trade-sac-dmc
python -m pip install --upgrade pip setuptools wheel

# Install the appropriate PyTorch build first, then:
pip install numpy==1.26.4 matplotlib tqdm gymnasium dm-control
```

Verify DMC:

```bash
python -c "from dm_control import suite; suite.load(domain_name='walker', task_name='walk'); print('dm_control OK')"
```

On a headless Linux server with a compatible NVIDIA/EGL setup:

```bash
export MUJOCO_GL=egl
```

`dm_control` supports GLFW, EGL, and OSMesa rendering backends; see the [official dm_control documentation](https://github.com/google-deepmind/dm_control) for platform-specific requirements.

### REDQ with MuJoCo/DMC

The REDQ research environment used Python 3.11, PyTorch 2.8.0, NumPy `<2.3`, MuJoCo 3.3.6, and dm-control 1.0.34. Use the PyTorch wheel appropriate for your machine rather than copying a CUDA-specific URL blindly.

```bash
conda create -n trade-redq python=3.11 pip -y
conda activate trade-redq
python -m pip install --upgrade pip setuptools wheel

# Install the appropriate PyTorch build first, then:
pip install "numpy<2.3" matplotlib tqdm scipy joblib
pip install mujoco==3.3.6 dm-control==1.0.34 "shimmy[dm-control]" "gymnasium[mujoco]"
pip install PyOpenGL PyOpenGL_accelerate
```

Gymnasium's MuJoCo dependencies can also be installed through `gymnasium[mujoco]`; see the [official Gymnasium MuJoCo documentation](https://gymnasium.farama.org/environments/mujoco/).

For headless EGL rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Meta-World

The TRADE Meta-World experiments use low-dimensional v2 task names such as `button-press-topdown-v2`. Install Meta-World only in the environment used for those experiments:

```bash
pip install metaworld gymnasium numpy torch tqdm
```

The current upstream Meta-World release uses newer v3 task names. For exact reproduction, use a Meta-World revision compatible with the v2 task API expected by the experiment code. Do not rename v2 tasks to v3 without revalidating observations, rewards, horizons, and benchmark semantics. See the [official Meta-World project](https://github.com/Farama-Foundation/Metaworld).

## Running Atari DDQN

Move into the included implementation:

```bash
cd Atari_DDQN
```

Run the fixed-period DDQN baseline:

```bash
python main.py --envs BreakoutNoFrameskip-v4 --modes ddqn --seeds 0
```

Run DDQN with TRADE:

```bash
python main.py \
  --envs BreakoutNoFrameskip-v4 \
  --modes hard_adaptive \
  --seeds 0 \
  --adaptive-c-min 500 \
  --adaptive-epsilon-tol 1e-3
```

Run both modes for multiple seeds:

```bash
python main.py \
  --envs BreakoutNoFrameskip-v4 \
  --modes ddqn hard_adaptive \
  --seeds 0 1 2
```

Important settings are defined in `Atari_DDQN/config.py`:

- `AVAILABLE_GPUS`: visible GPU indices used by workers;
- `ENV_IDS`: Atari task list;
- `SEEDS`: configured random seeds;
- `TOTAL_STEPS`: training budget;
- `FIXED_C`: fixed-update baseline interval;
- `ADAPTIVE_PARAMS`: `C_min` and decay-rebound tolerance;
- evaluation, replay-buffer, checkpoint, and performance settings.

Training outputs are written below `Atari_DDQN/ddqn_adaptive_results/` and are excluded from Git.

## Reproducibility notes

- Report results over multiple random seeds rather than a single run.
- Keep environment steps, evaluation intervals, and preprocessing identical when comparing fixed and adaptive target updates.
- Record the exact Python, PyTorch, CUDA, Gymnasium/ALE, MuJoCo, dm_control, and Meta-World versions used for a paper reproduction.
- On shared servers, set the GPU list in `config.py` to match the GPUs actually visible to the process.
- Generated checkpoints (`*.pth`) and result files (`*.pkl`) are intentionally not versioned.

## Citation

```bibtex
@article{zhang2026trade,
  title   = {Target-Network Refresh via Adaptive Decay Estimation in Deep Reinforcement Learning},
  author  = {Zhang, Hongming and Cheng, Xinbang and Shi, Rongye and Albrecht, Stefano V. and Bai, Fengshuo and Xu, Bo and M\"{u}ller, Martin},
  year    = {2026}
}
```
