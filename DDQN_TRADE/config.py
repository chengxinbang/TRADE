import os

# ===================== 硬件配置 =====================
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
NUM_WORKERS_PER_GPU = 1
MAX_PARALLEL_TASKS = len(AVAILABLE_GPUS) * NUM_WORKERS_PER_GPU

# ===================== 路径配置 =====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS_DIR = os.path.join(_THIS_DIR, "results")
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# ===================== 绘图输出配置 =====================
# 默认复用同一个目录，避免每次绘图都生成新的时间戳目录。
PLOT_USE_TIMESTAMP_DIR = True
PLOT_OUTPUT_SUBDIR = "plots_latest"


def get_env_paths(env_id):
    env_dir = os.path.join(BASE_RESULTS_DIR, env_id)
    pth_dir = os.path.join(env_dir, "checkpoints")

    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)

    return {
        "env_dir": env_dir,
        "pth_dir": pth_dir,
        "plot_root_dir": env_dir,
    }


# ===================== 论文口径实验配置 =====================
# The 30 Atari tasks reported in the manuscript's Table 1.
ENV_IDS = ['AlienNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CentipedeNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'NameThisGameNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4', 'VideoPinballNoFrameskip-v4']

SEEDS = [0, 1, 2, 3, 4]
FIXED_C = 30000

# ===================== 训练超参数（尽量贴近DQN/Double-DQN口径） =====================
TOTAL_STEPS = 10_000_000
BATCH_SIZE = 32
REPLAY_CAPACITY = 1_00_000
GAMMA = 0.99
TRAIN_FREQ = 4
WARMUP_STEPS = 50_000

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 100000

EVAL_INTERVAL = 100000
EVAL_EPISODES = 100
EVAL_MAX_FRAMES = 18_000
EVAL_EPSILON = 0.05

SAVE_INTERVAL = 10_000_000

ADAPTIVE_PARAMS = {
    "adaptive_C_min": 500,

}

# ===================== 性能开关（不改算法，仅影响运行效率） =====================
PERFORMANCE_OPTS = {
    "cudnn_benchmark": True,
    "torch_num_threads": 1,
    "pin_memory": True,
    "non_blocking": True,
    "mixed_precision": False,
    "deterministic": False,
}
