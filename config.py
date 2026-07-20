import os

# ===================== 硬件配置 =====================
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
NUM_WORKERS_PER_GPU = 1
MAX_PARALLEL_TASKS = len(AVAILABLE_GPUS) * NUM_WORKERS_PER_GPU

# ===================== 路径配置 =====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS_DIR = os.path.join(_THIS_DIR, "ddqn_adaptive_results")
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
# 49个Atari游戏，按简单到复杂的经验顺序排列（先Pong/Breakout等）
ENV_IDS = [
    # "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4", #已跑出
    "FreewayNoFrameskip-v4", #已跑出
    "BoxingNoFrameskip-v4", #已跑出
    "TennisNoFrameskip-v4", #已跑出
    "EnduroNoFrameskip-v4", #DDQN5
    "VideoPinballNoFrameskip-v4", #DDQN6
    "BeamRiderNoFrameskip-v4", #DDQN7
    "SpaceInvadersNoFrameskip-v4", #DDQN8
    "AssaultNoFrameskip-v4", #DDQN9
    "AsterixNoFrameskip-v4",
    "AlienNoFrameskip-v4", #0528-DDQN1
    "AmidarNoFrameskip-v4", #0528-DDQN1
    "BankHeistNoFrameskip-v4", #A800
    "BowlingNoFrameskip-v4", #A800
    "BattleZoneNoFrameskip-v4",#0529-DDQN1
    "CentipedeNoFrameskip-v4", #0529-DDQN1
    "DemonAttackNoFrameskip-v4", #A800
    "FishingDerbyNoFrameskip-v4", #A800
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    # "NameThisGameNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    # "RoadRunnerNoFrameskip-v4",
    # "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    # "StarGunnerNoFrameskip-v4",
    # "TimePilotNoFrameskip-v4",
    # "TutankhamNoFrameskip-v4",
    # "UpNDownNoFrameskip-v4",
    # "WizardOfWorNoFrameskip-v4",
    # "ZaxxonNoFrameskip-v4",
    # "AtlantisNoFrameskip-v4",
    # "AsteroidsNoFrameskip-v4",
    # "ChopperCommandNoFrameskip-v4",
    # "CrazyClimberNoFrameskip-v4",
    # "DoubleDunkNoFrameskip-v4",
    # "GravitarNoFrameskip-v4",
    # "PrivateEyeNoFrameskip-v4",
    # "VentureNoFrameskip-v4",
    # "MontezumaRevengeNoFrameskip-v4",
]

# ENV_IDS = [
#     # "BreakoutNoFrameskip-v4",      # 打砖块：DQN开山之作必用，最经典基准
#     # "SpaceInvadersNoFrameskip-v4", # 太空侵略者：DQN开山之作必用，最经典基准
#     # "MsPacmanNoFrameskip-v4",      # 吃豆人小姐：几乎所有Atari论文都会用
#     # "QbertNoFrameskip-v4",         # Q伯特：经典跳跃类游戏
#     # "RiverraidNoFrameskip-v4",     # 河流突袭：经典纵向射击游戏
#     # "SeaquestNoFrameskip-v4",      # 海洋探险：经典潜水艇射击游戏
#     # "BeamRiderNoFrameskip-v4",     # 光束骑手：经典太空射击游戏
#     "EnduroNoFrameskip-v4",        # 耐力赛：经典赛车游戏
#     "AssaultNoFrameskip-v4",       # 突袭：经典射击游戏
#     "VideoPinballNoFrameskip-v4",  # 弹珠台：独特的物理模拟游戏
#     "BoxingNoFrameskip-v4",        # 拳击：经典对抗类游戏
#     "TennisNoFrameskip-v4",        # 网球：经典对抗类游戏
#     "AsterixNoFrameskip-v4",       # 高卢英雄：经典横版过关游戏
#     "AlienNoFrameskip-v4",         # 异形：经典射击游戏
#     "CentipedeNoFrameskip-v4",     # 蜈蚣：经典射击游戏
# ]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
EVAL_EPISODES = 30
EVAL_MAX_FRAMES = 18_000
EVAL_EPSILON = 0.05

SAVE_INTERVAL = 10_000_000

ADAPTIVE_PARAMS = {
    "adaptive_C_min": 500,
    "adaptive_epsilon_tol": 1e-3,
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
