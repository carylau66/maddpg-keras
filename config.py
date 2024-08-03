
# 这个文件包含了诸如学习率、批处理大小、目标网络更新频率等超参数。
# 当智能体或动作空间的设定改变时，可能需要调整这些超参数以优化算法的性能。


# Number of Episodes
# NUM_EPISODES = 3000
NUM_EPISODES = 1000

# Checkpoints after which save models
CHECKPOINTS = 100

# Number of Steps in each episodes
# NUM_STEPS = 100
NUM_STEPS = 100

# For adding noise for exploration
STD_DEV = 0.2

# Number of experiences stored in buffer
NUM_BUFFER = 10000

# Batch size to select from buffer replay
BATCH_SIZE = 1

# Saved model path
MODEL_PATH = './saved_models/'

# Learning rate for actor-critic models
CRITIC_LR = 1e-4
ACTOR_LR = 5e-5

# Discount factor for future rewards
GAMMA = 0.95

# Used to update target networks

TAU = 0.005