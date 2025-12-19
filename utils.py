import numpy as np
import torch
import warnings
from datetime import datetime

# 全局配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# 初始化随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 忽略警告
warnings.filterwarnings('ignore')

# 公共常量
FRAUD_WEIGHT = 20.0  # 提高诈骗样本权重
EPOCHS = 150        # 增加训练轮次（配合早停）
LEARNING_RATE = 0.0003  # 降低学习率，更稳定
WEIGHT_DECAY = 1e-4     # 降低权重衰减