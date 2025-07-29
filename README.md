# CommonRoad-RL 强化学习自动驾驶项目

## 项目概述

本项目基于 [CommonRoad-RL](https://gitlab.lrz.de/tum-cps/commonroad-rl) 开源框架，实现了多场景强化学习训练系统。项目包含两部分：

1. **CommonRoad-RL 开源框架**：提供基础的强化学习环境和工具
2. **自定义强化学习实现**：基于DQN算法的多场景训练系统

## 快速开始

### 1. 安装 CommonRoad-RL 框架

CommonRoad-RL 是一个开源的强化学习环境，专门用于自动驾驶场景。您可以直接从官方仓库下载：

```bash
git clone https://gitlab.lrz.de/tum-cps/commonroad-rl.git
cd commonroad-rl
```

详细的安装和使用说明请参考官方文档：[CommonRoad-RL 官方仓库](https://gitlab.lrz.de/tum-cps/commonroad-rl)

### 2. 环境配置

```bash
# 创建conda环境
conda env create -n cr37 -f environment.yml

# 激活环境
conda activate cr37

# 安装依赖
pip install -e .
```

## 自定义强化学习实现

### 核心文件：`train_multi_scenario.py`

这是我基于DQN（Deep Q-Network）算法实现的多场景强化学习训练系统。

#### 主要特性

1. **DQN算法实现**
   - Q网络：使用PyTorch实现的多层感知机
   - 经验回放：打破样本相关性，提高训练稳定性
   - 目标网络：减少Q值估计偏差
   - ε-贪婪策略：平衡探索与利用

2. **多场景训练**
   - 支持多个CommonRoad场景同时训练
   - 提高模型的泛化能力
   - 增强对复杂交通环境的适应性

3. **网络架构**
   ```
   输入层：33维状态空间
   隐藏层：128个神经元（3层）
   输出层：25维动作空间
   ```

#### 核心组件

**QNetwork类**
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        # 多层感知机结构
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
```

**ReplayBuffer类**
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 固定容量队列
```

**DQNAgent类**
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 buffer_capacity=100000, batch_size=64, target_update_freq=1000):
        # 整合Q网络、目标网络、经验回放池
```

#### 使用方法

```bash
# 运行训练
python train_multi_scenario.py
```

训练参数配置：
- 总训练回合：100,000
- 学习率：1e-4
- 折扣因子：0.99
- 经验回放容量：100,000
- 批次大小：64
- 目标网络更新频率：1,000步

### 训练模型：`train_model_save_20250729_125944/`

这是训练了62,100回合的DQN模型。

#### 模型信息
- **模型文件**：`q_network_62100_episodes.pth`
- **训练回合**：62,100（目标100,000，完成62.1%）
- **状态维度**：33
- **动作维度**：25
- **隐藏层维度**：128
- **网络架构**：33 → 128 → 128 → 128 → 25

#### 加载模型

```python
import torch
from train_multi_scenario import QNetwork

# 创建网络
model = QNetwork(state_dim=33, action_dim=25)

# 加载训练好的权重
model.load_state_dict(torch.load('train_model_save_20250729_125944/q_network_62100_episodes.pth'))

# 设置为评估模式
model.eval()
```

### 训练数据：`training_data_20250729_125914/`

包含62,100回合训练的完整数据。

#### 数据文件
- `training_data.json`：原始训练数据
- `training_stats.json`：训练统计信息
- `average_reward_curve_62100.png`：奖励曲线图

#### 训练统计
- **总回合数**：62,100
- **平均奖励**：-347.47
- **最近平均奖励**：-60.39
- **最佳平均奖励**：-9.85
- **最差平均奖励**：-926.33

#### 查看训练数据

```python
import json
import matplotlib.pyplot as plt

# 读取训练统计
with open('training_data_20250729_125914/training_stats.json', 'r') as f:
    stats = json.load(f)

print(f"训练统计：{stats}")

# 显示奖励曲线
img = plt.imread('training_data_20250729_125914/average_reward_curve_62100.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

## 项目结构

```
commonroadrl/
├── train_multi_scenario.py              # 自定义DQN训练脚本
├── train_model_save_20250729_125944/   # 训练模型（62,100回合）
│   ├── q_network_62100_episodes.pth    # 模型权重文件
│   ├── model_info.json                 # 模型信息
│   └── README.md                       # 模型说明
├── training_data_20250729_125914/      # 训练数据（62,100回合）
│   ├── training_data.json              # 原始训练数据
│   ├── training_stats.json             # 训练统计
│   ├── average_reward_curve_62100.png  # 奖励曲线图
│   └── README.md                       # 数据说明
├── pickles_multi_scenarios/            # 多场景数据
│   ├── meta_scenario/                  # 场景元数据
│   ├── problem_train/                  # 训练场景
│   └── problem_test/                   # 测试场景
└── commonroad_rl/                      # CommonRoad-RL开源框架
```

## 技术特点

### 1. 算法优势
- **DQN算法**：结合深度学习和Q-learning
- **经验回放**：提高样本利用效率
- **目标网络**：稳定训练过程
- **ε-贪婪策略**：平衡探索与利用

### 2. 多场景训练
- **泛化能力**：在多个场景上训练，提高模型适应性
- **鲁棒性**：增强对复杂交通环境的处理能力
- **效率**：一次性训练多个场景，节省时间

### 3. 可配置性
- **超参数**：学习率、折扣因子、批次大小等可调
- **网络结构**：隐藏层数量和维度可配置
- **训练参数**：回合数、更新频率等可设置

## 使用建议

### 1. 继续训练
如果您想继续训练现有模型：

```python
# 加载现有模型
agent = DQNAgent(state_dim=33, action_dim=25)
agent.q_network.load_state_dict(torch.load('train_model_save_20250729_125944/q_network_62100_episodes.pth'))

# 继续训练
# ... 训练代码
```

### 2. 模型评估
```python
# 在测试场景上评估模型
env = gym.make("commonroad-v1", **test_env_kwargs)
model.eval()

for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            action = q_values.argmax().item()
        state, reward, done, _ = env.step(action)
```

### 3. 参数调优
根据训练数据，您可以调整以下参数：
- 增加训练回合数（当前62,100，目标100,000）
- 调整学习率（当前1e-4）
- 修改网络结构（当前3层隐藏层）
- 优化奖励函数配置

## 注意事项

1. **训练中断**：当前模型在62,100回合时中断，建议继续训练到100,000回合
2. **奖励曲线**：从训练数据看，模型仍在学习过程中，需要更多训练
3. **场景数据**：确保`pickles_multi_scenarios/`目录包含足够的训练场景
4. **硬件要求**：建议使用GPU加速训练过程

## 参考资料

- [CommonRoad-RL 官方文档](https://gitlab.lrz.de/tum-cps/commonroad-rl)
- [DQN算法论文](https://arxiv.org/abs/1312.5602)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [OpenAI Gym文档](https://gym.openai.com/docs/)

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues：[GitLab Issues](https://gitlab.lrz.de/tum-cps/commonroad-rl/-/issues)
- 邮箱：commonroad@lists.lrz.de

---

**注意**：本项目基于CommonRoad-RL开源框架，自定义部分仅包含`train_multi_scenario.py`文件及相关训练模型和数据。其他功能请参考CommonRoad-RL官方文档。 
