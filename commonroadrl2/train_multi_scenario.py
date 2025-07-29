import gym
import commonroad_rl.gym_commonroad
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os
import matplotlib.pyplot as plt

# 定义经验回放的存储结构：(状态, 动作, 奖励, 下一状态, 是否终止)
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# 1. 定义Q网络（函数近似器，替代Q表格）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        # 多层感知机结构（对应参考文档中的"神经网络拟合Q函数"）
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入层：状态维度 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出层：动作维度（每个动作的Q值）
        )
    
    def forward(self, x):
        # 输入状态，输出所有动作的Q值
        return self.fc(x)

# 2. 经验回放池（打破样本相关性）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 固定容量的队列，超出自动删除旧数据
    
    def add(self, state, action, reward, next_state, done):
        # 存入单条经验
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        # 随机采样批量经验（对应参考文档中的"随机抽取样本"）
        batch = random.sample(self.buffer, batch_size)
        # 转换为张量，便于神经网络计算
        states = torch.tensor([exp.state for exp in batch], dtype=torch.float32)
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long)  # 离散动作
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        next_states = torch.tensor([exp.next_state for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# 3. DQN智能体（整合网络、经验池、目标网络）
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 buffer_capacity=100000, batch_size=64, target_update_freq=1000):
        # Q网络（当前网络）
        self.q_network = QNetwork(state_dim, action_dim)
        # 目标网络（用于计算目标Q值，对应参考文档的"目标网络"）
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始权重相同
        self.target_network.eval()  # 目标网络不训练，只用于预测
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # Q值预测误差
        
        # 经验回放池（对应参考文档的"经验回放"）
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        
        # 其他超参数
        self.gamma = gamma  # 折扣因子
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.action_dim = action_dim
        self.step_count = 0  # 记录总步数，用于触发目标网络更新
        
        # ε-贪婪策略参数（平衡探索与利用）
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.05  # 提高最小探索率
        self.epsilon_decay = 0.9995  # 降低衰减速度，增加探索
    
    def select_action(self, state):
        # ε-贪婪策略选择动作
        if random.random() < self.epsilon:
            # 随机探索：随机选择动作
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # 将单个整数动作转换为MultiDiscrete格式的数组
        # 假设动作空间是[long_steps, lat_steps]的MultiDiscrete
        # 需要将单个索引转换为两个独立的索引
        long_steps = 5  # 假设纵向动作步数
        lat_steps = 5   # 假设横向动作步数
        
        long_idx = action_idx // lat_steps
        lat_idx = action_idx % lat_steps
        
        return np.array([long_idx, lat_idx])
    
    def learn(self):
        # 经验池数据不足时，不训练
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验池随机采样（打破时序相关性）
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值（Q网络预测）
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（目标网络预测，下一状态的最大Q值）
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]  # 下一状态的最大Q值
            target_q = rewards + (1 - dones) * self.gamma * next_q  # 目标Q值公式r + γ·maxQ(s',a';θ^-)
        
        # 计算损失并更新Q网络
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 每N步更新目标网络（复制当前网络权重）
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减ε（减少探索，增加利用）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 4. 主函数：环境交互与训练
def main():
    # 环境配置（简化观察空间以避免形状不匹配问题）
    env_kwargs = {
        "action_configs": {"action_type": "discrete"},  # DQN默认处理离散动作
        "goal_configs": {
            "observe_distance_goal_long": True,
            "observe_distance_goal_lat": True,
            "relax_is_goal_reached": True,
        },
        "lanelet_configs": {
            "observe_distance_togoal_via_referencepath": False,  # 禁用有问题的观察值
            "observe_route_reference_path": False,  # 禁用有问题的观察值
        },
        "reward_configs": {
            "hybrid_reward": {
                "reward_goal_reached": 200.0,
                "reward_collision": -200.0,
                "reward_off_road": -100.0,
                "reward_closer_to_goal_long": 10.0,
                "reward_closer_to_goal_lat": 10.0,
                "reward_reverse_driving": -50.0,
                "reward_off_road_time": -10.0,
                "reward_goal_distance": -1.0,
                "reward_goal_angle": -1.0,
                "reward_velocity_RL": -1.0,
                "reward_velocity_AL": -1.0,
                "reward_centering": -1.0,
                "reward_smoothness": -1.0,
                "reward_same_lane_goal": 0.0,
                "reward_time_out": -100.0,
                "reward_free_road": 0.0,
                "reward_comfort": 0.0,
                "reward_efficiency": 0.0,
                "reward_safety": 0.0,
                "reward_safe_distance_coef": 0.0,
                "reward_min_distance": 0.0,
                "reward_min_distance_coef": 0.0,
                "reward_min_distance_violation": 0.0,
                "reward_min_distance_violation_coef": 0.0,
                "reward_lane_center_offset": 0.0,
                "reward_lane_center_offset_coef": 0.0,
                "reward_velocity": 0.0,
                "reward_velocity_coef": 0.0,
                "reward_acceleration": 0.0,
                "reward_acceleration_coef": 0.0,
                "reward_jerk": 0.0,
                "reward_jerk_coef": 0.0,
                "reward_yaw_rate": 0.0,
                "reward_yaw_rate_coef": 0.0,
                "reward_action_change": 0.0,
                "reward_action_change_coef": 0.0,
                "reward_invalid_action": 0.0,
                "reward_invalid_action_coef": 0.0,
                "reward_invalid_state": 0.0,
                "reward_invalid_state_coef": 0.0,
                "reward_goal_reached_coef": 1.0,
                "reward_collision_coef": 1.0,
                "reward_off_road_coef": 1.0,
                "reward_reverse_driving_coef": 1.0,
                "reward_time_out_coef": 1.0,
                "reward_off_road_time_coef": 1.0,
                "reward_goal_distance_coef": 1.0,
                "reward_goal_angle_coef": 1.0,
                "reward_velocity_RL_coef": 1.0,
                "reward_velocity_AL_coef": 1.0,
                "reward_centering_coef": 1.0,
                "reward_smoothness_coef": 1.0,
                "reward_get_close_goal_time": 0.0,
                "reward_get_close_goal_time_coef": 1.0,
                "reward_goal_time": 0.0,
                "reward_goal_time_coef": 1.0,
                "reward_goal_distance_time": 0.0,
                "reward_goal_distance_time_coef": 1.0,
                "reward_goal_angle_time": 0.0,
                "reward_goal_angle_time_coef": 1.0,
                "reward_velocity_time": 0.0,
                "reward_velocity_time_coef": 1.0,
                "reward_acceleration_time": 0.0,
                "reward_acceleration_time_coef": 1.0,
                "reward_jerk_time": 0.0,
                "reward_jerk_time_coef": 1.0,
                "reward_yaw_rate_time": 0.0,
                "reward_yaw_rate_time_coef": 1.0,
                "reward_action_change_time": 0.0,
                "reward_action_change_time_coef": 1.0,
                "reward_invalid_action_time": 0.0,
                "reward_invalid_action_time_coef": 1.0,
                "reward_invalid_state_time": 0.0,
                "reward_invalid_state_time_coef": 1.0,
                "reward_goal_reached_time": 0.0,
                "reward_goal_reached_time_coef": 1.0,
                "reward_collision_time": 0.0,
                "reward_collision_time_coef": 1.0,
                "reward_off_road_time_coef": 1.0,
                "reward_reverse_driving_time": 0.0,
                "reward_reverse_driving_time_coef": 1.0,
                "reward_time_out_time": 0.0,
                "reward_time_out_time_coef": 1.0
            }
        },
        "vehicle_params": {"vehicle_type": VehicleType.BMW_320i, "vehicle_model": VehicleModel.PM},
        "meta_scenario_path": "pickles_multi_scenarios/meta_scenario",
        "train_reset_config_path": "pickles_multi_scenarios/problem_train",
        "test_reset_config_path": "pickles_multi_scenarios/problem_test",
        "reward_type": "hybrid_reward",
        "termination_configs": {"max_episode_steps": 200}
    }
    
    # 创建环境
    env = gym.make("commonroad-v1",** env_kwargs)
    state_dim = env.observation_space.shape[0]  # 状态维度
    # 处理MultiDiscrete动作空间
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n  # 离散动作数量
    else:
        # MultiDiscrete动作空间，计算总动作数
        action_dim = np.prod(env.action_space.nvec)
    
    # 初始化DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # 训练参数
    total_episodes = 100000   # 增加训练回合数，使用多场景
    log_dir = "./dqn_multi_scenario_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"开始多场景DQN训练，状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"使用场景数量: {len(os.listdir('pickles_multi_scenarios/problem_train'))}")
    print(f"核心机制：显式Q网络 + 目标网络 + 经验回放 + ε-贪婪策略")
    
    # 训练循环
    episode_rewards = [] # 用于存储每个回合的总奖励
    avg_rewards = [] # 用于存储滑动窗口平均奖励
    window_size = 100 # 滑动窗口大小
    
    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            # 执行动作，与环境交互
            next_state, reward, done, _ = env.step(action)
            # 将动作转换为单个索引用于存储
            action_idx = action[0] * 5 + action[1]  # 假设lat_steps=5
            # 存储经验到回放池
            agent.replay_buffer.add(state, action_idx, reward, next_state, done)
            # 训练Q网络
            agent.learn()
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward) # 记录每个回合的总奖励
        
        # 计算滑动窗口平均奖励
        if len(episode_rewards) >= window_size:
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_rewards.append(avg_reward)
        
        # 打印训练进度
        if (episode + 1) % 100 == 0:
            current_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"回合: {episode+1}, 总奖励: {total_reward:.2f}, 平均奖励(最近100回合): {current_avg:.2f}, ε: {agent.epsilon:.3f}")
    
    # 保存最终模型
    torch.save(agent.q_network.state_dict(), os.path.join(log_dir, "final_q_network.pth"))
    print("训练完成，Q网络模型已保存")

    # 绘制平均奖励曲线
    plt.figure(figsize=(12, 8))
    
    # 设置字体，优先使用系统可用的字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制滑动窗口平均奖励曲线
    if avg_rewards:
        episode_numbers = list(range(window_size, len(episode_rewards) + 1))
        plt.plot(episode_numbers, avg_rewards, 'b-', linewidth=2, label=f'Average Reward ({window_size} Episode Window)')
        plt.xlabel('Training Episodes', fontsize=12)
        plt.ylabel('Average Cumulative Reward', fontsize=12)
        plt.title('DQN Training - Average Reward Curve (Multi-Scenario)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 保存图表
        plt.savefig(os.path.join(log_dir, "average_reward_curve.png2"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Average reward curve saved to: {os.path.join(log_dir, 'average_reward_curve.png')}")
        
        # 打印最终统计信息
        print(f"\nTraining Statistics:")
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Recent {window_size} Episodes Average: {np.mean(episode_rewards[-window_size:]):.2f}")
        print(f"Final ε: {agent.epsilon:.3f}")
    else:
        print("Insufficient training episodes, cannot generate average reward curve")

    env.close()

if __name__ == "__main__":
    from commonroad.common.solution import VehicleType, VehicleModel  # 补充VehicleType和VehicleModel导入
    main()