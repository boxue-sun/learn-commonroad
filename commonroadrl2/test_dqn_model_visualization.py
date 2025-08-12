#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN模型可视化测试代码
用于测试train_multi_scenario.py训练的DQN模型，实现可视化观察小车的驾驶情况
借鉴test_improved.py的测试方法
"""

import gym
import commonroad_rl.gym_commonroad
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import time
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from PIL import Image
import glob
import imageio

class QNetwork(nn.Module):
    """Q网络定义，与训练时保持一致"""
    def __init__(self, state_dim=33, action_dim=25, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入层：状态维度 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出层：动作维度（每个动作的Q值）
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    """DQN智能体，用于加载和运行训练好的模型"""
    def __init__(self, state_dim, action_dim, model_path):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 加载Q网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.q_network.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.q_network.eval()
        
        print(f"模型已加载: {model_path}")
    
    def select_action(self, state, epsilon=0.0):
        """选择动作（测试时epsilon=0，完全利用）"""
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # 转换为MultiDiscrete格式
        long_steps = 5
        lat_steps = 5
        long_idx = action_idx // lat_steps
        lat_idx = action_idx % lat_steps
        
        return np.array([long_idx, lat_idx])

class DQNTestVisualizer:
    """DQN模型测试可视化器，借鉴test_improved.py的方法"""
    def __init__(self):
        # 环境配置，与train_multi_scenario.py保持一致
        self.env_kwargs = {
            "action_configs": {"action_type": "discrete"},
            "goal_configs": {
                "observe_distance_goal_long": True,
                "observe_distance_goal_lat": True,
                "relax_is_goal_reached": True,
            },
            "lanelet_configs": {
                "observe_distance_togoal_via_referencepath": False,
                "observe_route_reference_path": False,
            },
            "reward_configs": {
                "hybrid_reward": {
                    # ==================== 终止条件奖励 ====================
                "reward_goal_reached": 1000.0,      # 成功到达目标：大正奖励
                "reward_collision": -2000.0,        # 发生碰撞：大负奖励
                "reward_off_road": -1500.0,         # 偏离道路：中等负奖励
                "reward_time_out": -500.0,         # 超时：中等负奖励
                
                # ==================== 目标导向奖励 ====================
                "reward_closer_to_goal_long": 30.0,  # 向目标纵向移动：小正奖励
                "reward_closer_to_goal_lat": 20.0,   # 向目标横向移动：小正奖励
                "reward_goal_distance": -5.0,        # 距离目标惩罚：小负奖励
                "reward_goal_angle": -3.0,           # 目标角度偏差惩罚：小负奖励
                
                # ==================== 安全驾驶奖励 ====================
                "reward_reverse_driving": -100.0,     # 倒车惩罚：中等负奖励
                "reward_off_road_time": -20.0,       # 偏离道路时间惩罚：小负奖励
                
                # ==================== 驾驶质量奖励 ====================
                "reward_velocity_RL": 5.0,          # 纵向速度惩罚：小负奖励
                "reward_velocity_AL": -10.0,          # 横向速度惩罚：小负奖励
                "reward_centering": 15.0,            # 车道居中惩罚：小负奖励
                "reward_smoothness": -5.0,           # 驾驶平滑性惩罚：小负奖励
                "reward_orientation_to_ref": 20.0,
                
                # ==================== 其他奖励项（设为0，不启用） ====================
                "reward_same_lane_goal": 0.0,        # 同车道目标奖励
                "reward_free_road": 0.0,             # 自由道路奖励
                "reward_comfort": 0.0,               # 舒适性奖励
                "reward_efficiency": 0.0,            # 效率奖励
                "reward_safety": 0.0,                # 安全性奖励
                "reward_safe_distance_coef": 0.0,    # 安全距离系数
                "reward_min_distance": 0.0,          # 最小距离奖励
                "reward_min_distance_coef": 0.0,     # 最小距离系数
                "reward_min_distance_violation": 0.0, # 最小距离违规奖励
                "reward_min_distance_violation_coef": 0.0, # 最小距离违规系数
                "reward_lane_center_offset": 0.0,    # 车道中心偏移奖励
                "reward_lane_center_offset_coef": 0.0, # 车道中心偏移系数
                "reward_velocity": 0.0,              # 速度奖励
                "reward_velocity_coef": 0.0,         # 速度系数
                "reward_acceleration": 0.0,          # 加速度奖励
                "reward_acceleration_coef": 0.0,     # 加速度系数
                "reward_jerk": 0.0,                  # 加加速度奖励
                "reward_jerk_coef": 0.0,             # 加加速度系数
                "reward_yaw_rate": 0.0,              # 偏航角速度奖励
                "reward_yaw_rate_coef": 0.0,         # 偏航角速度系数
                "reward_action_change": 0.0,         # 动作变化奖励
                "reward_action_change_coef": 0.0,    # 动作变化系数
                "reward_invalid_action": 0.0,        # 无效动作奖励
                "reward_invalid_action_coef": 0.0,   # 无效动作系数
                "reward_invalid_state": 0.0,         # 无效状态奖励
                "reward_invalid_state_coef": 0.0,    # 无效状态系数
                
                # ==================== 奖励系数配置 ====================
                # 这些系数用于调整对应奖励项的权重，1.0表示正常权重
                "reward_goal_reached_coef": 1.2,     # 目标达成系数
                "reward_collision_coef": 1.5,        # 碰撞系数
                "reward_off_road_coef": 1.5,         # 偏离道路系数
                "reward_reverse_driving_coef": 1.0,  # 倒车系数
                "reward_time_out_coef": 1.0,         # 超时系数
                "reward_off_road_time_coef": 1.0,    # 偏离道路时间系数
                "reward_goal_distance_coef": 1.0,    # 目标距离系数
                "reward_goal_angle_coef": 1.0,       # 目标角度系数
                "reward_velocity_RL_coef": 1.0,      # 纵向速度系数
                "reward_velocity_AL_coef": 1.0,      # 横向速度系数
                "reward_centering_coef": 1.2,        # 居中系数
                "reward_smoothness_coef": 1.0,       # 平滑性系数
                "reward_orientation_to_ref_coef": 1.2,
                
                # ==================== 时间相关奖励（设为0，不启用） ====================
                "reward_get_close_goal_time": 0.0,   # 接近目标时间奖励
                "reward_get_close_goal_time_coef": 1.0, # 接近目标时间系数
                "reward_goal_time": 0.0,             # 目标时间奖励
                "reward_goal_time_coef": 1.0,        # 目标时间系数
                "reward_goal_distance_time": 0.0,    # 目标距离时间奖励
                "reward_goal_distance_time_coef": 1.0, # 目标距离时间系数
                "reward_goal_angle_time": 0.0,       # 目标角度时间奖励
                "reward_goal_angle_time_coef": 1.0,  # 目标角度时间系数
                "reward_velocity_time": 0.0,         # 速度时间奖励
                "reward_velocity_time_coef": 1.0,    # 速度时间系数
                "reward_acceleration_time": 0.0,     # 加速度时间奖励
                "reward_acceleration_time_coef": 1.0, # 加速度时间系数
                "reward_jerk_time": 0.0,             # 加加速度时间奖励
                "reward_jerk_time_coef": 1.0,        # 加加速度时间系数
                "reward_yaw_rate_time": 0.0,         # 偏航角速度时间奖励
                "reward_yaw_rate_time_coef": 1.0,    # 偏航角速度时间系数
                "reward_action_change_time": 0.0,    # 动作变化时间奖励
                "reward_action_change_time_coef": 1.0, # 动作变化时间系数
                "reward_invalid_action_time": 0.0,   # 无效动作时间奖励
                "reward_invalid_action_time_coef": 1.0, # 无效动作时间系数
                "reward_invalid_state_time": 0.0,    # 无效状态时间奖励
                "reward_invalid_state_time_coef": 1.0, # 无效状态时间系数
                "reward_goal_reached_time": 0.0,     # 目标达成时间奖励
                "reward_goal_reached_time_coef": 1.0, # 目标达成时间系数
                "reward_collision_time": 0.0,        # 碰撞时间奖励
                "reward_collision_time_coef": 1.0,   # 碰撞时间系数
                "reward_off_road_time_coef": 1.0,    # 偏离道路时间系数
                "reward_reverse_driving_time": 0.0,  # 倒车时间奖励
                "reward_reverse_driving_time_coef": 1.0, # 倒车时间系数
                "reward_time_out_time": 0.0,         # 超时时间奖励
                "reward_time_out_time_coef": 1.0     # 超时时间系数
                }
            },
            "vehicle_params": {"vehicle_type": VehicleType.BMW_320i, "vehicle_model": VehicleModel.PM},
            "meta_scenario_path": "pickles_scenario_DEU_AAH-1_11009_T-1/meta_scenario",
            "train_reset_config_path": "pickles_scenario_DEU_AAH-1_11009_T-1/problem_train",
            "test_reset_config_path": "pickles_scenario_DEU_AAH-1_11009_T-1/problem_test",
            "reward_type": "hybrid_reward",
            "termination_configs": {"max_episode_steps": 200},
            "render_configs": {
                "render_combine_frames": False,
                "render_follow_ego": True,
                "render_fps": 20,
                "render_skip_timesteps": 1,
                "render_range": 60
            }
        }
        
        # 创建结果目录
        self.results_dir = "./dqn_test_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # 初始化统计数据
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.off_road_count = 0
        
        # 存储每回合的图片路径
        self.episode_images = {}
        
    def create_renderer(self, cr_env):
        """创建渲染器和绘图参数"""
        renderer = MPRenderer(figsize=(12, 12))
        
        # 创建绘图参数
        draw_params = MPDrawParams()
        draw_params.time_begin = cr_env.current_step
        draw_params.time_end = cr_env.current_step
        draw_params.lanelet_network.lanelet.show_label = False
        draw_params.lanelet_network.lanelet.fill_lanelet = True
        draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
        draw_params.lanelet_network.traffic_sign.show_traffic_signs = "all"
        draw_params.lanelet_network.traffic_sign.show_label = False
        draw_params.lanelet_network.traffic_sign.scale_factor = 0.1
        draw_params.lanelet_network.intersection.draw_intersections = True
        draw_params.dynamic_obstacle.show_label = True
        draw_params.dynamic_obstacle.draw_icon = True
        
        return renderer, draw_params
    
    def render_frame(self, cr_env, renderer, draw_params):
        """渲染单帧"""
        # 绘制场景
        cr_env.scenario.draw(renderer, draw_params)
        
        # 绘制规划问题
        cr_env.planning_problem.draw(renderer)
        
        # 绘制观察结果
        cr_env.observation_collector.render(cr_env.render_configs, renderer)
        
        # 绘制自车
        ego_obstacle = DynamicObstacle(
            obstacle_id=cr_env.scenario.generate_object_id(),
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=Rectangle(length=cr_env.ego_action.vehicle.parameters.l,
                                   width=cr_env.ego_action.vehicle.parameters.w),
            initial_state=cr_env.ego_action.vehicle.state
        )
        
        ego_draw_params = MPDrawParams()
        ego_draw_params.time_begin = cr_env.current_step
        ego_draw_params.time_end = cr_env.current_step
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "red"
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "darkred"
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.alpha = 0.8
        ego_obstacle.draw(renderer, draw_params=ego_draw_params)
        
        # 如果需要跟随自车
        if cr_env.render_configs["render_follow_ego"]:
            x, y = cr_env.ego_action.vehicle.state.position
            range_val = cr_env.render_configs["render_range"]
            renderer.plot_limits = [x - range_val, x + range_val, y - range_val, y + range_val]
        
        # 添加状态信息
        velocity = cr_env.ego_action.vehicle.state.velocity
        acceleration = cr_env.ego_action.vehicle.state.acceleration
        orientation = cr_env.ego_action.vehicle.state.orientation
        info_text = f"Speed: {velocity:.2f} m/s\nAccel: {acceleration:.2f} m/s²\nOrientation: {orientation:.2f} rad"
        plt.gcf().text(0.02, 0.98, info_text, fontsize=8, va='top')
        
        return renderer
    
    def run_test_episode(self, env, agent, episode_num):
        """运行单个测试回合"""
        state = env.reset()
        episode_reward = 0
        frame_idx = 0
        done = False
        end_reason = "未知"
        trajectory = []  # 记录轨迹
        episode_frames = []  # 记录本回合的所有帧
        
        print(f"Episode {episode_num} 开始...")
        
        while not done:
            # 选择动作
            action = agent.select_action(state, epsilon=0.0)  # 测试时不探索
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 记录轨迹
            try:
                pos = env.ego_action.vehicle.state.position
                trajectory.append(pos.copy())
            except:
                pass
            
            # 每步都渲染
            should_render = True
            
            if should_render:
                try:
                    # 创建渲染器和绘图参数
                    renderer, draw_params = self.create_renderer(env)
                    
                    # 渲染当前帧
                    renderer = self.render_frame(env, renderer, draw_params)
                    
                    # 绘制轨迹
                    if len(trajectory) > 1:
                        trajectory_array = np.array(trajectory)
                        renderer.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                       'r-', linewidth=2, alpha=0.7, label='Ego Vehicle Trajectory')
                        renderer.ax.legend()
                    
                    # 保存图像
                    filename = f"{self.results_dir}/episode_{episode_num:02d}_frame_{frame_idx:04d}.png"
                    renderer.render(show=False, filename=filename, keep_static_artists=True)
                    episode_frames.append(filename)  # 添加到本回合帧列表
                    print(f"  保存帧 {filename} (步骤 {frame_idx})")
                    
                    # 显示实时图像
                    plt.pause(0.1)
                    
                except Exception as e:
                    print(f"  渲染帧时出错: {str(e)}")
            
            frame_idx += 1
            
            # 打印当前状态信息
            if frame_idx % 10 == 0 or done:
                try:
                    pos = env.ego_action.vehicle.state.position
                    vel = env.ego_action.vehicle.state.velocity
                    print(f"  步骤 {frame_idx}: 位置=({pos[0]:.1f}, {pos[1]:.1f}), 速度={vel:.1f} m/s")
                except:
                    pass
        
        # 更新统计信息
        self.episode_rewards.append(episode_reward)
        
        # 打印info字典内容以便调试
        print(f"  Info字典内容: {info}")
        
        # 检查终止原因
        if info.get('is_collision', False):
            self.collision_count += 1
            end_reason = "碰撞"
        elif info.get('is_time_out', False):
            self.timeout_count += 1
            end_reason = "超时"
        elif info.get('is_goal_reached', False):
            self.success_count += 1
            end_reason = "到达目标"
        elif info.get('is_off_road', False):
            self.off_road_count += 1
            end_reason = "偏离道路"
        elif info.get('termination_reason', None):
            # 检查termination_reason字段
            termination_reason = info.get('termination_reason')
            if 'collision' in str(termination_reason).lower():
                self.collision_count += 1
                end_reason = "碰撞"
            elif 'timeout' in str(termination_reason).lower():
                self.timeout_count += 1
                end_reason = "超时"
            elif 'goal' in str(termination_reason).lower():
                self.success_count += 1
                end_reason = "到达目标"
            else:
                end_reason = f"其他({termination_reason})"
        else:
            end_reason = "其他(未知)"
        
        print(f"Episode {episode_num} 结束 - 原因: {end_reason}, 总步数: {frame_idx}, 奖励: {episode_reward:.2f}")
        print(f"  轨迹长度: {len(trajectory)} 个点")
        
        # 保存本回合的帧列表
        self.episode_images[episode_num] = episode_frames
        
        plt.close()
        
        return episode_reward, end_reason
    
    def plot_test_results(self):
        """绘制测试结果图表"""
        # 创建结果图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制奖励曲线
        if self.episode_rewards:
            ax1.plot(self.episode_rewards, 'b-', linewidth=2)
            ax1.set_title('DQN Test Episode Rewards', fontsize=14)
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Total Reward', fontsize=12)
            ax1.grid(True, alpha=0.3)
        
        # 绘制平均奖励
        if len(self.episode_rewards) > 1:
            window_size = min(5, len(self.episode_rewards))
            avg_rewards = [np.mean(self.episode_rewards[max(0, i-window_size):i+1]) 
                          for i in range(len(self.episode_rewards))]
            ax2.plot(avg_rewards, 'g-', linewidth=2)
            ax2.set_title(f'Average Rewards ({window_size}-episode window)', fontsize=14)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Average Reward', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # 绘制统计饼图
        labels = ['成功', '碰撞', '超时', '偏离道路']
        sizes = [self.success_count, self.collision_count, self.timeout_count, self.off_road_count]
        colors = ['lightgreen', 'lightcoral', 'lightskyblue', 'orange']
        
        # 检查是否有数据
        if sum(sizes) > 0:
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
        
        ax3.set_title('Test Results Distribution', fontsize=14)
        
        # 绘制奖励分布
        if len(self.episode_rewards) > 0:
            ax4.hist(self.episode_rewards, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_title('Reward Distribution', fontsize=14)
            ax4.set_xlabel('Reward', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/dqn_test_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_episode_gif(self, episode_num, duration=200):
        """为单个回合创建GIF动画"""
        if episode_num not in self.episode_images:
            print(f"Episode {episode_num} 没有图片文件")
            return
        
        image_files = self.episode_images[episode_num]
        if not image_files:
            print(f"Episode {episode_num} 没有图片文件")
            return
        
        # 读取图片
        images = []
        for filename in image_files:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
        
        if not images:
            print(f"Episode {episode_num} 没有有效的图片文件")
            return
        
        # 创建GIF
        gif_filename = f"{self.results_dir}/episode_{episode_num:02d}_animation.gif"
        imageio.mimsave(gif_filename, images, duration=duration/1000.0)  # duration in seconds
        print(f"GIF动画已保存: {gif_filename}")
    
    def create_all_gifs(self, duration=200):
        """为所有回合创建GIF动画"""
        for episode_num in self.episode_images.keys():
            self.create_episode_gif(episode_num, duration)

def main():
    """主函数"""
    print("=== DQN模型可视化测试 ===")
    
    # 创建可视化器
    visualizer = DQNTestVisualizer()
    
    # 创建环境
    print("创建环境...")
    env = gym.make("commonroad-v1", **visualizer.env_kwargs)
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = np.prod(env.action_space.nvec)
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作空间: {env.action_space}")
    
    # 加载模型
    model_path = "/home/bx/Desktop/commonroad-rl/dqn_multi_scenario_logs/final_q_network_20000.pth"
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    # 创建DQN智能体
    agent = DQNAgent(state_dim, action_dim, model_path)
    
    # 运行测试回合
    n_test_episodes = 10  # 测试5个回合
    print(f"\n开始测试 {n_test_episodes} 个回合...")
    
    for episode in range(n_test_episodes):
        print(f"\n测试回合 {episode + 1}/{n_test_episodes}")
        reward, end_reason = visualizer.run_test_episode(env, agent, episode)
    
    # 打印最终统计结果
    print("\n=== 测试完成！最终统计 ===")
    print(f"总回合数: {len(visualizer.episode_rewards)}")
    print(f"平均奖励: {np.mean(visualizer.episode_rewards):.2f}")
    print(f"奖励标准差: {np.std(visualizer.episode_rewards):.2f}")
    print(f"成功率: {visualizer.success_count/n_test_episodes*100:.1f}%")
    print(f"碰撞率: {visualizer.collision_count/n_test_episodes*100:.1f}%")
    print(f"超时率: {visualizer.timeout_count/n_test_episodes*100:.1f}%")
    print(f"偏离道路率: {visualizer.off_road_count/n_test_episodes*100:.1f}%")
    
    # 绘制测试结果图表
    visualizer.plot_test_results()
    print(f"\n测试结果图表已保存至: {visualizer.results_dir}/dqn_test_results.png")
    
    # 创建GIF动画
    print("\n开始创建GIF动画...")
    visualizer.create_all_gifs(duration=200)  # 每帧200毫秒
    print("GIF动画创建完成！")
    
    # 关闭环境
    env.close()
    
    print(f"\n所有结果已保存至: {visualizer.results_dir}")
    print("=== 测试完成 ===")

if __name__ == "__main__":
    main() 
