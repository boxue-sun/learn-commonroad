#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multi_scenario5 测试脚本（不改动原有测试文件，新增本文件）
- 复用可视化与评估思路
- 适配 train_multi_scenario5 的阶段式（stage）状态增强：将 stage 独热向量拼入状态
- 阶段划分：基于初始到目标距离，设置 th1/th2/final_th，与训练时一致
"""

import argparse
import os
import random
import time

import gym
import commonroad_rl.gym_commonroad
import imageio
import numpy as np
import torch
import torch.nn as nn
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
import matplotlib.pyplot as plt
from PIL import Image


# ---------- 网络结构（与训练一致） ----------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, model_path: str):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.q_network.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.q_network.eval()
        print(f"模型已加载: {model_path}")

    def select_action(self, state_aug: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                s = torch.tensor(state_aug, dtype=torch.float32).unsqueeze(0)
                q = self.q_network(s)
                action_idx = int(q.argmax().item())
        # 映射到 MultiDiscrete(5,5)
        long_steps, lat_steps = 5, 5
        long_idx = action_idx // lat_steps
        lat_idx = action_idx % lat_steps
        return np.array([long_idx, lat_idx])


# ---------- 阶段式状态增强 ----------
class StageManager:
    def __init__(self, num_stages: int = 4):
        self.num_stages = num_stages
        self.stage = 0
        self.th1 = 50.0
        self.th2 = 20.0
        self.final_th = 1.0
        self.prev_dist = None

    @staticmethod
    def goal_distance(obs: np.ndarray) -> float:
        d_long = float(abs(obs[0])) if len(obs) > 0 else 0.0
        d_lat = float(abs(obs[1])) if len(obs) > 1 else 0.0
        return float(np.sqrt(d_long ** 2 + d_lat ** 2))

    def reset(self, obs: np.ndarray):
        d0 = self.goal_distance(obs)
        self.th1 = max(50.0, 0.5 * d0)
        self.th2 = max(20.0, 0.2 * d0)
        self.final_th = 1.0
        self.stage = 0
        self.prev_dist = d0

    def update_stage(self, obs: np.ndarray):
        curr = self.goal_distance(obs)
        if self.stage == 0 and curr <= self.th1:
            self.stage = 1
        elif self.stage == 1 and curr <= self.th2:
            self.stage = 2
        elif self.stage == 2 and curr <= self.final_th:
            self.stage = 3
        self.prev_dist = curr

    def build_aug_state(self, obs: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((self.num_stages,), dtype=np.float32)
        idx = min(max(self.stage, 0), self.num_stages - 1)
        one_hot[idx] = 1.0
        return np.concatenate([obs.astype(np.float32), one_hot], axis=0)


# ---------- 可视化辅助 ----------
class Visualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.episode_images = {}

    def create_renderer(self, cr_env):
        renderer = MPRenderer(figsize=(12, 12))
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

    def render_frame(self, cr_env, renderer, draw_params, frame_idx: int, episode_num: int):
        cr_env.scenario.draw(renderer, draw_params)
        cr_env.planning_problem.draw(renderer)
        cr_env.observation_collector.render(cr_env.render_configs, renderer)

        ego_obstacle = DynamicObstacle(
            obstacle_id=cr_env.scenario.generate_object_id(),
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=Rectangle(length=cr_env.ego_action.vehicle.parameters.l,
                                      width=cr_env.ego_action.vehicle.parameters.w),
            initial_state=cr_env.ego_action.vehicle.state,
        )
        ego_draw_params = MPDrawParams()
        ego_draw_params.time_begin = cr_env.current_step
        ego_draw_params.time_end = cr_env.current_step
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "red"
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "darkred"
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
        ego_draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.alpha = 0.8
        ego_obstacle.draw(renderer, draw_params=ego_draw_params)

        if cr_env.render_configs.get("render_follow_ego", True):
            x, y = cr_env.ego_action.vehicle.state.position
            rng = cr_env.render_configs.get("render_range", 60)
            renderer.plot_limits = [x - rng, x + rng, y - rng, y + rng]

        filename = os.path.join(self.results_dir, f"episode_{episode_num:02d}_frame_{frame_idx:04d}.png")
        renderer.render(show=False, filename=filename, keep_static_artists=True)
        self.episode_images.setdefault(episode_num, []).append(filename)

    def create_episode_gif(self, episode_num: int, duration_ms: int = 200):
        files = self.episode_images.get(episode_num, [])
        pil_images = []
        for f in files:
            if os.path.exists(f):
                try:
                    pil_images.append(Image.open(f).convert('RGB'))
                except Exception:
                    pass
        if pil_images:
            base_w, base_h = pil_images[0].size
            resized = [im if im.size == (base_w, base_h) else im.resize((base_w, base_h), Image.BILINEAR) for im in pil_images]
            # Convert to numpy arrays
            np_frames = [np.array(im) for im in resized]
            gif_path = os.path.join(self.results_dir, f"episode_{episode_num:02d}_animation.gif")
            imageio.mimsave(gif_path, np_frames, duration=duration_ms / 1000.0)
            print(f"GIF已保存: {gif_path}")


def build_env_kwargs():
    return {
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
        "vehicle_params": {"vehicle_type": VehicleType.BMW_320i, "vehicle_model": VehicleModel.PM},
        "meta_scenario_path": "pickles_scenario_DEU_AAH-4_2002_T-1/meta_scenario",
        "train_reset_config_path": "pickles_scenario_DEU_AAH-4_2002_T-1/problem_train",
        "test_reset_config_path": "pickles_scenario_DEU_AAH-4_2002_T-1/problem_test",
        "reward_type": "hybrid_reward",
        "termination_configs": {"max_episode_steps": 200},
        "render_configs": {
            "render_combine_frames": False,
            "render_follow_ego": True,
            "render_fps": 20,
            "render_skip_timesteps": 1,
            "render_range": 60,
        },
        "reward_configs": {
            "hybrid_reward": {
                "reward_goal_reached": 1000.0,
                "reward_collision": -2000.0,
                "reward_off_road": -1500.0,
                "reward_time_out": -500.0,
                "reward_friction_violation": -200.0,

                "reward_closer_to_goal_long": 40.0,
                "reward_closer_to_goal_lat": 30.0,

                "reward_reverse_driving": -100.0,
                "reward_safe_distance_coef": 0.0,
                "reward_same_lane_goal": 0.0,
                "reward_get_close_goal_time": 0.0,
                "reward_close_goal_velocity": 0.0,
                "reward_close_goal_orientation": 0.0,
                "reward_long_distance_reference_path": 0.0,
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Test for train_multi_scenario5 with stage-augmented state")
    parser.add_argument("--model", type=str, required=True, help="路径到已训练模型 .pth 文件")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--out", type=str, default="./dqn_test_results_5", help="输出目录")
    args = parser.parse_args()

    print("=== train_multi_scenario5 测试 ===")
    env_kwargs = build_env_kwargs()
    env = gym.make("commonroad-v1", **env_kwargs)

    # 计算动作维度
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        action_dim = int(np.prod(env.action_space.nvec))

    # 计算状态维度（原始）并扩展为带stage的维度
    raw_state_dim = env.observation_space.shape[0]
    num_stages = 4
    aug_state_dim = raw_state_dim + num_stages

    print(f"原始状态维度: {raw_state_dim}, 增强后状态维度: {aug_state_dim}, 动作维度: {action_dim}")

    # 加载模型
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在 {args.model}")
        return
    agent = DQNAgent(aug_state_dim, action_dim, args.model)

    vis = Visualizer(args.out)
    stage_mgr = StageManager(num_stages=num_stages)

    rewards, successes, collisions, timeouts, offroads = [], 0, 0, 0, 0

    for ep in range(args.episodes):
        print(f"\nEpisode {ep+1}/{args.episodes} 开始...")
        obs = env.reset()
        stage_mgr.reset(obs)
        state_aug = stage_mgr.build_aug_state(obs)
        done = False
        ep_reward = 0.0
        frame_idx = 0

        while not done:
            action = agent.select_action(state_aug, epsilon=0.0)
            next_obs, reward, done, info = env.step(action)

            # 更新阶段并构建下一增强状态（仅用于推理，不影响环境）
            stage_mgr.update_stage(next_obs)
            next_state_aug = stage_mgr.build_aug_state(next_obs)

            # 渲染与保存帧
            try:
                renderer, draw_params = vis.create_renderer(env)
                vis.render_frame(env, renderer, draw_params, frame_idx, ep)
            except Exception as e:
                # 渲染失败不影响测试
                pass

            state_aug = next_state_aug
            ep_reward += reward
            frame_idx += 1

        # 统计终止原因
        if info.get('is_goal_reached', False):
            successes += 1
        elif info.get('is_collision', False):
            collisions += 1
        elif info.get('is_time_out', False):
            timeouts += 1
        elif info.get('is_off_road', False):
            offroads += 1

        rewards.append(ep_reward)
        print(f"Episode {ep+1} 结束：回报={ep_reward:.2f}, 终止信息={info}")

        # 生成GIF
        vis.create_episode_gif(ep, duration_ms=200)

    # 汇总
    print("\n=== 测试完成 ===")
    if rewards:
        print(f"平均回报: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    total = max(1, args.episodes)
    print(f"成功: {successes/total*100:.1f}%, 碰撞: {collisions/total*100:.1f}%, 超时: {timeouts/total*100:.1f}%, 偏离: {offroads/total*100:.1f}%")
    print(f"输出目录: {args.out}")

    env.close()


if __name__ == "__main__":
    main() 
