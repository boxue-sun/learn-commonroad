"""
Module for generating CommonRoad scenarios using random actions
"""
import argparse
import os
import pickle
import multiprocessing

import gym
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.scenario import Tag
from commonroad.scenario.trajectory import Trajectory

from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

os.environ["KMP_WARNINGS"] = "off"


def get_parser():
    parser = argparse.ArgumentParser(description="Generates scenarios using random actions",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--meta_path", type=str, default=PATH_PARAMS["meta_scenario"],
                        help="Path to pickled meta scenarios")
    parser.add_argument("--scenario_path", type=str, default=PATH_PARAMS["test_reset_config"],
                        help="Path to pickled test scenarios")
    parser.add_argument("--output_path", "-o", type=str, default="random_scenarios",
                        help="Path to pickled test scenarios")
    parser.add_argument("--n_cpu", "-n", type=int, default=1, help="Number of cpu processes")
    parser.add_argument("--create_collision_action", "-collision", action="store_true",
        help="Store actions that caused collision")

    return parser


def create_scenarios(args):
    env = gym.make("commonroad-v1",
                   meta_scenario_path=args.meta_path,
                   test_reset_config_path=args.scenario_path,
                   logging_path=None,
                   play=True)
    env.reset()
    initial_state = env.ego_action.vehicle.initial_state
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            # Append ego trajectory
            ego_trajectory = Trajectory(1, env.ego_action.vehicle.state_list)
            ego_shape = Rectangle(width=env.ego_action.vehicle.parameters.w, length=env.ego_action.vehicle.parameters.l)
            ego_id = env.scenario.generate_object_id()
            ego_type = ObstacleType.CAR
            ego_prediction = TrajectoryPrediction(ego_trajectory, ego_shape)
            ego_dynamic_obstacle = DynamicObstacle(ego_id,
                                                   ego_type,
                                                   ego_shape,
                                                   initial_state,
                                                   ego_prediction)
            # Add ego to scenario
            env.scenario.add_objects(ego_dynamic_obstacle)
            # Write ego trajectory to scenario
            author = 'Xiao Wang'
            affiliation = 'Technical University of Munich, Germany'
            source = 'highD + random action for ego'
            tags = {Tag.INTERSTATE, Tag.MULTI_LANE, Tag.NO_ONCOMING_TRAFFIC, Tag.HIGHWAY}

            planning_problem_set = PlanningProblemSet([env.planning_problem])
            fw = CommonRoadFileWriter(env.scenario, planning_problem_set, author, affiliation, source, tags)
            filename = os.path.join(args.output_path, f"{env.benchmark_id}-{ego_id}.xml")
            print(filename)
            fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

            # Reset env
            env.reset()


def create_collisions_multiple_process(args):
    if args.n_cpu <= 1:
        results = create_collisions_single_process(args.meta_path, args.scenario_path, seed=1)
    else:
        with multiprocessing.Pool(processes=args.n_cpu) as pool:
            results = pool.starmap(
                create_collisions_single_process,
                [
                    (args.meta_path, os.path.join(args.scenario_path, str(i)), i+1) for i in range(args.n_cpu)
                ]
            )
    with open(os.path.join(args.output_path, "collision_actions.pkl"), "wb") as f:
        pickle.dump(results, f)


def create_collisions_single_process(meta_path, scenario_path, seed):
    env = gym.make("commonroad-v1",
                   meta_scenario_path=meta_path,
                   test_reset_config_path=scenario_path,
                   logging_path=None,
                   play=True)
    env.seed(seed)
    env.reset()
    initial_state = env.ego_action.vehicle.initial_state
    log_actions = {}
    actions = []
    while True:
        action = env.action_space.sample()
        actions.append(action)
        # actions[str(env.scenario.scenario_id)].append(action)
        obs, reward, done, info = env.step(action)
        if done:
            if info["is_collision"]:
                log_actions[str(env.scenario.scenario_id)] = actions

            # Reset env
            try:
                env.reset()
                actions = []
            except IndexError:
                return log_actions



if __name__ == "__main__":
    args = get_parser().parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.create_collision_action:
        create_collisions_multiple_process(args)
    else:
        create_scenarios(args)
