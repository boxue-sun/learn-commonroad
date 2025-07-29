import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad_rl.gym_commonroad.action import Vehicle
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector
from commonroad.geometry.shape import Shape
from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad.scenario.obstacle import ObstacleType, Obstacle, DynamicObstacle
from commonroad.prediction.prediction import Occupancy
import commonroad_dc.pycrcc as pycrcc


def check_collision_type(info, LOGGER, ego_vehicle: Vehicle, observation_collector: ObservationCollector,
                         scenario: Scenario, benchmark_id, lane_change_time_threshold, local_ccosy):
    other_vehicle_at_fault, reason = _current_collision_type_checking(
        ego_vehicle, observation_collector, scenario, lane_change_time_threshold, local_ccosy)
    if other_vehicle_at_fault:
        LOGGER.debug(f"Unexpected Collision not caused by ego vehicle in {benchmark_id} with reason: {reason}")

    info['valid_collision'] = not other_vehicle_at_fault
    info['collision_reason'] = reason


def _current_collision_type_checking(ego_vehicle: Vehicle, observation_collector: ObservationCollector,
                                     scenario: Scenario, lane_change_time_threshold, local_ccosy):

    # Determine collision participants
    collision_ego_vehicle = ego_vehicle.collision_object
    collision_checker = observation_collector.get_collision_checker()
    collision_objects = collision_checker.find_all_colliding_objects(collision_ego_vehicle)
    current_step = ego_vehicle.current_time_step

    # TODO: refactor once issue https://gitlab.lrz.de/cps/commonroad-drivability-checker/-/issues/17 is closed
    # get positions of all obstacles in the scenario
    obstacle_state_dict = dict()
    for obstacle in scenario.obstacles:
        obstacle_state = obstacle.state_at_time(current_step)
        if obstacle_state is not None:
            obstacle_state_dict[obstacle.obstacle_id] = obstacle_state

    collision_obstacles = {}
    for collision_object in collision_objects:
        collision_shape = collision_object.obstacle_at_time(current_step)
        if collision_shape.collide(observation_collector.road_edge["boundary_collision_object"]):
            return True, "other_outside_of_lanelet_network"
        for obstacle_id, obstacle_state in obstacle_state_dict.items():
            if isinstance(collision_shape, pycrcc.RectOBB) or isinstance(collision_shape, pycrcc.Circle):
                collision_shape_center = collision_shape.center()
            elif isinstance(collision_shape, pycrcc.RectAABB): # TODO: remove after dc adds center for RectAABB
                collision_shape_center = 0.5 * np.array([collision_shape.min_x() + collision_shape.max_x(),
                                                         collision_shape.min_y() + collision_shape.max_y()])
            else:
                raise NotImplementedError
            if np.allclose(collision_shape_center, obstacle_state.position):
                collision_obstacles[obstacle_id] = (obstacle_state, collision_shape)

    ego_shape = Rectangle(length=ego_vehicle.parameters.l,
                          width=ego_vehicle.parameters.w,
                          center=ego_vehicle.state.position,
                          orientation=ego_vehicle.state.orientation)
    ego_position = ego_vehicle.state.position

    # TODO: save lane results in pre-processing
    merge_results = create_lanes(scenario.lanelet_network, merging_length=1000.)
    lanes = [merge_result[1] for merge_result in merge_results]
    time_steps_back = int(lane_change_time_threshold / scenario.dt)

    for obstacle_id, (obstacle_state, collision_shape) in collision_obstacles.items():
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # get edges of obstacle shape and check if it was fully on the lanelet network
        if current_step == obstacle.initial_state.time_step:
            return True, "other_suddenly_appear"
        if obstacle.obstacle_type == ObstacleType.CAR:
            # compute contact point of collision
            contact_point = _find_collision_point(ego_shape, obstacle.occupancy_at_time(current_step))
            lanelets_contact_point = set(scenario.lanelet_network.find_lanelet_by_position([np.array(contact_point)])[0])

            def find_lanelet_id_for_state(observation_collector, state, scenario):
                lanelet_polygons, lanelet_polygons_sg = observation_collector._get_lanelet_polygons(str(scenario.scenario_id))
                lanelet_ids = observation_collector.sorted_lanelets_by_state(
                    scenario, state, lanelet_polygons, lanelet_polygons_sg
                )
                return lanelet_ids[0]

            def find_lanelet_ids(state_list, time_steps):
                if len(state_list) >= time_steps:
                    state_list = state_list[-time_steps:]
                return set([find_lanelet_id_for_state(observation_collector, state, scenario)
                                      for state in state_list])

            def check_lane_change(lanelet_ids, lanes):
                for lane in lanes:
                    if (lanelet_ids.issubset(set(lane))):
                        return False
                return True

            obstacle_state_list = [obstacle.state_at_time(time_step) for time_step in
                                   range(obstacle.initial_state.time_step, current_step + 1)]

            ego_lanelet_ids = find_lanelet_ids(ego_vehicle.state_list, time_steps_back)
            obstacle_lanelet_ids = find_lanelet_ids(obstacle_state_list, time_steps_back)
            obstacle_exceeds_lane = True
            ego_exceeds_lane = True
            for lanelet_contact_point in lanelets_contact_point:
                if lanelet_contact_point in obstacle_lanelet_ids:
                    obstacle_exceeds_lane = False
                if lanelet_contact_point in ego_lanelet_ids:
                    ego_exceeds_lane = False
            if obstacle_exceeds_lane:
                return True, "other_out_of_lane"
            if ego_exceeds_lane:
                return False, "ego_out_of_lane"

            # determine back collision
            s_ego, _ = local_ccosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
            s_obs, _ = local_ccosy.convert_to_curvilinear_coords(obstacle_state.position[0], obstacle_state.position[1])
            rear_collision = s_obs <= s_ego

            if rear_collision:
                # check if ego switched lane
                if check_lane_change(ego_lanelet_ids, lanes):
                    return False, "ego_cut_in"
                else:
                    return True, "other_collides_from_rear"
            else:
                # check if other switched lane
                if check_lane_change(obstacle_lanelet_ids, lanes):
                    return True, "other_cut_in"
                else:
                    return False, "ego_collides_from_rear"

    return False, "unsuccessful_determination"


def create_lanes(lanelet_network: LaneletNetwork, merging_length: float = 600.):
    """
    Creates lanes for road network

    :param merging_length: length for merging successors
    :param lanelet_network:
    :return:
    """
    lane_lanelets = []
    start_lanelets = []

    for lanelet in lanelet_network.lanelets:
        if len(lanelet.predecessor) == 0:
            start_lanelets.append(lanelet)
        else:
            predecessors = [lanelet_network.find_lanelet_by_id(pred_id) for pred_id in lanelet.predecessor]
            for pred in predecessors:
                if not lanelet.lanelet_type == pred.lanelet_type:
                    start_lanelets.append(lanelet)

    for lanelet in start_lanelets:
        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet, lanelet_network, merging_length)
        if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
            merged_lanelets.append(lanelet)
            merge_jobs.append([lanelet.lanelet_id])
        for idx in range(len(merged_lanelets)):
            lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))

    return lane_lanelets


def _find_collision_point(ego_shape: Shape, obstacle_occupancy: Occupancy, collision_checker: pycrcc.CollisionChecker=None):
    obstacle_vertices = obstacle_occupancy.shape.vertices

    # collision obstacle's corners in ego shape?
    for obstacle_vertice in obstacle_vertices:
        if ego_shape.contains_point(obstacle_vertice):
            return obstacle_vertice

    # ego's corner in obstacle's shape?
    # ego corners in collision obstacle's shape?
    for ego_vertice in ego_shape.vertices:
        if obstacle_occupancy.shape.contains_point(ego_vertice):
            return ego_vertice

    # neither of two cases
    if collision_checker is None:
        collision_checker = pycrcc.CollisionChecker()
        collision_checker.add_collision_object(create_collision_object(obstacle_occupancy.shape))
    ego_center = ego_shape.center
    for ego_vertice in ego_shape.vertices:
        intervals = collision_checker.raytrace(ego_center[0], ego_center[1], ego_vertice[0], ego_vertice[1], False)
        if len(intervals) > 0:
            return np.array([intervals[0][0], intervals[0][1]])

    raise ValueError