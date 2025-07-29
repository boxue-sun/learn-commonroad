import os
import numpy as np
from commonroad_rl.gym_commonroad import *
from commonroad_rl.gym_commonroad.utils.collision_type_checker import create_lanes
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

resource_path = resource_root("test_collision_type")
pickle_xml_scenarios(
    input_dir=os.path.join(resource_path),
    output_dir=os.path.join(resource_path, "pickles")
)

meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

output_path = output_root("test_collision_type")
visualization_path = os.path.join(output_path, "visualization")


@pytest.mark.parametrize(("benchmark_id", "check_collision_type", "expected_reason", "expected_validity"),
    [
        ("DEU_LocationALower-57_13_T-1", True, "other_collides_from_rear", 0),
        ("ZAM_Tjunction-1_129_T-1", True, "ego_cut_in", 1)
    ])
@module_test
@functional
def test_collision_type_check(benchmark_id, check_collision_type, expected_reason, expected_validity):
    env = gym.make("commonroad-v1",
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        visualization_path=visualization_path,
        check_collision_type=check_collision_type,
        vehicle_params={"vehicle_model": 2}
    )
    env.reset(benchmark_id=benchmark_id)
    # env.render()
    done = False
    while not done:
        if benchmark_id=="DEU_LocationALower-57_13_T-1":
            action = np.array([0., 0.])
        elif benchmark_id=="ZAM_Tjunction-1_129_T-1":
            if env.current_step < 8:
                action = np.array([0.05, 1.])
            else:
                action = np.array([-0.08, 0.3])
        obs, reward, done, info = env.step(action)

    assert info["collision_reason"]==expected_reason
    assert info["valid_collision"]==expected_validity


@pytest.mark.parametrize(("scenario_id", "expected_ids"),
    [("ZAM_Tjunction-1_129_T-1",
      [
          [50195, 50209, 50203],
          [50195, 50211, 50199],
          [50205, 50207, 50197],
          [50205, 50217, 50199],
          [50201, 50215, 50203],
          [50201, 50213, 50197]
      ])])
@unit_test
@functional
def test_create_lane(scenario_id, expected_ids):
    from commonroad.common.file_reader import CommonRoadFileReader
    fn = os.path.join(resource_path, f"{scenario_id}.xml")
    lanelet_network = CommonRoadFileReader(fn).open_lanelet_network()
    results = create_lanes(lanelet_network)
    merged_lanelet_ids = sorted([merged_lanelet_id for _, merged_lanelet_id in results])

    assert merged_lanelet_ids==sorted(expected_ids)
