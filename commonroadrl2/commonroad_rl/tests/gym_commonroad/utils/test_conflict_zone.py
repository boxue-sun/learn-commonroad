import os
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_rl.gym_commonroad.utils.conflict_zone import ConflictZone

from commonroad_rl.tests.common.path import resource_root
from commonroad_rl.tests.common.marker import *

resource_path = resource_root("test_gym_commonroad")


@pytest.mark.parametrize("scenario_id",
                         [
                             "DEU_AAH-1_11009_T-1",
                             "DEU_AAH-1_13008_T-1",
                             "DEU_AAH-1_100017_T-1",
                             "DEU_AAH-2_26003_T-1",
                             "DEU_AAH-3_320011_T-1",
                             "DEU_AAH-4_2002_T-1"
                         ])
@module_test
@functional
def test_conflict_zone_reset(scenario_id):
    conflict_zone = ConflictZone()
    file_name = os.path.join(resource_path, f"{scenario_id}.xml")
    scenario, _ = CommonRoadFileReader(file_name).open()
    conflict_zone.reset(scenario)
