#%matplotlib inline

import os
import argparse
import matplotlib.pyplot as plt
from IPython import display

# import functions to read xml file and visualize commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

def argsparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", help="path to xml file", type=str)
    parser.add_argument("--num_time_steps", "-n", type=int,  default=40,
                        help="number of timesteps to plot")
    parser.add_argument("--save_location", "-s", type=str, default=None, help="Path to directory to save rendered images to")
    parser.add_argument("--time_granularity", "-tg", type=int, default=1, help="Granularity for amount of timesteps in between plots")
    parser.add_argument("--start_at_time_step", "-st", type=int, default=0, help="Timestep to start plotting from (default is 0)")
    return parser.parse_args()

def main():
    args = argsparser()

    # generate path of the file to be opened
    file_path = args.path
    print(file_path)

    # read in the scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # plot the scenario for 40 time step, here each time step corresponds to 0.1 second
    start = args.start_at_time_step
    length = args.num_time_steps
    gran = args.time_granularity

    for i in range(start, length):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        # plot the scenario at different time step
        scenario.draw(rnd, draw_params={'time_begin': gran*i + start})
        # plot the planning problem set
        planning_problem_set.draw(rnd)
        
        rnd.render()
        if not (args.save_location is None):
            os.makedirs(args.save_location, exist_ok = True)
            plt.savefig(os.path.join(args.save_location,f"{scenario.scenario_id}-{i}.png"))
    plt.show()

if __name__ == "__main__":
    main()