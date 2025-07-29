# CommonRoad-RL

This repository contains a software package to solve motion planning problems on [CommonRoad](https://commonroad.in.tum.de) using Reinforcement Learning algorithms. We currently use the implementation for the RL algorithms from [OpenAI Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), but the package can be run with any standard (OpenAI Gym compatible) RL implementations.

## CommonRoad-RL in a nutshell
```python
import gym
import commonroad_rl.gym_commonroad

# kwargs overwrites configs defined in commonroad_rl/gym_commonroad/configs.yaml
env = gym.make("commonroad-v1",
		# if installed via pip add the paths to the data folders and uncomment the following two lines 
		# meta_scenario_path=meta_scenario_path, #path to meta scenario specification 
		# train_reset_config_path= training_data_path, #path to training pickels
               action_configs={"action_type": "continuous"},
               goal_configs={"observe_distance_goal_long": True, "observe_distance_goal_lat": True},
               surrounding_configs={"observe_lane_circ_surrounding": False,
               		     "fast_distance_calculation": False,
                                    "observe_lidar_circle_surrounding": True,
                                    "lidar_circle_num_beams": 20},
               reward_type="sparse_reward",
               reward_configs={"sparse_reward":{"reward_goal_reached": 50.,
                                      "reward_collision": -100.,
                                      "reward_off_road": -50.,
      					"reward_time_out": -10.,
					"reward_friction_violation": 0.}})

observation = env.reset()
for _ in range(500):
    # env.render() # rendered images with be saved under ./img
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
```
## Folder structure
```
commonroad-rl                                           
├─ commonroad_rl
│  ├─ doc                               # Folder for documentation         
│  ├─ gym_commonroad                    # Gym environment for CommonRoad scenarios
|     ├─ action                         # Action and Vehicle modules
|     ├─ observation                    # Observation modules
|     ├─ reward                         # Reward and Termination modules
|     ├─ utils                          # Utility functions for gym_commonroad
│     ├─ configs.yaml                   # Default config file for actions, observations, rewards, and termination conditions,
										  as well as for observation space optimization and reward coefficient optimization
│     ├─ commonroad_env.py              # CommonRoadEnv(gym.Env) class
│     └─ constants.py                   # Script to define path, vehicle, and draw parameters
│  ├─ hyperparams                       # Config files for default hyperparameters for various RL algorithms                                       
│  ├─ tests                             # Test system of commmonroad-rl.
│  ├─ tools                             # Tools to validate, visualize and analyze CommonRoad .xml files, as well as preprocess and convert to .pickle files.                                         
│  ├─ utils_run                         # Utility functions to run training, tuning and evaluating files                                      
│  ├─ README.md                                                      
│  ├─ evaluate_model.py                 # Script to evaluate a trained RL model on specific scenarios and visualize the scenario                
│  ├─ generate_solution.py              # Script to genearte CommonRoad solution files from trained RL models.
│  ├─ train_model.py                    # Script to train RL model or optimize hyperparameters or environment configurations           
│  ├─ sensitivity_analysis.py           # Script to run sensitivity analysis for a trained model
│  └─ plot_learning_curves.py           # Plot learning curves with provided training log files.                
├─ scripts                              # Bash scripts to install all dependencies, train and evaluate RL models, as well as generate CommonRoad solution files from trained RL models.
├─ README.md                                            
├─ commonroad_style_guide.rst           # Coding style guide for this project                
├─ environment.yml                                      
└─ setup.py                                      
```
## Installation

### Installation using Docker
Detailed instructions under ```./commonroad_rl/install_docker/readme_docker.md```

### Prerequisites 
This project should be run with conda. Make sure it is installed before proceeding with the installation.

1. [download & install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html), and init anaconda to work from the terminal. tested on conda 4.5; 4.9, 4.10
```
~/anaconda3/bin/conda init
# for minconda
~/miniconda3/bin/conda init
```
2. clone this repository
```
git clone https://gitlab.lrz.de/tum-cps/commonroad-rl.git
```
3. install build packages
```
sudo apt-get update
sudo apt-get install build-essential make cmake
```
4. setup a new conda env (or install packages to an existing conda env e.g. myenv `conda env update --name myenv --file environment.yml`)
```
conda env create -n cr37 -f environment.yml
git lfs pull
```
(optional) Install [`commonroad-interactive-scenarios`](https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios) 
if you want to evaluate a trained model with SUMO interactive scenarios.

5. (optional) install pip packages for the docs. If you want to use the jupyter notebook, also install jupyter.
```
source activate cr37
pip install -r commonroad_rl/doc/requirements_doc.txt
conda install jupyter
```

### Install mpi4py and commonroad-rl manually
```
conda install --quiet -y  -c conda-forge mpi4py==3.1.3
pip install -e .
```


### Test if installation succeeds

Further details of our test system refer to `./commonroad_rl/tests`.

```
source activate cr37
bash scripts/run_test.sh
```

## Usage

### Tutorials
To get to know the package, please check `./commonroad_rl/tutorials` for further details.

### Python scripts
The commonroad_rl folder contains the source files. There are Python scripts for training, evaluating, and visualizing models. The most important scrips are explained in `./commonroad_rl/README.md` and can be run with your Python executable. They are especially useful if you are developing a new feature or want to debug a specific part of the training.

### Bash scripts
If you tested your codes already and everything runs smoothly on your computer and you now want to run the real experiments on larger dataset, the bash scripts help you with that. The are located in `./scripts`. They can be used for training with PPO and TD3 and testing an agent. Always adapt the specific paths in the scripts to the corresponding paths on your machine and check the comments in the file to determine which arguments have to be provided.  

## References and Suggested Guides
 
1. [OpenAI Stable Baselines](https://stable-baselines.readthedocs.io/en/master/): the implementation of RL algorithms used in our project.
2. [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html): we do not use their implementations in our project. But they provide quite nice explanations of RL concepts.
3. [OpenAI Gym](https://gym.openai.com/docs/): general interface.
4. [OpenAI Safety Gym](https://openai.com/blog/safety-gym/): a special collection of Gyms for safe RL. Configurable as our project.

## Publication

If you use CommonRoad-RL in your paper, please cite:
```
@inproceedings{Wang2021,
	author = {Xiao Wang and  Hanna Krasowski and  Matthias Althoff},
	title = {{CommonRoad-RL}: A Configurable Reinforcement Learning Environment for Motion Planning of Autonomous Vehicles},
	booktitle = {Proc. of the IEEE International Conference on Intelligent Transportation Systems (ITSC)},
	year = {2021},
	pages={466--472},
}
```

Configurations and trained models used in our experiments in the paper can be downloaded [here](https://nextcloud.in.tum.de/index.php/s/n7oEr9dsyrqjgPZ).

Models trained with current version of code using the same configurations can be downloaded [here](https://nextcloud.in.tum.de/index.php/s/F8C9n2nWmfJy9pr)

## Contact:
commonroad@lists.lrz.de
