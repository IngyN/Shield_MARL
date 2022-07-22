# Shield_MARL

This repository contains the code for the [Safe multi-agent reinforcement learning via shielding](https://arxiv.org/pdf/2101.11196) paper. 

#### Prerequisites:
- Python 3.6+
- gym
- matplotlib 3.0.0 
- [multi-agent gridworld](https://github.com/IngyN/gym-grid-ma) for gridworld experiments
- [particle environment](https://github.com/openai/multiagent-particle-envs) for deep MARL experiments (modified to be discretized - code missing).

#### Code Structure:
- `CQLearning.py`: contains an implementation of CQ-Learning (non-deep) following the [Game Theory and Multi-agent Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-3-642-27645-3_14) book.
- `GridShield.py`: contains the implementation of the composed shielding method currently restricted to 2 agents per shield but code can be modified to accomodate more. 
- `QLearning.py` : contains an implementation of Q-Learning (also non-deep). 
- `Shield.py`: contains the implementation of the centralized shield method. 
- `parsing.py`: contains the options for running the code. 
- `smoothing.py` and `plotting.py`: for smoothing traces and plotting the accumulated rewards. 
- `run_exp_CQ.py` and `run_exp_QL.py`: to run experiments with QLearning or CQLearning. For example: `python run_exp_CQ.py -n 2 -p 1 -g 1 -i 10 -h 7 -q 0.12 -a 0.8 -w 1 -r 20 -t 1200 -e MIT_test_1` runs an experiment with CQLearning with 2 agents, with shielding active, with composed shielding, for 10 iterations, with 7 runs saved as traces, with a CQ test threshold of 0.12, with a learning rate of 0.8, a discount of 1, collision cost of 20 and 1200 episodes and the relevant produced files will have experiment name: MIT_test_1. 
- `/shields`: contains the centralized shield files produced by the [Slugs](https://github.com/VerifiableRobotics/slugs) tool and the folder `/shields/grid_shields` are the shields necessary for the composed shield method for each map. 
- `/shield_synthesis`: contains the files relevant for shield synthesis:   
    - `ControlParser.py` (I am not the author of this code): converts slugs output files to a json shield file.
    - `gen_shield_grid.py`: creates composed shields description files based on map and number of agents. 
    - `compile_all_grid.sh`: shell script file that takes a map name and number of agents and creates the shield files by calling the relevant code for composed shields.
    - `grid_world.py` and `grid_preprocessing.py` (based on my co-author Suda Bharadwaj's code): code used by the `gen_shield_grid.py` script. 
    - `maps.zip`: contains the map files, (mpe for multiagent particle environment). 
  
- `/maps`: contains the map information for each grid map. 
- `/graph_data`: containes some traces that were generated when running experiments. 

#### Notes:
- Code is provided as is and not actively maintained at the moment. However, I am happy to answer questions.
