from CQLearning import CQLearning
from plotting import plot_v2
from parsing import get_options
from custom_logger import CustomLogger
import numpy as np


# map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']
map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']

agents, shielding, iterations, display, save = get_options()
steps_test = 100
ep_test = 10

logger = CustomLogger(agents)


def format_data(steps, acc, coll, inter, ep):
    info = {}
    info['steps'] = steps
    info['acc_rewards'] = acc
    info['collisions'] = coll
    info['interferences'] = inter
    info['episodes'] = ep

    return info


# Loop over all maps
for m in map_names:
    cq = CQLearning(map_name=m, nagents=agents)
    i_step_max, i_episode_max, step_max, episode_max = cq.get_recommended_training_vars()

    train_data = []
    test_data = []

    for i in range(iterations):
        cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max)

        s, acc, coll, inter = cq.run(step_max=step_max, episode_max=episode_max,
                                     debug=False, shielding=shielding)
        train_data_i = format_data(s, acc, coll, inter, episode_max)
        plot_v2(train_data_i, agents=agents, iteration=i + 1, map=m, test=False, shielding=shielding, save=save,
                display=False)

        s2, acc2, coll2, inter2 = cq.run(step_max=steps_test, episode_max=ep_test,
                                         testing=True, debug=display, shielding=shielding)
        test_data_i = format_data(s2, acc2, coll2, inter2, ep_test)
        plot_v2(test_data_i, agents=agents, iteration=i + 1, map=m, test=True, shielding=shielding, save=save,
                display=display)

        train_data.append(train_data_i)
        test_data.append(test_data_i)

        cq.reset()

    # Log information
    logger.log_results(m, test_data, train_data, shielding, iterations)

# print(logger.df)
# print(logger.raw_df)
logger.save('CQ+shield')
