from CQLearning import CQLearning
from plotting import plot_v2
import numpy as np

# map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']
map_names = ['example']

agents = 2
shielding = False
steps_test = 50
ep_test = 10
iterations = 1
display = True
save = False


def format_data(steps, acc, coll, inter, ep):
    info = {}
    info['steps'] = steps
    info['acc_rewards'] = acc
    info['collisions'] = coll
    info['interferences'] = inter
    info['episodes'] = ep

    return info


# Loop over all maps
for i in range(iterations):
    for m in map_names:
        cq = CQLearning(map_name=m, nagents=agents)

        i_step_max, i_episode_max, step_max, episode_max = cq.get_recommended_training_vars()

        cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max)

        s, acc, coll, inter = cq.run(step_max=step_max, episode_max=episode_max, debug=False, shielding=shielding)
        train_data = format_data(s, acc, coll, inter, episode_max)
        plot_v2(train_data, agents=agents, map=m, test=False, shielding=shielding, save=save, display=display)

        # s2, _, _, _ = cq.run(step_max=steps_test, episode_max=ep_test, testing=True, debug=True, shielding=shielding)
        # print('steps test: \n', s2)
        #
        # plot_and_save(shielding=shielding,save=save, display = display)
