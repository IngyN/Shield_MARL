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
save = True


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

    # training
    s = np.zeros([iterations, episode_max], dtype=int)
    acc = np.zeros([iterations, episode_max, agents])
    coll = np.zeros([iterations, episode_max])
    inter = np.zeros([iterations, agents, episode_max])

    # testing
    s2 = np.zeros([iterations, ep_test], dtype=int)
    acc2 = np.zeros([iterations, ep_test, agents])
    coll2 = np.zeros([iterations, ep_test])
    inter2 = np.zeros([iterations, agents, ep_test])

    for i in range(iterations):
        cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max)

        s[i], acc[i], coll[i], inter[i] = cq.run(step_max=step_max, episode_max=episode_max,
                                                 debug=False, shielding=shielding)
        train_data = format_data(s[i], acc[i], coll[i], inter[i], episode_max)
        plot_v2(train_data, agents=agents, map=m, test=False, shielding=shielding, save=save, display=False)

        s2[i], acc2[i], coll2[i], inter2[i] = cq.run(step_max=steps_test, episode_max=ep_test,
                                                     testing=True, debug=True, shielding=shielding)
        test_data = format_data(s2[i], acc2[i], coll2[i], inter2[i], ep_test)
        plot_v2(test_data, agents=agents, map=m, test=True, shielding=shielding, save=save, display=display)

        cq.reset()
