from CQLearning import CQLearning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']
agents = 2

# Loop over all maps
for m in map_names:
    cq = CQLearning(map_name=m, nagents=agents)
    steps_test = 50
    ep_test = 10

    i_step_max, i_episode_max, step_max, episode_max = cq.get_recommended_training_vars()

    cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max)

    s, _, _ = cq.run(step_max=step_max, episode_max=episode_max, debug=False)
    print('steps train: \n', s)

    s2, _, _ = cq.run(step_max=steps_test, episode_max=ep_test, testing=True, debug=True)
    print('steps test: \n', s2)

    plt.ioff()
    fig = plt.figure(2)
    plt.plot(np.arange(1, episode_max + 1), s)
    fig.savefig('figures/cq_' + m + '_train.png', bbox_inches='tight')
    plt.title('Training steps')

    fig3 = plt.figure(3)
    plt.plot(np.arange(1, ep_test + 1), s2)
    plt.title('Testing steps')
    fig3.savefig('figures/cq_' + m + '_test.png', bbox_inches='tight')

    plt.show()
