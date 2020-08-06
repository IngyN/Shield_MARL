from QLearning import QLearning
from plotting import plot_v2
from parsing import get_options
from CustomLogger import CustomLogger
import numpy as np
from gym_grid.envs import GridEnv
from copy import deepcopy

# map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']
map_names = ['ISR', 'Pentagon', 'MIT', 'SUNY']
# map_names = ['ISR', 'Pentagon']

agents, shielding, iterations, display, save, grid, fair, extra, collision_cost, alpha, discount, episodes, d_max, \
t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop = get_options()
steps_test = 100
ep_test = 10
conv = False
convs = dict.fromkeys(map_names, [])
# collision_cost = 30
last = 500
logger = CustomLogger(agents)
print('Collision cost : ', collision_cost, ' - Shielding :', shielding)


def format_data(steps, coll, ep, acc):
    info = {}
    info['steps'] = steps
    info['collisions'] = coll
    info['episodes'] = ep
    info['acc_rewards'] = acc

    return info


def run_joint(env, nagents, qls, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False,
              epsilon=0.8, c_cost=10, noop=noop):
    alpha_index = 1
    for a in range(nagents):
        qls[a].discount = discount
    # print(pos)
    coef = (-0.05 + epsilon) / episode_max
    start_ep = 1
    steps = np.zeros([episode_max], dtype=int)
    coll = np.zeros([episode_max], dtype=int)
    action = np.zeros([nagents], dtype=int)
    acc_rew = np.zeros([episode_max, nagents])

    # print(action)
    stop = False
    for e in range(episode_max):
        env.reset()
        pos = deepcopy(env.pos)

        for s in range(step_max):

            if debug:
                env.render(episode=e + 1)

            ep = start_ep - e * coef
            for a in range(nagents):
                qls[a].alpha = alpha_index / (0.1 * s + 0.5)
                if not testing:
                    # print(pos[a])
                    action[a] = qls[a].action_selection(qls[a].qvalues[pos[a][0]][pos[a][1]], epsilon=ep)
                else:
                    action[a] = qls[a].action_selection(qls[a].qvalues[pos[a][0]][pos[a][1]], epsilon=0.05)

            obs, rew, info, done = env.step(action, collision_cost=c_cost, noop=noop)
            coll[e] += info['collisions']
            acc_rew[e] += rew

            for a in range(nagents):
                if not testing:
                    qls[a].update(pos[a], obs[a], rew[a], action[a], True)

            pos = deepcopy(obs)

            if np.all(done):
                steps[e] = s
                if debug:
                    env.render(episode=e + 1)
                break

        steps[e] = s
        # if e > 80:
        #     stop = True
        #     for stop_step in range(80):
        #         if steps[e] != steps[e - stop_step]:
        #             stop = False
        #
        # if stop:
        #     break
    # print(steps + 1)
    return steps, coll, acc_rew


# Loop over all maps
for m in map_names:

    env = GridEnv(nagents=agents, map_name=m, norender=False)  # set up ma environment
    qls = []
    # print(qls)
    # initialize array of QL agents.
    for a in range(agents):
        temp = QLearning(map_size=[env.nrows, env.ncols])
        qls.append(temp)
    # print(qls)
    i_step_max = 500
    i_episode_max = 1000

    if episodes is not None:
        i_episode_max = episodes

    train_data = []
    test_data = []
    done = False
    i = 0
    print('map : ', m)
    while not done:
        # print("\n *************************** map : ",m," iteration ", i+1, "/", iterations, "**************************")

        s, coll, acc = run_joint(env=env, nagents=agents, qls=qls, step_max=i_step_max, episode_max=i_episode_max,
                            discount=discount, c_cost=collision_cost, noop=noop)

        train_data_i = format_data(s, coll, i_episode_max, acc)

        s2, coll2, acc2 = run_joint(env, agents, qls, step_max=steps_test, episode_max=ep_test,
                              discount=discount, c_cost=collision_cost, noop=noop, testing=True, debug=False)

        test_data_i = format_data(s2, coll2, ep_test, acc2)

        train_data.append(train_data_i)
        test_data.append(test_data_i)

        i += 1
        if i >= iterations:
            done = True

        for a in range(agents):
            qls[a].reset()

    # Log information
    logger.log_results_QL(m, test_data, train_data, iterations)

logger.save('QL', extra=extra)
