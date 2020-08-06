from CQLearning import CQLearning
from plotting import plot_v2
from parsing import get_options, save_param
from CustomLogger import CustomLogger
import numpy as np


# map_names = ['example', 'ISR', 'Pentagon', 'MIT', 'SUNY']
# map_names = ['ISR', 'Pentagon', 'MIT', 'SUNY']
# map_names = ['ISR', 'Pentagon', 'MIT', 'SUNY']
map_names=['MIT']

agents, shielding, iterations, display, save, grid, fair, extra, collision_cost, alpha, discount, episodes , d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop = get_options()
steps_test = 100
ep_test = 10
conv = False
convs = dict.fromkeys(map_names, [])
# collision_cost = 30
last = 500
# noop = True
logger = CustomLogger(agents)
print('Collision cost : ', collision_cost, ' - Shielding :', shielding, ' - noop : ', noop, '- grid: ', grid)

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

    cq = CQLearning(map_name=m, nagents=agents, grid=grid, alpha=alpha, disc=discount,
                    d_max=d_max, t_thresh=t_thresh, c_thresh=c_thresh, c_max=c_max,
                    start_c=start_c, d=delta, ns=nsaved)

    i_step_max, i_episode_max, step_max, episode_max = cq.get_recommended_training_vars()

    if episodes is not None:
        episode_max = episodes

    train_data = []
    test_data = []
    done = False
    i = 0
    print('map : ', m)
    while not done:
        # print("\n *************************** map : ",m," iteration ", i+1, "/", iterations, "**************************")
        cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max, c_cost=collision_cost, noop=noop)

        s, acc, coll, inter = cq.run(step_max=step_max, episode_max=episode_max, noop = noop,
                                     debug=False, shielding=shielding, grid=grid, fair=fair, c_cost=collision_cost)

        # print('  ---- Conv : ', np.mean(acc[-last:]))

        convs[m] = convs[m] + [np.mean(acc[-last:])]

        train_data_i = format_data(s, acc, coll, inter, episode_max)
        plot_v2(train_data_i, agents=agents, iteration=i + 1, map=m, test=False, shielding=shielding, save=save,
                display=False)

        s2, acc2, coll2, inter2 = cq.run(step_max=steps_test, episode_max=ep_test, noop = noop,
                                         testing=True, debug=display, shielding=shielding, grid=grid, fair=fair, c_cost=collision_cost)
        test_data_i = format_data(s2, acc2, coll2, inter2, ep_test)
        plot_v2(test_data_i, agents=agents, iteration=i + 1, map=m, test=True, shielding=shielding, save=save,
                display=display)

        train_data.append(train_data_i)
        test_data.append(test_data_i)

        i+= 1
        if i >= iterations:
            done = True

        cq.reset()


    # Log information
    logger.log_results(m, test_data, train_data, shielding, iterations)

print('  ---- Convs : \n', convs)
# print(logger.df)
# print(logger.raw_df)
if shielding:
    date_str=logger.save('CQ+shield', grid=grid, fair=fair, extra=extra)
else:
    date_str=logger.save('CQ', grid=grid, fair=fair, extra=extra)

save_param(date_str,agents, shielding, iterations, display, save, grid, fair, extra, collision_cost, alpha, discount, episodes , d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop)

