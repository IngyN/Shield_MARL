import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from mpl_toolkits.axes_grid1 import host_subplot

def plot_v2(data, agents, map='example', iteration=1, test=False, shielding=False, save=True, display=False, grid=False):
    if display:
        plt.ioff()
    # Sent for figure
    font = {'size': 10}
    matplotlib.rc('font', **font)

    # Setup figure and subplots
    if not test:
        f1 = plt.figure(num=1, figsize=(9, 12), dpi=300)
        f1.set_figheight(9)
        f1.set_figwidth(12)
        f1.canvas.set_window_title('Train')
        f1.suptitle("Training Results for " + str(agents) + ' agents on the ' + map + ' map', fontsize=12)
    else:
        f1 = plt.figure(num=2, figsize=(9, 12))  # , dpi = 300)
        f1.set_figheight(9)
        f1.set_figwidth(12)
        f1.canvas.set_window_title('Test')
        f1.suptitle("Test Results for " + str(agents) + ' agents on the ' + map + ' map', fontsize=12)

    ax01 = plt.subplot2grid((2, 2), (0, 0))
    ax02 = plt.subplot2grid((2, 2), (0, 1))
    ax03 = plt.subplot2grid((2, 2), (1, 0))
    ax04 = plt.subplot2grid((2, 2), (1, 1))
    # plt.tight_layout()

    # Set titles of subplots
    ax01.set_title('Steps per Episode', size=10)
    ax02.set_title('Accumulated Rewards', size=10)
    ax03.set_title('Collisions', size=10)
    ax04.set_title('Interferences', size=10)

    # set y-limits
    if not test:
        ax01.set_ylim(0, 500)
    else:
        ax01.set_ylim(0, 200)
    ax02.set_ylim(-2000, 500)
    ax03.set_ylim(0, 200)
    ax04.set_ylim(0, 200)

    # set x-limits
    ax01.set_xlim(1, data['episodes'])
    ax02.set_xlim(1, data['episodes'])
    ax03.set_xlim(1, data['episodes'])
    ax04.set_xlim(1, data['episodes'])

    # Turn on grids
    ax01.grid(True)
    ax02.grid(True)
    ax03.grid(True)
    ax04.grid(True)

    # set label names
    # ax01.set_xlabel("Episodes")
    ax01.set_ylabel("Steps")
    # ax02.set_xlabel("Episodes")
    ax02.set_ylabel("Accumulated Reward")
    ax03.set_xlabel("Episodes")
    ax03.set_ylabel("Collisions")
    ax04.set_ylabel("Interferences")
    ax04.set_xlabel("Episodes")

    t = np.arange(1, data['episodes'] + 1)
    colors = ['b', 'g', 'r', 'c', 'm']

    # set plots
    p011, = ax01.plot(t, data['steps'], 'b-')  # steps

    acc_plots = []
    acc_labels = []
    for i in range(agents):
        p021, = ax02.plot(t, data['acc_rewards'][:, i], colors[i] + '-', label="agent " + str(i + 1))  # Acc rew
        acc_plots.append(p021)
        acc_labels.append(p021.get_label())

    p031, = ax03.plot(t, data['collisions'], 'b-')  # collisions

    inter_plots = []
    inter_labels = []
    if shielding:
        for i in range(agents):
            p041, = ax04.plot(t, data['interferences'][i, :], colors[i] + '-',
                              label="agent " + str(i + 1))  # interferences
            inter_plots.append(p041)
            inter_labels.append(p041.get_label())

    # set legends
    ax02.legend(acc_plots, acc_labels)
    if shielding:
        ax04.legend(inter_plots, inter_labels)

    plt.subplots_adjust(left=0.12, right=0.9, wspace=0.33, hspace=0.31)
    shield_txt = ['', 'shield_']
    if grid:
        dir_loc = 'figures/' + map + '/4_cq_grid_'
    else:
        dir_loc = 'figures/' + map + '/4_cq_'

    if save:
        if not test:
            f1.savefig(dir_loc + shield_txt[int(shielding)] + map + '_' + str(agents) + '_train_' + str(
                iteration) + '.png',
                       bbox_inches='tight')
        else:
            f1.savefig(dir_loc + shield_txt[int(shielding)] + map + '_' + str(agents) + '_test_' + str(
                iteration) + '.png',
                       bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()
