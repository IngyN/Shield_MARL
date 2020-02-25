import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from mpl_toolkits.axes_grid1 import host_subplot


def plot_and_save(data, agents, map='example', test=False, shielding=False, save=True, display=False):
    plt.ioff()

    fig = plt.figure(2)
    plt.plot(np.arange(1, data['episodes'] + 1), data['steps'])
    plt.title('Steps per episode')
    if save:
        if test:
            fig.savefig('figures/cq_' + m + '_' + str(agents) + '_test_steps.png', bbox_inches='tight')
        else:
            fig.savefig('figures/cq_' + m + '_' + str(agents) + '_train_steps.png', bbox_inches='tight')

    fig4 = plt.figure(4)
    plt.plot(np.arange(1, episode_max + 1), acc[:, 0])
    plt.plot(np.arange(1, episode_max + 1), acc[:, 1])
    plt.title('Accumulated rewards per agent per episode')
    if save:
        fig4.savefig('figures/cq_' + m + '_' + str(agents) + '_train_acc.png', bbox_inches='tight')

    fig5 = plt.figure(5)
    plt.plot(np.arange(1, episode_max + 1), coll)
    plt.title('Collisions')
    if save:
        fig5.savefig('figures/cq_' + m + '_' + str(agents) + '_train_coll.png', bbox_inches='tight')

    if shielding:
        fig6 = plt.figure(6)
        plt.plot(np.arange(1, episode_max + 1), inter[0, :])
        plt.plot(np.arange(1, episode_max + 1), inter[1, :])
        plt.title('Shield Interference')
        if save:
            fig6.savefig('figures/cq_' + m + '_' + str(agents) + '_train_inter.png', bbox_inches='tight')

    if display:
        plt.show()


def plot_v2(data, agents, map='example', test=False, shielding=False, save=True, display=False):
    plt.ioff()
    # Sent for figure
    # font = {'size': 12}
    # matplotlib.rc('font', **font)

    # Setup figure and subplots
    f1 = plt.figure(num=1, figsize=(12, 12))  # , dpi = 300)
    if not test:
        f1.suptitle("Training Results for " + str(agents) + ' agents on the ' + map + ' map', fontsize=14)
    else:
        f1.suptitle("Test Results for " + str(agents) + ' agents on the ' + map + ' map', fontsize=14)

    ax01 = plt.subplot2grid((2, 2), (0, 0))
    ax02 = plt.subplot2grid((2, 2), (0, 1))
    ax03 = plt.subplot2grid((2, 2), (1, 0))
    ax04 = plt.subplot2grid((2, 2), (1, 1))
    # plt.tight_layout()

    # Set titles of subplots
    ax01.set_title('Steps per episode')
    ax02.set_title('Accumulated rewards per agent per episode')
    ax03.set_title('Collisions per Episode')
    ax04.set_title('Interferences per Episode')

    # set y-limits
    ax01.set_ylim(0, 500)
    ax02.set_ylim(-2000, 500)
    ax03.set_ylim(0, 500)
    ax04.set_ylim(0, 500)

    # set x-limits
    ax01.set_xlim(0, data['episodes'])
    ax02.set_xlim(0, data['episodes'])
    ax03.set_xlim(0, data['episodes'])
    ax04.set_xlim(0, data['episodes'])

    # Turn on grids
    ax01.grid(True)
    ax02.grid(True)
    ax03.grid(True)
    ax04.grid(True)

    # set label names
    ax01.set_xlabel("Episodes")
    ax01.set_ylabel("Steps")
    ax02.set_xlabel("Episodes")
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

    # set lagends

    ax02.legend(acc_plots, acc_labels)
    if shielding:
        ax04.legend(inter_plots, inter_labels)

    # # Plot Update
    # figure(0)
    # p011.set_data(t, yp1)
    # p012.set_data(t, yp2)
    #
    # p021.set_data(t, yv1)
    # p022.set_data(t, yv2)
    #
    # p031.set_data(t, yp1)
    # p032.set_data(t, yv1)

    if save:
        if not test:
            f1.savefig('figures/cq_' + map + '_' + str(agents) + '_train.png', bbox_inches='tight')
        else:
            f1.savefig('figures/cq_' + map + '_' + str(agents) + '_test.png', bbox_inches='tight')

    if display:
        plt.show()
