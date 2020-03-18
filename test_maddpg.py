import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from Shield import Shield
from copy import deepcopy
from multiagent.environment import MultiAgentEnv
import matplotlib
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--map", type=str, default="example", help="name of the map")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=400, help="number of episodes")
    parser.add_argument("--nagents", type=int, default=2, help="number of agents")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="number of episodes to optimize at the same time")  # recommended = 1024
    parser.add_argument("--num-units", type=int, default=3, help="number of units in the mlp")
    parser.add_argument("--shield", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    # parser.add_argument("--benchmark", action="store_true", default=False)
    # parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    # parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(arglist):
    from gym_grid.envs import GridEnv
    env = GridEnv(nagents=arglist.nagents, map_name=arglist.map, norender=False)
    return env


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.nagents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    return trainers


def load_shield(map, env):
    dir = 'shields/collision_' + map + '_opt.shield'
    shield = Shield(env.nagents, start=env.start_pos, file=dir)
    return shield


def plot(steps):
    plt.ioff()
    plt.figure(2)
    plt.plot(np.arange(1, len(steps) + 1), steps)
    # fig.savefig('test.png', bbox_inches='tight')
    plt.title('Training steps')
    plt.show()


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist)
        # Create agent trainers
        obs_shape_n = [env.observation_space.shape[0] for i in range(env.nagents)]
        trainers = get_trainers(env, obs_shape_n, arglist)
        num_adversaries = 0
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        # if arglist.load_dir == "":
        #     arglist.load_dir = arglist.save_dir
        # if arglist.display or arglist.restore or arglist.benchmark:
        #     print('Loading previous state...')
        #     U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.nagents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        steps = np.zeros([arglist.num_episodes], dtype=int)
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        print('num_units :', arglist.num_units)

        if arglist.shield:
            pre_actions = np.zeros([env.nagents])
            shield = load_shield(arglist.map, env)
            shield.reset()

        print('Starting training...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # print(action_n.flatten())
            actions = [np.argmax(action_n[i]) for i in range(env.nagents)]

            if arglist.shield:  # shield check.
                pre_actions = deepcopy(actions)
                actions = shield.step(actions, env.goal_flag)
                punish = (pre_actions != actions)
                # if len(interference[punish]) > 0:
                #     idx_values = np.where(punish == True)[0]
                #     for idx in idx_values:
                #         interference[idx][e] += 1

            # environment step
            new_obs_n, rew_n, info_n, done_n = env.step(actions)

            episode_step += 1
            terminal = (episode_step >= arglist.max_episode_len)
            pre_obs_n = np.zeros([env.nagents])
            # add an extra experience here for shield. save new actions as [0, 0, 1, 0, 0] for 2
            if arglist.shield:  # punish pre_actions that were changed extra.
                if not np.all(punish == False):
                    for i in range(env.nagents):
                        pre_obs_n[i], _ = env.get_next_state(obs_n[i], pre_actions[i], done_n[i])
                    rew_shield = deepcopy(rew_n)
                    rew_shield[punish] = -10
                    action_bin = np.zeros([env.nagents, len(action_n[0])])
                    for i, agent in enumerate(trainers):
                        action_bin[i][actions[i]] = 1
                        agent.experience(obs_n[i], action_bin[i], rew_shield[i], pre_obs_n[i], done_n[i], terminal)

            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if np.all(done_n) or terminal:
                obs_n = env.reset()
                if arglist.shield:
                    shield.reset()
                # print('Episode step :', episode_step)
                episode_rewards.append(0)
                steps[len(episode_rewards) - 2] = episode_step
                episode_step = 0
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                terminal = True

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            # if arglist.benchmark:
            #     for i, info in enumerate(info_n):
            #         agent_info[-1][i].append(info_n['n'])
            #     if train_step > arglist.benchmark_iters and (done or terminal):
            #         file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
            #         print('Finished benchmarking, now saving...')
            #         with open(file_name, 'wb') as fp:
            #             pickle.dump(agent_info[:-1], fp)
            #         break
            #     continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render(episode=len(episode_rewards) + 1)
                continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.test:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

            # save model, display training output
            debug_rate = 50
            if terminal and (len(episode_rewards) % debug_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-debug_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) >= arglist.num_episodes:
                # rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                # with open(rew_file_name, 'wb') as fp:
                #     pickle.dump(final_ep_rewards, fp)
                # agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                # with open(agrew_file_name, 'wb') as fp:
                #     pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes, Mean rew/ep = {}.'.format(len(episode_rewards),
                                                                                   np.mean(episode_rewards)))
                print(steps[:-1])
                plot(steps[:-1])
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
