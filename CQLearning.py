import numpy as np
from QLearning import QLearning
from copy import deepcopy
from gym_grid.envs import GridEnv
from scipy.stats import ttest_ind, ttest_ind_from_stats
import random


class CQLearning:
    def __init__(self, map_name='example', nagents=2, nactions=5):
        self.nagents = nagents
        self.nactions = nactions
        self.map_name = map_name
        self.env = GridEnv(agents=nagents, map_name=map_name)  # TODO: set up ma environment
        self.nstates = self.env.nstates

        # individual/local info
        self.qvalues = np.zeros([nagents, self.nstates, nactions])
        self.marks = np.zeros([nagents, self.env.nrows, self.env.ncols], dtype=int)  # 0 = unmarked, 1= safe, 2= unsafe

        # Joint info
        self.dangerous_max = 50 * nagents
        self.joint_marks = np.zeros([self.dangerous_max, 3 * nagents], dtype=int)  # joint state marks
        dims = [self.dangerous_max, nagents]
        dims.extend([self.nstates] * nagents)
        dims.append(self.nactions)
        self.joint_qvalues = np.zeros(dims)

        # t-test vars
        self.test_threshold = 0.05  # 5%
        self.conf_threshold = 5
        self.start_conf = 10
        self.nsaved = 5  # history for observed/expected rewards

        self.W1 = np.ones([nagents, self.nstates, nactions, self.nsaved + 1]) * np.nan  # nan vals were not initialized

        self.initialize_qvalues()

    def state_to_index(self, i, j):
        return j + i * self.env.ncols

    def initialize_qvalues(self):
        single_env = GridEnv(agents=1, map_name=self.map_name)
        singleQL = QLearning([single_env.nrows, single_env.ncols])

        for a in range(self.nagents):
            qval, hist = singleQL.run(single_env, step_max=1000, episode_max=500, discount=0.9, debug=False, save=True,
                                      N=self.nsaved)
            self.qvalues[a] = deepcopy(qval)
            self.W1[a] = deepcopy(hist)

        self.W2 = deepcopy(self.W1)

    def greedy_select(self, qval, epsilon=0.7):
        flag = random.random()
        if flag >= epsilon:
            action = np.argmax(qval)  # choose from qvalues
        else:
            action = random.randint(0, self.nactions - 1)  # choose from random
        return action

    def action_selection(self, state, i):
        # TODO : select action based on marked and previous Q-values
        if self.marks[i][state[0]][state[1]] == 0:  # if local state is unmarked.
            a = self.greedy_select(self.qvalues[i][state[0]][state[1]])

        elif self.joint_marks:  # if safe
            a = self.greedy_select(self.qvalues[i][state[0]][state[1]])

        else:  # select from joint qvalues.
            a = self.greedy_select(self.joint_qvalues[i][state[0]][state[1]])

        return a

    def update(self, i, state, rewards):
        # TODO:analyze rewards with t-test and determine if dangerous.
        joint_state = 0  # TODO: fix this
        if self.is_dangerous(rewards, self.observed_rewards[state[0]][state[1]][i]):
            # mark state. for agent a
            # TODO figure this out
            pass

        elif self.marks[i][state[0]][state[1]] == 0 or self.joint_marks[joint_state] == 1:
            # do nothing
            pass

        else:
            # update joint qvalues
            self.joint_qvalues = 0  # TODO: update correctly

    def is_dangerous(self, reward, expected_rew):
        # TODO : take an action state combination +rewards and determine if state should e marked as dangerous
        t, p_val = ttest_ind(reward, expected_rew, equal_var=False)
        if p_val < self.threshold:
            # reject hypothesis of equal means -> dangerous
            pass
        else:
            # cannot reject -> dangerous
            pass

    def run(self, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False):

        alpha_index = 1
        self.discount = discount
        # print(pos)
        steps = np.zeros([episode_max], dtype=int)
        actions = np.zeros([self.nagents], dtype=int)
        stop = False

        for e in range(episode_max):
            self.env.reset()
            pos = self.env.pos

            for s in range(step_max):

                for a in range(self.nagents):  # action selection
                    actions[a] = self.action_selection(pos[a], a)

                # update environment and retrieve rewards:
                obs, rew, _, done = self.env.step(actions)

                for a in range(self.nagents):
                    state = obs[a]
                    self.update(a, obs[a], rew[a])  # update marks and qvalues


if __name__ == "__main__":
    cq = CQLearning()
