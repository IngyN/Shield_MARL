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

        # individual/local info
        self.qvalues = np.zeros([nagents, self.env.nrows, self.env.ncols, nactions])
        self.marks = np.zeros([nagents, self.env.nrows, self.env.ncols], dtype=int)  # 0 = unmarked, 1= safe, 2= unsafe

        # Joint info
        self.dangerous_max = 50
        dims = [self.dangerous_max]
        dims.extend([self.env.nrows, self.env.ncols, self.nactions] * nagents)
        dims.append(1)
        self.joint_marks = np.zeros([self.nagents, self.dangerous_max, 2 * self.nagents + 1],
                                    dtype=int)  # joint state marks
        # i, row,col,.., row,col, action_i, confidence
        # TODO check dims are applied correctly.
        dims = [self.dangerous_max, nagents]
        dims.extend([self.env.nrows, self.env.ncols] * nagents)
        dims.append(self.nactions)
        self.joint_qvalues = np.zeros(dims)

        # t-test vars
        self.test_threshold = 0.05  # 5%
        self.conf_threshold = 5
        self.conf_max = 50
        self.start_conf = 10
        self.nsaved = 5  # history for observed/expected rewards

        self.W1 = np.ones([nagents, self.env.nrows, self.env.ncols, nactions,
                           self.nsaved + 1]) * np.nan  # nan vals were not initialized

        # self.initialize_qvalues()

    def state_to_index(self, i, j):
        return j + i * self.env.ncols

    def initialize_qvalues(self):
        single_env = GridEnv(agents=1, map_name=self.map_name)
        singleQL = QLearning([single_env.nrows, single_env.ncols])

        for a in range(self.nagents):
            qval, hist = singleQL.run(single_env, step_max=400, episode_max=150, discount=0.9,
                                      debug=False, save=True, N=self.nsaved)
            self.qvalues[a] = deepcopy(qval)
            self.W1[a] = deepcopy(hist)

        self.W2 = deepcopy(self.W1)

    def greedy_select(self, qval, epsilon=0.4):
        flag = random.random()
        if flag >= epsilon:
            action = np.argmax(qval)  # choose from qvalues
        else:
            action = random.randint(0, self.nactions - 1)  # choose from random
        return action

    def retrieve_js(self, states, a):
        ret = np.ones([1, 2 * self.nagents + 1], dtype=int) * -1
        ind = []
        for i in range(self.joint_marks.shape[0]):
            if np.all(self.joint_marks[a][i][2 * a:2 * a + 2] == states[a]):
                if np.all(ret[0] == np.ones([2 * self.nagents + 1], dtype=int) * -1):
                    ret[0] = self.joint_marks[a][i]
                    ind.append(i)
                else:
                    ret = np.vstack([ret, self.joint_marks[a][i]])
                    ind.append(i)
        return ret, ind

    def action_selection(self, states, i):
        a = 0
        if self.marks[i][states[i][0]][states[i][1]] == 2:  # unsafe
            ret, indices = self.retrieve_js(states, i)
            if len(indices) == 0:  # no matching marks for joint states.
                a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])
                self.marks[i][states[i][0]][states[i][1]] = 0  # unmark since it wasnt found -> removed somewhere else
            else:  # check if state infor for the others is the same
                same = np.ones([ret.shape[0], self.nagents], dtype=int) * -1
                same[:, i] = 1
                for row, index in enumerate(ret):
                    for a in range(self.nagents):
                        if a != i:
                            if np.all(row[2 * a:2 * a + 2] == states[a]):
                                same[index] = 1
                found = False
                for row, j in enumerate(same):
                    if np.all(row == np.ones([self.nagents])):  # found one that matches. Should not have duplicates
                        found = True
                        if ret[j][-1] < self.conf_max:
                            self.joint_marks[i][indices[j]][-1] += 1
                        # TODO: greedy select from joint
                        a = self.greedy_select(self.joint_qvalues[i][states[i][0]][states[i][1]])
                    else:
                        if ret[j][-1] > 0:
                            self.joint_marks[i][indices[j]][-1] -= 1
                            if self.joint_marks[i][indices[j]][-1] < self.conf_threshold:
                                # TODO: remove from joint Qvalues
                                pass
                if not found:
                    a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])

        else:  # unmarked or safe
            a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])

        return a

    def update(self, states, obs, rewards, actions):
        # TODO:analyze rewards with t-test and determine if dangerous.
        joint_state = states  # TODO: fix this

        for i in range(self.nagents):
            if self.is_dangerous(self.W2[i][states[i][0]][states[i][1]][actions[i]][:-1],
                                 self.W1[i][i][states[i][0]][states[i][1]][actions[i]][:-1]):
                # mark state. for agent a
                # TODO figure this out
                pass

            elif self.marks[i][state[0]][state[1]] == 0 or self.joint_marks[joint_state] == 1:
                # do nothing
                pass

            else:
                # update joint qvalues
                self.joint_qvalues = 0  # TODO: update correctly

    def is_dangerous(self, rk, rewards, expected_rew):
        # TODO : take an action state combination +rewards and determine if state should e marked as dangerous
        flag = False
        t, p_val = ttest_ind(rewards, expected_rew, equal_var=False)  # TODO: make two sample t-test
        if p_val < self.test_threshold:
            # reject hypothesis of equal means -> dangerous
            t, p_val = ttest_ind(rk, rewards, equal_var=False)  # single sample
            if p_val < self.test_threshold:  # check if rk < mean of W2
                # tODO: add to js and update marks.
                flag = True

        return flag

    def update_W(self, pos, actions, rew):

        for i in range(self.nagents):

            index = self.W1[i][pos[0]][pos[1]][actions[i]][-1]
            index2 = self.W2[i][pos[0]][pos[1]][actions[i]][-1]
            nextval = self.W1[pos[0]][pos[1]][actions[i]][int(index)]

            if np.isnan(index):  # check if never updated
                index = 0
                self.W1[pos[0]][pos[1]][actions[i]][int(index)] = rew[i]
                self.W1[i][pos[0]][pos[1]][actions[i]][-1] = (index + 1) % self.nsaved

            elif np.isnan(nextval):  # check if there are still spaces left
                self.W1[pos[0]][pos[1]][actions[i]][int(index)] = rew[i]
                self.W1[i][pos[0]][pos[1]][actions[i]][-1] = (index + 1) % self.nsaved

            elif np.isnan(index2):  # add to W2
                index2 = 0
                self.W2[pos[0]][pos[1]][actions[i]][int(index2)] = rew[i]
                self.W2[i][pos[0]][pos[1]][actions[i]][-1] = (index2 + 1) % self.nsaved
            else:
                self.W2[pos[0]][pos[1]][actions[i]][int(index2)] = rew[i]
                self.W2[i][pos[0]][pos[1]][actions[i]][-1] = (index2 + 1) % self.nsaved


    def run(self, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False):

        alpha_index = 1
        self.discount = discount
        # print(pos)
        steps = np.zeros([episode_max], dtype=int)
        actions = np.zeros([self.nagents], dtype=int)

        for e in range(episode_max):
            self.env.reset()

            pos = self.env.pos

            for s in range(step_max):

                self.alpha = alpha_index / (0.1 * s + 0.5)

                for a in range(self.nagents):
                    actions[a] = self.action_selection(pos, a)  # select actions for each agent

                # update environment and retrieve rewards:
                obs, rew, _, done = self.env.step(actions)  # sample rewards and new states
                self.update_W(pos, actions, rew)  # Update observed rewards. l. 11 in pseudocode.

                self.update(pos, obs, rew, actions)  # update marks and qvalues

                pos = deepcopy(obs[0])

                if done:
                    steps[e] = s
                    break

if __name__ == "__main__":
    cq = CQLearning(nagents=3)
    print('done testing initialization')
    cq.joint_marks = np.array([[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                               [[2, 3, 5, 5, 7, 8, 2], [2, 4, 5, 6, 7, 7, 2], [3, 3, 5, 6, 7, 8, 2]],
                               [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]])
    r, ind = cq.retrieve_js(states=np.array([[2, 2], [5, 6], [6, 6]]), a=1)
    print('Ind: ', ind)
    print('Ret: ', r)
