import numpy as np
from QLearning import QLearning
from copy import deepcopy
from gym_grid.envs import GridEnv
from scipy.stats import ttest_ind, ttest_1samp
import random, time


class CQLearning:
    def __init__(self, map_name='example', nagents=2, nactions=5, norender=False):
        self.nagents = nagents
        self.nactions = nactions
        self.map_name = map_name
        self.env = GridEnv(agents=nagents, map_name=map_name, norender=False)  # set up ma environment
        self.targets = self.env.targets
        self.start = self.env.start_pos

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
        self.mark_index = np.zeros([nagents], dtype=int)

        # i, row,col,.., row,col, confidence
        # TODO check dims are applied correctly.
        dims = [nagents, self.dangerous_max]
        dims.extend([self.env.nrows, self.env.ncols] * nagents)
        dims.append(self.nactions)
        self.joint_qvalues = np.zeros(dims)

        # t-test vars
        self.test_threshold = 0.2  # 5%
        self.test_threshold2 = 0.2
        self.conf_threshold = 1
        self.conf_max = 50
        self.start_conf = 10
        self.nsaved = 5  # history for observed/expected rewards

        self.W1 = np.ones([nagents, self.env.nrows, self.env.ncols, nactions,
                           self.nsaved + 1]) * np.nan  # nan vals were not initialized

    def state_to_index(self, i, j):
        return j + i * self.env.ncols

    def initialize_qvalues(self):
        single_env = GridEnv(agents=1, map_name=self.map_name)
        singleQL = QLearning([single_env.nrows, single_env.ncols])

        for a in range(self.nagents):
            singleQL.reset()
            single_env.set_start(self.start[a])
            single_env.set_targets(self.targets[a])
            # print('i:', a,'start :', self.start[a], ' - target: ', self.targets[a])
            qval, hist = singleQL.run(single_env, step_max=250, episode_max=200, discount=0.9,
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
        # print(states)
        if self.marks[i][states[i][0]][states[i][1]] == 2:  # unsafe
            ret, indices = self.retrieve_js(states, i)
            if len(indices) == 0:  # no matching marks for joint states.
                a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])
                self.marks[i][states[i][0]][states[i][1]] = 0  # unmark since it wasnt found -> removed somewhere else
            else:  # check if state in for for the others is the same
                same = np.ones([ret.shape[0], self.nagents], dtype=int) * -1
                same[:, i] = 1
                for index, row in enumerate(ret):
                    for a in range(self.nagents):
                        if a != i:
                            if np.all(row[2 * a:2 * a + 2] == states[a]):
                                same[index] = 1
                found = False
                for j, row in enumerate(same):
                    if np.all(row == np.ones([self.nagents])):  # found one that matches. Should not have duplicates
                        found = True
                        if ret[j][-1] < self.conf_max:
                            self.joint_marks[i][indices[j]][-1] += 1
                            if self.joint_marks[i][indices[j]][-1] > self.conf_max:
                                self.joint_marks[i][indices[j]][-1] = self.conf_max
                        temp = np.append(i, ret[j][:-1])
                        a = self.greedy_select(self.joint_qvalues.item(tuple(temp)))
                        break
                    else:
                        if ret[j][-1] > 0:
                            self.joint_marks[i][indices[j]][-1] -= 1
                            if self.joint_marks[i][indices[j]][-1] < self.conf_threshold:
                                self.joint_marks[i][indices[j]][-1] = 0  # will be overwritten

                if not found:
                    a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])

        else:  # unmarked or safe
            a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]])

        return a

    def update(self, states, obs, rewards, actions):  # TODO: test.
        joint_state = states.flatten()

        for i in range(self.nagents):
            if self.is_dangerous(rewards[i], self.W2[i][states[i][0]][states[i][1]][actions[i]][:-1],
                                 self.W1[i][states[i][0]][states[i][1]][actions[i]][:-1]):
                #  Mark both joint and local state + update index.
                self.marks[i][states[i][0]][states[i][1]] = 2
                temp = np.append(states.flatten(), self.start_conf)
                self.mark_index[i] = self.find_next_index(i)
                if self.mark_index[i] != -1:
                    self.joint_marks[i][self.mark_index[i]] = temp

            elif self.marks[i][states[i][0]][states[i][1]] == 0:
                # if marked as safe -> do not update local # TODO: check if we should update here too.
                pass

            elif self.mark_index[i] != -1 and np.all(self.joint_marks[i][self.mark_index[i]][:-1] == joint_state):
                # update joint qvalues
                index_joint = np.append(joint_state, actions[i])
                # new val using new states
                m_value = self.alpha * (rewards[i] + self.discount * max(self.qvalues[i][obs[i][0]][obs[i][1]]))
                # old val
                o_value = (1 - self.alpha) * self.joint_qvalues[i][self.mark_index[i]].item(tuple(index_joint))
                self.joint_qvalues[i][self.mark_index[i]].itemset(tuple(index_joint), o_value + m_value)

    def is_dangerous(self, rk, rewards, expected_rew, debug=True):
        #  take an action state combination +rewards and determine if state should e marked as dangerous
        flag = False
        t, p_val = ttest_ind(rewards, expected_rew, equal_var=True, nan_policy='omit')  # two  independent sample t-test
        # if debug:
        #     print('test 1: r', rewards, ' e_r:', expected_rew, ' p_val:', p_val, ' t:', t)
        if p_val < self.test_threshold:
            # reject hypothesis of equal means -> dangerous
            t, p_val = ttest_1samp(rewards, rk)  # single sample
            if debug:
                print('test 2: rk', rk, ' rew:', rewards, ' p_val:', p_val, ' t:', t)
            if p_val < self.test_threshold2:  # check if rk < mean of W2
                flag = True
        return flag

    def find_next_index(self, i):  # find next available index for mark
        next = int(self.mark_index[i])
        if next == -1:
            next = 0
        start = next
        first = True
        # either a mark with confidence = 0
        while self.joint_marks[i][next][-1] != 0:
            if (not first and start == next):  # if we looped over everything
                next = -1
                break
            first = False
            next = (next + 1) % self.dangerous_max

        return next

    def update_W(self, pos, actions, rew):

        for i in range(self.nagents):
            p = pos[i]
            index = self.W1[i][p[0]][p[1]][actions[i]][-1]
            index2 = self.W2[i][p[0]][p[1]][actions[i]][-1]
            if np.isnan(index):
                index = 0
            nextval = self.W1[i][p[0]][p[1]][actions[i]][int(index)]

            if np.isnan(index):  # check if never updated
                index = 0
                self.W1[i][p[0]][p[1]][actions[i]][int(index)] = rew[i]
                self.W1[i][p[0]][p[1]][actions[i]][-1] = (index + 1) % self.nsaved

            elif np.isnan(nextval):  # check if there are still spaces left
                self.W1[i][p[0]][p[1]][actions[i]][int(index)] = rew[i]
                self.W1[i][p[0]][p[1]][actions[i]][-1] = (index + 1) % self.nsaved

            elif np.isnan(index2):  # add to W2
                index2 = 0
                self.W2[i][p[0]][p[1]][actions[i]][int(index2)] = rew[i]
                self.W2[i][p[0]][p[1]][actions[i]][-1] = (index2 + 1) % self.nsaved
            else:
                self.W2[i][p[0]][p[1]][actions[i]][int(index2)] = rew[i]
                self.W2[i][p[0]][p[1]][actions[i]][-1] = (index2 + 1) % self.nsaved


    def run(self, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False):

        alpha_index = 1
        self.discount = discount
        # print(pos)
        steps = np.zeros([episode_max], dtype=int)
        actions = np.zeros([self.nagents], dtype=int)

        for e in range(episode_max):
            self.env.reset()

            if debug:
                print('episode : ', e + 1)

            for s in range(step_max):

                self.alpha = alpha_index / (0.1 * s + 0.5)
                pos = deepcopy(self.env.pos)
                if debug:
                    self.env.render(episode=e + 1)
                    time.sleep(0.1)

                for a in range(self.nagents):
                    actions[a] = self.action_selection(pos, a)  # select actions for each agent

                # update environment and retrieve rewards:
                obs, rew, _, done = self.env.step(actions)  # sample rewards and new states
                self.update_W(pos, actions, rew)  # Update observed rewards. l. 11 in pseudocode.

                self.update(pos, obs, rew, actions)  # update marks and qvalues

                if debug:  # and s > step_max*0.6:
                    print('action:', actions, '- done:', done, '\t- goal_flag:', self.env.goal_flag, '- rewards:', rew,
                          '\t- pos:', pos.flatten(), '- obs:', obs.flatten())
                    # print('js :', self.joint_marks[0])
                # pos = deepcopy(obs)

                if done:
                    steps[e] = s
                    if debug:
                        self.env.render(episode=e + 1)
                    break

        return steps, self.joint_qvalues, self.qvalues

if __name__ == "__main__":
    cq = CQLearning(map_name='ISR')
    cq.initialize_qvalues()

    cq.run(step_max=20, episode_max=3, debug=True)
