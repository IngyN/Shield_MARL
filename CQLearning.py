import numpy as np
from QLearning import QLearning
from Shield import Shield
from copy import deepcopy
from gym_grid.envs import GridEnv
from scipy.stats import ttest_ind, ttest_1samp
from scipy import stats
import random, time
import matplotlib
import matplotlib.pyplot as plt


class CQLearning:
    def __init__(self, map_name='example', nagents=2, nactions=5):
        self.nagents = nagents
        self.nactions = nactions
        self.map_name = map_name
        self.env = GridEnv(nagents=nagents, map_name=map_name, norender=False)  # set up ma environment
        self.targets = self.env.targets
        self.start = self.env.start_pos
        self.alpha = 1
        self.discount = 0.9

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
        dims = [nagents]
        dims.extend([self.env.nrows, self.env.ncols] * nagents)
        dims.append(self.nactions)
        self.joint_qvalues = np.zeros(dims)

        # t-test vars
        self.test_threshold = 0.35
        self.test_threshold2 = 0.2
        self.conf_threshold = 1
        self.conf_max = 50
        self.start_conf = 20
        self.delta = 2  # buffer for checking if rk < mean(rewards)
        self.nsaved = 5  # history for observed/expected rewards

        # Rewards History
        self.W1 = np.ones([nagents, self.env.nrows, self.env.ncols, nactions,
                           self.nsaved + 1]) * np.nan  # nan vals were not initialized
        self.W2 = np.ones([nagents, self.env.nrows, self.env.ncols, nactions,
                           self.nsaved + 1]) * np.nan  # nan vals were not initialized

    def reset(self):
        self.__init__(self.map_name, self.nagents, self.nactions)

    # Initialize the local q-values using Q-learning for each agent as well as rewards history W1.
    def initialize_qvalues(self, step_max=250, episode_max=200):
        single_env = GridEnv(nagents=1, map_name=self.map_name)
        singleQL = QLearning([single_env.nrows, single_env.ncols])

        for a in range(self.nagents):
            singleQL.reset()
            single_env.set_start(self.start[a])
            single_env.set_targets(self.targets[a])
            # print('i:', a,'start :', self.start[a], ' - target: ', self.targets[a])
            qval, hist = singleQL.run(single_env, step_max=step_max, episode_max=episode_max, discount=0.9,
                                      debug=False, save=True, N=self.nsaved, epsilon=0.9)
            self.qvalues[a] = deepcopy(qval)
            self.W1[a] = deepcopy(hist)

        self.W2 = deepcopy(self.W1)

    # Greedy select based on a randomness value epsilon
    def greedy_select(self, qval, epsilon=0.4):
        flag = random.randint(0, 99) / 100
        if flag >= epsilon or epsilon == 0.0:
            if np.count_nonzero(qval) < 0.5 * self.nactions:  # if qval is empty, choose randomly
                action = random.randint(0, self.nactions - 1)
            else:
                action = np.argmax(qval)  # choose from qvalues
        else:
            action = random.randint(0, self.nactions - 1)  # choose from random
        return action

    # Retrieve the joint states that contain the current state for agent a
    def retrieve_js(self, states, a):
        ret = np.ones([1, 2 * self.nagents + 1], dtype=int) * -1
        ind = []
        for i in range(self.dangerous_max):
            if np.all(self.joint_marks[a][i][2 * a:2 * a + 2] == states[a]):
                if np.all(ret[0] == np.ones([2 * self.nagents + 1], dtype=int) * -1):
                    ret[0] = self.joint_marks[a][i]
                    ind.append(i)
                else:
                    ret = np.vstack([ret, self.joint_marks[a][i]])
                    ind.append(i)
        return ret, ind

    # Select an action based on marks and current state for one agent i
    def action_selection(self, states, i, epsilon=0.4):
        a = 0
        # print(states)
        if self.marks[i][states[i][0]][states[i][1]] == 2:  # unsafe
            ret, indices = self.retrieve_js(states, i)
            if len(indices) == 0:  # no matching marks for joint states.
                a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]], epsilon=epsilon)
                self.marks[i][states[i][0]][states[i][1]] = 0  # unmark since it wasnt found -> removed somewhere else
            else:  # check if state in for for the others is the same
                same = np.ones([ret.shape[0], self.nagents], dtype=int) * -1
                same[:, i] = 1
                for index, row in enumerate(ret):
                    for a in range(self.nagents):
                        if a != i:
                            if np.all(row[2 * a:2 * a + 2] == states[a]):
                                same[index] = 1
                                break
                found = False
                for j, row in enumerate(same):
                    if np.all(row == np.ones([self.nagents])):  # found one that matches. Should not have duplicates
                        found = True
                        if ret[j][-1] < self.conf_max:
                            self.joint_marks[i][indices[j]][-1] += 1
                            if self.joint_marks[i][indices[j]][-1] > self.conf_max:
                                self.joint_marks[i][indices[j]][-1] = self.conf_max

                        temp = self.get_qval(i, indices[j], ret[j][:-1])
                        a = self.greedy_select(temp, epsilon=epsilon)
                    else:
                        if ret[j][-1] > 0:
                            self.joint_marks[i][indices[j]][-1] -= 1
                            if self.joint_marks[i][indices[j]][-1] < self.conf_threshold:
                                self.joint_marks[i][indices[j]][-1] = 0  # will be overwritten

                if not found:
                    a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]], epsilon=epsilon)

        else:  # unmarked or safe
            a = self.greedy_select(self.qvalues[i][states[i][0]][states[i][1]], epsilon=epsilon)

        return a

    # Retrieve the joint Q-values for current joint state
    def get_qval(self, i, ind, js):
        qval = self.joint_qvalues[i]
        for e in js:
            qval = qval[e]

        return qval

    # Update Q-values and marks based on the taken actions and received rewards
    def update(self, states, obs, rewards, actions):
        joint_state = states.flatten()

        for i in range(self.nagents):
            if self.env.goal_flag[i]:
                pass
            elif self.is_dangerous(rewards[i], self.W2[i][states[i][0]][states[i][1]][actions[i]][:-1],
                                   self.W1[i][states[i][0]][states[i][1]][actions[i]][:-1]):
                #  Mark both joint and local state + update index.
                self.marks[i][states[i][0]][states[i][1]] = 2
                temp = np.append(states.flatten(), self.start_conf)
                self.mark_index[i], found = self.find_next_index(i, joint_state)
                if not found:
                    self.joint_marks[i][self.mark_index[i]] = temp

            if self.marks[i][states[i][0]][states[i][1]] == 0:
                # if marked as safe -> do not update local
                pass

            else:
                self.mark_index[i], found = self.find_next_index(i, joint_state)
                if found and np.all(self.joint_marks[i][self.mark_index[i]][:-1] == joint_state):
                    # update joint qvalues
                    index_joint = np.append(joint_state, actions[i])
                    # new val using new states
                    m_value = self.alpha * (rewards[i] + self.discount * max(self.qvalues[i][obs[i][0]][obs[i][1]]))
                    # old val
                    o_value = (1 - self.alpha) * self.joint_qvalues[i].item(tuple(index_joint))
                    self.joint_qvalues[i].itemset(tuple(index_joint), o_value + m_value)

    # check if statistical tests detect current state as dangerous.
    def is_dangerous(self, rk, rewards, expected_rew, debug=False, v2=True):
        #  take an action state combination +rewards and determine if state should e marked as dangerous
        flag = False
        t_crit1 = round(stats.t.ppf(q=self.test_threshold, df=self.nsaved - 1), 3)
        n_crit1 = t_crit1 if t_crit1 < 0 else -t_crit1

        t_crit2 = round(stats.t.ppf(q=self.test_threshold2, df=self.nsaved - 1), 3)
        t, p_val = ttest_ind(rewards, expected_rew, equal_var=True, nan_policy='omit')  # two  independent sample t-test

        if debug:
            print('test 1: r', rewards, ' e_r:', expected_rew, ' p_val:', p_val, ' t:', t)

        if (t > abs(t_crit1) or t < n_crit1) and p_val < self.test_threshold:
            # reject hypothesis of equal means -> dangerous

            if not v2:
                t, p_val = ttest_1samp(rewards, rk)  # negative direction single sample t-test
                t = -t  # probably shouldn't do this, but it makes it work
                p_val = p_val * 2  # get one-tailed value
                if debug:
                    print('test 2: rk', rk, ' rew:', rewards, ' p_val:', p_val, ' t:', t)

                if t < t_crit2 and p_val < self.test_threshold2:  # fail to reject, check if rk < mean of W2
                    flag = True

            elif rk + self.delta < np.mean(rewards):
                flag = True

        return flag

    # find the next index to mark in joint_marks -> js in pseudo-code
    def find_next_index(self, i, entry):  # find next available index for mark
        next = int(self.mark_index[i])
        if next == -1:
            next = 0
        start = next
        first = True
        found = False
        # check if already exists
        for ind in range(0, self.dangerous_max):
            if np.all(self.joint_marks[i][ind][:-1] == entry):  # already exists
                next = ind
                found = True
                break

        # either a mark with confidence = 0
        if not found:
            while self.joint_marks[i][next][-1] != 0:
                if not first and start == next:  # if we looped over everything
                    next = -1
                    break

                first = False
                next = (next + 1) % self.dangerous_max

        return next, found

    # updated the observed rewards for W1 and W2
    def update_W(self, pos, actions, rew):

        for i in range(self.nagents):
            p = pos[i]
            index = self.W1[i][p[0]][p[1]][actions[i]][-1]
            index2 = self.W2[i][p[0]][p[1]][actions[i]][-1]
            if np.isnan(index):  # check if never updated
                index = 0
            nextval = self.W1[i][p[0]][p[1]][actions[i]][int(index)]

            if np.isnan(nextval):  # check if there are still spaces left
                self.W1[i][p[0]][p[1]][actions[i]][int(index)] = rew[i]
                self.W1[i][p[0]][p[1]][actions[i]][-1] = (index + 1) % self.nsaved

            else:
                if np.isnan(index2):  # add to W2
                    index2 = 0
                self.W2[i][p[0]][p[1]][actions[i]][int(index2)] = rew[i]
                self.W2[i][p[0]][p[1]][actions[i]][-1] = (index2 + 1) % self.nsaved

    def reset_W(self):
        self.W2 = deepcopy(self.W1)

    def get_recommended_training_vars(self):  # save recommended training vars.
        # joint :
        step_max = 250
        episode_max = 200

        if self.map_name == 'example':
            pass
        elif self.map_name == 'ISR':
            pass
        elif self.map_name == 'Pentagon':
            pass
        elif self.map_name == 'MIT':
            step_max = 500
            episode_max = 500

        elif self.map_name == 'SUNY':
            step_max = 500
            episode_max = 500

        # independent
        i_step_max = 250
        i_episode_max = 200
        if self.map_name == 'example':
            pass
        elif self.map_name == 'ISR':
            pass
        elif self.map_name == 'Pentagon':
            pass
        elif self.map_name == 'MIT':
            i_step_max = 500
            i_episode_max = 1000

        elif self.map_name == 'SUNY':
            i_step_max = 500
            i_episode_max = 1000

        return i_step_max, i_episode_max, step_max, episode_max

    def load_shield(self):
        dir = 'shields/collision_' + self.map_name + '_opt.shield'
        self.shield = Shield(self.nagents, start=self.start, file=dir)

    # non shielded running of the algorithm
    def run(self, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False, shielding=False):

        alpha_index = 1
        self.discount = discount
        start_ep = 0.85
        # print(pos)
        steps = np.zeros([episode_max], dtype=int)
        actions = np.zeros([self.nagents], dtype=int)

        if shielding:
            pre_actions = np.zeros([self.nagents], dtype=int)
            self.load_shield()

        for e in range(episode_max):  # loop over episodes
            self.env.reset()
            if shielding:
                self.shield.reset()
            # self.reset_W()
            done = False

            if debug:
                print('episode : ', e + 1)

            for s in range(step_max):  # loop over steps

                self.alpha = alpha_index / (0.1 * s + 0.5)
                # ep = 1 / (0.6 * e + 3) + 0.1
                ep = start_ep - e * 0.0002 if e > episode_max * 0.87 else start_ep
                pos = deepcopy(self.env.pos)  # update pos based in the environment

                if debug:
                    self.env.render(episode=e + 1, speed=1)

                for a in range(self.nagents):  # greedy select an action
                    if not testing:
                        actions[a] = self.action_selection(pos, a, epsilon=ep)  # select actions for each agent
                    else:
                        actions[a] = self.action_selection(pos, a, epsilon=0.05)

                if shielding:
                    pre_actions = deepcopy(actions)
                    actions = self.shield.step(actions)
                    punish = (pre_actions != actions)

                # update environment and retrieve rewards:
                obs, rew, _, done = self.env.step(actions)  # sample rewards and new states

                if shielding:  # punish pre_actions that were changed extra.
                    if not np.all(punish == False):
                        rew_shield = deepcopy(rew)
                        rew_shield[punish] = -10
                        self.update_W(pos, pre_actions, rew_shield)
                        self.update(pos, obs, rew_shield, pre_actions)

                self.update_W(pos, actions, rew)  # Update observed rewards. l. 11 in pseudo-code.
                self.update(pos, obs, rew, actions)  # update marks and qvalues

                # if debug:  # and s > step_max*0.6:
                #     print('action:',actions,'- done:', done, '\t- goal_flag:', self.env.goal_flag, '- rewards:', rew,
                #           '\t- pos:', pos.flatten(), '- obs:', obs.flatten())

                if done:  # if all agents have reached their goal -> episode is finished
                    steps[e] = s
                    if debug:
                        self.env.render(episode=e + 1, speed=1)
                    break

            if not done:  # if episode terminated without agents reaching their goals
                steps[e] = step_max - 1

        return steps + 1, self.joint_qvalues, self.qvalues


def full_test(cq, shielding=False):
    ep_train = 600
    steps_train = 500
    steps_test = 50
    ep_test = 10

    # MIT
    # ep_train = 500
    # steps_train = 500
    # steps_test = 50
    # ep_test = 10
    cq.initialize_qvalues(step_max=steps_train, episode_max=ep_train)

    s, _, _ = cq.run(step_max=steps_train, episode_max=ep_train, debug=False, shielding=shielding)
    print('steps train: \n', s)

    s2, _, _ = cq.run(step_max=steps_test, episode_max=ep_test, testing=True, debug=True, shielding=shielding)
    print('steps test: \n', s2)

    plt.ioff()
    plt.figure(2)
    plt.plot(np.arange(1, ep_train + 1), s)
    # fig.savefig('test.png', bbox_inches='tight')
    plt.title('Training steps')

    plt.figure(3)
    plt.plot(np.arange(1, ep_test + 1), s2)
    plt.title('Testing steps')
    plt.show()


def min_test(cq):
    cq.initialize_qvalues()
    s, _, _ = cq.run(step_max=20, episode_max=90, debug=False)
    print('steps train: \n', s)


def shield_test(cq):
    cq.initialize_qvalues()
    s, _, _ = cq.run(step_max=20, episode_max=90, debug=False, shielding=True)
    print('steps train: \n', s)


if __name__ == "__main__":
    cq = CQLearning(map_name='Pentagon', nagents=2)
    # MIT
    # cq.initialize_qvalues(episode_max= 1000, step_max=500)

    full_test(cq=cq, shielding=False)
    # min_test(cq=cq)
    # shield_test(cq=cq)
