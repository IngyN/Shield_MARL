import numpy as np
import random
import time
from copy import deepcopy
from gym_grid.envs import GridEnv


class QLearning:
    def __init__(self, map_size, nactions=5):
        self.qvalues = np.zeros([map_size[0], map_size[1], nactions])
        self.map_dim = map_size
        self.nactions = nactions

    def action_selection(self, qval, epsilon=0.8):
        flag = random.random()
        if flag >= epsilon:
            # choose from qvalues
            action = np.argmax(qval)
        else:
            # choose from random
            action = random.randint(0, self.nactions - 1)

        return action

    def update(self, pos, obs, reward, action, debug):
        m_value = max(self.qvalues[obs[0]][obs[1]])
        o_value = (1 - self.alpha) * self.qvalues[pos[0]][pos[1]][action]
        self.qvalues[pos[0]][pos[1]][action] = self.alpha * (reward + self.discount * m_value) + o_value

    def reset(self):
        self.qvalues = np.zeros([self.map_dim[0], self.map_dim[1], self.nactions])

    def run(self, env, step_max=500, episode_max=2000, discount=0.9, testing=False, noop=False, debug=False, save=False,
            N=10,
            epsilon=0.8, c_cost=10):

        alpha_index = 1
        self.discount = discount
        # print(pos)
        steps = np.zeros([episode_max], dtype=int)
        stop = False
        if save:
            hist = np.ones([env.nrows, env.ncols, self.nactions, N + 1], dtype=int) * np.nan

        for e in range(episode_max):
            env.reset()
            pos = deepcopy(env.pos[0])

            if debug:
                print('episode ', e + 1)
            for s in range(step_max):

                if debug:
                    env.render(episode=e + 1)

                self.alpha = alpha_index / (0.1 * s + 0.5)
                if not testing:
                    action = self.action_selection(self.qvalues[pos[0]][pos[1]], epsilon=epsilon)
                else:
                    action = self.action_selection(self.qvalues[pos[0]][pos[1]], epsilon=0)
                obs, rew, _, done = env.step([action], noop=noop, collision_cost=c_cost)

                if save:
                    index = hist[pos[0]][pos[1]][action][-1]
                    if index < 0 or np.isnan(index):
                        index = 0

                    hist[pos[0]][pos[1]][action][int(index)] = rew
                    hist[pos[0]][pos[1]][action][-1] = (index + 1) % N

                if not testing:
                    self.update(pos, obs[0], rew, action, True)

                if debug:  # and s > step_max*0.6:
                    print('action:', action, ' -- done: ', done, ' -- goal_flag:', env.goal_flag, ' --- rewards: ', rew,
                          ' -- pos: ', pos, ' -- obs: ', obs)

                pos = deepcopy(obs[0])

                if done:
                    if debug:
                        env.render(episode=e + 1)
                    break

            steps[e] = s
            if e > 80:
                stop = True
                for stop_step in range(80):
                    if steps[e] != steps[e - stop_step]:
                        stop = False

            if stop:
                break
        # print(steps + 1)
        if save:
            return self.qvalues, hist
        else:
            return self.qvalues


if __name__ == "__main__":
    env = GridEnv(nagents=1, norender=False, padding=True)
    env.set_start(np.array([2, 4]))
    env.set_targets(np.array([2, 1]))
    singleQL = QLearning([env.nrows, env.ncols])
    env.render()
    # input('next')
    qval = singleQL.run(env, step_max=400, episode_max=100, discount=0.9, debug=False)
    # input('end')
    print(
        '----------------------------------------------------------------- \n end of training -----------------------------------------------------------------')
    singleQL.run(env, step_max=400, episode_max=5, testing=True, debug=True)
    env.final_render()
