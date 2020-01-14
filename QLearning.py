import numpy as np
import random
import time
from copy import deepcopy
from gym_grid.envs import GridEnv


class QLearning:
    def __init__(self, map_size, nactions=5):
        self.qvalues = np.zeros([map_size[0], map_size[1], nactions])
        self.nactions = nactions

    def action_selection(self, qval, epsilon=0.7):
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

    def run(self, env, step_max=500, episode_max=2000, discount=0.9, testing=False, debug=False):

        alpha_index = 1
        self.discount = discount
        # print(pos)
        steps = np.zeros([episode_max], dtype = int)
        stop = False

        for e in range(episode_max):
            env.reset()
            pos = deepcopy(env.pos[0])

            for s in range(step_max):

                if debug:
                    env.render()

                self.alpha = alpha_index / (0.1 * s + 0.5)
                if not testing:
                    action = self.action_selection(self.qvalues[pos[0]][pos[1]])
                else:
                    action = self.action_selection(self.qvalues[pos[0]][pos[1]], epsilon=0)
                obs, rew, _, done = env.step([action])
                if not testing:
                    self.update(pos, obs[0], rew, action, True)

                if debug:  # and s > step_max*0.6:
                    print('action:', action, ' -- done: ', done,  ' -- goal_flag:', env.goal_flag, ' --- rewards: ', rew, ' -- pos: ', pos, ' -- obs: ', obs)

                pos = deepcopy(obs[0])

                if done:
                    if debug:
                        env.render()
                    print('episode ', e+1, ' done')
                    break

            steps[e] = s
            if e > 200:
                stop = True
                for stop_step in range(80):
                    if steps[e] != steps[e - stop_step]:
                        stop = False

            if stop:
                break
        print(steps+1)
        return self.qvalues


if __name__ == "__main__":
    env = GridEnv(agents=1, map_name='MIT')
    singleQL = QLearning([env.nrows, env.ncols])
    env.render()
    # input('next')
    qval = singleQL.run(env, step_max=1000, episode_max=500, discount=0.9, debug=False)
    # input('end')
    print('----------------------------------------------------------------- \n end of training -----------------------------------------------------------------')
    singleQL.run(env, step_max=400, episode_max=5, testing=True, debug=True)
    env.final_render()