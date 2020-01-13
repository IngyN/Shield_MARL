import numpy as np
import random
import time
from gym_grid.envs import GridEnv


class QLearning:
    def __init__(self, map_size, nactions=5):
        self.qvalues = np.zeros([map_size[0], map_size[1], nactions])
        self.nactions = nactions

    def action_selection(self, qval, epsilon=0.7):
        # TODO : select action based on  previous Q-values or random
        flag = random.random()
        if flag >= epsilon:
            # choose from qvalues
            action = np.argmax(qval)
        else:
            # choose from random
            action = random.randint(0, self.nactions - 1)

        return action

    def update(self, pos, obs, reward, action):
        # TODO:update q-values
        m_value = max(self.qvalues[obs[0]][obs[1]])
        o_value = (1 - self.alpha) * self.qvalues[pos[0]][pos[1]][action]
        self.qvalues[pos[0]][pos[1]][action] = self.alpha * (reward + self.discount * m_value) + o_value

    def run(self, env, step_max=500, episode_max=1000, debug=False):

        env.reset()
        alpha_index = 1
        self.discount = 0.9
        pos = env.pos[0]
        # print(pos)
        steps = np.zeros([episode_max])
        stop = False

        for e in range(episode_max):
            for s in range(step_max):
                self.alpha = alpha_index / (0.1 * s + 1)
                action = self.action_selection(self.qvalues[pos[0]][pos[1]])
                obs, rew, _, _ = env.step([action])
                self.update(pos, obs[0], rew, action)

                if debug:  # and s > step_max*0.6:
                    time.sleep(2)
                    env.render()

                pos = obs[0]  # TODO check this - update position from observation
                goal_flag = env.goal_flag

                if goal_flag == 1:
                    break

            steps[e] = s
            if e > 200:
                stop = True
                for stop_step in range(80):
                    if steps[e] != steps[e - stop_step]:
                        stop = False

            if stop:
                break

        return self.qvalues


if __name__ == "__main__":
    env = GridEnv(agents=1)
    singleQL = QLearning([env.nrows, env.ncols])
    env.render()
    input('next')
    singleQL.run(env, step_max=400, episode_max=20, debug=True)
    input('end')
