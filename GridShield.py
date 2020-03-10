import json
import numpy as np


class GridShield:
    # Composed shield version
    def __init__(self, env, nagents=2, start=np.array([[7, 0], [6, 3]]), file='example/example_2_agents_'):
        self.nagents = nagents
        base = 'shields/grid_shields/'
        res = [3, 3]

        smap = np.zeros([env.nrows, env.ncols])
        for i in range(env.nrows):
            for j in range(env.ncols):
                # 1 based shield number
                shieldnum = int((i / res[0])) * int(env.ncols / (res[1])) + int(j / res[0]) + 1
                smap[i][j] = shieldnum

        self.nshields = int(np.max(smap))
        self.shield_json = []
        self.current_state = np.zeros([self.nshields])
        self.start_state = np.zeros([self.nshields])
        self.agent_pos = np.ones([self.nagents, 2]) * -1
        self.agent_shields = np.ones([self.nagents]) * -1

        # loading Shields.
        for sh in range(self.nshields):
            f = open(base + file + str(sh))
            self.shield_json.append(json.load(f))
            f.close()

        # check in which shields agents are starting
        for i in range(self.nagents):
            sh_ind = smap[start[i][0]][start[i][1]]
            self.agent_shields[i] = sh_ind

        # find the start state for each shield.
        for sh in range(self.nshields):
            ag = np.where(self.agent_shields == sh)[0]

            found = self._find_start_state(start, sh, ag)
            if found:
                self.current_state = self.start_state
            else:
                print('State not found for shield : ', str(sh))

    # search shield to find right start state based on agent start states
    # i = index of shield, agents = indices of agents in shield i
    def _find_start_state(self, start, i, agents):
        found = False

        for ind in range(len(self.shield_json[i])):
            cur = self.shield_json[i][str(ind)]['State']
            condition = np.zeros(agents)

            for i in range(len(agents)):
                str_x = 'x' + str(i)
                str_y = 'y' + str(i)
                if start[i][0] == cur[str_x] and start[i][1] == cur[str_y]:
                    condition[i] = 1

            if np.count_nonzero(condition) == agents:
                found = True
                self.start_state[i] = ind  # state number.
                for a in agents:
                    self.agent_pos[a][0] = np.where(agents == a)[0][0]
                    self.agent_pos[a][1] = i
                break

        return found

    #  use actions and current shield state to determine if action is dangerous.
    # tODO decide if step on one or more shields.
    def step(self, act, goal_flag):
        actions = np.zeros([self.nagents], dtype=int)
        successors = np.array(self.shield_json[str(self.current_state)]['Successors'])

        for s in successors:
            cur = self.shield_json[str(s)]['State']
            condition = np.zeros(self.nagents)
            for i in range(self.nagents):
                a_str = 'a' + str(i)
                if goal_flag[i] == 1:
                    condition[i] = 1 if (cur[a_str] == 0) else 0
                else:
                    condition[i] = 1 if (cur[a_str] == act[i]) else 0

            if np.all(condition):  # found the correct successor
                for i in range(self.nagents):
                    s_str = 'shield' + str(i)
                    actions[i] = cur[s_str]
                    self.current_state = s
                break

        return actions

    def reset(self):
        # reset to start state.
        self.current_state = self.start_state


# for testing
if __name__ == "__main__":
    shield = Shield()
    actions = np.array([4, 3])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
    actions = np.array([4, 2])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
    actions = np.array([4, 3])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
