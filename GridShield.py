import json
import numpy as np
from copy import deepcopy
import random


class GridShield:
    # Composed shield version
    def __init__(self, env, nagents=2, start=np.array([[7, 0], [6, 3]]), file='example/example_2_agents_'):
        self.nagents = nagents
        base = 'shields/grid_shields/'
        self.res = [3, 3]
        self.max_per_shield = 2

        self.smap = np.zeros([env.nrows, env.ncols])
        for i in range(env.nrows):
            for j in range(env.ncols):
                # 1 based shield number
                shieldnum = int((i / self.res[0])) * int(env.ncols / (self.res[1])) + int(j / self.res[0]) + 1
                self.smap[i][j] = shieldnum

        self.nshields = int(np.max(self.smap))
        self.shield_json = []
        self.current_state = np.zeros([self.nshields])
        self.start_state = np.zeros([self.nshields])
        self.agent_pos = np.ones([self.nagents, 2]) * -1
        self.states = np.reshape(np.arange(0, self.res[0] * self.res[1]), self.res)

        # loading Shields.
        for sh in range(self.nshields):
            f = open(base + file + str(sh))
            self.shield_json.append(json.load(f))
            f.close()

        # check in which shields agents are starting
        for i in range(self.nagents):
            sh_ind = self.smap[start[i][0]][start[i][1]]
            self.agent_pos[i][1] = sh_ind

        # find the start state for each shield.
        for sh in range(self.nshields):
            ag = np.where(self.agent_pos[:, 1] == sh)[0]

            if len(ag) > 0:  # if there are agents starting in that shield
                found = self._find_start_state(start, sh, ag)
                if found:
                    self.current_state[sh] = self.start_state[sh]
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
    # Step -> all shields, assumption : both agents cannot have already been in the shield and both have the same idx
    # TODO be careful with which agent is 0 and which is 1.
    def step(self, actions, pos, goal_flag, env):
        act = deepcopy(actions)
        shield_pos = np.zeros([self.nagents])
        # update which agents are in which shields
        for i in range(self.nagents):
            shield_pos[i] = self.smap[pos[i][0]][pos[i][1]]

        for sh in range(self.nshields):
            ag_sh = np.where(shield_pos == sh)[0]  # which agents are in shield sh

            if len(ag_sh) > self.max_per_shield:
                print('Error too many agents in shield : ', sh)

            elif len(ag_sh) == 1:  # there are agents for this shield
                if self.agent_pos[ag_sh[0]][0] == 0 and self.agent_pos[ag_sh[0]][1] == sh:
                    act[ag_sh[0]] = self.step_one(sh, actions[ag_sh[0]], goal_flag[ag_sh[0]], agent0=ag_sh[0])
                elif self.agent_pos[ag_sh[0]][0] == 1 and self.agent_pos[ag_sh[0]][1] == sh:
                    act[ag_sh[0]] = self.step_one(sh, actions[ag_sh[0]], goal_flag[ag_sh[0]], agent1=ag_sh[0])
                else:  # it's a new shield
                    act[ag_sh[0]] = self.step_one(sh, actions[ag_sh[0]], goal_flag[ag_sh[0]], agent0=ag_sh[0])

            elif len(ag_sh) == 2:
                a0 = None
                a1 = None
                if self.agent_pos[ag_sh[0]][1] == sh:
                    if self.agent_pos[ag_sh[0]][0] == 0:
                        a0 = ag_sh[0]
                        a1 = ag_sh[1]
                    else:
                        a0 = ag_sh[1]
                        a1 = ag_sh[0]

                elif self.agent_pos[ag_sh[1]][1] == sh:
                    if self.agent_pos[ag_sh[1]][0] == 0:
                        a0 = ag_sh[1]
                        a1 = ag_sh[0]
                    else:
                        a0 = ag_sh[0]
                        a1 = ag_sh[1]

                else:  # both agents are new
                    r = random.randint(0, 1)
                    a0 = ag_sh[r]
                    a1 = ag_sh[1 - r]

                act[ag_sh] = self.step_one(sh, actions[ag_sh], goal_flag[ag_sh], agent0=a0, agent1=a1)

        return act

    # TODO update to work for 1 shield
    # + update self.agent_pos
    def step_one(self, sh, act, goal_flag, agent0=None, agent1=None):
        actions = np.zeros([self.nagents], dtype=int)
        successors = np.array(self.shield_json[sh][str(self.current_state)]['Successors'])

        for s in successors:
            cur = self.shield_json[sh][str(s)]['State']
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
