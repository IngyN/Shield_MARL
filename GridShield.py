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

    # Transform 2d coordinate to inner shield state.
    def _to_state(self, pos):
        sh = self.smap[pos[0]][pos[1]]
        corner = np.array([(int(sh / self.res[0])) * self.res[0], (sh % self.res[1]) * self.res[1]])
        in_shield_pos = pos - corner

        state = self.states[in_shield_pos[0]][in_shield_pos[1]]

        return state

    #  use actions and current shield state to determine if action is dangerous.
    # Step -> all shields, assumption : both agents cannot have already been in the shield and both have the same idx
    # Assumption : when 2 in shield and 1 entering -> wait one turn until on agent leaves.
    # TODO be careful with which agent is 0 and which is 1.
    # tODO : add a_req calculations
    def step(self, actions, pos, goal_flag, env):
        act = np.ones([self.nagents])
        a_states = np.zeros([self.nagents], dtype=int)
        a_req = np.ones([self.nagents, 2], dtype=int) * -1  # desired shield state
        pos_shield = np.zeros([self.nagents])
        desired_shield = np.zeros([self.nagents])
        desired = np.zeros([self.nagents, 2])  # desired coordinates
        shield_idx = np.ones([self.nagents, 2]) * -1  # desired agent index in shield

        # update which agents are in which shields
        for i in range(self.nagents):
            desired[i] = env.get_next_state(pos[i], actions[i], goal_flag[i])
            pos_shield[i] = self.smap[pos[i][0]][pos[i][1]]
            if not goal_flag[i]:
                desired_shield[i] = self.smap[desired[i][0]][desired[i][1]]
            else:
                desired_shield[i] = pos_shield[i]
            a_states[i] = self._to_state(pos[i])
            a_req[i][0] = self._to_state(desired[i])
            if pos_shield[i] != desired_shield[i]:  # Exiting /entering
                a_req[i][1] = 9
            else:
                a_req[i][1] = -1

        # Loop over shields
        for sh in range(self.nshields):
            ag_sh = np.where(pos_shield == sh)[0]  # which agents are in shield sh
            des_sh = np.where(desired_shield == sh)[0]
            des_sh = np.setdiff1d(des_sh, ag_sh)  # only in desired (not already in these shields)
            ex_sh = []  # agents exiting

            # find exiting agents
            for a in ag_sh:
                if desired_shield[a] != sh and a_req[a][1] == 9:
                    ex_sh.append(a)

            # TODO fix exiting and entering
            if len(ag_sh) > self.max_per_shield:
                print('Error too many agents in shield : ', sh)

            elif len(ag_sh) == 0 and len(des_sh) > 0:  # there are no current agents but there are new ones
                if len(des_sh) > 2:
                    for i in range(2, len(des_sh)):
                        act[des_sh[i]] = 0

                if des_sh[0] > des_sh[1]:
                    temp = self.step_one(sh, [goal_flag[des_sh[0]], goal_flag[des_sh[1]]],
                                         [a_req[des_sh[0]], a_req[des_sh[1]]],
                                         [a_states[des_sh[0]], a_states[des_sh[1]]],
                                         agent0=des_sh[0], agent1=des_sh[1])
                    shield_idx[des_sh[0]][1] = 0
                    shield_idx[des_sh[1]][1] = 1
                    act[des_sh[0]] = act[des_sh[0]] and temp[0]
                    act[des_sh[1]] = act[des_sh[1]] and temp[1]
                else:
                    temp = self.step_one(sh, [goal_flag[des_sh[1]], goal_flag[des_sh[0]]],
                                         [a_req[des_sh[1]], a_req[des_sh[0]]],
                                         [a_states[des_sh[1]], a_states[des_sh[0]]],
                                         agent0=des_sh[1], agent1=des_sh[0])
                    shield_idx[des_sh[0]][1] = 1
                    shield_idx[des_sh[1]][1] = 0
                    act[des_sh[0]] = act[des_sh[0]] and temp[1]
                    act[des_sh[1]] = act[des_sh[1]] and temp[0]

            elif len(ag_sh) == 1:  # there is one agent for this shield -> can accept 1 new
                if len(des_sh) > 1:
                    for i in range(1, len(des_sh)):  # only one is accepted so all others are denied movement.
                        act[des_sh[i]] = 0

                if self.agent_pos[ag_sh[0]][0] == 0:
                    if len(des_sh) > 0:
                        if des_sh[0] > ag_sh[0]:  # make sure the order in a_req is consistent with agent order
                            temp = self.step_one(sh, [goal_flag[ag_sh[0]], goal_flag[des_sh[0]]],
                                                 [a_req[ag_sh[0]], a_req[des_sh[0]]],
                                                 [a_states[ag_sh[0]], a_states[des_sh[0]]],
                                                 agent0=ag_sh[0], agent1=des_sh[0])
                            shield_idx[des_sh[0]][1] = 1
                            shield_idx[ag_sh[0]][0] = 0
                            act[ag_sh[0]] = act[ag_sh[0]] and temp[0]
                            act[des_sh[0]] = act[des_sh[0]] and temp[1]
                        else:
                            temp = self.step_one(sh, [goal_flag[des_sh[0]], goal_flag[ag_sh[0]]],
                                                 [a_req[des_sh[0]], a_req[ag_sh[0]]],
                                                 [a_states[des_sh[0]], a_states[ag_sh[0]]],
                                                 agent0=ag_sh[0], agent1=des_sh[0])
                            shield_idx[des_sh[0]][1] = 1
                            shield_idx[ag_sh[0]][0] = 0
                            act[ag_sh[0]] = act[ag_sh[0]] and temp[1]
                            act[des_sh[0]] = act[des_sh[0]] and temp[0]
                    else:
                        temp = self.step_one(sh, goal_flag[ag_sh[0]],
                                             a_req[ag_sh[0]], a_states[ag_sh[0]], agent0=ag_sh[0])
                        shield_idx[ag_sh[0]][0] = 0
                        act[ag_sh[0]] = act[ag_sh[0]] and temp

                elif self.agent_pos[ag_sh[0]][0] == 1:
                    if len(des_sh) > 0:
                        if des_sh[0] > ag_sh[0]:
                            temp = self.step_one(sh, [goal_flag[ag_sh[0]], goal_flag[des_sh[0]]],
                                                 [a_req[ag_sh[0]], a_req[des_sh[0]]],
                                                 [a_states[ag_sh[0]], a_states[des_sh[0]]],
                                                 agent1=ag_sh[0], agent0=des_sh[0])
                            shield_idx[des_sh[0]][1] = 0
                            shield_idx[ag_sh[0]][0] = 1
                            act[ag_sh[0]] = act[ag_sh[0]] and temp[0]
                            act[des_sh[0]] = act[des_sh[0]] and temp[1]
                        else:
                            temp = self.step_one(sh, [goal_flag[des_sh[0]], goal_flag[ag_sh[0]]],
                                                 [a_req[des_sh[0]], a_req[ag_sh[0]]],
                                                 [a_states[des_sh[0]], a_states[ag_sh[0]]],
                                                 agent1=ag_sh[0], agent0=des_sh[0])
                            shield_idx[des_sh[0]][1] = 0
                            shield_idx[ag_sh[0]][0] = 1
                            act[ag_sh[0]] = act[ag_sh[0]] and temp[1]
                            act[des_sh[0]] = act[des_sh[0]] and temp[0]
                    else:
                        temp = self.step_one(sh, goal_flag[ag_sh[0]],
                                             a_req[ag_sh[0]], a_states[ag_sh[0]], agent1=ag_sh[0])
                        shield_idx[ag_sh[0]][0] = 1
                        act[ag_sh[0]] = act[ag_sh[0]] and temp


            elif len(ag_sh) == 2:  # no more space
                for a in des_sh:
                    act[a] = 0

                if self.agent_pos[ag_sh[0]][0] == 0:
                    a0 = ag_sh[0]
                    a1 = ag_sh[1]
                    shield_idx[ag_sh[0]][0] = 0
                    shield_idx[ag_sh[1]][0] = 1
                else:
                    a0 = ag_sh[1]
                    a1 = ag_sh[0]
                    shield_idx[ag_sh[0]][0] = 1
                    shield_idx[ag_sh[1]][0] = 0

                temp = self.step_one(sh, goal_flag[ag_sh], a_req[ag_sh], a_states[ag_sh], agent0=a0, agent1=a1)
                act[ag_sh] = act[ag_sh] and temp

        for i in range(self.nagents):
            if act[i]:
                self.agent_pos[i] = [shield_idx[i][1], desired_shield[i]]
            else:
                self.agent_pos[i] = [shield_idx[i][0], desired_shield[i]]

        # act says which ones need to be changed
        actions[not act] = 0  # blocked = 0
        return actions

    def get_arr_idx(self, agent0=None, agent1=None):
        idx = []
        if agent0 is None or agent1 is None:
            idx = [0]
        elif agent1 is not None and agent0 is not None:
            if agent0 > agent1:
                idx = [1, 0]
            else:
                idx = [0, 1]  # agent0 < agent1
        return idx

    # Processes one shield and relevant agents
    def step_one(self, sh, goal_flag, a_req, a_states, agent0=None, agent1=None):

        idx = self.get_arr_idx(agent0=agent0, agent1=agent1)
        act = np.zeros([len(goal_flag)], dtype=int)
        successors = np.array(self.shield_json[sh][str(self.current_state)]['Successors'])

        for s in successors:
            cur = self.shield_json[sh][str(s)]['State']
            condition = np.zeros(len(goal_flag))

            for i in range(len(goal_flag)):
                a_str = 'a' + str(i)
                a_req_str = 'a_req' + str(i)

                if goal_flag[idx[i]] == 1:
                    condition[idx[i]] = (cur[a_str] == a_states[idx[i]] and cur[a_req_str] == a_states[idx[i]])
                else:
                    condition[idx[i]] = (cur[a_str] == a_states[idx[i]] and cur[a_req_str] == a_req[idx[i]])

            if np.all(condition):  # found the correct successor
                for i in range(len(goal_flag)):
                    s_str = 'a_shield' + str(i)
                    act[idx[i]] = cur[s_str]

                self.current_state[sh] = s
                break

        return act

    def reset(self):
        # reset to start state.
        self.current_state = self.start_state


# for testing
if __name__ == "__main__":
    shield = GridShield()
    actions = np.array([4, 3])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
    actions = np.array([4, 2])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
    actions = np.array([4, 3])
    sactions = shield.step(actions)
    print('Start actions:', actions, 'Shield actions:', sactions)
