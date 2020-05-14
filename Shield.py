import json
import numpy as np


class Shield:
    # Centralized shield version
    def __init__(self, nagents=2, start=np.array([[7, 0], [6, 3]]), file='shields/collision_ISR_opt.shield'):
        self.current_state = 0
        self.nagents = nagents

        f = open(file)
        self.shield_json = json.load(f)

        found = self._find_start_state(start)
        if found:
            self.current_state = self.start_state
            # print('Start state = ', self.start_state)
        else:
            print('Error finding start state!!! ')
            exit(1)

        f.close()

    # search shield to find right start state based on agent start states
    def _find_start_state(self, start):
        found = False

        for ind in range(len(self.shield_json)):
            cur = self.shield_json[str(ind)]['State']
            condition = np.zeros(self.nagents)

            for i in range(self.nagents):
                str_x = 'x' + str(i)
                str_y = 'y' + str(i)
                if start[i][0] == cur[str_x] and start[i][1] == cur[str_y]:
                    condition[i] = 1

            if np.count_nonzero(condition) == self.nagents:
                found = True
                self.start_state = ind  # state number.
                break

        return found

    #  use actions and current shield state to determine if action is dangerous.
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
