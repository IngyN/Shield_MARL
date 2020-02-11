import json


class Shield:

    # Centralized shield version
    def __init__(self, start, file='shields/collision_ISR_opt.shield'):
        # TODO : asdd reading the file for shield here.
        # find a more efficient way to load shield?
        self.current_state = 0

        f = open(file)
        self.shield_json = json.load(f)

        self._find_start_state(start)

        f.close()

    def _find_start_state(self, start):
        #TODO search shield to find right start state based on current state
        self.start_state = 0
        pass

    def step(self, actions):
        # TODO use info to determine next state
        pass

    def reset(self):
        #TODO reset to start state.
        self.current_state = self.start_state


# for testing
if __name__ == "__main__":
    Shield()
