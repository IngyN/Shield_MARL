import os, sys
import numpy as np
from enum import Enum
from grid_preprocessing import preprocessing

class Action(Enum):
    stay = 0
    up = 1
    down = 2
    left = 3
    right = 4

    def all_actions(self):
        return [self.stay, self.up, self.down, self.left, self.right ]

class GridWorld:
    def __init__(self, map, name, n):

        self.collision = True
        self.distance = False
        self.coop = False
        self.shared_goal = False
        self.nagents = int(n)
        self.res =[3,3] # width, length / columns, rows

        self.states = np.zeros(self.res)
        self.state_to_pos = np.zeros(np.concatenate((self.res, 2), axis = None))
        for i in range(self.res[1]):
            for map_j in range(self.res[0]):
                self.states[i][map_j] = i * self.res[1] + map_j
                self.state_to_pos[i][map_j] = [i, map_j]

        # print(self.state_to_pos)
        self.nactions = 5
        self.obstacle = False

        self.process_map(map)

        # read map
        with open('maps/'+map+".txt") as textFile:
            self.map = [line.split() for line in textFile]
        self.map = [[int(e) for e in row] for row in self.map ]

        # grid preprocessing
        self.map, self.smap, self.coord = preprocessing(self.map, self.res)

        self.nshields = int(np.max(self.smap))
        print("Number of Shields: ", self.nshields, '\n')
        self.shield_max = [2]* int(self.nshields) # tODO -> make it based on shield morphology

        file = open('grid_shields/'+map+'/shield_num.txt','w')
        file.write(str(self.nshields))
        file.close()

        self.map = np.asarray(self.map)
        self.nstates = np.count_nonzero(self.map ==0)
        print(self.map)
        print(self.smap)
        print('Initialization complete')
        print('Number of reachable states : ', self.nstates)
        # print(self.coord)
        # e_s = self.find_entries(4)
        # print(self.get_next(1, 4, e_s))

        if "fair" in name:
            self.fair= True
            print("****** Fair option selected ******")
        else:
            self.fair = False

    def get_state (self, x,y):
        return x*self.ncols + y


    def is_target(self, n, x, y):

        if self.targets[n][0] == x and self.targets[n][1] == y:
                return True
        return False

    def get_next(self, state,shield_num, entry_states):
        valid = set()
        obs = set()
        corner = self.coord[shield_num]
        move = [[0,-1], [1,0], [0,1], [-1, 0]] # left/down/right/up
        disp = self.state_to_pos[int(state/self.res[1])][int(state%self.res[0])]
        # print(state, disp)
        pos = [int(corner[0]+disp[0]), int(corner[1]+disp[1])]
        # print(pos)
        if self.smap [corner[0]][corner[1]] != self.smap [pos[0]][pos[1]]:
            print("Error: invalid shield and position combination")
        else:
            # print("valid combination ")
            state = self.states[pos[0]-corner[0]][pos[1]-corner[1]]
            valid.add(state)
            
            for i in range(len(move)):
                    
                    map_i = np.clip(pos[0] + move[i][0] , 0, self.map.shape[0]-1)
                    map_j = np.clip(pos[1] + move[i][1], 0, self.map.shape[1]-1)

                    # print(map_i, map_j)
                    # print(self.map[map_i][map_j])
                    if self.smap[map_i][map_j] != shield_num+1 and state in entry_states:
                        valid.add(9) # outside shield shield_num

                    elif self.smap[map_i][map_j] != shield_num+1: # outside but not a valid exiting state
                        pass

                    elif self.map[map_i][map_j] : # obstacle
                        obs.add(int(self.states[map_i-corner[0]][map_j-corner[1]]))

                    elif not self.map[map_i][map_j]:
                        valid.add(int(self.states[map_i-corner[0]][map_j-corner[1]]))

        return valid, obs

    def get_next_req(self, state, shield_num, entry_states, debug=False):
        valid = set()
        corner = self.coord[shield_num]
        move = [[0,-1], [1,0], [0,1], [-1, 0]] # left/down/right/up
        # print(state, shield_num, entry_states)
        disp = self.state_to_pos[int(state/self.res[1])][int(state%self.res[0])]
        pos = [int(corner[0]+disp[0]), int(corner[1]+disp[1])]

        valid.add(state)

        for i in range(len(move)):
                
                map_i = np.clip(pos[0] + move[i][0] , 0, self.map.shape[0]-1)
                map_j = np.clip(pos[1] + move[i][1], 0, self.map.shape[1]-1)

                if self.smap[map_i][map_j] != shield_num+1 and state in entry_states:
                    valid.add(9) # outside shield shield_num

                elif self.smap[map_i][map_j] != shield_num+1: # outside but not a valid exiting state
                    pass

                else:
                    valid.add(int(self.states[map_i-corner[0]][map_j-corner[1]]))

        return valid

    def find_entries(self, shield_num, req=False):
        corner = self.coord[shield_num]
        entry_states = set()

        # check top
        if corner[0] > 0 : # TODO make it so if there is a block outside shield -> still counts for a_req. 
            for i in range(self.res[0]) : 
                if (req or not self.map[corner[0] -1][corner[1] + i]) and (req or not self.map[corner[0]][corner[1] + i]):
                    entry_states.add(i)

        # check bottom
        if corner[0] + self.res[1] < self.map.shape[0] :
            for i in range(self.res[0]) : 
                if (req or not self.map[corner[0] + self.res[1]][corner[1] + i]) and (req or not self.map[corner[0] + self.res[1]-1][corner[1] + i]):
                    entry_states.add(i + 6)

        # check right
        r =[2,5,8]
        if corner[1] + self.res[0] < self.map.shape[1] :
            for i in range(self.res[1]) : 
                if (req or not self.map[corner[0] + i][corner[1] + self.res[0]]) and (req or not self.map[corner[0] + i][corner[1] + self.res[0]- 1]):
                    entry_states.add(r[i])
        # check left
        l = [0,3,6]        
        if corner[1] > 0 :
            for i in range(self.res[1]) : 
                if (req or not self.map[corner[0] + i][corner[1]-1]) and (req or not self.map[corner[0] + i][corner[1]]):
                    entry_states.add(l[i])

        return entry_states

    def find_obstacles(self, shield_num):
        corner = self.coord[shield_num]
        obs_states = set()

        for i in range(0,  self.res[1]):
            for map_j in range (0, self.res[0]):
                if self.map[corner[0] + i][corner[1] + map_j] :
                    obs_states.add(int(self.states[i][map_j]))

        return obs_states

    def process_map(self, map):
        if map == "simple":
            self.targets = [[0,1],[0,1], [0,1], [0,1]]
            self.init = [[3,0], [3,2], [1,0], [1,2]]
            self.shared_goal = True
            # self.distance = True
            # self.collision = False
        elif map == "example":       
            # example map
            self.targets = [[0,2],[2,0]]
            if self.collision:
                self.init = [[2, 0], [2, 4]]
            elif self.distance:
                self.targets = [[0,2],[0,2]]
                self.init = [[2, 0], [2, 1]]

        elif map == "ISR":       
            # ISR map
            self.targets = [[6,3],[7,0],[6,1],[9,3] ]
            if self.collision:
                self.init = [[7, 0], [6, 3], [7,4], [6,1]]
            elif self.distance:
                self.targets = [[0,5],[0,5], [0,5], [0,5]]
                self.init = [[9, 2], [9, 3], [8,3], [7,2]]


        elif map =="MIT":
            # MIT map
            self.targets = [[3,16],[3,0], [0, 12], [6,6]]
            if self.collision:
                self.init = [[3, 0], [3, 16], [6,6], [0,12]]
            elif self.distance:
                self.targets = [[3,16],[3,16], [3,16],[3,16]]
                self.init = [[3, 0], [3, 1], [4,1], [1,1]]

        elif map == "Pentagon":
            # Pentagon
            self.targets = [[4,10],[3,7], [3,5], [6,5]]
            if self.collision:
                self.init = [[3, 7], [4, 10], [6,5], [3,5]]
            elif self.distance:
                self.targets = [[4,10],[4,10],[4,10],[4,10]]
                self.init = [[8, 0], [8, 1], [7,1], [8,2]]

        elif map == "SUNY":
            # SUNY map
            self.targets = [[8,17],[3,22], [6,11], [5,19]]
            if self.collision:
                self.init = [[3, 22], [8, 17], [5,19], [7,14]]
            elif self.distance:
                self.targets = [[3,22],[3,22],[3,22],[3,22]]
                self.init = [[1, 1], [2, 1], [3,1], [4,1]]

        elif map == "SUNYvar":
            # SUNY map
            self.targets = [[7,7],[3,22], [6,13], [8,17]]
            if self.collision:
                self.init = [[3, 22], [7, 7], [8,17], [6,13]]
            elif self.distance:
                self.targets = [[3,22],[3,22],[3,22],[3,22]]
                self.init = [[1, 1], [2, 1], [3,1], [4,1]]

        else :
            # unrecognized map
            print("unrecognized map error")
            exit(1)

def write_to_slugs_test_simple(infile,gw):

    for s in range(gw.nshields):
        print('Generating shield ',s)
        entry_states = gw.find_entries(s)
        entry_states_req = gw.find_entries(s, req=True)
        obs_states = gw.find_obstacles(s)
        oob= gw.res[0]*gw.res[1] # out of bounds state

        filename = 'grid_shields/temp/'+infile+'_'+str(s)+'.structuredslugs'
        file = open(filename,'w')

        for i in range(gw.shield_max[s]): 
            file.write('[INPUT]\n')
            file.write('a_req{}:0...{}\n'.format(i,oob))
            file.write('a{}:0...{}\n\n'.format(i, oob))

            file.write('[OUTPUT]\n')
            file.write('same{}:0...{}\n'.format(i, 1))
            file.write('a_shield{}\n\n'.format(i))

        file.write('\n[ENV_INIT]\n')
        for i in range(gw.shield_max[s]):
            file.write('a_req{} = {}\n'.format(i,oob))
            file.write('a{} = {}\n'.format(i, oob))

        file.write('\n[SYS_INIT]\n')
        for i in range(gw.shield_max[s]):
            file.write('a_shield{}\n'.format(i))
            file.write('same{}={}\n'.format(i, 0))

        file.write('\n[ENV_TRANS]\n')
        states_list = np.reshape(gw.states, (1,-1))[0]
        states_list = states_list.astype(int)

        
        for i in range(gw.shield_max[s]):
            str_req = ''
            str_a = ''
            for j in states_list : # j = state within shield

                # a_req transition
                v_req = gw.get_next_req(j, s, entry_states_req) 
                str_req += 'a{} = {} -> a_req{} = {} '.format(i,j,i,j)
                # print('s', s, ' j:', j, ' valid_req: ', v_req)
                for k in v_req:
                    if int(k) != j:
                        str_req+=' \\/ a_req{} = {}'.format(i, int(k))
                str_req += '\n'

                # a transition
                v_a, _ = gw.get_next(j, s, entry_states)
                str_a += 'a{} = {} -> a{}\' = {} '.format(i,j,i,j)
                for k in v_a:
                    if int(k) != j:
                        str_a += ' \\/ (a_req{} = {} /\\ a{}\' = {}) '.format(i, int(k), i, int(k))
                str_a +='\n'

            # add entering the shield
            str_req += 'a{} = {} -> a_req{} = {} '.format(i,oob,i,oob)
            for e in entry_states_req:
                str_req+=' \\/ a_req{} = {}'.format(i, int(e))
            str_req += '\n'

            str_a += 'a{} = {} -> a{}\' = {} '.format(i,oob,i,oob)
            for e in entry_states:
                str_a+=' \\/ (a_req{} = {} /\\ a{}\' = {})'.format(i, int(e), i, int(e))
            str_a +='\n'
                    
            file.write(str_req+'\n')
            file.write(str_a+'\n')
            # obstacles
            stri = ''
            for o in obs_states:
                stri+= 'a{} != {} /\\ '.format(i,o)
            stri = stri[:-3] + '\n'
            file.write(stri)

            # shield functionality
            file.write('!a_shield{} -> a{}\' = a{}\n\n'.format(i,i,i))

        # writing env_trans
        file.write('\n[ENV_LIVENESS]\n')

        file.write('\n[SYS_TRANS]\n')
        for i in range(gw.shield_max[s]):
            for j in range(i+1, gw.shield_max[s]):
                file.write('a{}\' != a{}\' | a{} = {} | a{} = {} | a{}\' = {} | a{}\' = {} \n'.format(i,j,i,oob,j,oob,i,oob,j,oob))
                file.write('(a{}\' != a{} /\\ a{}\' != a{}) | a{} = {} | a{} = {}\n'.format(i,j,j,i,i,oob,j,oob))

            file.write('same{}\' = {} <-> (a_shield{}\')\n'.format(i,0, i))

        # Writing sys_liveness
        # reach target
        file.write('\n[SYS_LIVENESS]\n')

        if gw.fair :
            # fairness for interference
            file.write('a_shield0\' | a_req0\'=a1\' \n')
            file.write('a_shield1\' | a0\'=a_req1\' \n')


        # Writing env_liveness
        file.close()




# main
if __name__ == "__main__":

    map = sys.argv[1]
    # file = open("map.txt")
    # map = file.read()
    # map = map [:-1]
    # file.close()

    # read number of agents
    n= int(sys.argv[2])
    # file = open("num_agents.txt")
    # n = file.read()
    # n = n [:-1]
    # file.close()

    name= sys.argv[3]
    # file = open("name.txt")
    # name = file.read()
    # name = name [:-1]
    # file.close()
    print("----------------------------------------------------------------------------------\nGenerating Grid Shields for map:", map, "\n----------------------------------------------------------------------------------\n")
    gw = GridWorld(map, name, n)

    write_to_slugs_test_simple(name, gw)
