import pandas as pd
import numpy as np
import datetime
from copy import deepcopy


# Log results into a data frame
class CustomLogger:

    def __init__(self, nagents):
        self.cols = ['Map', 'Shield', '# iterations']
        self.raw_cols = deepcopy(self.cols)
        self.base = ['ep', 'avg_steps', 'avg_coll']
        self.raw_base = ['ep', 'steps', 'coll']
        self.ext = ['_train', '_test']
        self.raw_ext = ['it_index']

        for ex in self.ext:
            for b in self.base:
                self.cols.append(b + ex)

            for b in self.raw_base:
                self.raw_cols.append(b + ex)

            for i in range(nagents):
                self.cols.append('avg_acc_' + str(i) + ex)
                self.cols.append('avg_inter_' + str(i) + ex)

                self.raw_cols.append('acc_' + str(i) + ex)
                self.raw_cols.append('inter_' + str(i) + ex)

        self.nagents = nagents
        self.df = pd.DataFrame(columns=self.cols)

        self.cols.append(self.raw_ext)
        self.raw_df = pd.DataFrame(columns=self.raw_cols)

    # add entries for QL versions
    def log_results_QL(self, map, test_data, train_data, iterations):
        entry = {}

        entry['Map'] = map
        entry[self.cols[2]] = iterations
        ep_train = train_data[0]['episodes']
        ep_test = test_data[0]['episodes']

        # aggregate data training
        sum_steps = 0
        sum_coll = 0
        sum_acc = np.zeros([self.nagents])

        for it in range(iterations):
            sum_steps += np.sum(train_data[it]['steps'])
            sum_coll += np.sum(train_data[it]['collisions'])
            for i in range(self.nagents):
                sum_acc[i] += np.sum(train_data[it]['acc_rewards'][:, i])

        # train data
        entry[self.cols[3]] = ep_train
        entry[self.cols[4]] = sum_steps / (ep_train * iterations)
        entry[self.cols[5]] = sum_coll / (ep_train * iterations)
        entry['total_coll'] = sum_coll / iterations

        cur_ind = 6
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_train * iterations)
            cur_ind += 2

        # aggregate data testing
        sum_steps = 0
        sum_coll = 0
        sum_acc = np.zeros([self.nagents])
        sum_inter = np.zeros([self.nagents])

        for it in range(iterations):
            sum_steps += np.sum(test_data[it]['steps'])
            sum_coll += np.sum(test_data[it]['collisions'])
            for i in range(self.nagents):
                sum_acc[i] += np.sum(test_data[it]['acc_rewards'][:, i])

        # train data
        entry[self.cols[cur_ind]] = ep_test
        entry[self.cols[cur_ind + 1]] = sum_steps / (ep_test * iterations)
        entry[self.cols[cur_ind + 2]] = sum_coll / (ep_test * iterations)
        entry['total_coll_test'] = sum_coll / (iterations)

        cur_ind = cur_ind + 3
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_test * iterations)
            cur_ind += 2


        self.df = self.df.append(entry, ignore_index=True)

        raw_entry = {}
        raw_entry['Map'] = map
        raw_entry[self.raw_cols[2]] = iterations
        raw_entry[self.raw_cols[3]] = ep_train
        train_steps = train_data[0]['steps']
        test_steps = test_data[0]['steps']

        # save raw
        for it in range(iterations):
            for ep in range(ep_train):
                cur_ind = 4
                raw_entry[self.raw_cols[cur_ind]] = train_data[it]['steps'][ep]
                cur_ind += 1
                raw_entry[self.raw_cols[cur_ind]] = train_data[it]['collisions'][ep]
                for i in range(self.nagents):
                    raw_entry[self.raw_cols[cur_ind + 1]] = train_data[it]['acc_rewards'][ep][i]
                    cur_ind += 2

                cur_ind += 1
                raw_entry[self.raw_cols[cur_ind]] = ep_test
                cur_ind += 1
                if ep < ep_test:
                    raw_entry[self.raw_cols[cur_ind]] = test_data[it]['steps'][ep]
                    raw_entry[self.raw_cols[cur_ind + 1]] = test_data[it]['collisions'][ep]
                    cur_ind += 1
                    for i in range(self.nagents):
                        raw_entry[self.raw_cols[cur_ind + 1]] = test_data[it]['acc_rewards'][ep][i]
                        cur_ind += 2
                else:
                    raw_entry[self.raw_cols[cur_ind]] = np.nan
                    raw_entry[self.raw_cols[cur_ind + 1]] = np.nan
                    cur_ind += 1
                    for i in range(self.nagents):
                        raw_entry[self.raw_cols[cur_ind + 1]] = np.nan
                        raw_entry[self.raw_cols[cur_ind + 2]] = np.nan
                        cur_ind += 2

                self.raw_df = self.raw_df.append(raw_entry, ignore_index=True)

    # construct a row and add it to the data frame
    def log_results(self, map, test_data, train_data, shielding, iterations, save=False, display=False):

        entry = {}

        entry['Map'] = map
        entry['Shield'] = int(shielding)
        entry[self.cols[2]] = iterations
        ep_train = train_data[0]['episodes']
        ep_test = test_data[0]['episodes']

        # aggregate data training
        sum_steps = 0
        sum_coll = 0
        sum_acc = np.zeros([self.nagents])
        sum_inter = np.zeros([self.nagents])

        for it in range(iterations):
            sum_steps += np.sum(train_data[it]['steps'])
            sum_coll += np.sum(train_data[it]['collisions'])
            for i in range(self.nagents):
                sum_acc[i] += np.sum(train_data[it]['acc_rewards'][:, i])
                sum_inter[i] += np.sum(train_data[it]['interferences'][i, :])

        # train data
        entry[self.cols[3]] = ep_train
        entry[self.cols[4]] = sum_steps / (ep_train * iterations)
        entry[self.cols[5]] = sum_coll / (ep_train * iterations)
        entry['total_coll'] = sum_coll / iterations

        cur_ind = 6
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_train * iterations)
            entry[self.cols[cur_ind + 1]] = sum_inter[i] / (ep_train * iterations)
            cur_ind += 2

        # aggregate data testing
        sum_steps = 0
        sum_coll = 0
        sum_acc = np.zeros([self.nagents])
        sum_inter = np.zeros([self.nagents])

        for it in range(iterations):
            sum_steps += np.sum(test_data[it]['steps'])
            sum_coll += np.sum(test_data[it]['collisions'])
            for i in range(self.nagents):
                sum_acc[i] += np.sum(test_data[it]['acc_rewards'][:, i])
                sum_inter[i] += np.sum(test_data[it]['interferences'][i, :])

        # train data
        entry[self.cols[cur_ind]] = ep_test
        entry[self.cols[cur_ind + 1]] = sum_steps / (ep_test * iterations)
        entry[self.cols[cur_ind + 2]] = sum_coll / (ep_test * iterations)
        entry['total_coll_test'] = sum_coll / (iterations)

        cur_ind = cur_ind + 3
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_test * iterations)
            entry[self.cols[cur_ind + 1]] = sum_inter[i] / (ep_test * iterations)
            cur_ind += 2

        self.df = self.df.append(entry, ignore_index=True)

        raw_entry = {}
        raw_entry['Map'] = map
        raw_entry['Shield'] = int(shielding)
        raw_entry[self.raw_cols[2]] = iterations
        raw_entry[self.raw_cols[3]] = ep_train
        train_steps = train_data[0]['steps']
        test_steps = test_data[0]['steps']

        # save raw
        for it in range(iterations):
            for ep in range(ep_train):
                cur_ind = 4
                raw_entry[self.raw_cols[cur_ind]] = train_data[it]['steps'][ep]
                cur_ind += 1
                raw_entry[self.raw_cols[cur_ind]] = train_data[it]['collisions'][ep]
                for i in range(self.nagents):
                    raw_entry[self.raw_cols[cur_ind + 1]] = train_data[it]['acc_rewards'][ep][i]
                    raw_entry[self.raw_cols[cur_ind + 2]] = train_data[it]['interferences'][i][ep]
                    cur_ind += 2

                cur_ind += 1
                raw_entry[self.raw_cols[cur_ind]] = ep_test
                cur_ind += 1
                if ep < ep_test:
                    raw_entry[self.raw_cols[cur_ind]] = test_data[it]['steps'][ep]
                    raw_entry[self.raw_cols[cur_ind + 1]] = test_data[it]['collisions'][ep]
                    cur_ind += 1
                    for i in range(self.nagents):
                        raw_entry[self.raw_cols[cur_ind + 1]] = test_data[it]['acc_rewards'][ep][i]
                        raw_entry[self.raw_cols[cur_ind + 2]] = test_data[it]['interferences'][i][ep]
                        cur_ind += 2
                else:
                    raw_entry[self.raw_cols[cur_ind]] = np.nan
                    raw_entry[self.raw_cols[cur_ind + 1]] = np.nan
                    cur_ind += 1
                    for i in range(self.nagents):
                        raw_entry[self.raw_cols[cur_ind + 1]] = np.nan
                        raw_entry[self.raw_cols[cur_ind + 2]] = np.nan
                        cur_ind += 2

                self.raw_df = self.raw_df.append(raw_entry, ignore_index=True)

    def save(self, alg, grid=False, fair=False, extra=None):
        # save the dataframe to a csv file.
        if grid:
            if fair:
                alg = 'grid_fair_'+alg
            else:
                alg = 'grid_' + alg

        if extra is not None:
            alg = alg + '_' + extra + '_'

        temp = list(self.df.columns.values)

        cols_to_print = [temp[0]]+ temp[2:8] + temp[10:14] + temp[16:]


        #print(cols_to_print)

        print(self.df[cols_to_print])

        date_str = datetime.datetime.now().strftime('_%H_%M_%d_%b_%Y')
        self.df.to_csv('logs/summary_' + alg + '_' + str(self.nagents) + date_str + '.csv', sep='\t', encoding='utf-8',
                       index=False)

        self.raw_df.to_csv('logs/raw_' + alg + '_' + str(self.nagents) + date_str + '.csv', sep='\t', encoding='utf-8',
                           index=False)

        return alg + '_' + str(self.nagents) + date_str