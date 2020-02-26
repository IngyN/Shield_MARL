import pandas as pd
import numpy as np


# Log results into a data frame
class CustomLogger:

    def __init__(self, nagents):
        self.cols = ['Shield', '# iterations']
        self.base = ['ep', 'avg_steps', 'avg_coll']
        self.ext = ['_train', '_test']

        for ex in self.ext:
            for b in self.base:
                self.cols.append(b + ex)
            for i in range(nagents):
                self.cols.append('avg_acc_' + str(i) + ex)
                self.cols.append('avg_inter_' + str(i) + ex)

        self.nagents = nagents
        self.df = pd.DataFrame(columns=self.cols)

    # construct a row and add it to the data frame
    def log_results(self, test_data, train_data, shielding, iterations, save=False, display=False):

        entry = {}

        entry['Shield'] = int(shielding)
        entry[self.cols[1]] = iterations
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
        entry[self.cols[2]] = ep_train
        entry[self.cols[3]] = sum_steps / (ep_train)
        entry[self.cols[4]] = sum_coll / (ep_train)

        cur_ind = 5
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_train)
            entry[self.cols[cur_ind + 1]] = sum_inter[i] / (ep_train)
            cur_ind += 2

        # aggregate data testing
        for it in range(iterations):
            sum_steps += np.sum(test_data[it]['steps'])
            sum_coll += np.sum(test_data[it]['collisions'])
            for i in range(self.nagents):
                sum_acc[i] += np.sum(test_data[it]['acc_rewards'][:, i])
                sum_inter[i] += np.sum(test_data[it]['interferences'][i, :])

        # train data
        entry[self.cols[cur_ind]] = ep_test
        entry[self.cols[cur_ind + 1]] = sum_steps / (ep_test)
        entry[self.cols[cur_ind + 2]] = sum_coll / (ep_test)

        cur_ind = cur_ind + 3
        for i in range(self.nagents):
            entry[self.cols[cur_ind]] = sum_acc[i] / (ep_test)
            entry[self.cols[cur_ind + 1]] = sum_inter[i] / (ep_test)
            cur_ind += 2

        self.df = self.df.append(entry, ignore_index=True)

        print(self.df)

    def save(self):
        # save the dataframe to a csv file.
        pass
