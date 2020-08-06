import numpy as np
import pandas as pd
import sys, getopt

cols = ['Map', 'Shield', '# iterations', 'ep_train', 'steps_train', 'coll_train',
        'acc_0_train', 'inter_0_train', 'acc_1_train', 'inter_1_train']
data_cols = ['coll_train', 'acc_0_train', 'inter_0_train', 'acc_1_train', 'inter_1_train']
base = ['Map', 'Shield', 'Grid']
extra = ['std_dev_0', 'std_dev_1', 'mean_acc', 'mean_std']

# parse arguments
def get_options(debug=False):
    opts, args = getopt.getopt(
        sys.argv[1:],
        'f:',
        ['file'],
    )

    filename = None

    for opt, arg in opts:
        if opt in ('-f', '--file'):
            filename = str(arg)
            if debug:
                print(opt + ': ' + arg)

    return filename



def process_df(filename):

    df = pd.read_csv(filename, index_col=False, usecols=cols, sep='\t')

    print(df)

    num_iterations = df['# iterations'].iloc[0]
    num_episodes = df['ep_train'].iloc[0]
    maps = df['Map'].unique()

    sum_df = pd.DataFrame(columns=base + data_cols + extra)
    print(sum_df)

    row = {}
    row['Shield'] = df['Shield'].iloc[0]
    row['Grid'] = 1 if 'grid' in filename else 0

    for m, map in enumerate(maps):
        row['Map'] = map

        for e in range(num_episodes):
            for col in data_cols:
                sum = 0
                data = []
                for i in range(num_iterations):
                    sum += df[col].iloc[e + i * num_episodes + (num_episodes * num_iterations) * m]
                    data.append(df[col].iloc[e + i * num_episodes + (num_episodes * num_iterations) * m])

                row[col] = sum / 10.0
                if 'acc' in col:
                    row['std_dev_' + col[4:5]] = np.std(data)

            sum_df = sum_df.append(row, ignore_index=True)

    sum_df['mean_acc'] = sum_df[[data_cols[1], data_cols[3]]].mean(axis=1)
    sum_df['mean_std'] = sum_df[['std_dev_0', 'std_dev_1']].mean(axis=1)

    print(sum_df)
    print(sum_df.describe())

    new_file = 'graph_data/' + filename[5:-4] + '_'
    # sum_df.to_csv(new_file, encoding='utf-8')
    for map in maps:
        print('---- Saving ' + map)
        temp = sum_df[sum_df['Map'] == map]
        temp.to_csv(new_file+map+'.csv', encoding='utf-8')

if __name__ == "__main__":
    filename = get_options(debug=True)

    if filename is None:
        print('Error: No path provided \n')
        exit(1)

    else:
        process_df(filename)
