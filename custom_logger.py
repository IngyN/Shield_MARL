import pandas as pd


def log_results(test_data, train_data, nagents, iterations, save=False, display=False):
    cols = ['Shield', '# iterations']
    base = ['ep', 'avg_steps', 'avg_acc']
    ext = ['_train', '_test']

    for ex in ext:
        for b in base:
            cols.append(b + ex)
        for i in range(nagents):
            cols.append('avg_coll_' + str(i) + ex)
            cols.append('avg_inter_' + str(i) + ex)

    print(cols)

    df = pd.DataFrame(columns=cols, index=[0, 1])

    print(df)
