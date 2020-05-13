import sys, getopt

''


def get_options(debug=False):
    opts, args = getopt.getopt(
        sys.argv[1:],
        'n:p:i:t:d:s:g:f:e:r:a:d:m:q:c:x:y:z:h:',
        ['nagents', 'shielding', 'iterations', 'episodes', 'display', 'save', 'grid', 'fair', 'extra', 'rew',
         'alpha', 'disc', 'd_max', 't_thresh', 'c_thresh', 'c_max', 'start_c', 'delta', 'nsaved'],
    )

    agents = 2
    shielding = False
    iterations = 10
    display = False
    save = True
    grid = False
    fair = False
    extra = None
    coll_cost = 30
    alpha = 1
    discount = 0.9
    episodes = None
    d_max = 50
    t_thresh = 0.35
    c_thresh = 1
    c_max = 50
    start_c = 20
    delta = 2
    nsaved = 5

    for opt, arg in opts:
        if opt in ('-n', '--nagents'):
            agents = int(arg)
            if debug:
                print(opt + ': ' + arg)

        elif opt in ('-p', '--shielding'):
            if debug:
                print(opt + ':' + arg + ':', bool(int(arg)))
            shielding = bool(int(arg))

        elif opt in ('-i', '--iterations'):
            iterations = int(arg)
            if debug:
                print(opt + ': ' + arg)

        elif opt in ('-d', '--display'):
            display = bool(int(arg))
            if debug:
                print(opt + ': ' + arg)

        elif opt in ('-s', '--save'):
            if debug:
                print(opt + ':' + arg + ':')
            save = bool(int(arg))

        elif opt in ('-g', '--grid'):
            if debug:
                print(opt + ':' + arg + ':')
            grid = bool(int(arg))

        elif opt in ('-f', '--fair'):
            if debug:
                print(opt + ':' + arg + ':')
            fair = bool(int(arg))

        elif opt in ('-e', '--extra'):
            if debug:
                print(opt + ':' + arg + ':')
            extra = str(arg)

        elif opt in ('-r', '--rew'):
            coll_cost = int(arg)

        elif opt in ('-a', '--alpha'):
            if debug:
                print(opt + ':' + arg + ':')
            alpha = float(arg)

        elif opt in ('-d', '--disc'):
            if debug:
                print(opt + ':' + arg + ':')
            discount = float(arg)

        elif opt in ('-t', '--episodes'):
            if debug:
                print(opt + ':' + arg + ':')
            episodes = int(arg)

        elif opt in ('-m', '--d_max'):
            if debug:
                print(opt + ':' + arg + ':')
            d_max = int(arg)

        elif opt in ('-q', '--t_thresh'):
            if debug:
                print(opt + ':' + arg + ':')
            t_thresh = int(arg)

        elif opt in ('-c', '--c_thresh'):
            if debug:
                print(opt + ':' + arg + ':')
            c_thresh = int(arg)

        elif opt in ('-x', '--c_max'):
            if debug:
                print(opt + ':' + arg + ':')
            c_max = int(arg)

        elif opt in ('-y', '--start_c'):
            if debug:
                print(opt + ':' + arg + ':')
            start_c = int(arg)

        elif opt in ('-z', '--delta'):
            if debug:
                print(opt + ':' + arg + ':')
            delta = int(arg)

        elif opt in ('-h', '--nsaved'):
            if debug:
                print(opt + ':' + arg + ':')
            nsaved = int(arg)

        else:
            if debug:
                print('invalid')

    return agents, shielding, iterations, display, save, grid, fair, extra, coll_cost, alpha, discount, episodes, d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved


if __name__ == "__main__":
    print(get_options(debug=True))
