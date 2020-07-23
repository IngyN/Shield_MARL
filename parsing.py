import sys, getopt

''


def get_options(debug=False):
    opts, args = getopt.getopt(
        sys.argv[1:],
        'n:p:i:t:d:s:g:f:e:r:a:w:m:q:c:x:y:z:h:u:',
        ['nagents', 'shielding', 'iterations', 'episodes', 'display', 'save', 'grid', 'fair', 'extra', 'rew',
         'alpha', 'disc', 'd_max', 't_thresh', 'c_thresh', 'c_max', 'start_c', 'delta', 'nsaved', 'noop'],
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
    alpha = 0.9
    discount = 0.9
    episodes = None
    d_max = 50
    t_thresh = 0.1
    c_thresh = 1
    c_max = 70
    start_c = 20
    delta = 1
    nsaved = 15
    noop = False

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

        elif opt in ('-w', '--disc'):
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
            t_thresh = float(arg)

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
            delta = float(arg)

        elif opt in ('-h', '--nsaved'):
            if debug:
                print(opt + ':' + arg + ':')
            nsaved = int(arg)

        elif opt in ('-u', '--noop'):
            if debug:
                print(opt + ':' + arg + ':')
            noop = bool(int(arg))
        else:
            if debug:
                print('invalid')

    return agents, shielding, iterations, display, save, grid, fair, extra, coll_cost, alpha, discount, episodes, d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop

def save_param(date_str, agents, shielding, iterations, display, save, grid, fair, extra, coll_cost, alpha, discount, episodes, d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop):

    file = 'params/'+date_str+'.txt'

    f = open(file, 'w')

    f.write(f'shielding '+str(shielding)+'\n')
    f.write(f'iterations: {iterations} \n')
    f.write(f'grid: {grid} \n')
    f.write(f'fair: {fair} \n')
    f.write(f'collision cost: {coll_cost} \n')
    f.write(f'alpha: {alpha} \n')
    f.write(f'discount: {discount} \n')
    f.write(f'episodes: {episodes} \n')
    f.write(f'd_max: {d_max} \n')
    f.write(f't_thresh: {t_thresh} \n')
    f.write(f'c_thresh: {c_thresh} \n')
    f.write(f'c_max: {c_max} \n')
    f.write(f'start_c: {start_c} \n')
    f.write(f'delta: {delta} \n')
    f.write(f'nsaved: {nsaved} \n')
    f.write(f'noop: {noop} \n')

    f.close()

if __name__ == "__main__":
    print(get_options(debug=True))
