import sys, getopt


def get_options(debug=False):
    opts, args = getopt.getopt(
        sys.argv[1:],
        'n:p:i:d:s:g:f:e:',
        ['nagents', 'shielding', 'iterations', 'display', 'save', 'grid', 'fair', 'extra'],
    )

    agents = 2
    shielding = False
    iterations = 10
    display = False
    save = True
    grid = False
    fair = False
    extra = None

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
        else:
            if debug:
                print('invalid')

    return agents, shielding, iterations, display, save, grid, fair, extra


if __name__ == "__main__":
    print(get_options(debug=True))
