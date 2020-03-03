import sys, getopt


def get_options(debug=False):
    opts, args = getopt.getopt(
        sys.argv[1:],
        'n:p:i:d:s:',
        ['nagents', 'shielding', 'iterations', 'display', 'save'],
    )

    agents = 2
    shielding = True
    iterations = 1
    display = True
    save = True

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

        else:
            if debug:
                print('invalid')

    return agents, shielding, iterations, display, save


if __name__ == "__main__":
    print(get_options())
