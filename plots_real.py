import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle

plt.style.use('ggplot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('private FRL')
    parser.add_argument('--name', default='global_d100')
    parser.add_argument('--noise_mult', type=float, default=1.)
    parser.add_argument('--C', type=float, default=1.)
    parser.add_argument('--Q', type=int, default=1)
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--show_legend', action='store_true')
    args = parser.parse_args()

    if 'emnist' in args.name:
        alpha_list = [0, 0.1, 1., 10., -1]
    elif 'stack' in args.name:
        alpha_list = [0, 1., 10., 100., -1]

    from collections import defaultdict
    curves = defaultdict(list)
    epscurves = {}
    test_iters = {}

    for fname in glob.glob(os.path.join('res', args.name, 'out_*.pkl')):
        res = pickle.load(open(fname, 'rb'))
        for kv, curv in res:
            if kv['C'] != args.C:
                continue
            if args.step is not None and kv['step'] != args.step:
                continue
            if kv['noise_mult'] == args.noise_mult:
                if args.test:
                    assert len(curv['accs_test']) == len(curv['test_iters'])
                    curves[kv['alpha'], kv['noise_mult']].append(curv['accs_test'])
                else:
                    curves[kv['alpha'], kv['noise_mult']].append(np.convolve(curv['accs'], np.ones(20) / 20., 'valid'))
                epscurves[kv['alpha'], kv['noise_mult']] = curv['epss']
                test_iters[kv['alpha'], kv['noise_mult']] = curv['test_iters']

    fig, ax1 = plt.subplots(figsize=(5,3))

    def sortkey(k):
        a, n = k
        return (1e8 if a < 0 else a, n)

    for (alpha, nse) in sorted(curves.keys(), key=sortkey):
        if alpha not in alpha_list:
            continue
        iters = test_iters[alpha, nse]
        lss = np.array(curves[alpha, nse])
        ls = lss.max(axis=0)[:args.end]
        ax1.plot(iters[:ls.shape[0]], ls, label='$\\alpha Q$ = {}'.format('$\\infty$' if alpha < 0 else alpha / args.Q))

    ax1.set_xlabel('iterations')
    if args.test:
        ax1.set_ylabel('test accuracy')
    else:
        ax1.set_ylabel('batch accuracy')

    if args.show_legend:
        ax1.legend(loc='lower right')

    for (alpha, nse), eps in epscurves.items():
        if alpha > 0 and nse > 0:
            ax2 = ax1.twinx()
            ax2.plot(eps[:args.end], 'g--')
            ax2.set_ylabel('$\\epsilon$', color='g')
            ax2.grid(False)
            break

    fig.tight_layout()
    plt.title(f'privacy noise $\\sigma$ = {args.noise_mult}')

    if args.outfile:
        plt.savefig(args.outfile, pad_inches=0, bbox_inches='tight')

