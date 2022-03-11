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
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--Q', type=int, default=1)
    parser.add_argument('--show_legend', action='store_true')
    parser.add_argument('--outfile', default=None)
    args = parser.parse_args()

    res = []
    for fname in glob.glob(os.path.join('res', args.name, 'out_*.pkl')):
        res += pickle.load(open(fname, 'rb'))
    print('loaded data.')

    from collections import defaultdict
    curves = defaultdict(list)
    epscurves = {}

    for kv, ls, eps in res:
        if kv['C'] != args.C:
            continue
        if args.step is not None and kv['step'] != args.step:
            continue
        if kv['noise_mult'] == args.noise_mult:
            curves[kv['alpha'], kv['noise_mult']].append(ls)
            epscurves[kv['alpha'], kv['noise_mult']] = eps

    fig, ax1 = plt.subplots(figsize=(5,3))
    for (alpha, nse), eps in epscurves.items():
        if alpha > 0 and nse > 0:
            ax2 = ax1.twinx()
            ax2.plot(eps[:args.end], 'g--')
            ax2.set_ylabel('$\\epsilon$', color='g')
            ax2.grid(False)
            break

    def sortkey(k):
        a, n = k
        return (1e8 if a < 0 else a, n)

    for (alpha, nse) in sorted(curves.keys(), key=sortkey):
        if alpha not in [0, 0.3, 1., 3., -1]:
            continue
        lss = np.array(curves[alpha, nse])
        ls = lss.min(axis=0)
        ax1.semilogy(ls[:args.end], label='$\\alpha Q$ = {}'.format('$\\infty$' if alpha < 0 else alpha / args.Q))

    ax1.set_xlabel('iterations')
    ax1.set_ylabel('excess risk')

    if args.show_legend:
        ax1.legend()
    fig.tight_layout()
    plt.title(f'privacy noise $\\sigma$ = {args.noise_mult}')

    if args.outfile:
        plt.savefig(args.outfile, pad_inches=0, bbox_inches='tight')

