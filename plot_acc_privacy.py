import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle

plt.style.use('ggplot')

def pareto(points):
    newpoints = [points[0]]
    idx = 0
    for eps, acc in points[1:]:
        preveps, prevacc = newpoints[-1]
        if eps < preveps:
            continue
        if acc >= prevacc:
            if eps < preveps + 1e-2:
                newpoints.pop()
            newpoints.append((eps, acc))

    return newpoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser('private FRL')
    parser.add_argument('--name', default='global_d100')
    parser.add_argument('--eval_iter', type=float, default=-1)
    parser.add_argument('--C', type=float, default=1.)
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--epsmin', type=float, default=1e-1)
    parser.add_argument('--epsmax', type=float, default=1e3)
    parser.add_argument('--addeps', type=float, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--show_legend', action='store_true')
    parser.add_argument('--dsname', default='Stackoverflow')
    args = parser.parse_args()

    from collections import defaultdict
    curves = defaultdict(dict)
    epscurves = {}
    test_iters = {}

    for fname in glob.glob(os.path.join('res', args.name, 'out_*.pkl')):
        res = pickle.load(open(fname, 'rb'))
        for kv, curv in res:
            if kv['C'] != args.C:
                continue
            if args.step is not None and kv['step'] != args.step:
                continue
            assert len(curv['accs_test']) == len(curv['test_iters'])
            it = curv['test_iters'][args.eval_iter]
            a, nse = kv['alpha'], kv['noise_mult']
            if (a not in curves or nse not in curves[a]
                 or curv['accs_test'][args.eval_iter] > curves[a][nse][1]):
                curves[a][nse] = (curv['epss'][it], curv['accs_test'][args.eval_iter])

    points = {}
    for a in curves:
        points[a] = sorted(curves[a].values())

    all_points = []
    for a, pts in points.items():
        if a != 0:
            all_points.extend(pts)
    if args.addeps is not None:
        all_points.append((args.addeps, points[0][0][1]))
    all_points = sorted(all_points)
    all_points.insert(0, (args.epsmin, points[0][0][1]))


    pareto_curve = np.array(pareto(all_points))

    plt.figure(figsize=(5, 3))

    def sortkey(k):
        a, n = k
        return (1e8 if a < 0 else a, n)

    def sortalpha(a):
        return 1e8 if a < 0 else a

    for alpha in sorted(points.keys(), key=sortalpha):
        if alpha not in [0, 0.1, 1., 10., 100., -1]:
            continue
        if alpha == 0:
            acc = points[alpha][0][1]
            plt.semilogx([args.epsmin, args.epsmax], [acc, acc], label='local only')
        elif alpha == -1:
            data = np.array(points[alpha])
            eps, acc = data[:,0], data[:,1]
            plt.semilogx(eps, acc, label='global only')
        else:
            data = np.array(points[alpha])
            eps, acc = data[:,0], data[:,1]
            plt.semilogx(eps, acc, 'g', alpha=0.2, linewidth=0.8)

    plt.semilogx(pareto_curve[:,0], pareto_curve[:,1], 'k-', label='best personalized', linewidth=3.)
    plt.xlabel('privacy $\\epsilon$')
    plt.ylabel('accuracy')

    if args.show_legend:
        plt.legend(loc='lower left')

    plt.xlim(args.epsmin, args.epsmax)
    plt.gca().invert_xaxis()

    plt.title(f'accuracy vs privacy ({args.dsname})')

    if args.outfile:
        plt.savefig(args.outfile, pad_inches=0, bbox_inches='tight')

