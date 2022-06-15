import argparse
import collections
import itertools
import numpy as np
import os
import pickle
import time
from opacus.accountants.rdp import RDPAccountant

from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import jax_datasets as datasets

from main_real import EMNIST, Stackoverflow

fdim = 32

init_random_params, predict_features = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(fdim),
    stax.Relu,
    # stax.Dense(10), linear layer done explicitly below
)

def predict(params, w, theta, x):
    features = predict_features(params, x)
    return jnp.sum(features[:,:,None] * (w[None,:,:] + theta[None,:,:]), axis=1)

def loss(params, w, theta, x, y):
    return jnp.mean((predict(params, w, theta, x) - y) ** 2)

def clipped_grad(params, w, theta, l2_norm_clip, x, y):
    gp, gw, gth = grad(loss, (0, 1, 2))(params, w, theta, x, y)
    grads, tree_def = tree_flatten(gp)
    total_grad_norm = jnp.sqrt(jnp.sum(gw ** 2) + jnp.array([jnp.sum(g.ravel() ** 2) for g in grads]).sum())
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_grads = [g / divisor for g in grads]
    return tree_unflatten(tree_def, normalized_grads), gw / divisor, gth

@jit
def private_grad(params, w, thetas, xs, ys, rng, l2_norm_clip, noise_mult, num_users):
    cgp, cgw, gth = vmap(clipped_grad, (None, None, 0, None, 1, 1))(
        params, w, thetas, l2_norm_clip, xs, ys)
    # p
    cgp_flat, grads_treedef = tree_flatten(cgp)
    cgp_tot = [g.mean(0) for g in cgp_flat]
    rngp, rngw = random.split(rng)
    rngs = random.split(rngp, len(cgp_tot))
    cgp_noised = [g + l2_norm_clip * noise_mult * random.normal(r, g.shape) / num_users
                  for r, g in zip(rngs, cgp_tot)]
    # w
    cgw_tot = cgw.mean(0)
    cgw_noised = cgw_tot + l2_norm_clip * noise_mult * random.normal(rngw, cgw_tot.shape) / num_users
    return tree_unflatten(grads_treedef, cgp_noised), cgw_noised, gth


def training_priv(ds, step=0.5, stepw=None, steptheta=None, stepw_prob=None, 
                  C=10., noise_mult=0.05, delta=1e-4, eval_delta=500):
    key = random.PRNGKey(42)
    _, params = init_random_params(key, (-1, 28, 28, 1))
    w = np.random.randn(fdim, ds.n_classes) / np.sqrt(fdim)
    thetas = np.zeros((ds.N, fdim, ds.n_classes))

    priv = RDPAccountant()

    if stepw is None:
        stepw = step
    if steptheta is None:
        steptheta = step

    ls = []
    accs = []
    accs_test = []
    test_iters = []
    epss = []

    @jit
    def update_params(params, params_grad):
        ps, tree_def = tree_flatten(params)
        gs, gtree_def = tree_flatten(params_grad)
        ps_new = [p - stepw * g for (p, g) in zip(ps, gs)]
        return tree_unflatten(tree_def, ps_new)

    def eval_test(params, w, thetas):
        acc = 0
        cnt = 0
        for i, (X, y) in enumerate(ds.test_datasets):
            preds = predict(params, w, thetas[i], X.reshape(-1, 28, 28, 1))
            acc += np.sum(preds.argmax(1) == y)
            cnt += len(y)

        return acc / cnt

    for i, (ids, X, y) in enumerate(ds.batches()):
        L = len(ids)
        X = X[:,:L,:]  # [B,N,d]
        X = X.reshape(X.shape[0], X.shape[1], 28, 28, 1)
        y = y[:,:L,:]  # [B,N,K]
        y -= 0.1
        if i % eval_delta == 0:
            print(i, 'eval test acc...', end='', flush=True)
            t = time.time()
            acc_test = eval_test(params, w, thetas)
            print(acc_test, '({:.2f})'.format(time.time() - t))
            test_iters.append(i)
            accs_test.append(acc_test)


        rng = random.fold_in(key, i)  # get new key for new random numbers
        params_grad, w_grad, thetas_grad = private_grad(params, w, thetas[ids],
                X, y, rng, C, noise_mult, L)
        if steptheta > 0:
            thetas[ids] -= L * steptheta * thetas_grad

        if stepw > 0:
            w -= stepw * w_grad
            params = update_params(params, params_grad)

            priv.step(noise_multiplier=noise_mult, sample_rate=L/ds.N)

        eps, best_alpha = priv.get_privacy_spent(delta=delta)
        if i % eval_delta == 0:
            print('eps:', eps)
        epss.append(eps)
    return {'ls': ls, 'accs': accs, 'test_iters': test_iters, 'accs_test': accs_test, 'epss': epss}


grid_emnist = {
    'step': [0.05, 0.1, 0.2, 0.5, 1., 2., 5.],
    'alpha': [0, 0.1, 0.3, 1., 3., 10., 30., 100., -1],
    'noise_mult': [0, 0.1, 0.3, 1., 3., 10., 30., 100.],
    'C': [0.1, 1., 10.],
    'rand': [False],
}

grid_stackoverflow = {
    'step': [2., 5., 10., 20., 50., 100., 150., 200., 250.],
    'alpha': [0, 0.1, 0.3, 1., 3., 10., 30., 100., -1],
    'noise_mult': [0.02, 0.05, 0.2, 0.5, 2., 5.],
    'C': [0.01],
    'rand': [False],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EMNIST CNN')
    parser.add_argument('task_id', type=int)
    parser.add_argument('num_tasks', type=int)
    parser.add_argument('--task_offset', type=int, default=0)
    parser.add_argument('--name', default='emnist_test')
    parser.add_argument('--dataset', default='emnist')
    parser.add_argument('--clients_per_round', type=int, default=10, help='dimension')
    parser.add_argument('--N', type=int, default=3383, help='num users')
    parser.add_argument('--b', type=int, default=10, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='num epochs')
    parser.add_argument('--eval_delta', type=int, default=500, help='how often to eval')
    parser.add_argument('--delta', type=float, default=1e-4, help='delta DP parameter')
    parser.add_argument('--interactive', action='store_true')
    # options used only for interactive (otherwise use grid)
    parser.add_argument('--step', type=float, default=1., help='step-size')
    parser.add_argument('--alpha', type=float, default=-1, help='alpha')
    parser.add_argument('--noise_mult', type=float, default=0, help='DP noise')
    parser.add_argument('--C', type=float, default=10., help='DP clip')
    parser.add_argument('--rand', action='store_true', help='rand strategy')

    args = parser.parse_args()

    if args.dataset == 'emnist':
        ds = EMNIST(N=args.N, clients_per_round=args.clients_per_round, batch_size=args.b, num_epochs=args.num_epochs)
        grid = grid_emnist
    elif args.dataset == 'stackoverflow':
        ds = Stackoverflow(N=args.N, clients_per_round=args.clients_per_round, batch_size=args.b, num_epochs=args.num_epochs)
        grid = grid_stackoverflow
    else:
        assert False

    if not args.interactive:
        os.makedirs(os.path.join('res', args.name), exist_ok=True)

        outfile = os.path.join('res', args.name, f'out_{args.task_offset + args.task_id}.pkl')

    if args.interactive:
        grid = {
            'step': [args.step],
            'alpha': [args.alpha],
            'noise_mult': [args.noise_mult],
            'C': [args.C],
            'rand': [args.rand],
        }

    results = []
    from itertools import product

    for i, vals in enumerate(product(*grid.values())):
        if i % args.num_tasks != (args.task_id - 1):
            continue
        kv = dict(zip(grid.keys(), vals))
        print(kv, flush=True)

        kwargs = {'C': kv['C'], 'noise_mult': kv['noise_mult'], 'eval_delta': args.eval_delta, 'delta': args.delta}

        if kv['alpha'] <= 1. and kv['alpha'] >= 0:
            if kv['rand']:
                if kv['alpha'] == 0 or kv['alpha'] == 1:
                    continue
                curves = training_priv(ds, step=kv['step'], stepw_prob=kv['alpha'], **kwargs)
            else:
                curves = training_priv(ds, stepw=kv['alpha'] * kv['step'], steptheta=kv['step'], **kwargs)

        elif kv['rand']:
            continue

        elif kv['alpha'] == -1: # -1 means alpha = infinity, global only
            curves = training_priv(ds, stepw=kv['step'], steptheta=0, **kwargs)

        else: # alpha >= 1
            curves = training_priv(ds, stepw=kv['step'], steptheta=kv['step'] / kv['alpha'], **kwargs)

        results.append((kv, curves))
        if not args.interactive:
            pickle.dump(results, open(outfile, 'wb'))
