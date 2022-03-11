import argparse
import collections
import itertools
import numpy as np
import os
import pickle
import time
from opacus.accountants.rdp import RDPAccountant

import tensorflow_federated as tff
import tensorflow as tf


class EMNIST(object):
    def __init__(self, N=3383, clients_per_round=50, batch_size=10, num_epochs=1, seed=42):
        self.d = 784
        self.n_classes = 10
        self.N = N
        self.rd = np.random.RandomState(seed)

        tr, te = tff.simulation.datasets.emnist.load_data(cache_dir='data/')

        self.batch_size = batch_size
        self.L = clients_per_round
        self.num_epochs = num_epochs

        def preprocess_train(dataset):
            def batch_format_fn(element):
                """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
                return collections.OrderedDict(x=tf.reshape(element['pixels'], [-1, 784]),
                                               y=tf.reshape(element['label'], [-1, 1]))

            return dataset.repeat(1).batch(10000).map(batch_format_fn)

        def preprocess_test(dataset):
            def batch_format_fn(element):
                """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
                return collections.OrderedDict(x=tf.reshape(element['pixels'], [-1, 784]),
                                               y=tf.reshape(element['label'], [-1, 1]))

            return dataset.repeat(1).batch(1000).map(batch_format_fn)

        self.tr = tr.preprocess(preprocess_train)
        self.te = te.preprocess(preprocess_test)

        self.train_datasets = []
        self.num_batches = []
        for cid in self.tr.client_ids[:self.N]:
            for elem in self.tr.create_tf_dataset_for_client(cid):
                X = elem['x'].numpy()
                y = elem['y'].numpy().flatten()
                self.train_datasets.append((X, y))
                self.num_batches.append(len(y) // batch_size)

        self.test_datasets = []
        for cid in self.tr.client_ids[:self.N]:
            for elem in self.te.create_tf_dataset_for_client(cid):
                X = elem['x'].numpy()
                y = elem['y'].numpy().flatten()
                self.test_datasets.append((X, y))

    def eval_test(self, w, thetas):
        acc = 0
        cnt = 0
        for i, (X, y) in enumerate(self.test_datasets):
            preds = X.dot(w + thetas[i])
            acc += np.sum(preds.argmax(1) == y)
            cnt += len(y)

        return acc / cnt

    def batches(self):
        clients = list(range(self.N))
        data = [itertools.chain.from_iterable([list(self.rd.permutation(self.num_batches[i])) for _ in range(self.num_epochs)]) for i in range(self.N)]

        xs = np.zeros((self.batch_size, self.L, self.d))
        ys = np.zeros((self.batch_size, self.L, self.n_classes))
        while clients:
            self.rd.shuffle(clients)
            ids = []
            xs[:] = 0.
            ys[:] = 0.
            i = 0
            while i < self.L and clients and i < len(clients):
                j = next(data[clients[i]], None) # batch number
                if j is not None:
                    XX, yy = self.train_datasets[clients[i]]
                    y = yy[j*self.batch_size:(j+1)*self.batch_size]
                    ys[np.arange(y.shape[0]),i, y] = 1.
                    xs[:y.shape[0],i,:] = XX[j*self.batch_size:(j+1)*self.batch_size]
                    ids.append(clients[i])
                    i += 1
                else:
                    del clients[i]
            if ids:
                yield (ids, xs, ys) 


class Stackoverflow(object):
    def __init__(self, N=500, clients_per_round=20, batch_size=10, num_epochs=1, seed=42):
        self.train_datasets = pickle.load(open('stackoverflow_train_500.pkl', 'rb'))[:N]
        self.test_datasets = pickle.load(open('stackoverflow_test_500.pkl', 'rb'))[:N]

        self.d = self.train_datasets[0][0].shape[1]
        self.n_classes = self.train_datasets[0][1].shape[1]
        self.N = N
        self.rd = np.random.RandomState(seed)

        self.batch_size = batch_size
        self.L = clients_per_round
        self.num_epochs = num_epochs

        self.num_batches = []
        for X, y in self.train_datasets:
            self.num_batches.append(X.shape[0] // batch_size)

    def eval_test(self, w, thetas):
        acc = 0
        cnt = 0
        for i, (X, y) in enumerate(self.test_datasets[:self.N]):
            preds = X.dot(w + thetas[i])
            acc += np.sum(y[np.arange(y.shape[0]),preds.argmax(1)])
            cnt += len(y)

        return acc / cnt

    def batches(self):
        clients = list(range(self.N))
        data = [itertools.chain.from_iterable([list(self.rd.permutation(self.num_batches[i])) for _ in range(self.num_epochs)]) for i in range(self.N)]

        xs = np.zeros((self.batch_size, self.L, self.d))
        ys = np.zeros((self.batch_size, self.L, self.n_classes))
        while clients:
            self.rd.shuffle(clients)
            ids = []
            xs[:] = 0.
            ys[:] = 0.
            i = 0
            while i < self.L and clients and i < len(clients):
                j = next(data[clients[i]], None) # batch number
                if j is not None:
                    XX, yy = self.train_datasets[clients[i]]

                    ys[:yy.shape[0],i,:] = yy[j*self.batch_size:(j+1)*self.batch_size]
                    xs[:yy.shape[0],i,:] = XX[j*self.batch_size:(j+1)*self.batch_size]
                    ids.append(clients[i])
                    i += 1
                else:
                    del clients[i]
            if ids:
                yield (ids, xs, ys) 


def training_priv(ds, step=0.5, stepw=None, steptheta=None,
                  C=10., noise_mult=0.05, delta=1e-4, eval_delta=500):
    w = np.zeros((ds.d, ds.n_classes))
    thetas = np.zeros((ds.N, ds.d, ds.n_classes))

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

    for i, (ids, X, y) in enumerate(ds.batches()):
        L = len(ids)
        X = X[:,:L,:]  # [B,N,d]
        y = y[:,:L,:]  # [B,N,K]
        y -= 0.1
        pred = np.sum(X[:,:,:,None] * (w[None,None,:,:] + thetas[None,ids,:,:]), axis=2)
        if i % eval_delta == 0:
            print(i, 'eval test acc...', end='', flush=True)
            t = time.time()
            acc_test = ds.eval_test(w, thetas)
            print(acc_test, '({:.2f})'.format(time.time() - t))
            test_iters.append(i)
            accs_test.append(acc_test)

        thetas_grad = np.sum(X[:,:,:,None] * (pred - y)[:,:,None,:], axis=0) / ds.batch_size # [N,d,K]
        if steptheta > 0:
            thetas[ids] -= steptheta * thetas_grad

        thetas_grad /= np.clip(np.linalg.norm(thetas_grad, axis=(1,2)) / C, a_min=1., a_max=None)[:,None,None]
        w_grad = np.mean(thetas_grad, axis=0)
        if stepw > 0:
            w -= stepw * (w_grad + noise_mult * C * np.random.randn(ds.d, ds.n_classes) / L)
            priv.step(noise_multiplier=noise_mult, sample_rate=L/ds.N)

        eps, best_alpha = priv.get_privacy_spent(delta=delta)
        if i % eval_delta == 0:
            print('eps:', eps)
        epss.append(eps)
    return {'ls': ls, 'accs': accs, 'test_iters': test_iters, 'accs_test': accs_test, 'epss': epss}


grid_emnist = {
    'step': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
    'alpha': [0, 0.1, 0.3, 1., 3., 10., 30., 100., -1],
    # 'noise_mult': [0, 0.1, 0.3, 1., 3., 10., 30., 100.],
    'noise_mult': [2.,  5., 20., 50., 70.],
    'C': [1., 10., 100.],
}

grid_stackoverflow = {
    'step': [2., 5., 10., 20., 50., 100., 150., 200., 250.],
    'alpha': [0, 0.1, 0.3, 1., 3., 10., 30., 100., -1],
    # 'noise_mult': [0, 0.01, 0.1, 1., 10., 100.],
    'noise_mult': [0.02, 0.05, 0.2, 0.5, 2., 5.],
    'C': [0.001, 0.01, 0.1, 1., 10.],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EMNIST/Stackoverflow')
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
            curves = training_priv(ds, stepw=kv['alpha'] * kv['step'], steptheta=kv['step'], **kwargs)

        elif kv['alpha'] == -1: # -1 means alpha = infinity, global only
            curves = training_priv(ds, stepw=kv['step'], steptheta=0, **kwargs)

        else: # alpha >= 1
            curves = training_priv(ds, stepw=kv['step'], steptheta=kv['step'] / kv['alpha'], **kwargs)

        results.append((kv, curves))
        if not args.interactive:
            pickle.dump(results, open(outfile, 'wb'))
