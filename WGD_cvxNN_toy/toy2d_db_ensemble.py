import argparse
import torch
import numpy as np
# import matplotlib.pyplot as plt
from NN_utils import *
from sampler import *
from math import ceil
from MMD import MMD, cal_MMD
import os
from dlog_p import *
import pickle
from time import time

def get_parser():
    parser = argparse.ArgumentParser(description='linear')
    parser.add_argument("--max_iter", type=int, default=60)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--lbd", type=float, default=50)
    parser.add_argument("--lbd_decay", type=float, default=0.95)
    parser.add_argument("--num_neurons", type=int, default=200)
    parser.add_argument("--thres", type=int, default=1e-4)
    parser.add_argument("--lr_nn", type=float, default=1e-3)
    parser.add_argument("--iter_nn", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed_init", type=int, default=1)
    parser.add_argument("--seed_alg", type=int, default=1)
    parser.add_argument("--sampler", type=str, default='cvx_NN',choices=['cvx_NN', 'NN', 'KDE'])
    return parser

def main():
    start = time()
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    sampler = args.sampler
    lbd = args.lbd
    lbd_decay = args.lbd_decay
    max_iter = args.max_iter
    lr = args.lr
    sample_num = args.sample_num


    prob_name = 'double_banana'
    pi = p_double_banana
    log_pi = lambda X: np.log(p_double_banana(X))
    dlog_pi = dlog_p_double_banana

    with (open('SVN_particles_seed5.pi', "rb")) as f:
        X_SVN = pickle.load(f)
    X_SVN = X_SVN.T


    N = 50
    d = 2
    np.random.seed(args.seed_init)
    X = np.random.randn(N,d)
    X_init = X.copy()
    interval = 1

    method_str = ''

    if sampler=='cvx_NN':
        cvx_NN_opts = {'sample_num':sample_num,'seed':args.seed_alg}
        X_end, results, X_history, flag = WGD_cvx_NN(X_init, dlog_pi, log_pi, lr=lr,max_iter=max_iter,interval=interval, lbd=lbd,cvx_NN_opts=cvx_NN_opts, 
                                                            verbose=True, track_X=True,lbd_decay=lbd_decay)

        method_str = 'cvx_NN_lbd_{:.1e}_decay_{}'.format(lbd,lbd_decay)

    elif sampler=='NN':
        NN_setting = {'activation': 'ReLU_sq', 'num_neurons': args.num_neurons, 'iter_num': args.iter_nn, 'add_poly': False, 'cubic_reg':True, 
              'lbd':lbd,'lr': args.lr_nn,'thres':args.thres}
        X_end, results, X_history = WGD_NN(X_init, dlog_pi, log_pi, lr=1e-3, max_iter=max_iter, NN_opts=NN_setting, 
                                     interval=interval, verbose=True, track_X=True,lbd_decay=lbd_decay)

        method_str = 'NN_lbd_{:.1e}_decay_{}_nnum_{}_thres_{}'.format(lbd,lbd_decay, args.num_neurons, args.thres)

    MMD_NN = cal_MMD(X_history, X_SVN)

    save_folder = 'results/toy2d_db'
    os.makedirs(save_folder, exist_ok=True)

    file_name = '{}/iter_{}_{}_iter_{}_lr_{:.1e}_seed_{}.pi'.format(save_folder, max_iter,method_str, max_iter, lr, args.seed_init)


    with open(file_name, 'wb') as f:
        pickle.dump([MMD_NN,X_end, results, X_history],f)

    end = time()
    print('It takes {:.3f} s.'.format(end-start))

if __name__ == '__main__':
    main()