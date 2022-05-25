# MMD.py

import numpy as np

def MMD(X,Y,bandwidth=1.):
    # MMD based on Gaussian kernel
    # X, Y are with shapes N*d and M*d
    N, d = X.shape
    M, _ = Y.shape
    h = bandwidth
    X_sum = np.sum(X**2,axis=1)
    Y_sum = np.sum(Y**2,axis=1)
    Dxx = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])
    Dxy = -2*np.matmul(X, Y.T)+X_sum.reshape([-1,1])+Y_sum.reshape([1,-1])
    Dyy = -2*np.matmul(Y, Y.T)+Y_sum.reshape([-1,1])+Y_sum.reshape([1,-1])
    Kxx = np.exp(-Dxx/(2*h))
    Kxy = np.exp(-Dxy/(2*h))
    Kyy = np.exp(-Dyy/(2*h))
    res = 1./N**2*np.sum(Kxx.ravel())-2./(M*N)*np.sum(Kxy.ravel())+1./M**2*np.sum(Kyy.ravel())
    res = res*(2*np.pi*h)**(-d/2)
    return res

def cal_MMD(X_history, X_MCMC):
    x_num = len(X_history)
    MMD_list = []
    for Xi in X_history:
        MMD_list.append(MMD(X_MCMC,Xi))
        MMD_np = np.array(MMD_list)
    return MMD_np