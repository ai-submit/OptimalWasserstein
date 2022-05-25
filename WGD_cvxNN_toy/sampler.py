import numpy as np 
from NN_utils import *
from cvx_nn import *

def Langevin_MCMC(X_init, dlog_pi, lr=1e-1, max_iter = 100, verbose=False, interval=10, track_X=False):
    X = X_init.copy()
    N, d = X.shape
    k = 0
    results = np.zeros(max_iter)
    X_history = []
    while k<max_iter:
        grad_log_pi = dlog_pi(X)
        X = X+lr*grad_log_pi+np.sqrt(2*lr)*np.random.randn(N,d)
        if verbose and k%interval==0:
            print('iter: {}'.format(k))
            if track_X:
                X_history.append(X.copy())
        k+=1
    return X, X_history

def WGD_KDE(X_init, dlog_pi, lr = 1e-1, max_iter = 100, verbose=False, interval=10, track_X=False):
    X = X_init.copy()
    k = 0
    results = np.zeros(max_iter)
    X_history = []
    while k<max_iter:
        grad_log_pi = dlog_pi(X)
        d_log_rho, ibw = dlog_rho_KDE(X)
        X = X+lr*(grad_log_pi-d_log_rho)
        if verbose and k%interval==0:
            print('iter: {}'.format(k))
            if track_X:
                X_history.append(X.copy())
        k+=1
    return X, X_history

def WGD_NN(X_init, dlog_pi, log_pi, lr = 1e-1, max_iter=100, NN_opts={}, verbose=False, interval=10, 
           track_X = False, lbd_decay=1.):
    X = X_init.copy()
    N, d = X.shape
    k = 0
    results = np.zeros([max_iter,6])
    if verbose:
        print('{:>4s} {:>10s} {:>8s} {:>10s} {:>10s} {:>5s} {:>10s} {:>10s}'.format('iter', 'log_pi', 'movement', 'loss_init',
         'loss_end', 'siter', 'relerr','lambda'))
    
    X_history = []

    if NN_opts.get('cubic_reg',-1)==-1:
        NN_opts['cubic_reg'] = False

    while k<max_iter:
        grad_log_pi = dlog_pi(X)
        X_t = torch.Tensor(X)
        grad_log_pi_t = torch.Tensor(grad_log_pi)
        if k==0:
            grad_NN, state_dict, info, _ = dlog_rho_NN(X_t, grad_log_pi_t, **NN_opts)
        else:
            grad_NN, state_dict, info, _ = dlog_rho_NN(X_t, grad_log_pi_t, use_cache=True, state_dict = state_dict,
                **NN_opts)

        grad_NN_np = grad_NN.detach().numpy()
        X = X-lr*grad_NN_np

        if lbd_decay<1.:
            if NN_opts['cubic_reg']:
                NN_opts['lbd'] = NN_opts['lbd']*lbd_decay
            else:
                NN_opts['weight_decay'] = NN_opts['weight_decay']*lbd_decay

        movement = np.mean(np.sqrt(np.sum(grad_NN_np**2,1)))*lr
        log_pi_avg = np.mean(log_pi(X))

        results[k, :] = [log_pi_avg, movement, info[0],info[1],info[2], info[3]]

        if NN_opts['cubic_reg']:
            reg = NN_opts['lbd'] 
        else:
            reg = NN_opts['weight_decay']

        if verbose and k%interval==0:
            print('{:4d} {:10.2e} {:8.2e} {:10.2e} {:10.2e} {:5d} {:10.2e} {:10.2e}'.format(k, log_pi_avg, movement, info[0],info[1],info[2], info[3], reg))
            if track_X:
                X_history.append(X.copy())
        k+=1

    return X, results, X_history

def WGD_cvx_NN(X_init, dlog_pi, log_pi, lr = 1e-1, max_iter=100, actv='ReLU_sqr', lbd=1e-3, cvx_NN_opts={}, verbose=False, interval=10, 
           track_X = False, lbd_decay=1, lbd_inc_iter = 10):
    X = X_init.copy()
    N, d = X.shape
    k = 0
    results = np.zeros([max_iter,4])
    if verbose:
        print('{:>4s} {:>10s} {:>8s} {:>10s} {:>10s} {:>10s}'.format('iter', 'log_pi', 'movement', 'loss_cvx', 'lambda', 'num_neurons'))
    
    X_history = []

    while k<max_iter:
        grad_log_pi = dlog_pi(X)

        lbd = lbd*lbd_decay 

        grad_NN_cvx, p_star, flag, num_neurons = cvx_nn_relu_dual(X,grad_log_pi,lbd,**cvx_NN_opts)

        if flag==0:
            print('lambda becomes larger')
            lbd = lbd/lbd_decay**(lbd_inc_iter+1)
            dlog_rho_X = 0
        else:
            dlog_rho_X = -grad_NN_cvx-grad_log_pi
        
        X = X-lr*dlog_rho_X
        
        # X = X+lr*(grad_NN_cvx+grad_log_pi)

        movement = np.mean(np.sqrt(np.sum((grad_NN_cvx+grad_log_pi)**2,1)))*lr
        log_pi_avg = np.mean(log_pi(X))

        results[k, :] = [log_pi_avg, movement, p_star,lbd]

        if verbose and k%interval==0:
            print('{:4d} {:10.2e} {:8.2e} {:10.2e} {:10.2e} {:10d}'.format(k, log_pi_avg, movement, p_star,lbd, num_neurons))
            if track_X:
                X_history.append(X.copy())
        k+=1

    return X, results, X_history, 1