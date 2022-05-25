# WGD_utils.py

import numpy as np

def kernel_mat(x,y,ibw,dtype,opts={}):
    # Derivatives of Gaussian kernel
    # Input:
    #       x, y  --- 1*d vector
    #       ibw   --- inverse of bandwidth
    #       dtype --- form: a0b
    #                 a 1: di x 2: didj x 3: didi x 4: laplace x
    #                 b 1: di y 2: didj y 3: didi y 4: laplace y
    # Output:
    #       K

    if opts.get('pre_comp',-1)==-1:
        opts['pre_comp'] = 0

    x = x.reshape([1,-1])
    y = y.reshape([1,-1])

    _, d = x.shape
    kxy = np.exp(-np.linalg.norm(x-y)**2/2*ibw)

    dxy = x-y

    if dtype==0:
        K = kxy;
    elif dtype==100:
        K = -dxy*ibw*kxy;
    elif dtype==400:
        K = (np.linalg.norm(dxy)**2*ibw-d)*(ibw*kxy);

    return K

def kernel_mat_group(X,ibw,dtype,opts={}):
    # Input:
    #       X N*d vector
    #       ibw: inverse of bandwidth
    #       dtype:
    # Output:
    #       K

    N, d = X.shape

    if dtype not in [0,100,400]:
        raise Exception('Invalid dtype.')

    if dtype==0 or dtype==400:
        row, col = 1, 1
    elif dtype==100:
        row, col = d, 1

    K = np.zeros([N*row,N*col])

    for i in range(N):
        for j in range(N):
            K[i*row:(i+1)*row,j*col:(j+1)*col] = kernel_mat(X[i,:],X[j,:],ibw,dtype,opts).reshape([row,col]);
    return K

def kernel_mat_uni(X,ibw,dtype,cache={}):

    # Input:
    #       X N*d vector
    #       ibw: inverse of bandwidth
    #       dtype:
    # Output:
    #       K

    N, d = X.shape

    if dtype not in [0,100,400]:
        raise Exception('Invalid dtype.')

    if cache!={}:
        dXX = cache['dXX']
    else:
        X_sum = np.sum(X**2,axis=1)
        dXX = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])

    kXX = np.exp(-dXX/2*ibw)
    if dtype==0:
        K = kXX
    elif dtype==400:
        K = kXX*ibw*(dXX*ibw-d)
    elif dtype==100:
        tmp1 = X.reshape([N,d,1])
        tmp2 = X.T.reshape([1,d,N])
        kXX_tmp = np.reshape(kXX,[N,1,N])
        K = (tmp2-tmp1)*kXX_tmp*ibw
        K = K.reshape([N*d,N])
    return K

def dlog_rho_kernel(X, grad, ibw=-1, lbd=0,lbd2=0, normalize=False, cache={}, use_sketch=False, sketch_dim=0):
    # approximate nabla log rho via RKHS

    N, d = X.shape
    X_sum = np.sum(X**2,axis=1)

    if cache=={}:
        dXX = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])
        cache = {'dXX':dXX}
    else:
        dXX = cache['dXX']

    if ibw==-1:
        ibw = 1/(0.5*np.median(dXX.ravel())/np.log(N+1))

    K0 = kernel_mat_uni(X,ibw,0,cache)
    K1 = kernel_mat_uni(X,ibw,100,cache)
    K2 = kernel_mat_uni(X,ibw,400,cache)

    if normalize:
        K0 = K0*(2*pi*ibw)**(-d/2)
        K1 = K1*(2*pi*ibw)**(-d/2)
        K2 = K2*(2*pi*ibw)**(-d/2)

    K2_sum = np.sum(K2,1)-np.matmul(K1.T,grad.reshape([N*d]))
    if use_sketch:
        if sketch_dim>0:
            m = sketch_dim
        else:
            m = N
        idx = np.choice(N*d,m)
        K1_sub = K1[idx,:]/np.sqrt(m/N)
        K_mat = np.matmul(K1_sub.T, K1_sub)+N*lbd*K0+N*lbd2*np.eye(N)
    else:
        K_mat = np.matmul(K1.T,K1)+N*lbd*K0+N*lbd2*np.eye(N)

    alpha_ = np.linalg.solve(K_mat,K2_sum)
    dlog_rho = np.reshape(np.matmul(K1,alpha_),[N,d])
    return dlog_rho

def dlog_rho_kernel_ver2(K0, K1, K2, grad, lbd=0,lbd2=0,use_sketch=False, sketch_dim=0):
    # approximate nabla log rho via RKHS
    N, d = grad.shape
    K2_sum = np.sum(K2,1)-np.matmul(K1.T,grad.reshape([N*d]))
    if use_sketch:
        if sketch_dim>0:
            m = sketch_dim
        else:
            m = N
        idx = np.choice(N*d,m)
        K1_sub = K1[idx,:]/np.sqrt(m/N)
        K_mat = np.matmul(K1_sub.T, K1_sub)+N*lbd*K0+N*lbd2*np.eye(N)
    else:
        K_mat = np.matmul(K1.T,K1)+N*lbd*K0+N*lbd2*np.eye(N)

    alpha_ = np.linalg.solve(K_mat,K2_sum)
    dlog_rho = np.reshape(np.matmul(K1,alpha_),[N,d])
    return dlog_rho

def dlog_rho_KDE(X,ibw=-1,cache={}):
    N, d = X.shape
    if cache!={}:
        dXX = cache['dXX']
    else:
        X_sum = np.sum(X**2,axis=1)
        dXX = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])
    if ibw==-1:
        ibw = 1/(0.5*np.median(dXX.ravel())/np.log(N+1))
    Kxy = np.exp(-dXX*0.5*ibw)
    dxKxy = np.matmul(Kxy,X)
    sumKxy = np.sum(Kxy,1).reshape([-1,1])
    dlog_rho = ibw*(dxKxy/sumKxy-X)
    return dlog_rho