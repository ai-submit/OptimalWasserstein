
import cvxpy as cp
# import mosek
import numpy as np
# from NN_models import *
import torch
# import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0


def cvx_nn_gen_mask(X, sample_num = 200, seed = 0):
    ## Finite approximation of all possible sign patterns
    N, d = X.shape

    np.random.seed(seed)
    dmat = drelu(X@np.random.randn(d,sample_num)).T
    dmat = (np.unique(dmat,axis=0))

    return dmat

def cvx_nn_gen_mask_target(X, target_num = 200, seed = 0):
    ## Finite approximation of all possible sign patterns
    N, d = X.shape

    np.random.seed(seed)
    dmat = drelu(X@np.random.randn(d,target_num*2)).T

    dmat = (np.unique(dmat,axis=0))

    count = 0

    while dmat.shape[0]<target_num:
        count = count+1
        dmat_aug = drelu(X@np.random.randn(d,target_num*2)).T
        dmat = np.concatenate([dmat,dmat_aug],axis=0)
        dmat = (np.unique(dmat,axis=0))
        if count>=10:
            print('Error: cannot find enough masking matrices. Try a smaller target_num.')
            break
        
    dmat = dmat[:target_num,:]
    return dmat



def cvx_nn_relu_find_reg(X, sample_num = 200, target_num=0,seed=0):
    N, d = X.shape

    e = np.zeros([d+1,1])
    e[-1] = 1
    
    if target_num==0:
        dmat = cvx_nn_gen_mask(X, sample_num, seed)
    else:
        dmat = cvx_nn_gen_mask_target(X, target_num, seed)

    m = dmat.shape[0]
    Y_aug = cp.Variable([N,d+1])
    a = cp.Variable(1)
    r_p = cp.Variable([m,N+1])
    r_m = cp.Variable([m,N+1])

    H0 = np.eye(d+1)
    H0[-1,-1]=-1
    
    I_aug = np.eye(d+1)
    I_aug[-1,-1] = 0

    constraints = [r_p>=0, r_m>=0,Y_aug[:,-1]==0]
    X_aug = np.concatenate([X,np.zeros([N,1])],1)

    for j in range(m):
        dj = dmat[j,:]
        XDj = dj.reshape([-1,1])*X_aug
        ABj = 2*np.sum(dj)*I_aug-Y_aug.T@XDj-XDj.T@Y_aug
        h_sum_m = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_m[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
        h_sum_m = cp.reshape(h_sum_m,[d+1,1])
        Smj = ABj+r_m[j,0]*H0+h_sum_m@e.T+e@h_sum_m.T+a*e@e.T
        h_sum_p = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_p[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
        h_sum_p = cp.reshape(h_sum_p,[d+1,1])
        Spj = -ABj+r_p[j,0]*H0+h_sum_p@e.T+e@h_sum_p.T+a*e@e.T

        constraints.append(Smj>>0)
        constraints.append(Spj>>0)


    p_star = cp.Problem(cp.Minimize(a),constraints).solve()

    return p_star


def cvx_nn_relu_dual(X, V, lbd, sample_num = 200 , target_num =0, seed=0, cvx_solver='SCS', cvx_opts={}):
    N, d = X.shape
    
    tlbd = 3*2**(-5/3)*N*lbd

    e = np.zeros([d+1,1])
    e[-1] = 1
    
    if target_num==0:
        dmat = cvx_nn_gen_mask(X, sample_num, seed)
    else:
        dmat = cvx_nn_gen_mask_target(X, target_num, seed)

    m = dmat.shape[0]
    Y_aug = cp.Variable([N,d+1])
    a = cp.Variable(1)
    r_p = cp.Variable([m,N+1])
    r_m = cp.Variable([m,N+1])

    H0 = np.eye(d+1)
    H0[-1,-1]=-1

    I_aug = np.eye(d+1)
    I_aug[-1,-1] = 0

    constraints = [r_p>=0, r_m>=0,Y_aug[:,-1]==0]
    X_aug = np.concatenate([X,np.zeros([N,1])],1)
    V_aug = np.concatenate([V,np.zeros([N,1])],1)


    for j in range(m):
        dj = dmat[j,:]
        XDj = dj.reshape([-1,1])*X_aug
        ABj = 2*np.sum(dj)*I_aug-Y_aug.T@XDj-XDj.T@Y_aug
        h_sum_m = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_m[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
        h_sum_m = cp.reshape(h_sum_m,[d+1,1])
        Smj = ABj+r_m[j,0]*H0+h_sum_m@e.T+e@h_sum_m.T+tlbd*e@e.T
        h_sum_p = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_p[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
        h_sum_p = cp.reshape(h_sum_p,[d+1,1])
        Spj = -ABj+r_p[j,0]*H0+h_sum_p@e.T+e@h_sum_p.T+tlbd*e@e.T

        constraints.append(Smj>>0)
        constraints.append(Spj>>0)

    if cvx_solver == 'MOSEK':
        solver = cp.MOSEK
    else: 
        solver = cp.SCS


    p_star = cp.Problem(cp.Maximize(-0.5*cp.sum_squares(Y_aug+V_aug)),constraints).solve(solver=solver,**cvx_opts)

    if np.isinf(p_star):
        print('Lambda is too small')
        return 0, 0, 0, 0


    return Y_aug.value[:,:-1], p_star/N, 1, m