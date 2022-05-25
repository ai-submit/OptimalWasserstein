# dlog_p.py

import numpy as np 

def model_double_banana(d=2, a=1, b=100, nobs=1, std=0.3, seed=5, import_data=False):
    if import_data==False:
        np.random.seed(seed)
        u_true = np.random.randn(1,d)
        noise = std*np.random.randn(nobs,1)
        u1 = u_true[0,0]
        u2 = u_true[0,1]
        y = np.log( (a - u1)**2 + b*(u2 - u1**2)**2 ) + noise
    else:
        y = 3.57857342 # from SVN random seed
    model_F = lambda u: np.log( (a - u[0])**2 + b*(u[1] - u[0]**2)**2 )
    return y, std, model_F

def prior_double_banana(d=2):
    m0 = np.zeros([d,1])
    C0i = np.eye(d)
    return m0, C0i

def p_double_banana(X):
    N, d = X.shape
    p = np.zeros([N,1])
    y, std, model_F = model_double_banana()
    m0, C0i = prior_double_banana()
    for i in range(N):
        x = X[i,:].reshape([-1,1])
        Fx= model_F(x)
        mlprr = 0.5*(x - m0).T @ C0i @ (x - m0)
        misfit = (y - Fx) / std;
        mllkd  = 0.5*np.sum(misfit**2);
        p[i] = np.exp( -(mllkd + mlprr) )
    return p

def forward_solve_double_banana(x,a=1,b=100):
    J = np.zeros([2,1])
    J[0] = ( 2*( x[0] - a - 2*b*x[0]*(x[1]- x[0]**2) ) )/( (x[0]-a)**2 + b*(x[1] - x[0]**2)**2 )
    J[1] = 2*b*(x[1]- x[0]**2)/( (x[0]-a)**2 + b*(x[1] - x[0]**2)**2 )
    return J


def dlog_p_double_banana(X):
    N, d = X.shape
    dlog_p = np.zeros([N,d])
    y, std, model_F = model_double_banana()
    m0, C0i = prior_double_banana()
    for i in range(N):
        xi = X[i,:].reshape([-1,1])
        Fx= model_F(xi)
        J = forward_solve_double_banana(xi)
        dlog_p[i,:] = -(C0i@(xi-m0)+J * (Fx - y)/std**2).reshape([1,-1])
    return dlog_p

    