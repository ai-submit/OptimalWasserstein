# cvx_nn_square_actv.py

import cvxpy as cp
import mosek
import numpy as np
# from NN_models import *
import torch
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0

def convex_NN_sqr(X,V,lbd,eps=1e-7):
    N, d = X.shape
    
    tlbd = 3*2**(-2/3)*N*lbd

    mu = np.sum(X,0).reshape([-1,1])
    I = np.eye(d)
    y = cp.Variable(N)
#     beta = cp.Variable(1)

    y_list = []
    beta_list = []
    pstar_list = []
    for i in range(d):
        ei = np.zeros([d,1])
        ei[i] = 1
        muei = mu@ei.T
        muei = (muei+muei.T)
        A = X.T@cp.diag(y)@X-muei
#         S1 = beta*I-A
#         S2 = beta*I+A
#         constraints = [S1>>0, S2>>0, beta<=tlbd/2]
        constraints = [tlbd/2*I-A>>0, tlbd/2*I+A>>0]
        objective = -0.5*cp.sum_squares(y+V[:,i])
        p_star = cp.Problem(cp.Maximize(objective),constraints).solve(eps=eps)
        y_list.append(y.value)
# #         beta_list.append(beta.value)
        pstar_list.append(p_star)
    d_opt = np.sum(np.array(pstar_list))
    # print('The optimal dual objective is {:.3f}'.format(d_opt))
    
    return d_opt, y_list

def convex_NN_sqr_bidual(X,V,lbd,eps=1e-7):
    N, d = X.shape
    
    tlbd = 3*2**(-2/3)*N*lbd

    mu = np.sum(X,0).reshape([-1,1])
    I = np.eye(d)
    pstar_list = []
    S_list = []
    
    for i in range(d):
        ei = np.zeros([d,1])
        ei[i] = 1
        muei = mu@ei.T
        muei = (muei+muei.T)
        vi = V[:,i]
        
        S = cp.Variable((d,d), symmetric=True)
        z = cp.Variable(N)
        constraints = []
        for n in range(N):
            xn = X[n,:]
            constraints.append(z[n] == xn.T@S@xn)
        
        pstar = cp.Problem(cp.Minimize(0.5*cp.sum_squares(vi+z)+2*ei.T@S@mu+tlbd/2*cp.norm(S,'nuc')),constraints).solve(eps=eps)
        pstar_list.append(pstar-0.5*np.linalg.norm(vi)**2)
        S_list.append(S.value)
    p_opt = np.sum(np.array(pstar_list))
    
    W_list = []
    U_list = []
    for i in range(d):
        Si = S_list[i]
        ei = np.zeros([d,1])
        ei[i] = 1
        u, W = np.linalg.eig(Si)
        U = u.reshape([-1,1])@ei.reshape([1,-1])
        W_list.append(W)
        U_list.append(U)
        
    W = np.concatenate(W_list,1)
    U = np.concatenate(U_list,0)
    
    return p_opt, S_list, (W,U)

def build_quad_NN(X,V,lbd,W,U):
    N,d = X.shape
    
    m = W.shape[1]
    W_p = W.copy()
    U_p = U.copy()
    for i in range(m):
        alpha = 2**(-1/9)*(np.linalg.norm(W[:,i])/np.linalg.norm(U[i,:],1))**(1/3)
        U_p[i,:] = U[i,:]*alpha**2
        W_p[:,i] = W[:,i]/alpha
        
    Net = Feedforward_vector_poly(d,m,'zero',False, 2)
    state_dict = Net.state_dict()
    state_dict['fc1_poly.weight'] = torch.Tensor(W_p.T)
    state_dict['fc2_poly.weight'] = torch.Tensor(U_p.T)
    Net.load_state_dict(state_dict)
    
    X_t = torch.Tensor(X)
    V_t = torch.Tensor(V)
    
    grad_NN = Net(X_t)
    divergence_NN = Net.divergence(X_t)
    V_t = -X_t
    loss = torch.sum(grad_NN**2)+2*torch.sum(divergence_NN)+2*torch.sum(grad_NN*V_t)
    Wt = list(Net.fc1_poly.parameters())[0]
    Ut = list(Net.fc2_poly.parameters())[0]
    loss = loss+N*lbd*(torch.sum(torch.norm(Wt,2,dim=1)**3)+torch.sum(torch.norm(Ut,1,dim=0)**3))
    loss = loss/2
    p_opt_NN = loss.item()
    return p_opt_NN, state_dict

def convex_NN_sqr_primal_from_dual(X,V,lbd,y_list,thres1=1e-3, thres2=1e-3,debug=False):
    N, d = X.shape
    
    tlbd = 3*2**(-5/3)*N*lbd

    mu = np.sum(X,0).reshape([-1,1])
    I = np.eye(d)
    # recover primal variables
    w_list = []
    u_list = []
    for i in range(d):
        if debug:
            print(i)
        y_np = y_list[i]
        ei = np.zeros([d,1])
        ei[i] = 1
        muei = mu@ei.T
        muei = (muei+muei.T)
        S_np = X.T@np.diag(y_np)@X-muei
        val, vec = np.linalg.eig(S_np)
#         betai = beta_list[i]

        w_list_sub = []
        sign_list_sub = []
#         if debug:
#             print('beta: {:.3f}'.format(betai.item()))
        for j in range(d):
            eig_val = val[j]
            if debug:
                print('{}-th eigenvalue: {:.3f}'.format(j,eig_val))
                print('difference: {:.3e}'.format(np.abs(np.abs(eig_val)-tlbd/2).item()))
#             if np.abs(np.abs(eig_val)-betai)<thres1:
            if np.abs(np.abs(eig_val)-tlbd/2)<thres1:
                if debug:
                    print('hit')
                w_list_sub.append(vec[:,j])
                sign_list_sub.append(np.sign(eig_val))
        W = np.array(w_list_sub).T
        b = -(y_np+V[:,i])
        C = (X@W)**2
        m0 = C.shape[1]
#         u = np.linalg.solve(C.T@C,C.T@b)
        u_cp = cp.Variable(m0)
        p_sub = cp.Problem(cp.Minimize(cp.sum_squares(C@u_cp-b))).solve()
        u = u_cp.value
        
        if np.linalg.norm(b-C@u,np.inf)>thres2:
            
            print('Error. {}-th residual: {:.3e}'.format(i,np.linalg.norm(b-C@u,np.inf)))
        for k in range(len(u)):
            if u[k]*sign_list_sub[k]<0:
                print('Sign Error')
        U = u.reshape([-1,1])@ei.reshape([1,-1])
        w_list.append(W)
        u_list.append(U)
        
    W = np.concatenate(w_list,1)
    U = np.concatenate(u_list,0)
    
    return W,U

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

def cvx_nn_relu_primal(X, V, lbd, sample_num = 200 ,seed=0):
    N, d = X.shape
    
    tlbd = 3*2**(-5/3)*N*lbd

    e = np.zeros([d+1,1])
    e[-1] = 1
    
    # Finite approximation of all possible sign patterns
    np.random.seed(seed)
    dmat = drelu(X@np.random.randn(d,sample_num)).T

    dmat = (np.unique(dmat,axis=0))

    m = dmat.shape[0]
    
    H0 = np.eye(d+1)
    H0[-1,-1]=-1
    
    B0 = np.eye(d+1)
    B0[-1,-1]=0
    
    X_aug = np.concatenate([X,np.zeros([N,1])],1)
    V_aug = np.concatenate([V,np.zeros([N,1])],1)
    
    S_p_list = []
    S_m_list = []
    
    constraints = []
    objective = 0
    
    Z = 0
    for j in range(m):
        S_p_j = cp.Variable((d+1,d+1), symmetric=True)
        S_m_j = cp.Variable((d+1,d+1), symmetric=True)
        
        S_p_list.append(S_p_j)
        S_m_list.append(S_m_j)
        constraints.append(S_p_j>>0)
        constraints.append(S_m_j>>0)
        
        constraints.append(cp.trace(S_p_j@H0)<=0)
        constraints.append(cp.trace(S_m_j@H0)<=0)
        dj = dmat[j,:]
        DjX_H = (2*dj.reshape([-1,1])-1)*X_aug
        
        DjX = dj.reshape([-1,1])*X_aug
        
        Z = Z+2*DjX@(S_p_j-S_m_j)
        objective = objective+2*np.sum(dj)*cp.trace((S_p_j-S_m_j)@B0)+tlbd*cp.trace((S_m_j+S_p_j)@e@e.T)
        
        
        for i in range(N):
            DjXi = DjX_H[i,:].reshape([-1,1])
            constraints.append(cp.trace(S_p_j@e@DjXi.T)<=0)
            constraints.append(cp.trace(S_m_j@e@DjXi.T)<=0)
            
    objective = objective+0.5*cp.sum_squares(Z+V_aug)-0.5*np.linalg.norm(V_aug)**2
    p_star = cp.Problem(cp.Minimize(objective),constraints).solve()

    if np.isinf(p_star):
        print('Lambda is too small')
        return 0, 0, 0, 0
    
    Sp_v_list = []
    Sm_v_list = []
    
    for j in range(m):
        Sp_v_list.append(S_p_list[j].value)
        Sm_v_list.append(S_m_list[j].value)


    return Z.value[:,:-1], p_star/N, (Sp_v_list,Sm_v_list), 1

def cvx_nn_vec_relu_find_reg(X, i, sample_num = 200 ,seed=0):
    N, d = X.shape

    e = np.zeros([d+1,1])
    e[-1] = 1
    
    ## Finite approximation of all possible sign patterns
    np.random.seed(seed)
    dmat = drelu(X@np.random.randn(d,sample_num)).T

    dmat = (np.unique(dmat,axis=0))

    m = dmat.shape[0]
    y = cp.Variable(N)
    a = cp.Variable(1)
    r_p = cp.Variable([m,N+1])
    r_m = cp.Variable([m,N+1])
    
    ei = np.zeros([d+1,1])
    ei[i] = 1
    H0 = np.eye(d+1)
    H0[-1,-1]=-1

    constraints = [r_p>=0, r_m>=0]
    X_aug = np.concatenate([X,np.zeros([N,1])],1)


    for j in range(m):
        dj = dmat[j,:]
        XDj = dj.reshape([-1,1])*X_aug
        XDje = np.sum(XDj,0).reshape([-1,1])
        ABj = XDje@ei.T+ei@XDje.T-XDj.T@cp.diag(y)@XDj
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

def cvx_nn_vec_relu_dual(X, V, lbd, sample_num = 200 ,seed=0):
    N, d = X.shape
    
    tlbd = 3*2**(-2/3)*N*lbd

    e = np.zeros([d+1,1])
    e[-1] = 1
    
    # Finite approximation of all possible sign patterns
    np.random.seed(seed)
    dmat = drelu(X@np.random.randn(d,sample_num)).T

    dmat = (np.unique(dmat,axis=0))

    m = dmat.shape[0]
    y = cp.Variable(N)
    a = cp.Variable(1)
    r_p = cp.Variable([m,N+1])
    r_m = cp.Variable([m,N+1])
    
    y_list = []
    X_aug = np.concatenate([X,np.zeros([N,1])],1)
    H0 = np.eye(d+1)
    H0[-1,-1]=-1
    for i in range(d):
        ei = np.zeros([d+1,1])
        ei[i] = 1

        constraints = [r_p>=0, r_m>=0]


        for j in range(m):
            dj = dmat[j,:]
            XDj = dj.reshape([-1,1])*X_aug
            XDje = np.sum(XDj,0).reshape([-1,1])
            ABj = XDje@ei.T+ei@XDje.T-XDj.T@cp.diag(y)@XDj
    #         h_sum_m = 0
    #         for k in range(N):
    #             h_sum_m = h_sum_m+r_m[j,k+1]*(1-2*dj[k])*X_aug[k,:]
            h_sum_m = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_m[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
            h_sum_m = cp.reshape(h_sum_m,[d+1,1])
            Smj = ABj+r_m[j,0]*H0+h_sum_m@e.T+e@h_sum_m.T+tlbd/2*e@e.T
            h_sum_p = cp.sum(cp.multiply(cp.reshape(cp.multiply(r_p[j,1:],(1-2*dj)),[N,1]),X_aug),axis=0)
    #         h_sum_p = 0
    #         for k in range(N):
    #             h_sum_p = h_sum_p+r_p[j,k+1]*(1-2*dj[k])*X_aug[k,:]
            h_sum_p = cp.reshape(h_sum_p,[d+1,1])
            Spj = -ABj+r_p[j,0]*H0+h_sum_p@e.T+e@h_sum_p.T+tlbd/2*e@e.T

            constraints.append(Smj>>0)
            constraints.append(Spj>>0)


        p_star = cp.Problem(cp.Maximize(-0.5*cp.sum_squares(y+V[:,i])),constraints).solve()

        if np.isinf(p_star):
            print('Lambda is too small')
            return 0, 0, 0

        y_v = y.value
        y_list.append(y_v)
    y_all = np.array(y_list).T
    return y_all, p_star/N, 1

# def train_NN_square_actv(X,grad_log_post,lbd,num_neurons, ReLU=False, verbose=False, iter_num=100, lr=1e-1,interval=10,thres=1e-3):
#     N, d = X.shape
#     if ReLU:
#         Net = Feedforward_vector_basic(d,num_neurons,'ReLU_sq', False)
#     else:
#         Net = Feedforward_vector_poly(d,num_neurons,'zero', False, 2)
#     optimizer = torch.optim.Adam(Net.parameters(), lr=lr,weight_decay=0)
#     use_vector = True
#     i = 0
#     results = np.zeros(iter_num)
#     while i<iter_num:
#         if i>0:
#             loss_prev = loss.item()
#         optimizer.zero_grad()

#         if use_vector:
#             grad_NN = Net(X)
#             divergence_NN = Net.divergence(X)
#             loss = torch.sum(grad_NN**2)+2*torch.sum(divergence_NN)+2*torch.sum(grad_NN*grad_log_post)
#             if ReLU:
#                 Wt = list(Net.fc1.parameters())[0]
#                 Ut = list(Net.fc2.parameters())[0]
#             else:
#                 Wt = list(Net.fc1_poly.parameters())[0]
#                 Ut = list(Net.fc2_poly.parameters())[0]
#             loss = loss+N*lbd*(torch.sum(torch.norm(Wt,2,dim=1)**3)+torch.sum(torch.norm(Ut,1,dim=0)**3))
#             loss = loss/2

#         if i==0:
#             loss_init = loss.item()
#         loss.backward()
#         optimizer.step()
#         results[i] = loss.item()
#         if verbose and i%interval==0:
#             print('[inner] siter: {} loss: {:.3e}'.format(i, loss.item()))
#         loss_item = loss.item()
#         if i>0:
#             rel_err = np.abs(loss_prev-loss_item)/(np.abs(loss_item)+1)
#             if rel_err<thres and loss_item<0:
#                 break
#         i+=1
#     results = results[:i-1]
#     loss_end = loss.item()
#     return grad_NN, Net.state_dict(), [loss_init, loss_end, i, rel_err], results