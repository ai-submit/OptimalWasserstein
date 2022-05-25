# NN_utils.py

import numpy as np
import torch
from NN_models import *

def dlog_rho_NN(X, grad_log_post, num_neurons=20, activation='Sigmoid', 
                add_poly=False, poly_degree=2, use_vector=False, 
                freeze_second_layer=False,use_cache = False, 
                state_dict=None, iter_num = 100, lr=1e-3, weight_decay=0, thres=1e-3,
                verbose=False, interval=100, cstop=0, use_bias=False, lbd=0, cubic_reg=False):
    N, d = X.shape
    if use_vector:
        if add_poly:
            Net = Feedforward_vector_poly(d,num_neurons,activation, poly_degree=poly_degree, use_bias=use_bias)
        else:
            Net = Feedforward_vector_basic(d,num_neurons,activation, use_bias=use_bias)
    else:
        # Net = Feedforward(d,num_neurons,activation, add_poly=add_poly, poly_num = poly_num)
        if add_poly:
            Net = Feedforward_poly(d,num_neurons,activation, poly_degree)
        else:
            Net = Feedforward_basic(d,num_neurons,activation)
    # advanced setting
    if freeze_second_layer:
        for name, param in Net.named_parameters():
            if 'fc2' in name:
                param.require_grad=False

    if use_cache:
        Net.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr,weight_decay=weight_decay)
    i=0
    results = np.zeros(iter_num)
    while i<iter_num:
        if i>0:
            loss_prev = loss.item()
        optimizer.zero_grad()
        if use_vector:
            grad_NN = Net(X)
            divergence_NN = Net.divergence(X)
            loss = torch.sum(grad_NN**2)+2*torch.sum(divergence_NN)+2*torch.sum(grad_NN*grad_log_post)
            loss = loss/N
            if cubic_reg:
                if add_poly:
                    Wt = list(Net.fc1_poly.parameters())[0]
                    Ut = list(Net.fc2_poly.parameters())[0]
                else:
                    Wt = list(Net.fc1.parameters())[0]
                    Ut = list(Net.fc2.parameters())[0]
                loss = loss + lbd*(torch.sum(torch.norm(Wt,2,dim=1)**3)+torch.sum(torch.norm(Ut,1,dim=0)**3))
        else:
            grad_NN = Net.grad(X)
            laplace_NN = Net.laplace(X)
            loss = torch.sum(grad_NN**2)+2*torch.sum(laplace_NN)+2*torch.sum(grad_NN*grad_log_post)
            loss = loss/N
            if cubic_reg:
                if add_poly:
                    Wt = list(Net.fc1_poly.parameters())[0]
                    Ut = list(Net.fc2_poly.parameters())[0]
                else:
                    Wt = list(Net.fc1.parameters())[0]
                    Ut = list(Net.fc2.parameters())[0]
                loss = loss + lbd*(torch.sum(torch.norm(Wt,2,dim=1)**3)+torch.sum(torch.norm(Ut,1,dim=0)**3))
        if i==0:
            loss_init = loss.item()
        loss.backward()
        optimizer.step()
        results[i] = loss.item()
        if verbose and i%interval==0:
            print('[inner] siter: {} loss: {:.3e}'.format(i, loss.item()))
        loss_item = loss.item()
        if i>0:
            rel_err = np.abs(loss_prev-loss_item)/(np.abs(loss_item)+1)
            if rel_err<thres and loss_item<0:
                break
        i+=1
        if cstop==1:
            if i==iter_num and loss_item>0:
                iter_num = iter_num+1000
            results_new = np.zeros(iter_num)
            results_new[:i-1] = results[:i-1]
            results = results_new
    results = results[:i-1]

    loss_end = loss.item()
    return grad_NN, Net.state_dict(), [loss_init, loss_end, i, rel_err], results

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
    return dlog_rho, ibw
