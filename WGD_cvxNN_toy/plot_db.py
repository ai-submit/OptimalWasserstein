# plot_db.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from NN_utils import *
from sampler import *
from math import ceil
from MMD import MMD, cal_MMD
import os
from dlog_p import *
import pickle

prob_name = 'double_banana'
pi = p_double_banana
log_pi = lambda X: np.log(p_double_banana(X))
dlog_pi = dlog_p_double_banana

with (open('SVN_particles_seed5.pi', "rb")) as f:
     X_SVN = pickle.load(f)
X_SVN = X_SVN.T

lr_list = [0.001]
lbd_list = [0.5]
max_iter = 100
cvx_NN_result_list = []
for lr in lr_list:
    for lbd in lbd_list:
        file_name = './results/toy2d_db/iter_{}_cvx_NN_lbd_{:.1e}_decay_0.95_iter_{}_lr_{:.1e}_seed_1.pi'.format(max_iter,lbd,max_iter,lr)
        cvx_NN_result = {}
        cvx_NN_result['print_name'] = 'cvx_NN lr {:.1e} lbd {}'.format(lr,lbd)
        with open(file_name,'rb') as f:
            MMD_NN,X_end, results,_ = pickle.load(f)
        cvx_NN_result['MMD'] = MMD_NN
        cvx_NN_result['X_end'] = X_end
        
        cvx_NN_result_list.append(cvx_NN_result)

NN_result_list = []
for lr in lr_list:
    for lbd in lbd_list:
        file_name = './results/toy2d_db/iter_{}_NN_lbd_{:.1e}_decay_0.95_nnum_200_thres_0.0001_iter_{}_lr_{:.1e}_seed_1.pi'.format(max_iter,lbd,max_iter,lr)
        NN_result = {}
        NN_result['print_name'] = 'NN lr {:.1e} lbd {}'.format(lr,lbd)
        with open(file_name,'rb') as f:
            MMD_NN,X_end, results,_ = pickle.load(f)
        NN_result['MMD'] = MMD_NN
        NN_result['X_end'] = X_end
        
        NN_result_list.append(NN_result)

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1,1,figsize=(12,4))
for cvx_NN_result in cvx_NN_result_list:
    MMD_NN = cvx_NN_result['MMD']
    print_name = 'WGD-cvxNN'
    ax.plot(MMD_NN,label=print_name)
for NN_result in NN_result_list:
    MMD_NN = NN_result['MMD']
    print_name = 'WGD-NN'
    ax.plot(MMD_NN,label=print_name)
# ax.set_yscale('log')
ax.set_xlabel('iteration')
ax.set_ylabel('MMD')
plt.legend()
os.makedirs('figures/', exist_ok=True)
fig.savefig('figures/{}_MMD_cvx_NN.png'.format(prob_name),bbox_inches='tight')

xnum, ynum = 100, 100
xmin, ymin = -2, -2
xmax, ymax = 2, 3
x = (np.arange(1,xnum+1)/xnum-0.5)*(xmax-xmin)+(xmax+xmin)/2
y = (np.arange(1,ynum+1)/ynum-0.5)*(ymax-ymin)+(ymax+ymin)/2
X1,Y1 = np.meshgrid(x,y)
Z = np.zeros([xnum,ynum,2])
Z[:,:,0]=X1
Z[:,:,1]=Y1
Z_aux = np.reshape(Z,[xnum*ynum,2])
Z_ans = pi(Z_aux)
Z_plot = np.reshape(Z_ans,[xnum,ynum])

fig, ax = plt.subplots(1,2,figsize=(10,4))
alpha_v = 0.6

ax_sub=ax[0]
cvx_NN_result = cvx_NN_result_list[0]
X_result = cvx_NN_result['X_end']
ax_sub.scatter(X_result[:,0],X_result[:,1])
ax_sub.contour(X1,Y1,Z_plot,alpha=alpha_v)
ax_sub.set_title('WGD-cvxNN')
ax_sub=ax[1]
NN_result = NN_result_list[0]
X_result = NN_result['X_end']
ax_sub.scatter(X_result[:,0],X_result[:,1])
ax_sub.contour(X1,Y1,Z_plot,alpha=alpha_v)
ax_sub.set_title('WGD-NN')
fig.savefig('figures/{}_MMD_final.png'.format(prob_name),bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(5,4))
alpha_v = 0.6
ax_sub=ax

X_result = X_SVN
ax_sub.scatter(X_result[:,0],X_result[:,1])
ax_sub.contour(X1,Y1,Z_plot,alpha=alpha_v)
ax_sub.set_title('SVN')

fig.savefig('figures/{}_MMD_final_ref.png'.format(prob_name),bbox_inches='tight')