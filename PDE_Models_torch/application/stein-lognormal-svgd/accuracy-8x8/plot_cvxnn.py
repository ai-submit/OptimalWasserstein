import pickle
import numpy as np
import matplotlib.pyplot as plt

dSet = [81, 65, 256, 1024]
dPara = np.array(dSet)
# NSet = [256, 256, 256, 256]
NSet = [64, 64, 64, 64]
# NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 95

fontsize = 12

colors = ['g', 'k', 'm', 'c', 'b', 'y', 'r','g', 'k', 'm', 'c', 'b', 'y', 'r']
markers = ['x-', 'd-', 'o-', '*-', 's-', '<-', '*-','x-', 'd-', 'o-', '*-', 's-', '<-', '*-']

case = 0
d, N = dSet[case], NSet[case]

# SVGD = {}
# SVGD['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# SVGD['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# SVGD['label'] = 'SVGD'
# SVGD['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_SVGD'.format(d,N)

# pSVGD = {}
# pSVGD['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# pSVGD['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pSVGD['label'] = 'pSVGD'
# pSVGD['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_SVGD'.format(d,N)

# pSVGD_post = {}
# pSVGD_post['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# pSVGD_post['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pSVGD_post['label'] = 'pSVGD-post'
# pSVGD_post['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_SVGD'.format(d,N)

# WGF = {}
# WGF['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# WGF['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# WGF['label'] = 'WGF'
# WGF['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_WGF'.format(d,N)

pWGF_nn_1 = {}
pWGF_nn_1['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_nn_1['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_nn_1['label'] = 'WGD-NN_1'
pWGF_nn_1['label'] = 'pWGD-NN'
pWGF_nn_1['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_NN_ReLU_sq_200_lbd_1_decay95'.format(d,N)

pWGF_nn_2 = {}
pWGF_nn_2['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_nn_2['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_nn_2['label'] = 'WGD-NN_2'
pWGF_nn_2['label'] = 'pWGD-NN'
pWGF_nn_2['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_NN_ReLU_sq_200_lbd_2_decay95'.format(d,N)

pWGF_nn_5 = {}
pWGF_nn_5['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_nn_5['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_nn_5['label'] = 'WGD-NN_5'
pWGF_nn_5['label'] = 'pWGD-NN'
pWGF_nn_5['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_NN_ReLU_sq_200_lbd_5_decay95'.format(d,N)

pWGF_nn_10 = {}
pWGF_nn_10['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_nn_10['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_nn_10['label'] = 'WGD-NN_10'
pWGF_nn_10['label'] = 'pWGD-NN'
pWGF_nn_10['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_NN_ReLU_sq_200_lbd_10_decay95'.format(d,N)

# pWGF_b5 = {}
# pWGF_b5['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_b5['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_b5['label'] = 'pWGF d={} b5'.format(d)
# pWGF_b5['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_batch5'.format(d,N)

pWGF_cvx_1 = {}
pWGF_cvx_1['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_cvx_1['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_cvx_1['label'] = 'WGD-cvxNN_1'
pWGF_cvx_1['label'] = 'pWGD-cvxNN'
pWGF_cvx_1['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_cvxNN_lbd_1_decay95'.format(d,N)

pWGF_cvx_2 = {}
pWGF_cvx_2['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_cvx_2['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_cvx_2['label'] = 'WGD-cvxNN_2'
pWGF_cvx_2['label'] = 'pWGD-cvxNN'
pWGF_cvx_2['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_cvxNN_lbd_2_decay95'.format(d,N)

pWGF_cvx_5 = {}
pWGF_cvx_5['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_cvx_5['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_cvx_5['label'] = 'WGD-cvxNN_5'
pWGF_cvx_5['label'] = 'pWGD-cvxNN'
pWGF_cvx_5['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_cvxNN_lbd_5_decay95'.format(d,N)

pWGF_cvx_10 = {}
pWGF_cvx_10['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_cvx_10['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_cvx_10['label'] = 'WGD-cvxNN_10'
pWGF_cvx_10['label'] = 'pWGD-cvxNN'
pWGF_cvx_10['filename'] = 'data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_cvxNN_lbd_10_decay95'.format(d,N)

# pWGF_post = {}
# pWGF_post['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_post['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
# pWGF_post['label'] = 'pWGF-post'
# pWGF_post['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_post'.format(d,N)


# algorithms = [SVGD, pSVGD, pSVGD_post, WGF, pWGF, pWGF_post]
# algorithms = [SVGD, pSVGD, WGF, pWGF, pWGF_b5]
algorithms = [pWGF_nn_1, pWGF_nn_2, pWGF_nn_5, pWGF_nn_10, pWGF_cvx_1, pWGF_cvx_2, pWGF_cvx_5, pWGF_cvx_10]
algorithm = [[pWGF_nn_1, pWGF_cvx_1], [pWGF_nn_2, pWGF_cvx_2], [pWGF_nn_5, pWGF_cvx_5], [pWGF_nn_10, pWGF_cvx_10]]
lbd = [1, 2, 5, 10]

for i in range(Ntrial):
    print(i)
    for algo in algorithms:
        filename = '{}_{}.p'.format(algo['filename'],i+1)
        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2norm = data_save["meanErrorL2norm"]
        varianceErrorL2norm = data_save["varianceErrorL2norm"]
        # print(algo['meanErrorL2norm'].shape)
        algo['meanErrorL2norm'][i, :Niter] = meanErrorL2norm[:Niter]
        algo['varianceErrorL2norm'][i, :Niter] = varianceErrorL2norm[:Niter]

    
for k in range(4):
    fig1=plt.figure(1)

    interval = 10
    iter_num = Niter
    iters = np.arange(iter_num)

    # print(iters[0:200:5])
    # print(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0)))[0:200:5])
    for (j, algo) in enumerate(algorithm[k]):
        for i in range(Ntrial):
            plt.plot(np.log10(algo['meanErrorL2norm'][i, :]), colors[j], alpha=0.2)
        # plt.plot(np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
        plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

    plt.legend(fontsize=fontsize)
    plt.xlabel("# iterations", fontsize=fontsize)
    plt.ylabel("Log10(RMSE of mean)", fontsize=fontsize)

    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tick_params(axis='both', which='minor', labelsize=fontsize)

    filename = "figure/error_mean_"+str(lbd[k])+".pdf"
    fig1.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_mean_"+str(lbd[k])+".eps"
    fig1.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()


    fig2 = plt.figure(2)

    for (j, algo) in enumerate(algorithm[k]):
        for i in range(Ntrial):
            plt.plot(np.log10(algo['varianceErrorL2norm'][i, :]), colors[j], alpha=0.2)
        # plt.plot(np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
        plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

    plt.legend(fontsize=fontsize)
    plt.xlabel("# iterations", fontsize=fontsize)
    plt.ylabel("Log10(RMSE of variance)", fontsize=fontsize)

    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tick_params(axis='both', which='minor', labelsize=fontsize)

    filename = "figure/error_variance_"+str(lbd[k])+".pdf"
    fig2.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_variance_"+str(lbd[k])+".eps"
    fig2.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()


fig1=plt.figure(1)

interval = 10
iter_num = Niter
iters = np.arange(iter_num)

# print(iters[0:200:5])
# print(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0)))[0:200:5])
for (j, algo) in enumerate(algorithms[4:]):
    for i in range(Ntrial):
        plt.plot(np.log10(algo['meanErrorL2norm'][i, :]), colors[j], alpha=0.2)
    # plt.plot(np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
    plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

plt.legend(fontsize=fontsize)
plt.xlabel("# iterations", fontsize=fontsize)
plt.ylabel("Log10(RMSE of mean)", fontsize=fontsize)

plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.tick_params(axis='both', which='minor', labelsize=fontsize)

filename = "figure/error_mean.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


fig2 = plt.figure(2)

for (j, algo) in enumerate(algorithms[4:]):
    for i in range(Ntrial):
        plt.plot(np.log10(algo['varianceErrorL2norm'][i, :]), colors[j], alpha=0.2)
    # plt.plot(np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
    plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

plt.legend(fontsize=fontsize)
plt.xlabel("# iterations", fontsize=fontsize)
plt.ylabel("Log10(RMSE of variance)", fontsize=fontsize)

plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.tick_params(axis='both', which='minor', labelsize=fontsize)

filename = "figure/error_variance.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()