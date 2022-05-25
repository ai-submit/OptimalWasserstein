Optimal Neural Network Approximation of Wasserstein Gradient Direction via Convex Optimization

The PDE model example is implemented in PDE_Models_torch, using FEniCS as backend solver

The other models are implemented in nonPDE_Models_torch, using HIPS/autograd to compute gradients



## Conda

```
conda create -n fenicsproject -c conda-forge fenics mpi4py scipy sympy matplotlib pytorch torchvision torchaudio -c pytorch
```



# WGF_cvxNN

To prepare the environment, run

```
conda install -c conda-forge cvxpy 
```

to install `cvxpy` under the environment `fenicsproject`. To use Mosek as the cvxpy inner solver instead of SCS, install Mosek from https://www.mosek.com.



The scripts `demo_pWGF_NN` and `demo_pWGF_cvxNN` are written for debugging and the scripts `run_pWGF_NN` and `run_pWGF_cvxNN` are written for running the program. 



To adjust the precision of the SCS solver to 1e-3, add `--SCS_eps 1e-3` in the script. 

