from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
from ..modeling.variables import STATE, PARAMETER
from ..algorithms.lowRankOperator import LowRankOperator
from mpi4py import MPI
import time
from .BM_utils import partial_copy
from .WGD_utils import *
from .NN_utils import *
from .cvx_nn import *

class GradientDescent:
    # solve the optimization problem by Newton method with separated linear system
    def __init__(self, model, particle, variation, kernel, options, comm):
        self.model = model  # forward model
        self.particle = particle  # set of particles, pn = particles[m]
        self.variation = variation
        self.kernel = kernel
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.save_kernel = options["save_kernel"]
        self.add_number = options["add_number"]
        self.add_step = options["add_step"]
        self.save_step = options["save_step"]
        self.save_number = options["save_number"]
        self.search_size = options["search_size"]
        self.plot = options["plot"]
        if options.get('WGF',-1) == -1:
            options["WGF"] = False
        self.WGF = options["WGF"]
        if options.get('WGF_kernel',-1)==-1:
            options['WGF_kernel'] = False
        if options.get('WGF_NN',-1)==-1:
            options['WGF_NN'] = False
        self.WGF_kernel = options['WGF_kernel']
        self.WGF_NN = options['WGF_NN']
        self.WGF_cvxNN = options['WGF_cvxNN']
        self.lbd_decay = options['lbd_decay']
        
        if self.WGF_NN:
            self.NN_opts = options['NN_opts']

        if self.WGF_cvxNN:
            self.cvxNN_opts = options['cvxNN_opts']
            self.lbd = options['lbd']


        # if options.get('lbd',-1)==-1:
        #     options['lbd'] = 1e-2
        # if options.get('lbd2',-1)==-1:
        #     options['lbd2'] = 1e-4

        # self.lbd = options['lbd']
        # self.lbd2 = options['lbd2']

        self.it = 0
        self.converged = False
        self.reason = 0

        self.gradient_norm = np.zeros(self.particle.number_particles_all)
        self.gradient_norm_init = np.zeros(self.particle.number_particles_all)
        self.pg_phat = np.zeros(self.particle.number_particles_all)
        self.pstep_norm = np.zeros(self.particle.number_particles_all)
        self.step_norm = np.zeros(self.particle.number_particles_all)
        self.step_norm_init = np.zeros(self.particle.number_particles_all)
        self.relative_grad_norm = np.zeros(self.particle.number_particles_all)
        self.relative_step_norm = np.zeros(self.particle.number_particles_all)
        self.tol_gradient = np.zeros(self.particle.number_particles_all)
        self.final_grad_norm = np.zeros(self.particle.number_particles_all)
        self.cost_new = np.zeros(self.particle.number_particles_all)
        self.reg_new = np.zeros(self.particle.number_particles_all)
        self.misfit_new = np.zeros(self.particle.number_particles_all)
        self.alpha = 1.e-2 * np.ones(self.particle.number_particles_all)
        self.n_backtrack = np.zeros(self.particle.number_particles_all)

        self.data_save = dict()

        self.data_save["nCores"] = self.nproc
        self.data_save["particle_dimension"] = particle.particle_dimension
        self.data_save["dimension"] = []
        self.data_save["gradient_norm"] = []
        self.data_save["pg_phat"] = []
        self.data_save["step_norm"] = []
        self.data_save["relative_grad_norm"] = []
        self.data_save["relative_step_norm"] = []
        self.data_save["cost_new"] = []
        self.data_save["reg_new"] = []
        self.data_save["misfit_new"] = []
        self.data_save["iteration"] = []
        self.data_save["meanL2norm"] = []
        self.data_save["moment2L2norm"] = []
        self.data_save["meanErrorL2norm"] = []
        self.data_save["varianceL2norm"] = []
        self.data_save["varianceErrorL2norm"] = []
        self.data_save["sample_trace"] = []
        self.data_save["cost_mean"] = []
        self.data_save["cost_std"] = []
        self.data_save["cost_moment2"] = []
        self.data_save["d"] = []
        self.data_save["d_average"] = []

        if self.model.qoi is not None:
            self.data_save["qoi_std"] = []
            self.data_save["qoi_mean"] = []
            self.data_save["qoi_moment2"] = []
            self.qoi_mean = 0.
            self.qoi_std = 0.
            self.qoi_moment2 = 0.

        self.time_communication = 0.
        self.time_computation = 0.

    def gradientSeparated(self, gradient, m):

        if self.save_kernel:
            kernel_value = self.kernel.value_set[m]
            kernel_gradient = self.kernel.gradient_set[m]
        else:
            kernel_value = self.kernel.values(m)
            kernel_gradient = self.kernel.gradients(m)

        if self.kernel.delta_kernel:
            gp_misfit = self.particle.generate_vector()
            gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
            gradient.axpy(kernel_value[self.rank][m], gp_misfit)
            gradient.axpy(-1.0, kernel_gradient[self.rank][m])
        else:
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation over particle set
                    gp_misfit = self.particle.generate_vector()
                    gp_misfit.set_local(self.variation.gradient_gather[p][ie].get_local())
                    gradient.axpy(kernel_value[p][ie], gp_misfit)
                    gradient.axpy(-1.0, kernel_gradient[p][ie])

            # also use the particle to compute the expectation
            if m >= self.particle.number_particles:
                gp_misfit = self.particle.generate_vector()
                gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
                gradient.axpy(kernel_value[self.rank][m], gp_misfit)
                gradient.axpy(-1.0, kernel_gradient[self.rank][m])

            # this is not needed in Newton method, because both sides are divided by # particles
            gradient[:] /= self.nproc*self.particle.number_particles + (m >= self.particle.number_particles)

        if self.options["is_projection"]:
            if self.options["is_precondition"]:
                if self.options["type_approximation"] is 'hessian':
                    A = self.variation.hessian_misfit_average + np.eye(self.variation.hessian_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)
                elif self.options["type_approximation"] is 'fisher':
                    A = self.variation.fisher_misfit_average + np.eye(self.variation.fisher_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)

            gradient_norm = np.sqrt(gradient.inner(gradient))
        else:
            if self.options["is_precondition"]:
                gradient_tmp = self.model.generate_vector(PARAMETER)
                gradient_tmp.axpy(1.0, gradient)
                d = np.divide(self.variation.d_average, (1+self.variation.d_average))
                hessian_misfit = LowRankOperator(d, self.variation.U_average)
                gradient_misfit = self.model.generate_vector(PARAMETER)
                hessian_misfit.mult(gradient_tmp, gradient_misfit)

                self.model.prior.Rsolver.solve(gradient, gradient_tmp)
                gradient.axpy(-1.0, gradient_misfit)

            tmp = self.particle.generate_vector()
            self.model.prior.Msolver.solve(tmp, gradient)
            gradient_norm = np.sqrt(gradient.inner(tmp))

        return gradient_norm

    def gradientSeparated_WGF(self, gradient, m):

        if self.save_kernel:
            kernel_value = self.kernel.value_set[m]
            kernel_gradient = self.kernel.gradient_set[m]
        else:
            kernel_value = self.kernel.values(m)
            kernel_gradient = self.kernel.gradients(m)

        gp_misfit = self.particle.generate_vector()
        gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
        gradient.axpy(1.0, gp_misfit)

        gp_misfit = self.particle.generate_vector()
        kernel_sum = 1e-16 # safeguard for division
        for p in range(self.nproc):
            for ie in range(self.particle.number_particles):  # take the expectation over particle set
                gp_misfit.axpy(1, kernel_gradient[p][ie])
                kernel_sum = kernel_sum+kernel_value[p][ie]
        gradient.axpy(-1./kernel_sum, gp_misfit)

        if self.options["is_projection"]:
            if self.options["is_precondition"]:
                if self.options["type_approximation"] is 'hessian':
                    A = self.variation.hessian_misfit_average + np.eye(self.variation.hessian_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)
                elif self.options["type_approximation"] is 'fisher':
                    A = self.variation.fisher_misfit_average + np.eye(self.variation.fisher_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)

            gradient_norm = np.sqrt(gradient.inner(gradient))
        else:
            if self.options["is_precondition"]:
                gradient_tmp = self.model.generate_vector(PARAMETER)
                gradient_tmp.axpy(1.0, gradient)
                d = np.divide(self.variation.d_average, (1+self.variation.d_average))
                hessian_misfit = LowRankOperator(d, self.variation.U_average)
                gradient_misfit = self.model.generate_vector(PARAMETER)
                hessian_misfit.mult(gradient_tmp, gradient_misfit)

                self.model.prior.Rsolver.solve(gradient, gradient_tmp)
                gradient.axpy(-1.0, gradient_misfit)

            tmp = self.particle.generate_vector()
            self.model.prior.Msolver.solve(tmp, gradient)
            gradient_norm = np.sqrt(gradient.inner(tmp))

        return gradient_norm

    def gradientSeparated_WGF_kernel(self, gradient, m, dlog_rho_X):

        # gp_misfit = self.particle.generate_vector()
        # gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
        # gradient.axpy(1.0, gp_misfit)

        gp_misfit = self.particle.generate_vector()
        gp_misfit.set_local(dlog_rho_X[m,:])
        gradient.axpy(-1., gp_misfit)

        if self.options["is_projection"]:
            if self.options["is_precondition"]:
                if self.options["type_approximation"] is 'hessian':
                    A = self.variation.hessian_misfit_average + np.eye(self.variation.hessian_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)
                elif self.options["type_approximation"] is 'fisher':
                    A = self.variation.fisher_misfit_average + np.eye(self.variation.fisher_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient.get_local())
                    gradient.set_local(gradient_array)

            gradient_norm = np.sqrt(gradient.inner(gradient))
        else:
            if self.options["is_precondition"]:
                gradient_tmp = self.model.generate_vector(PARAMETER)
                gradient_tmp.axpy(1.0, gradient)
                d = np.divide(self.variation.d_average, (1+self.variation.d_average))
                hessian_misfit = LowRankOperator(d, self.variation.U_average)
                gradient_misfit = self.model.generate_vector(PARAMETER)
                hessian_misfit.mult(gradient_tmp, gradient_misfit)

                self.model.prior.Rsolver.solve(gradient, gradient_tmp)
                gradient.axpy(-1.0, gradient_misfit)

            tmp = self.particle.generate_vector()
            self.model.prior.Msolver.solve(tmp, gradient)
            gradient_norm = np.sqrt(gradient.inner(tmp))

        return gradient_norm

    def get_grad(self):
        grad = np.zeros([self.particle.number_particles, self.particle.dimension])
        for m in range(self.particle.number_particles):
            grad[m,:] = self.variation.gradient_gather[self.rank][m].get_local().copy()
        return grad


    def communication(self, phat):

        time_communication = time.time()

        phat_array = np.empty([self.particle.number_particles_all, self.particle.dimension], dtype=float)
        phat_gather_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
        for n in range(self.particle.number_particles_all):
            phat_array[n, :] = phat[n].get_local()
        self.comm.Allgather(phat_array, phat_gather_array)

        phat_gather = [[self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p
                       in range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.particle.number_particles_all):
                phat_gather[p][n].set_local(phat_gather_array[p, n, :])

        self.time_communication += time.time() - time_communication

        return phat_gather

    def solve(self):
        # use gradient decent method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        line_search = self.options["line_search"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)

        self.it = 0
        self.converged = False

        if self.save_number:
            self.kernel.save_values(self.save_number, self.it)
            self.particle.save(self.save_number, self.it)
            self.variation.save_eigenvalue(self.save_number, self.it)
            if self.plot:
                self.variation.plot_eigenvalue(self.save_number, self.it)

        self.cost_mean, self.cost_std = 0., 0.

        if self.WGF_NN:
            NN_opts = self.NN_opts

        while self.it < max_iter and (self.converged is False):
            # sync
            self.kernel.alpha = self.alpha
            self.kernel.it = self.it+1

            if self.options["type_parameter"] is 'vector' and self.plot:
                self.particle.plot_particles(self.particle, self.it)

            if self.particle.mean_posterior is None:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace = self.particle.statistics()
                self.meanErrorL2norm, self.varianceErrorL2norm = 0., 0.
            else:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace, \
                self.meanErrorL2norm, self.varianceErrorL2norm = self.particle.statistics()

            if self.rank == 0:
                print("# samples = ", self.nproc * self.particle.number_particles_all,
                      " mean = ", self.meanL2norm, "mean error = ", self.meanErrorL2norm,
                      " variance = ", self.varianceL2norm, " variance error = ", self.varianceErrorL2norm,
                      " trace = ", self.sample_trace, "dimension = ", self.particle.dimension)

            if self.model.qoi is not None:
                self.qoi_mean, self.qoi_std, self.qoi_moment2 = self.variation.qoi_statistics()

                if self.rank == 0:
                    print("# samples = ", "qoi_mean = ", self.qoi_mean, "qoi_std = ", self.qoi_std,
                          "cost_mean = ", self.cost_mean, "cost_std = ", self.cost_std)

            time_computation = time.time()

            # calculate pstep
            phat = [self.particle.generate_vector() for m in range(self.particle.number_particles)]
            self.pg_phat = np.ones(self.particle.number_particles)


            if self.WGF and self.WGF_kernel:
                grad = self.get_grad()
                # if self.options["is_projection"]:
                #     X = np.zeros([self.particle.number_particles, self.particle.dimension])
                #     for m in range(self.particle.number_particles):
                #         X[m,:] = self.particle.coefficients[m].get_local().copy()
                # else:
                #     X = self.particle.particles_array.copy()
                # dlog_rho_X = dlog_rho_kernel(X, grad, ibw=1./self.kernel.scale[0], lbd=self.lbd, lbd2=self.lbd2)
                K0 = self.kernel.compute_kernel_mat(0)
                K1 = self.kernel.compute_kernel_mat(100)
                K2 = self.kernel.compute_kernel_mat(400)
                dlog_rho_X = dlog_rho_kernel_ver2(K0, K1, K2, grad, lbd=1e-3, lbd2=1e-3)
            elif self.WGF and self.WGF_NN:
                grad = self.get_grad()
                if self.options["is_projection"]:
                    X = np.zeros([self.particle.number_particles, self.particle.dimension])
                    for m in range(self.particle.number_particles):
                        X[m,:] = self.particle.coefficients[m].get_local().copy()
                else:
                    X = self.particle.particles_array.copy()

                nn_dim = self.particle.dimension
                X_t = torch.Tensor(X)
                grad_t = torch.Tensor(grad)
                if self.it == 0 or nn_dim!=nn_dim_old:
                    grad_NN, state_dict, info, results = dlog_rho_NN(X_t, grad_t, **NN_opts)
                else:
                    grad_NN, state_dict, info, results = dlog_rho_NN(X_t, grad_t, use_cache=True, state_dict = state_dict,
                        **NN_opts)

                if self.lbd_decay<1.:
                    if NN_opts['cubic_reg']:
                        NN_opts['lbd'] = NN_opts['lbd']*self.lbd_decay
                    else:
                        NN_opts['weight_decay'] = NN_opts['weight_decay']*self.lbd_decay

                dlog_rho_X = grad_NN.detach().numpy()
                # print(np.linalg.norm(dlog_rho_X))
                nn_dim_old = self.particle.dimension
            elif self.WGF and self.WGF_cvxNN:
                grad = self.get_grad()
                if self.options["is_projection"]:
                    X = np.zeros([self.particle.number_particles, self.particle.dimension])
                    for m in range(self.particle.number_particles):
                        X[m,:] = self.particle.coefficients[m].get_local().copy()
                else:
                    X = self.particle.particles_array.copy()

                self.lbd = self.lbd*self.lbd_decay 

                nn_dim = self.particle.dimension
                if self.it == 0 or nn_dim!=nn_dim_old:
                    self.cvxNN_opts['target_num'] = 0
                    grad_NN_cvx, p_star, flag, num_mask = cvx_nn_relu_dual(X,grad,self.lbd,**self.cvxNN_opts)
                    self.cvxNN_opts['target_num'] = num_mask
                else:
                    grad_NN_cvx, p_star, flag, num_mask = cvx_nn_relu_dual(X,grad,self.lbd,**self.cvxNN_opts)

                if flag==0:
                    print('Error: choose a larger lambda')
                    self.lbd = self.lbd/(self.lbd_decay)**11
                    dlog_rho_X = 0
                else:
                    dlog_rho_X = -grad_NN_cvx-grad

                nn_dim_old = self.particle.dimension

            for m in range(self.particle.number_particles):  # solve for each particle
                # evaluate gradient
                gradient = self.particle.generate_vector()
                if self.WGF:
                    if self.WGF_kernel or self.WGF_NN:
                        self.gradient_norm[m] = self.gradientSeparated_WGF_kernel(gradient, m, dlog_rho_X)
                    else:
                        self.gradient_norm[m] = self.gradientSeparated_WGF(gradient, m)
                else:
                    self.gradient_norm[m] = self.gradientSeparated(gradient, m)

                phat[m].axpy(-1.0, gradient)
                self.pg_phat[m] = gradient.inner(phat[m])
                # set tolerance for gradient iteration
                if self.it == 0:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                if self.particle.number_particles > self.particle.number_particles_old:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

            phat = self.communication(phat)  # gather the coefficients to all processors

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                pstep[m].axpy(1.0, phat[self.rank][m])

                # for p in range(self.nproc):
                #     for n in range(self.particle.number_particles):
                #         pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                #     if m >= self.particle.number_particles:
                #         pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                if self.options["is_projection"]:
                    self.pstep_norm[m] = np.sqrt(pstep[m].inner(pstep[m]))
                    pstep_m = pstep[m].get_local()
                    for r in range(self.particle.coefficient_dimension):
                        deltap[m].axpy(pstep_m[r], self.particle.bases[r])
                else:
                    phelp = self.model.generate_vector(PARAMETER)
                    self.model.prior.M.mult(pstep[m], phelp)
                    self.pstep_norm[m] = np.sqrt(pstep[m].inner(phelp))
                    deltap[m].axpy(1.0, pstep[m])

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.step_norm_init[m] = self.pstep_norm[m]

            if self.it == 0:
                self.step_norm_init = self.pstep_norm

            self.alpha = self.search_size * np.ones(self.particle.number_particles_all)
            # self.alpha = np.power(self.it+1., -0.5) * self.search_size * np.ones(self.particle.number_particles_all) #gradient_norm_max_one #

            self.n_backtrack = np.zeros(self.particle.number_particles_all)
            self.cost_new = np.zeros(self.particle.number_particles_all)
            self.reg_new = np.zeros(self.particle.number_particles_all)
            self.misfit_new = np.zeros(self.particle.number_particles_all)

            for m in range(self.particle.number_particles_all):
                # compute the old cost
                x = self.variation.x_all[m]
                cost_old, reg_old, misfit_old = self.model.cost(x)
                self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                if line_search:
                    # do line search
                    descent = 0
                    x_star = self.model.generate_vector()
                    while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                        # update the parameter
                        x_star[PARAMETER].zero()
                        x_star[PARAMETER].axpy(1., x[PARAMETER])
                        x_star[PARAMETER].axpy(self.alpha[m], deltap[m])
                        # update the state at new parameter
                        x_star[STATE].zero()
                        x_star[STATE].axpy(1., x[STATE])
                        self.model.solveFwd(x_star[STATE], x_star)

                        # evaluate the cost functional, here the potential
                        self.cost_new[m], self.reg_new[m], self.misfit_new[m] = self.model.cost(x_star)

                        # Check if armijo conditions are satisfied
                        if m < self.particle.number_particles:
                            if (self.cost_new[m] < cost_old + self.alpha[m] * c_armijo * self.pg_phat[m]) or \
                                    (-self.pg_phat[m] <= self.options["gdm_tolerance"]):
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5
                                # print("alpha = ", alpha[m])
                        else:  # we do not have pg_phat for m >= particle.number_particles
                            if self.cost_new[m] < cost_old:
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5

            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(self.cost_new**2)

            # compute the norm of the step/direction to move
            self.step_norm = self.pstep_norm * self.alpha

            # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
            self.particle.move(self.alpha, pstep)

            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            self.time_computation += time.time() - time_computation

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:5} {1:5} {2:8} {3:15} {4:15} {5:15} {6:15} {7:15} {8:14}".format(
                        "it", "cpu", "id", "cost", "misfit", "reg", "||g||L2", "||m||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e} {8:14e}".format(
                    self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m]))
                for m in range(self.particle.number_particles, self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m], 0., self.alpha[m]))

            # save data
            gradient_norm = np.empty([self.nproc, len(self.gradient_norm)], dtype=float)
            self.comm.Allgather(self.gradient_norm, gradient_norm)
            pg_phat = np.empty([self.nproc, len(self.pg_phat)], dtype=float)
            self.comm.Allgather(self.pg_phat, pg_phat)
            step_norm = np.empty([self.nproc, len(self.step_norm)], dtype=float)
            self.comm.Allgather(self.step_norm, step_norm)
            relative_grad_norm = np.empty([self.nproc, len(self.relative_grad_norm)], dtype=float)
            self.comm.Allgather(self.relative_grad_norm, relative_grad_norm)
            relative_step_norm = np.empty([self.nproc, len(self.relative_step_norm)], dtype=float)
            self.comm.Allgather(self.relative_step_norm, relative_step_norm)
            cost_new = np.empty([self.nproc, len(self.cost_new)], dtype=float)
            self.comm.Allgather(self.cost_new, cost_new)

            if self.rank == 0:
                self.data_save["gradient_norm"].append(np.mean(gradient_norm))
                # self.data_save["pg_phat"].append(pg_phat)
                self.data_save["step_norm"].append(np.mean(step_norm))
                # self.data_save["relative_grad_norm"].append(relative_grad_norm)
                # self.data_save["relative_step_norm"].append(relative_step_norm)
                # self.data_save["cost_new"].append(cost_new)
                self.data_save["dimension"].append(self.particle.dimension)
                self.data_save["cost_new"].append(cost_new)
                self.data_save["iteration"].append(self.it)
                self.data_save["meanL2norm"].append(self.meanL2norm)
                self.data_save["moment2L2norm"].append(self.moment2L2norm)
                self.data_save["meanErrorL2norm"].append(self.meanErrorL2norm)
                self.data_save["varianceL2norm"].append(self.varianceL2norm)
                self.data_save["varianceErrorL2norm"].append(self.varianceErrorL2norm)
                self.data_save["sample_trace"].append(self.sample_trace)
                self.data_save["cost_mean"].append(self.cost_mean)
                self.data_save["cost_std"].append(self.cost_std)
                self.data_save["cost_moment2"].append(self.cost_moment2)
                self.data_save["d"].append(self.variation.d)
                self.data_save["d_average"].append(self.variation.d_average_save)
                if self.model.qoi is not None:
                    self.data_save["qoi_mean"].append(self.qoi_mean)
                    self.data_save["qoi_std"].append(self.qoi_std)
                    self.data_save["qoi_moment2"].append(self.qoi_moment2)

                N = self.nproc*self.particle.number_particles_all
                if self.WGF:
                    if self.WGF_kernel:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_kernel.p"
                    elif self.WGF_NN:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_NN.p"
                    elif self.WGF_cvxNN:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_cvxNN.p"
                    else:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                                   "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                                   str(self.options["is_projection"])+"_WGF.p"
                else:
                    filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_SVGD.p"

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_gradient[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles_all):  # should use _all
                if self.n_backtrack[m] < max_backtracking_iter:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = False
                self.reason = 2
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if self.step_norm[m] > self.step_norm_init[m]*self.options["step_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 4
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m-self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init, m-self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m-self.particle.number_particles_add, 0.)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m-self.particle.number_particles_add, 0.)

            # update variation, kernel, and hessian before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
            self.kernel.update(self.particle, self.variation)

            # save the particles for visualization and plot the eigenvalues at each particle
            if self.save_number and np.mod(self.it, self.save_step) == 0:
                self.kernel.save_values(self.save_number, self.it)
                self.particle.save(self.save_number, self.it)
                self.variation.save_eigenvalue(self.save_number, self.it)
                if self.plot:
                    self.variation.plot_eigenvalue(self.save_number, self.it)

    def solve_batchwise(self, batchsize = 5):
        # use gradient decent method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        line_search = self.options["line_search"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)

        self.it = 0
        self.converged = False

        if self.save_number:
            self.kernel.save_values(self.save_number, self.it)
            self.particle.save(self.save_number, self.it)
            self.variation.save_eigenvalue(self.save_number, self.it)
            if self.plot:
                self.variation.plot_eigenvalue(self.save_number, self.it)

        self.cost_mean, self.cost_std = 0., 0.

        self.kernel.batch_proj = True

        num_group = 1

        while self.it < max_iter and (self.converged is False):
            # sync
            self.kernel.alpha = self.alpha
            self.kernel.it = self.it+1

            if self.options["type_parameter"] is 'vector' and self.plot:
                self.particle.plot_particles(self.particle, self.it)

            if self.particle.mean_posterior is None:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace = self.particle.statistics()
                self.meanErrorL2norm, self.varianceErrorL2norm = 0., 0.
            else:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace, \
                self.meanErrorL2norm, self.varianceErrorL2norm = self.particle.statistics()

            if self.rank == 0:
                print("# samples = ", self.nproc * self.particle.number_particles_all,
                      " mean = ", self.meanL2norm, "mean error = ", self.meanErrorL2norm,
                      " variance = ", self.varianceL2norm, " variance error = ", self.varianceErrorL2norm,
                      " trace = ", self.sample_trace, "dimension = ", self.particle.dimension,
                      " num_group = ", num_group)

            if self.model.qoi is not None:
                self.qoi_mean, self.qoi_std, self.qoi_moment2 = self.variation.qoi_statistics()

                if self.rank == 0:
                    print("# samples = ", "qoi_mean = ", self.qoi_mean, "qoi_std = ", self.qoi_std,
                          "cost_mean = ", self.cost_mean, "cost_std = ", self.cost_std)

            time_computation = time.time()
            # update variation, kernel, and hessian before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold

            
            # print('shuffle index')
            idx_all = np.arange(self.particle.dimension)
            num_group = self.particle.dimension//batchsize
            if self.particle.dimension-num_group*batchsize>0:
                num_group = num_group+1
            # np.random.shuffle(idx_all)

            for j in range(num_group):
                self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
                relative_step_norm = np.max(self.relative_step_norm)
                relative_step_norm_reduce = np.zeros(1, dtype=float)
                self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
                self.relative_step_norm = relative_step_norm_reduce[0]
                dimension_old = self.particle.dimension
                # self.variation.update(self.particle, self.it, self.relative_step_norm)
                dimension_new = self.particle.dimension
                self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
                if dimension_new != dimension_old:
                    print('detect dimension change')
                    idx_all = np.arange(self.particle.dimension)
                    num_group = self.particle.dimension//batchsize
                    if self.particle.dimension-num_group*batchsize>0:
                        num_group = num_group+1
                    # np.random.shuffle(idx_all)
                    continue

                
                if j<num_group-1:
                    group_idx = idx_all[j*num_group:(j+1)*num_group]
                else:
                    group_idx = idx_all[j*num_group:]
                self.kernel.group_idx = group_idx
                # project particle based on group_idx
                particle_proj_diff = []
                for m in range(self.particle.number_particles):
                    particle_proj_diff_member = self.particle.generate_vector()
                    if self.options["is_projection"]:
                        pm_array = self.particle.coefficients[m].get_local().copy()
                        pm_array = pm_array-partial_copy(pm_array, group_idx)
                    else:
                        pm_array = self.particle.particles_array[m].copy()
                        pm_array = pm_array-partial_copy(pm_array, group_idx)
                    # print(pm_array.shape)
                    # print(particle_proj_diff_member.get_local().shape)
                    particle_proj_diff_member.set_local(pm_array)
                    particle_proj_diff.append(particle_proj_diff_member)

                # self.kernel.update(self.particle, self.variation)
                self.particle.move(-np.ones(self.particle.number_particles),particle_proj_diff)
                self.kernel.update(self.particle, self.variation)
                self.particle.move(np.ones(self.particle.number_particles),particle_proj_diff)

                phat = [self.particle.generate_vector() for m in range(self.particle.number_particles)]
                self.pg_phat = np.ones(self.particle.number_particles)

                for m in range(self.particle.number_particles):  # solve for each particle
                    # evaluate gradient
                    gradient = self.particle.generate_vector()
                    if self.WGF:
                        self.gradient_norm[m] = self.gradientSeparated_WGF(gradient, m)
                    else:
                        self.gradient_norm[m] = self.gradientSeparated(gradient, m)

                    gradient.set_local(partial_copy(gradient.get_local(),group_idx))
                    phat[m].axpy(-1.0, gradient)
                    self.pg_phat[m] = gradient.inner(phat[m])
                    # set tolerance for gradient iteration
                    if self.it == 0:
                        self.gradient_norm_init[m] = self.gradient_norm[m]
                        self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                    if self.particle.number_particles > self.particle.number_particles_old:
                        self.gradient_norm_init[m] = self.gradient_norm[m]
                        self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                phat = self.communication(phat)  # gather the coefficients to all processors

                # step for particle update, pstep(x) = sum_n phat_n k_n(x)
                self.pstep_norm = np.zeros(self.particle.number_particles_all)
                pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
                deltap = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles_all)]
                for m in range(self.particle.number_particles_all):
                    pstep[m].axpy(1.0, phat[self.rank][m])

                    # projection
                    pstep[m].set_local(partial_copy(pstep[m].get_local(),group_idx))

                    # for p in range(self.nproc):
                    #     for n in range(self.particle.number_particles):
                    #         pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                    #     if m >= self.particle.number_particles:
                    #         pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                    if self.options["is_projection"]:
                        self.pstep_norm[m] = np.sqrt(pstep[m].inner(pstep[m]))
                        pstep_m = pstep[m].get_local()
                        for r in range(self.particle.coefficient_dimension):
                            deltap[m].axpy(pstep_m[r], self.particle.bases[r])
                    else:
                        phelp = self.model.generate_vector(PARAMETER)
                        self.model.prior.M.mult(pstep[m], phelp)
                        self.pstep_norm[m] = np.sqrt(pstep[m].inner(phelp))
                        deltap[m].axpy(1.0, pstep[m])

                    if self.particle.number_particles_all > self.particle.number_particles_all_old:
                        if m >= self.particle.number_particles_old \
                                and m < self.particle.number_particles_all - self.particle.number_particles_add:
                            self.step_norm_init[m] = self.pstep_norm[m]



                if self.it == 0:
                    self.step_norm_init = self.pstep_norm

                self.alpha = self.search_size * np.ones(self.particle.number_particles_all)
                # self.alpha = np.power(self.it+1., -0.5) * self.search_size * np.ones(self.particle.number_particles_all) #gradient_norm_max_one #

                self.n_backtrack = np.zeros(self.particle.number_particles_all)
                self.cost_new = np.zeros(self.particle.number_particles_all)
                self.reg_new = np.zeros(self.particle.number_particles_all)
                self.misfit_new = np.zeros(self.particle.number_particles_all)


                for m in range(self.particle.number_particles_all):
                    # compute the old cost
                    x = self.variation.x_all[m]
                    cost_old, reg_old, misfit_old = self.model.cost(x)
                    self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                    if line_search:
                        # do line search
                        descent = 0
                        x_star = self.model.generate_vector()
                        while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                            # update the parameter
                            x_star[PARAMETER].zero()
                            x_star[PARAMETER].axpy(1., x[PARAMETER])
                            x_star[PARAMETER].axpy(self.alpha[m], deltap[m])
                            # update the state at new parameter
                            x_star[STATE].zero()
                            x_star[STATE].axpy(1., x[STATE])
                            self.model.solveFwd(x_star[STATE], x_star)

                            # evaluate the cost functional, here the potential
                            self.cost_new[m], self.reg_new[m], self.misfit_new[m] = self.model.cost(x_star)

                            # Check if armijo conditions are satisfied
                            if m < self.particle.number_particles:
                                if (self.cost_new[m] < cost_old + self.alpha[m] * c_armijo * self.pg_phat[m]) or \
                                        (-self.pg_phat[m] <= self.options["gdm_tolerance"]):
                                    cost_old = self.cost_new[m]
                                    descent = 1
                                else:
                                    self.n_backtrack[m] += 1
                                    self.alpha[m] *= 0.5
                                    # print("alpha = ", alpha[m])
                            else:  # we do not have pg_phat for m >= particle.number_particles
                                if self.cost_new[m] < cost_old:
                                    cost_old = self.cost_new[m]
                                    descent = 1
                                else:
                                    self.n_backtrack[m] += 1
                                    self.alpha[m] *= 0.5

                # compute the norm of the step/direction to move
                self.step_norm = self.pstep_norm * self.alpha

                # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
                self.particle.move(self.alpha, pstep)


            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(self.cost_new**2)


            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            self.time_computation += time.time() - time_computation

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:5} {1:5} {2:8} {3:15} {4:15} {5:15} {6:15} {7:15} {8:14}".format(
                        "it", "cpu", "id", "cost", "misfit", "reg", "||g||L2", "||m||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e} {8:14e}".format(
                    self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m]))
                for m in range(self.particle.number_particles, self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m], 0., self.alpha[m]))

            # save data
            gradient_norm = np.empty([self.nproc, len(self.gradient_norm)], dtype=float)
            self.comm.Allgather(self.gradient_norm, gradient_norm)
            pg_phat = np.empty([self.nproc, len(self.pg_phat)], dtype=float)
            self.comm.Allgather(self.pg_phat, pg_phat)
            step_norm = np.empty([self.nproc, len(self.step_norm)], dtype=float)
            self.comm.Allgather(self.step_norm, step_norm)
            relative_grad_norm = np.empty([self.nproc, len(self.relative_grad_norm)], dtype=float)
            self.comm.Allgather(self.relative_grad_norm, relative_grad_norm)
            relative_step_norm = np.empty([self.nproc, len(self.relative_step_norm)], dtype=float)
            self.comm.Allgather(self.relative_step_norm, relative_step_norm)
            cost_new = np.empty([self.nproc, len(self.cost_new)], dtype=float)
            self.comm.Allgather(self.cost_new, cost_new)

            if self.rank == 0:
                self.data_save["gradient_norm"].append(np.mean(gradient_norm))
                # self.data_save["pg_phat"].append(pg_phat)
                self.data_save["step_norm"].append(np.mean(step_norm))
                # self.data_save["relative_grad_norm"].append(relative_grad_norm)
                # self.data_save["relative_step_norm"].append(relative_step_norm)
                # self.data_save["cost_new"].append(cost_new)
                self.data_save["dimension"].append(self.particle.dimension)
                self.data_save["cost_new"].append(cost_new)
                self.data_save["iteration"].append(self.it)
                self.data_save["meanL2norm"].append(self.meanL2norm)
                self.data_save["moment2L2norm"].append(self.moment2L2norm)
                self.data_save["meanErrorL2norm"].append(self.meanErrorL2norm)
                self.data_save["varianceL2norm"].append(self.varianceL2norm)
                self.data_save["varianceErrorL2norm"].append(self.varianceErrorL2norm)
                self.data_save["sample_trace"].append(self.sample_trace)
                self.data_save["cost_mean"].append(self.cost_mean)
                self.data_save["cost_std"].append(self.cost_std)
                self.data_save["cost_moment2"].append(self.cost_moment2)
                self.data_save["d"].append(self.variation.d)
                self.data_save["d_average"].append(self.variation.d_average_save)
                if self.model.qoi is not None:
                    self.data_save["qoi_mean"].append(self.qoi_mean)
                    self.data_save["qoi_std"].append(self.qoi_std)
                    self.data_save["qoi_moment2"].append(self.qoi_moment2)

                N = self.nproc*self.particle.number_particles_all
                if self.WGF:
                    if self.WGF_kernel:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_kernel.p"
                    elif self.WGF_NN:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_NN.p"
                    elif self.WGF_cvxNN:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_WGF_cvxNN.p"
                    else:
                        filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                                   "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                                   str(self.options["is_projection"])+"_WGF.p"
                else:
                    filename = "data/data"+"_nDimensions_"+str(self.particle.particle_dimension) + \
                               "_nCores_"+str(self.nproc)+"_nSamples_"+str(N)+"_isProjection_"+\
                               str(self.options["is_projection"])+"_SVGD.p"

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_gradient[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles_all):  # should use _all
                if self.n_backtrack[m] < max_backtracking_iter:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = False
                self.reason = 2
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if self.step_norm[m] > self.step_norm_init[m]*self.options["step_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 4
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m-self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init, m-self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m-self.particle.number_particles_add, 0.)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m-self.particle.number_particles_add, 0.)


            # save the particles for visualization and plot the eigenvalues at each particle
            if self.save_number and np.mod(self.it, self.save_step) == 0:
                self.kernel.save_values(self.save_number, self.it)
                self.particle.save(self.save_number, self.it)
                self.variation.save_eigenvalue(self.save_number, self.it)
                if self.plot:
                    self.variation.plot_eigenvalue(self.save_number, self.it)

    def solve_adam(self):
        # D.P. Kingma, J. Ba. Adam: A method for stochastic optimization. https://arxiv.org/pdf/1412.6980.pdf
        # to be tuned, it does not seem to work well
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]

        max_backtracking_iter = self.options["max_backtracking_iter"]

        gradient_norm = np.zeros(self.particle.number_particles)
        gradient_norm_init = np.zeros(self.particle.number_particles)
        tol_gradient = np.zeros(self.particle.number_particles)
        cost_new = np.zeros(self.particle.number_particles)
        reg_new = np.zeros(self.particle.number_particles)
        misfit_new = np.zeros(self.particle.number_particles)

        self.it = 0
        alpha = 1.e-3 * np.ones(self.particle.number_particles)
        beta_1 = 0.9
        beta_2 = 0.999
        mt_1 = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles)]
        mt_2 = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles)]

        vt_1 = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles)]
        vt_2 = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles)]

        epsilon = 1.e-8

        while self.it < max_iter and (self.converged is False):
            # always update variation and kernel before compute the gradient
            self.variation.update(self.particle)
            self.kernel.update(self.particle, self.variation)

            gt = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles)]
            n_backtrack = np.zeros(self.particle.number_particles)

            for m in range(self.particle.number_particles):  # solve for each particle
                # evaluate gradient
                gradient = self.model.generate_vector(PARAMETER)
                if self.WGF:
                    if self.WGF_kernel:
                        self.gradient_norm[m] = self.gradientSeparated_WGF_kernel(gradient, m)
                    else:
                        self.gradient_norm[m] = self.gradientSeparated_WGF(gradient, m)
                else:
                    gradient_norm[m] = self.gradientSeparated(gradient, m)

                gt[m].axpy(1.0, gradient)

                if self.it == 0:
                    gradient_norm_init[m] = gradient_norm[m]
                    tol_gradient[m] = max(abs_tol, gradient_norm_init[m] * rel_tol)

                mt_2[m].axpy(beta_1, mt_1[m])
                mt_2[m].axpy((1-beta_1), gt[m])
                mt_1[m].zero()
                mt_1[m].axpy(1./(1-np.power(beta_1, self.it+1)), mt_2[m])

                vt_2[m].axpy(beta_2, vt_1[m])
                gt_array = gt[m].get_local()
                gt[m].set_local(gt_array**2)
                vt_2[m].axpy((1-beta_2), gt[m])
                vt_1[m].axpy(1./(1-np.power(beta_2, self.it+1)), vt_2[m])

                mt_array = mt_1[m].get_local()
                vt_array = vt_1[m].get_local()
                gt[m].set_local(mt_array/(np.sqrt(vt_array) + epsilon))

                # update particles
                self.particle.particles[m].axpy(-alpha[m], gt[m])

                # update moments
                mt_1[m].zero()
                mt_1[m].axpy(1.0, mt_2[m])
                vt_1[m].zero()
                vt_1[m].axpy(1.0, vt_2[m])

                # compute cost
                x_star = self.model.generate_vector()
                x_star[PARAMETER].zero()
                x_star[PARAMETER].axpy(1., self.particle.particles[m])
                x_star[STATE].zero()
                self.model.solveFwd(x_star[STATE], x_star)

                cost_new[m], reg_new[m], misfit_new[m] = self.model.cost(x_star)

            if (self.rank == 0) and (print_level >= -1):
                print("\n{0:5} {1:8} {2:15} {3:15} {4:15} {5:14} {6:14}".format(
                    "It", "id", "cost", "misfit", "reg", "||g||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e}".format(
                        self.it, m, cost_new[m], misfit_new[m], reg_new[m], gradient_norm[m], alpha[m]))

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = gradient_norm[m]
                if gradient_norm[m] > tol_gradient[m]:
                    done = False
            if done:
                self.converged = True
                self.reason = 1
                break

            done = True
            for m in range(self.particle.number_particles):
                if n_backtrack[m] < max_backtracking_iter:
                    done = False
            if done:
                self.converged = False
                self.reason = 2
                break

            self.it += 1
