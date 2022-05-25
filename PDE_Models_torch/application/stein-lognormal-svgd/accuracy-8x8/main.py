from model_lognormal import *

import os
import time
import pickle
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='linear')
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--WGF", action='store_true')
    parser.add_argument("--WGF_kernel", action='store_true')
    parser.add_argument("--WGF_NN", action='store_true')
    parser.add_argument("--WGF_cvxNN", action='store_true')
    parser.add_argument("--projection", action='store_true')
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--type_scaling", type=int, default=1)
    parser.add_argument("--lbd", type=float, default=1)
    parser.add_argument("--lbd_decay", type=float, default=1)
    parser.add_argument("--type_metric", type=str, default="prior")
    parser.add_argument("--use_batch", action='store_true')
    parser.add_argument("--line_search", action='store_true')
    parser.add_argument("--activation", type=str, default="ReLU_sq")
    parser.add_argument("--cvx_solver", type=str, default="SCS")
    parser.add_argument("--SCS_eps", type=float, default=1e-4)
    parser.add_argument("--mosek_eps", type=float, default=1e-4)
    parser.add_argument("--mosek_eps_small", type=float, default=1e-8)
    parser.add_argument("--cvx_verbose", action='store_true')
    parser.add_argument("--num_neurons", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--sub_iter", type=int, default=200)
    parser.add_argument("--sub_thres", type=float, default=1e-2)
    parser.add_argument("--sub_lr", type=float, default=1e-3)
    parser.add_argument("--add_poly", action='store_true')
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--cubic_reg", action='store_true')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    # check the stein/options to see all possible choices
    options["type_optimization"] = "gradientDescent"
    options["is_projection"] = args.projection
    options["tol_projection"] = 1.e-1
    options["type_projection"] = "fisher"
    options["is_precondition"] = False
    options["type_approximation"] = "fisher"
    options["coefficient_dimension"] = 10
    options["add_dimension"] = 5

    options["number_particles"] = 64
    options["number_particles_add"] = 0
    options["add_number"] = 0
    options["add_step"] = 5
    options["add_rule"] = 1

    print(args.type_metric)

    options["type_scaling"] = args.type_scaling
    options["type_metric"] = args.type_metric  # posterior_average
    options["kernel_vectorized"] = False

    options['WGF'] = args.WGF
    options['WGF_NN'] = args.WGF_NN
    options['WGF_cvxNN'] = args.WGF_cvxNN
    options["WGF_kernel"] = args.WGF_kernel
    # options['lbd'] = args.lbd
    # options['lbd2'] = args.lbd2

    options["type_Hessian"] = "lumped"
    options["low_rank_Hessian"] = False
    options["rank_Hessian"] = 20
    options["rank_Hessian_tol"] = 1.e-4
    options["low_rank_Hessian_average"] = False
    options["rank_Hessian_average"] = 20
    options["rank_Hessian_average_tol"] = 1.e-4
    options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

    options["max_iter"] = args.max_iter
    options["step_tolerance"] = 1e-7
    options["step_projection_tolerance"] = 1e-3
    options["line_search"] = args.line_search
    options["search_size"] = args.lr
    options["max_backtracking_iter"] = 10
    options["cg_coarse_tolerance"] = 0.5e-2
    options["print_level"] = -1
    options["plot"] = False
    # options['max_backtracking_iter'] = 10

    NN_opts = {}
    NN_opts['activation'] = args.activation
    NN_opts['num_neurons'] = args.num_neurons
    NN_opts['iter_num'] = args.sub_iter
    NN_opts['thres'] = args.sub_thres
    NN_opts['add_poly'] = args.add_poly
    NN_opts['poly_degree'] = args.poly_degree
    NN_opts['verbose'] = args.verbose
    NN_opts['lr'] = args.sub_lr
    NN_opts['lbd'] = args.lbd
    NN_opts['cubic_reg'] = args.cubic_reg

    options['NN_opts'] = NN_opts

    cvxNN_opts = {}
    cvxNN_opts['sample_num'] = args.sample_num
    cvxNN_opts['cvx_solver'] = args.cvx_solver
    cvxsolver_opts = {}
    if args.cvx_solver=='SCS':
        cvxsolver_opts['eps'] = args.SCS_eps
    elif args.cvx_solver=='MOSEK':
        mosek_params = {}
        mosek_params['MSK_DPAR_BASIS_REL_TOL_S'] = args.mosek_eps_small
        mosek_params['MSK_DPAR_CHECK_CONVEXITY_REL_TOL'] = args.mosek_eps_small
        mosek_params['MSK_DPAR_DATA_SYM_MAT_TOL'] = args.mosek_eps_small
        mosek_params['MSK_DPAR_INTPNT_CO_TOL_INFEAS'] = args.mosek_eps_small
        mosek_params['MSK_DPAR_INTPNT_QO_TOL_INFEAS'] = args.mosek_eps_small
        mosek_params['MSK_DPAR_BASIS_TOL_S'] = args.mosek_eps
        mosek_params['MSK_DPAR_BASIS_TOL_X'] = args.mosek_eps
        mosek_params['MSK_DPAR_DATA_TOL_X'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_CO_TOL_DFEAS'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_CO_TOL_MU_RED'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_CO_TOL_PFEAS'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_CO_TOL_REL_GAP'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_QO_TOL_DFEAS'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_QO_TOL_MU_RED'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_QO_TOL_PFEAS'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_QO_TOL_REL_GAP'] = args.mosek_eps
        mosek_params['MSK_DPAR_INTPNT_TOL_DFEAS'] = args.mosek_eps


        cvxsolver_opts['mosek_params'] = mosek_params
    cvxNN_opts['cvx_opts'] = cvxsolver_opts
    cvxNN_opts['cvx_verbose'] = args.cvx_verbose

    options['cvxNN_opts'] = cvxNN_opts

    options['lbd'] = args.lbd
    options['lbd_decay'] = args.lbd_decay

    # generate particles
    particle = Particle(model, options, comm)

    filename = 'data/mcmc_dili_sample.p'
    if os.path.isfile(filename):
        print("set reference for mean and variance")
        data_save = pickle.load(open(filename, 'rb'))
        mean = model.generate_vector(PARAMETER)
        mean.set_local(data_save["mean"])
        variance = model.generate_vector(PARAMETER)
        variance.set_local(data_save["variance"])
        particle.mean_posterior = mean
        particle.variance_posterior = variance

    # evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
    variation = Variation(model, particle, options, comm)

    # evaluate the kernel and its gradient at given particles
    kernel = Kernel(model, particle, variation, options, comm)

    t0 = time.time()

    solver = GradientDescent(model, particle, variation, kernel, options, comm)

    if args.use_batch:
        solver.solve_batchwise(batchsize=args.batch_size)
    else:
        solver.solve()

    print("GradientDecent solving time = ", time.time() - t0)

if __name__ == '__main__':
    main()
