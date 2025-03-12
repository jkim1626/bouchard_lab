import numpy as np
import os
from sko.PSO import PSO
from unit_test_NMR import solve_lowpass_with_sparse_beta, solve_beta_least_squares, get_parameters

def objective_function(params, Y_noisy, Xint):
    """
    Define loss function.
    'params' represents the variables PSO is optimizing.
    """
    sparse_beta  = bool(round(params[0]))  # Ensure boolean for sparse_beta
    alpha_lasso  = params[1]
    cutoff_freq  = params[2]
    filter_order = int(round(params[3]))  # Ensure integer for filter order
    max_iter     = int(round(params[4]))  # Ensure integer for iterations

    # Run Deconvolution Algorithm
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        sparse_beta=sparse_beta,
        alpha_lasso=alpha_lasso,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        max_iter=max_iter
    )

    # Compute Error as Residual
    Y_fit = Xint @ beta_lp + Z_lp
    error = np.linalg.norm(Y_noisy - Y_fit) ** 2
    
    return error  # Minimize this error

def main():
    # Load data
    num_points = 120001  # Assumed number of points in each file
    data_folder = "new_data"
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 6)]
    
    Xint = np.zeros((num_points, len(all_files)))
    for j, fpath in enumerate(all_files):
        Xint[:, j] = np.loadtxt(fpath, delimiter=',')
    
    test_folder = "test_3"
    test_name = "synthetic_spectrum_1.txt"
    test_file = os.path.join(test_folder, test_name)
    Y_noisy = np.loadtxt(test_file, delimiter=',')

    # PSO Optimization
    pso = PSO(
        func=lambda params: objective_function(params, Y_noisy, Xint), 
        n_dim=5,
        pop=30,
        max_iter=50,
        lb=[0,0.01,0,2,0],
        ub=[1,0.1,0.5,8,50],
        w=0.7, c1=1.5, c2=1.5
    )

    beta_ls = solve_beta_least_squares(Xint, Y_noisy)

    param_solutions = pso.run()
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint, 
        sparse_beta=param_solutions[0],
        alpha_lasso=param_solutions[1],
        cutoff_freq=param_solutions[2],
        filter_order=param_solutions[3],
        max_iter=param_solutions[4]
    )

    true_beta_folder = "test_3"
    true_beta_file = "spectrum_ratios_1.txt"

    true_beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = np.loadtxt(true_beta_path, delimiter=',')
    
    def rmse(a,b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(beta_lp, true_beta)
    rmse_ls = rmse(beta_ls, true_beta)

    print(rmse_lp, '\t', rmse_ls)
    if rmse_lp < rmse_ls:
        print("Better")

    col_width = 20

    # Print header
    print(f"{'beta_lp':<{col_width}} {'beta_ls':<{col_width}} {'true_beta':<{col_width}}")

    # Print each pair of values
    for beta_lp_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f} {true_beta_val:<{col_width}.10f}")

if __name__ == "__main__":
    main()