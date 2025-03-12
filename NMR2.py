import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso
from scipy.optimize import nnls, minimize

# Low-Pass Baseline Filtering
def butter_lowpass_filter(signal, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

# Solve Beta using Least Squares
def solve_beta_least_squares(X, R):
    beta, _ = nnls(X, R)
    return beta

# Solve Beta using Lasso Regression
def solve_beta_lasso(X, R, alpha_lasso):
    model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=5000, warm_start=True, positive=True)
    model.fit(X, R)
    return model.coef_

# Alternating Solver: (Beta-step) + (Z-step)
def alternating_solver(Y, X, sparse_beta=False, 
                       alpha_lasso=0.01, cutoff_freq=0.01, 
                       filter_order=4, max_iter=10, tol=1e-6):

    Y = Y.flatten()
    m, n = X.shape
    beta = np.zeros(n)
    Z = np.zeros(m)

    for iteration in range(1, max_iter + 1):
        # --- (1) Beta-step ---
        R = Y - Z  # Residual ignoring current baseline

        if sparse_beta:
            beta_new = solve_beta_lasso(X, R, alpha_lasso)
        else:
            beta_new = solve_beta_least_squares(X, R)

        # --- (2) Z-step ---
        R2 = Y - X @ beta_new  # Residual ignoring new beta fit
        Z_new = butter_lowpass_filter(R2, cutoff_freq, order=filter_order)  # Smooth baseline estimation

        # Check for convergence
        db = np.linalg.norm(beta_new - beta)
        dz = np.linalg.norm(Z_new - Z)
        beta = beta_new
        Z = Z_new

        if db < tol and dz < tol:
            break

    return beta, Z

# NEW FUNCTION: Post-optimization of beta values
def optimize_beta_post_process(Y, X, beta_initial, Z_initial, cutoff_freq=0.01, filter_order=4):
    """
    Refine beta values after alternating solver solution using constrained optimization.
    
    Parameters:
    Y (numpy.ndarray): Experimental data
    X (numpy.ndarray): Dictionary matrix
    beta_initial (numpy.ndarray): Initial beta values from alternating solver
    Z_initial (numpy.ndarray): Initial baseline from alternating solver
    
    Returns:
    numpy.ndarray: Optimized beta values
    numpy.ndarray: Updated baseline
    """
    
    def objective(beta):
        """Objective function for optimization"""
        beta_reshaped = beta.reshape(-1)
        # Calculate residual with Z fixed
        residual = Y - X @ beta_reshaped - Z_initial
        return np.sum(residual**2)  # Sum of squared residuals
    
    # Setup bounds (all beta values must be non-negative)
    bounds = [(0, None) for _ in range(len(beta_initial))]
    
    # Run optimization
    result = minimize(
        objective,
        beta_initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 100}
    )
    
    beta_optimized = result.x
    
    # Optionally recalculate Z with optimized beta
    residual = Y - X @ beta_optimized
    Z_optimized = butter_lowpass_filter(residual, cutoff_freq, order=filter_order)
    
    return beta_optimized, Z_optimized

# Load metabolite files to create Xint dictionary
def load_Xint(num_points, metabolite_files):    
    Xint = np.zeros((num_points, len(metabolite_files)))
    for i, file_path in enumerate(metabolite_files):
        data = np.loadtxt(file_path, delimiter=',')
        if data.shape[0] != num_points:
            raise ValueError(f"Dimension mismatch: {file_path} has {data.shape[0]} points, expected {num_points}")
        Xint[:, i] = data
    print(f"\nLoaded Xint with {len(metabolite_files)} metabolites. Shape of Xint: {Xint.shape}")
    
    return Xint

# Load experimental file for Y
def load_Y(num_points, file_path):
    Y_noisy = np.loadtxt(file_path, delimiter=',')
    if len(Y_noisy) != num_points: 
        raise ValueError(f"Experimental data has length {len(Y_noisy)}, "
                         f"but Xint expects length {num_points}.")
    print(f"Loaded experimental data from {file_path}. Shape of Y_noisy: {Y_noisy.shape}")
    
    return Y_noisy

# Load true beta values
def load_Beta(path):
    true_beta = np.loadtxt(path, delimiter=',')
    
    return true_beta

# Calculate the RMSE and residual scores of different methods
def calc_scores(Y, Xint, beta_ls, beta_lp, beta_opt, true_beta, Z_lp, Z_opt):
    def rmse(a,b):
        return np.sqrt(np.mean((a-b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(true_beta, beta_lp)
    rmse_opt = rmse(true_beta, beta_opt)
    rmse_ls = rmse(true_beta, beta_ls)

    def resid(Y, X, beta, Z=None):
        if Z is None:
            return np.linalg.norm(Y - (X @ beta))
        else:
            return np.linalg.norm(Y - (X @ beta + Z))
            
    resid_lp_norm = resid(Y, Xint, beta_lp, Z_lp)
    resid_opt_norm = resid(Y, Xint, beta_opt, Z_opt)
    resid_ls_norm = resid(Y, Xint, beta_ls)

    return rmse_lp, rmse_opt, rmse_ls, resid_lp_norm, resid_opt_norm, resid_ls_norm

# Helper function to output results
def output(beta_lp, beta_opt, beta_ls, true_beta, rmse_lp, rmse_opt, rmse_ls, resid_lp_norm, resid_opt_norm, resid_ls_norm):
    print("\n====== Final Results =====")
    print(f"beta_lp (Low-pass) vs beta_opt (Optimized) vs beta_ls (Unconstrained LS):")
    
    col_width = 20
    print(f"{'beta_lp':<{col_width}} {'beta_opt':<{col_width}} {'beta_ls':<{col_width}} {'true_beta':<{col_width}}")
    for beta_lp_val, beta_opt_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_opt, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_opt_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f} {true_beta_val:<{col_width}.10f}")

    # Print RMSE values for all approaches    
    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"Optimized RMSE: {rmse_opt}")
    print(f"LS RMSE: {rmse_ls}")

    # Print normalized residual values for all approaches 
    print(f"\n--- Residual Norms wrt Y ---")
    print(f"LP + baseline vs Y      : {resid_lp_norm:.25f}")
    print(f"Optimized + baseline vs Y: {resid_opt_norm:.25f}")
    print(f"LS vs Y                 : {resid_ls_norm:.25f}")

# Test function
def test(data_folder, test_folder, test_file, true_beta_folder, true_beta_file, count_lp, count_opt):
    num_points = 120001

    # Load Metabolite Dictionary
    metabolite_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1,21)]
    Xint = load_Xint(num_points, metabolite_files)
    
    # Load Experimental Data
    Y_path = os.path.join(test_folder, test_file)
    Y = load_Y(num_points, Y_path)

    # Load True Beta Values
    beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = load_Beta(beta_path)

    # Run Alternating Solver and Naive LS approach
    beta_lp, Z_lp = alternating_solver(Y, Xint, sparse_beta=False, alpha_lasso=0.01, 
                                     cutoff_freq=0.01, filter_order=4, max_iter=10, tol=1e-6)
    beta_ls = solve_beta_least_squares(Xint, Y)
    
    # NEW: Apply post-optimization to refine beta values
    beta_opt, Z_opt = optimize_beta_post_process(Y, Xint, beta_lp, Z_lp)

    # Calculate RMSE and Residual Scores, then output results
    rmse_lp, rmse_opt, rmse_ls, resid_lp_norm, resid_opt_norm, resid_ls_norm = calc_scores(
        Y, Xint, beta_ls, beta_lp, beta_opt, true_beta, Z_lp, Z_opt)
    
    output(beta_lp, beta_opt, beta_ls, true_beta, rmse_lp, rmse_opt, rmse_ls, 
           resid_lp_norm, resid_opt_norm, resid_ls_norm)
    
    # Count wins
    if rmse_lp < rmse_ls:
        count_lp += 1
    if rmse_opt < rmse_ls:
        count_opt += 1
        
    # Output optimization improvement
    if rmse_opt < rmse_lp:
        print(f"\nOptimization IMPROVED RMSE by {((rmse_lp - rmse_opt) / rmse_lp * 100):.2f}%")
    else:
        print(f"\nOptimization DID NOT IMPROVE RMSE (change: {((rmse_lp - rmse_opt) / rmse_lp * 100):.2f}%)")

    # Return results as a dictionary instead of using global variables
    test_results = {
        'rmse_lp': rmse_lp,
        'rmse_opt': rmse_opt,
        'rmse_ls': rmse_ls
    }
    
    return count_lp, count_opt, test_results

def main():
    data_folder = "new_data"
    test_folder = "test_3"
    true_beta_folder = "test_3"
    
    count_lp = 0      # Count for LP baseline wins vs LS
    count_opt = 0     # Count for optimized LP wins vs LS
    count_opt_vs_lp = 0  # Count for optimized LP outperforming regular LP
    
    for i in range(1,16):
        print(f"\n\n========== Processing Test Case {i} ==========")
        test_file = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"
        count_lp, count_opt, test_results = test(data_folder, test_folder, test_file, true_beta_folder, true_beta_file, count_lp, count_opt)
        
        # Check if the optimized version outperformed regular LP for this test case
        rmse_lp, rmse_opt = test_results['rmse_lp'], test_results['rmse_opt']
        if rmse_opt < rmse_lp:
            count_opt_vs_lp += 1
    
    print(f"\n========== Final Results ==========")
    print(f"Low-pass Baseline Filtering outperformed Least Squares in {count_lp} out of 15 cases.")
    print(f"Optimized Low-pass method outperformed Least Squares in {count_opt} out of 15 cases.")
    print(f"Optimized Low-pass method outperformed regular Low-pass in {count_opt_vs_lp} out of 15 cases.")
    
    if count_opt > count_lp:
        print(f"Post-optimization improved overall performance vs LS in {count_opt - count_lp} cases.")
    elif count_opt < count_lp:
        print(f"Post-optimization reduced overall performance vs LS in {count_lp - count_opt} cases.")
    else:
        print("Post-optimization did not change the number of wins vs Least Squares.")

if __name__ == "__main__":
    main()