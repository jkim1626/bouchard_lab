import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso
from scipy.optimize import nnls

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

# Calculate the RMSE and residual scores of LS and LP approaches
def calc_scores(Y, Xint, beta_ls, beta_lp, true_beta):
    def rmse(a,b):
        return np.sqrt(np.mean((a-b)**2)) if len(a) == len(b) else np.nan
    rmse_lp = rmse(true_beta, beta_lp)
    rmse_ls = rmse(true_beta, beta_ls)

    def resid(Y,a,b):
        return np.linalg.norm(Y - (a @ b))
    resid_lp_norm = resid(Y, Xint, beta_lp)
    resid_ls_norm = resid(Y, Xint, beta_ls)

    return rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm

# Helper function to output results
def output(beta_lp, beta_ls, true_beta, rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm):
    print("====== Final Results =====")
    print(f"beta_lp (Low-pass) vs beta_ls (Unconstrained LS):")
    
    col_width = 20
    print(f"{'beta_lp':<{col_width}} {'beta_ls':<{col_width}} {'true_beta':<{col_width}}")
    for beta_lp_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f} {true_beta_val:<{col_width}.10f}")

    # Print RMSE values for both approaches    
    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"LS RMSE: {rmse_ls}")

    # Print normalized residual values for both approaches 
    print(f"\n--- Residual Norms wrt Y (dictionary portion only) ---")
    print(f"LP dictionary vs Y  : {resid_lp_norm:.25f}")
    print(f"LS dictionary vs Y  : {resid_ls_norm:.25f}")

# Test function
def test(data_folder, test_folder, test_file, true_beta_folder, true_beta_file, count):
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
    beta_lp,_ = alternating_solver(Y, Xint, sparse_beta=False, alpha_lasso=0.01, cutoff_freq=0.01, filter_order=4, max_iter=10, tol=1e-6)
    beta_ls = solve_beta_least_squares(Xint, Y)

    # Calculate RMSE and Residual Scores, then output results
    rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm = calc_scores(Y, Xint, beta_ls, beta_lp, true_beta)
    output(beta_lp, beta_ls, true_beta, rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm)
    if rmse_lp < rmse_ls:
        count += 1

    return count

def main():
    data_folder = "new_data"
    test_folder = "test_0"
    true_beta_folder = "test_0"
    
    count = 0
    for i in range(1,4):
        test_file = f"test{i}.txt"
        true_beta_file = f"answer{i}.txt"
        count = test(data_folder,test_folder,test_file,true_beta_folder,true_beta_file, count)
    
    print(f"\nLow-pass Baseline Filtering outperformed Least Squares in {count} out of 15 cases.")

if __name__ == "__main__":
    main()