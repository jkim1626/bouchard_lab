import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso
from scipy.optimize import nnls
from sko.PSO import PSO

# Low-Pass Baseline Filtering
def butter_lowpass_filter(signal, cutoff_freq, order=4):
    """
    Applies a forward-backward Butterworth filter to 'signal', producing a low-pass result.
    - cutoff_freq should be in [0,1], where 1 is the Nyquist frequency (0.5 of sample rate).
    - order sets the steepness of the filter's roll-off.
    - filtfilt(...) ensures zero phase shift.
    
    This is used to create a smooth, slowly varying baseline (Z).
    """
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

# Solve Beta using Least Squares
def solve_beta_least_squares(X, R):
    """
    Solve for beta using Non-Negative Least Squares (NNLS):
        beta = argmin || R - X beta ||^2  subject to beta >= 0
    """
    beta, _ = nnls(X, R)
    return beta

# Solve Beta using Lasso Regression
def solve_beta_lasso(X, R, alpha_lasso):
    """
    Solves for beta using Lasso Regression:
        beta = argmin 0.5 || R - X beta ||^2 + alpha_lasso * || beta ||_1
    """
    model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=5000, warm_start=True, positive=True)
    model.fit(X, R)
    return model.coef_

# Alternating Solver: (Beta-step) + (Z-step)
def solve_lowpass_with_sparse_beta(
    Y, X, 
    sparse_beta=False, 
    alpha_lasso=0.01,
    cutoff_freq=0.01, 
    filter_order=4,
    max_iter=10, 
    tol=1e-6
    ):
    """
    Approach:
      1) Beta-step:
         - If sparse_beta=False: solve a standard LS problem for beta:
             min_beta || (Y - Z) - X beta ||^2
         - If sparse_beta=True : solve a Lasso problem for beta:
             min_beta 0.5|| (Y - Z) - X beta ||^2 + alpha_lasso * ||beta||_1
      2) Z-step:
         - Z is updated by a low-pass filter:
             Z = butter_lowpass_filter( Y - X beta, cutoff_freq, filter_order )

    We iterate these two steps until convergence or reaching max_iter.

    Arguments:
    ----------
    Y           : array of shape (m,) or (m,1) - the observed data (noisy).
    X           : array of shape (m,n)         - dictionary columns for known spectra.
    cutoff_freq : float - the low-pass cutoff (0.0 - 0.5 typical).
    filter_order: int   - order of Butterworth filter.
    alpha_lasso : float - L1 penalty strength for Lasso if sparse_beta=True.
    sparse_beta : bool  - if True, use Lasso for the beta-step; else standard LS.
    max_iter    : int   - maximum outer iterations.
    tol         : float - stopping criterion for changes in beta and Z.

    Returns:
    --------
    beta        : final coefficient array (n,).
    Z           : final baseline/unmodeled portion (m,).
    """

    Y = Y.flatten()
    m, n = X.shape
    beta = np.zeros(n)
    Z = np.zeros(m)

    for iteration in range(1, max_iter + 1):
        beta_old = beta.copy()
        Z_old = Z.copy()

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

# Objective function for PSO to minimize residual error
def objective_function(beta, Y_noisy, Xint):
    """
    Define loss function.
    'params' represents the variables PSO is optimizing.
    """
    sparse_beta  = bool(round(beta[20]))  # Ensure boolean for sparse_beta
    alpha_lasso  = beta[21]
    cutoff_freq  = beta[22]
    filter_order = int(round(beta[23]))  # Ensure integer for filter order

    # Run Deconvolution Algorithm
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        sparse_beta=sparse_beta,
        alpha_lasso=alpha_lasso,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        max_iter=50
    )

    # Compute Error as Residual
    Y_fit = Xint @ beta_lp + Z_lp
    error = np.linalg.norm(Y_noisy - Y_fit) ** 2
    
    return error 

# Get parameters from User
def get_parameters():
    """
    Prompts user for parameters (or use default values):
        n_ref        (20)   : Number of reference spectra
        sparse_beta  (no)   : Boolean to activate sparse prior on beta
        alpha_lasso  (0.01) : L1 penality strength for sparse prior on beta
        cutoff_freq  (0.2   : Low pass cutoff frequency 
        fiter_order  (2)    : Filter order for butterworth filter
    """
    
    # Helper function for user input with defaults
    def input_with_default(prompt, default):
        """
        Prompt user to input parameters or use a default value instead
        """
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            val = default
        return val

    # Prompt: number of reference columns 
    n_ref_str = input_with_default("How many known reference spectra? (1-20)", "20")
    n_ref = int(n_ref_str)

    # Prompt: Ask about sparse prior
    sparse_str = input_with_default("Activate sparse prior on beta? (Yes/No)", "no").lower()
    sparse_beta = True if sparse_str == "yes" else False

    alpha_lasso_str = "0.0"
    if sparse_beta:
        alpha_lasso_str = input_with_default("Enter alpha_lasso (L1 penalty strength, e.g. 0.01~0.1)", "0.01")
    alpha_lasso = float(alpha_lasso_str)

    # Low-pass parameters
    cutoff_freq_str  = input_with_default("Enter low-pass cutoff freq (0.0~0.5 typical)", "0.2")
    cutoff_freq      = float(cutoff_freq_str)
    filter_order_str = input_with_default("Enter filter order (2~8 typical)", "2")
    filter_order     = int(filter_order_str)
    num_iter_str = input_with_default("Enter number of max iterations", "20")
    num_iter = int(num_iter_str)

    return n_ref, sparse_beta, alpha_lasso, cutoff_freq, filter_order, num_iter

# Load metabolite files for Xint 
def create_dict(num_points, all_files):    
    Xint = np.zeros((num_points, 20))
    for i, file_path in enumerate(all_files):
        Xint[:, i] = np.loadtxt(file_path, delimiter=',')
        
    print(f"\nLoaded Xint. Using {len(all_files)} references. Shape of Xint: {Xint.shape}")

    return Xint

# Load experimental file for Y
def load_Y(num_points, folder, file):
    Y_path = os.path.join(folder, file)
    Y_noisy = np.loadtxt(Y_path, delimiter=',')
    if len(Y_noisy) != num_points: # This file must have the same length as each column in Xint.
        raise ValueError(f"Data in {file} has length {len(Y_noisy)}, "
                         f"but Xint expects length {num_points}.")

    print(f"Loaded experimental data from {file}. Shape of Y_noisy: {Y_noisy.shape}")
    
    return Y_noisy

# Main test file 
def test(data_folder, test_folder_name, test_file_name, true_beta_folder, true_beta_file):

    # -- (1) Load metabolite files to be added into Xint 
    num_points = 120001  # Assumed number of points in each file
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 21)]
    Xint = create_dict(num_points, all_files)

    # -- (3a) Load the experimental data 
    Y_noisy = load_Y(num_points, test_folder_name, test_file_name)

    # -- (3b) Solve beta values using least squares technique (for benchmark)
    beta_ls = solve_beta_least_squares(Xint, Y_noisy)

    # -- (3b) Initialize an array of beta values as 0's
    beta_initial = np.zeros(24)

    # -- (3c) Initialize parameters from user
    n_ref, sparse_beta, alpha_lasso, cutoff_freq, filter_order, num_iter = get_parameters()
    
    beta_initial[:20] = beta_ls[:20]
    beta_initial[20] = sparse_beta
    beta_initial[21] = alpha_lasso
    beta_initial[22] = cutoff_freq
    beta_initial[23] = filter_order

    # Run PSO Optimization on Beta Values
    pso = PSO(
        func=lambda beta_initial: objective_function(beta_initial, Y_noisy, Xint), 
        n_dim=24,
        pop=15,
        max_iter=20,
        lb=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01, 0.01, 2],
        ub=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,0.1, 0.5, 8],
        w=0.7, c1=1.5, c2=1.5
    )
    solution, _ = pso.run()

    beta_lp = solution[:20]

    # -- (5a) Compute dictionary-only portion from the low-pass approach amd naive LS approach
    Y_fit_lp_dict = Xint @ beta_lp
    Y_fit_ls = Xint @ beta_ls

    # -- (5b) Calculate residuals for both approaches
    resid_lp_dict = Y_noisy - Y_fit_lp_dict
    resid_ls      = Y_noisy - Y_fit_ls
    
    # -- (5c) Normalize residuals for both approaches
    resid_lp_norm = np.linalg.norm(resid_lp_dict)
    resid_ls_norm = np.linalg.norm(resid_ls)

    # -- (6a) Load true beta values for given experimental file (If "ground truth" is known)
    true_beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = np.loadtxt(true_beta_path, delimiter=',')
    
    # -- (6b) Calculate RMSE values for both approaches (If "ground truth" is known)
    def rmse(a,b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(beta_lp, true_beta)
    rmse_ls = rmse(beta_ls, true_beta)

    # -- (7a) Print beta values for both approaches and true betas
    mode_str = "(Low-pass + Lasso)" if bool(round(solution[20])) else "(Low-pass, no sparse prior)"
    print("====== Final Results =====")
    print(f"beta_lp {mode_str} vs beta_ls (Unconstrained LS):")
    
    col_width = 20
    print(f"{'beta_lp':<{col_width}} {'beta_ls':<{col_width}} {'true_beta':<{col_width}}")
    for beta_lp_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f} {true_beta_val:<{col_width}.10f}")

    # -- (7b) Print RMSE values for both approaches    
    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"LS RMSE: {rmse_ls}")

    # -- (7c) Print normalized residual values for both approaches 
    print(f"\n--- Residual Norms wrt Y (dictionary portion only) ---")
    print(f"LP dictionary vs Y  : {resid_lp_norm:.25f}")
    print(f"LS dictionary vs Y  : {resid_ls_norm:.25f}")

    return rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm

# Change folders/files or add iterations
def main():
    data_folder = "new_data"
    test_folder_name = "test_3"
    true_beta_folder = "test_3"
    count = 0

    # Iterate through all files and check results 
    for i in range(1,3):
        test_file_name = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"
        rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm = test(data_folder, test_folder_name, test_file_name, true_beta_folder, true_beta_file)
        if rmse_lp < rmse_ls:
            print(f"\n{test_file_name} has a lower RMSE lp value than RMSE ls \n")
            count += 1
    
    print(f"\nNumber of files that where LP RMSE was less than LS RMSE: {count}")

if __name__ == "__main__":
    main()