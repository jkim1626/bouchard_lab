import numpy as np
import os
import multiprocessing 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso
from scipy.optimize import nnls
from sko.PSO import PSO

class PSOParallel(PSO):
    """
    A subclass of PSO that evaluates each particle's fitness in parallel
    using multiprocessing.
    """
    def cal_y(self):
        """
        Overridden method to compute self.Y (fitness) for all particles.
        
        self.X has shape (pop, dim). We'll map each row (one particle)
        to a worker in the pool, collecting the results in self.Y.
        """
        # X is shape (pop, dim)
        X_all = self.X.copy()  # (pop, dim)

        # Use a context manager to create the pool
        with multiprocessing.Pool() as pool:
            # Map each row X_all[i] to the objective function
            results = pool.map(self._evaluate_single_particle, X_all)

        self.Y = np.array(results)

    def _evaluate_single_particle(self, x_particle):
        """
        x_particle is a 1D numpy array of shape (dim,).
        We call self.func(x_particle) to get its fitness.
        """
        return self.func(x_particle)

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

# Solve Beta using Lasso Regression (If sparse beta)
def solve_beta_lasso(X, R, alpha_lasso):
    """
    Solves for beta using Lasso Regression:
        beta = argmin 0.5 || R - X beta ||^2 + alpha_lasso * || beta ||_1
    """
    model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=5000, warm_start=True)
    model.fit(X, R)
    return model.coef_

# Alternating Solver: (Beta-step) + (Z-step)
def solve_lowpass_with_sparse_beta(
    Y, X, 
    sparse_beta=False, 
    alpha_lasso=0.01,
    cutoff_freq=0.01, 
    filter_order=4,
    max_iter=20, 
    tol=1e-5
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
def objective_function(params, Y_noisy, Xint):
    """
    Define loss function.
    'params' represents the variables PSO is optimizing.
    """
    sparse_beta  = bool(round(params[0]))  # Ensure boolean for sparse_beta
    alpha_lasso  = params[1]
    cutoff_freq  = params[2]

    # Run Deconvolution Algorithm
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        sparse_beta=sparse_beta,
        alpha_lasso=alpha_lasso,
        cutoff_freq=cutoff_freq,
    )

    # Compute Error as Residual
    Y_fit = Xint @ beta_lp + Z_lp
    error = np.linalg.norm(Y_noisy - Y_fit)
    
    return error  

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
    """
    1) Prompt user for parameters (or use default values).
    2) Load Xint (dictionary of reference spectra).
    3) Load the experimental data Y 
    4) Solve Y = Xint * beta + Z with the low-pass approach.
    5) Calculate and normalize for both LS and LP approaches
    6) Calculate RMSE values for LS and LP approaches (If using synthetic data)
    7) Plot and print results.
    """

    # -- (1) Load metabolite files to be added into Xint 
    num_points = 120001  # Assumed number of points in each file
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 21)]
    Xint = create_dict(num_points, all_files)

    # -- (3) Load the experimental data 
    Y_noisy = load_Y(num_points, test_folder_name, test_file_name)

    # PSO Optimization
    pso = PSO(
        func=lambda params: objective_function(params, Y_noisy, Xint), 
        n_dim=3,
        pop=10,
        max_iter=20,
        lb=[0, 0.001, 0.01],
        ub=[1, 0.1, 0.5],
        w=0.7, c1=1.5, c2=1.5
    )

    best_x, best_y = pso.run()
    print(f"\nOptimal Parameters: {best_x}")

    # -- (4) Solve Y = Xint * beta + Z using the low-pass approach
    beta_ls = solve_beta_least_squares(Xint, Y_noisy)
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        sparse_beta = bool(round(best_x[0])),
        alpha_lasso = best_x[1],
        cutoff_freq = best_x[2],
        max_iter = 10,
        tol=1e-5
    )

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

    # -- (7a) Print beta values for both approaches 
    mode_str = "(Low-pass + Lasso)" if bool(round(best_x[0])) else "(Low-pass, no sparse prior)"
    print("====== Final Results =====")
    print(f"beta_lp {mode_str} vs beta_ls (Unconstrained LS):")
    
    # Define a fixed column width
    col_width = 20

    # Print each pair of values 
    for beta_lp_val, beta_ls_val in zip(beta_lp, beta_ls):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f}")

    # -- (7b) Print RMSE values for both approaches    
    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"LS RMSE: {rmse_ls}")

    # -- (7c) Print normalized residual values for both approaches 
    print(f"\n--- Residual Norms wrt Y (dictionary portion only) ---")
    print(f"LP dictionary vs Y  : {resid_lp_norm:.25f}")
    print(f"LS dictionary vs Y  : {resid_ls_norm:.25f}")
    
    if (rmse_lp < rmse_ls):
        print("\nBETTER")

# Change folders/files or add iterations
def main():
    data_folder = "New_Dict"
    test_folder_name = "test_3"
    true_beta_folder = "test_3"
    
    for i in range(1, 16):
        test_file_name = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"
        test(data_folder, test_folder_name, test_file_name, true_beta_folder, true_beta_file)

if __name__ == "__main__":
    main()