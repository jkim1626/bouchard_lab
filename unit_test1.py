import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso

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
    Solve for beta values using standard least squares:
        beta = argmin || R - X beta ||^2
    """
    beta, _, _, _ = np.linalg.lstsq(X, R, rcond=None)
    return beta

# Solve Beta using Lasso Regression
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
    Y, X, cutoff_freq=0.01, filter_order=4,
    alpha_lasso=0.01, sparse_beta=False,
    max_iter=10, tol=1e-6
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
            print(f"Converged at iteration {iteration}.")
            break

    return beta, Z

# Helper function for user input with defaults
def input_with_default(prompt, default):
    """
    Prompt user to input parameters or use a default value instead
    """
    val = input(f"{prompt} [{default}]: ").strip()
    if not val:
        val = default
    return val

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

    print(f"Loaded experimental data from {file}. Shape of Y_noisy: {Y_noisy.shape}\n")
    
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

    # -- (1) Prompt Users for Parameters (or use default values)
    n_ref, sparse_beta, alpha_lasso, cutoff_freq, filter_order, num_iter = get_parameters()

    # -- (2a) Load metabolite files to be added into Xint 
    num_points = 120001  # Assumed number of points in each file
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 21)]
    Xint = create_dict(num_points, all_files)

    # -- (3) Load the experimental data 
    Y_noisy = load_Y(num_points, test_folder_name, test_file_name)

    # -- (4) Solve Y = Xint * beta + Z using the low-pass approach
    beta_ls = solve_beta_least_squares(Xint, Y_noisy)
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        cutoff_freq  = cutoff_freq,
        filter_order = filter_order,
        alpha_lasso  = alpha_lasso,
        sparse_beta  = sparse_beta,
        max_iter     = num_iter
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
    true_beta = np.loadtxt(true_beta_path)
    
    # -- (6b) Calculate RMSE values for both approaches (If "ground truth" is known)
    def rmse(a,b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(beta_lp, true_beta)
    rmse_ls = rmse(beta_ls, true_beta)

    # -- (7a) Print beta values for both approaches 
    mode_str = "(Low-pass + Lasso)" if sparse_beta else "(Low-pass, no sparse prior)"
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


    """

    # (E) Plot
    plt.figure(figsize=(10, 8))

    # 1) Top subplot: measured vs. dictionary fits
    plt.subplot(3,1,1)
    plt.plot(Y_noisy, 'k', lw=1, label='Experimental Data (1.txt)')
    plt.plot(Y_fit_ls, 'r', lw=1, label='LS Dict Only')
    plt.plot(Y_fit_lp_dict, 'b', lw=1, label=mode_str + " Dict Only")
    plt.legend()
    plt.title(f"Experimental Data vs. LS vs. LP Dict  {mode_str}")

    # 2) Middle subplot: residuals wrt Y (dictionary portion only)
    plt.subplot(3,1,2)
    plt.plot(resid_ls, 'r', label='LS Residual')
    plt.plot(resid_lp_dict, 'b', label='LP Residual')
    plt.legend()
    plt.title("Residuals wrt Y (Ignoring Baseline Z)")

    # 3) Bottom subplot: baseline
    plt.subplot(3,1,3)
    plt.plot(Z_lp, 'g', lw=1, label='Z (Low-pass baseline)')
    plt.legend()
    plt.title("Unmodeled / Baseline Z (Low-pass)")
    
    plt.tight_layout()
    plt.show()
    """

    # Compute dictionary contribution:
    Xbeta = Xint @ beta_lp
    residual = Y_noisy - (Xbeta + Z_lp)
    plt.figure(figsize=(10, 8))

    # --- Subplot 1: Show data vs. total reconstruction ---
    plt.subplot(3, 1, 1)
    plt.plot(Y_noisy, 'k', lw=1, label='Data (Y)')
    plt.plot(Xbeta + Z_lp, 'r', lw=1, label='Reconstruction (Xβ + Z)')
    plt.title("Data vs. Reconstructed Signal")
    plt.legend()

    # --- Subplot 2: Show dictionary-only vs. baseline ---
    plt.subplot(3, 1, 2)
    plt.plot(Xbeta, 'b', lw=1, label='Dictionary Part (Xβ)')
    plt.plot(Z_lp, 'g', lw=1, label='Baseline (Z)')
    plt.title("Decomposed Parts: Dictionary vs. Baseline")
    plt.legend()

    # --- Subplot 3: Residual ---
    plt.subplot(3, 1, 3)
    plt.plot(residual, 'm', lw=1, label='Residual = Y - (Xβ + Z)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residual")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Change folders/files or add iterations
def main():
    data_folder = "new_data"
    test_folder_name = "test_1"
    test_file_name = "synthetic_spectrum_1.txt"
    true_beta_folder = "test_1"
    true_beta_file = "spectrum_ratios_1.txt"
    test(data_folder, test_folder_name, test_file_name, true_beta_folder, true_beta_file)

if __name__ == "__main__":
    main()