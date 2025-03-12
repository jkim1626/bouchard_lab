import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso

###############################################################################
#                   Generation of Synthetic NMR-Like Spectra                  #
###############################################################################
def lorentzian(x, x0, gamma, amplitude=1.0):
    """
    Creates a single Lorentzian peak at position x0 with half-width-at-half-max gamma.
    amplitude is the peak's height.
    
    Lorentzian functional form:
        L(x) = amplitude * ( (gamma/2)^2 / ( (x - x0)^2 + (gamma/2)^2 ) )
    """
    return amplitude * ((gamma/2.0)**2 / ((x - x0)**2 + (gamma/2.0)**2))

def generate_lorentzian_spectrum(num_points=1024, num_peaks=3, max_gamma=10.0):
    """
    Generate a random 1D Lorentzian spectrum of length num_points.
    - Each spectrum has 'num_peaks' random Lorentzian peaks.
    - Normalizes the max amplitude to 1.0 for consistency.

    This is used to simulate dictionary columns (Xint) or synthetic data (Y).
    In real use-cases, you would load actual NMR data from a file and
    store them in Xint columns, rather than call this function.
    """
    x = np.arange(num_points, dtype=float)
    spectrum = np.zeros(num_points, dtype=float)

    for _ in range(num_peaks):
        # Random center, half-width, amplitude
        center = np.random.uniform(0, num_points)
        gamma  = np.random.uniform(1.0, max_gamma)
        amp    = np.random.uniform(0.2, 1.0)
        spectrum += lorentzian(x, center, gamma, amplitude=amp)

    # Normalize so that the peak amplitude is 1.0
    max_val = spectrum.max()
    if max_val > 0:
        spectrum /= max_val

    return spectrum

###############################################################################
#                           Low-Pass Baseline Filtering                       #
###############################################################################
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

###############################################################################
#                   Alternating Solver:  (Beta-step) + (Z-step)               #
###############################################################################
def solve_lowpass_with_sparse_beta(
    Y, X, cutoff_freq=0.01, filter_order=4,
    alpha_lasso=0.01, sparse_beta=False,
    max_iter=10, tol=1e-6
):
    """
    Model:  Y = X beta + Z
    
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
    # Initialize beta and Z to zero
    beta = np.zeros(n)
    Z    = np.zeros(m)

    for iteration in range(1, max_iter+1):
        beta_old = beta.copy()
        Z_old    = Z.copy()

        #--- (1) Beta-step ---
        # Residual ignoring current baseline
        R = Y - Z

        if sparse_beta:
            # Lasso regression for beta
            model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=5000, warm_start=True)
            model.fit(X, R)
            beta_new = model.coef_
        else:
            # Standard Least Squares for beta
            beta_new, _, _, _ = np.linalg.lstsq(X, R, rcond=None)

        #--- (2) Z-step ---
        # Residual ignoring the new dictionary fit
        R2 = Y - X @ beta_new
        # Low-pass filter that residual to create a smooth baseline
        Z_new = butter_lowpass_filter(R2, cutoff_freq, order=filter_order)

        # Check for convergence
        db = np.linalg.norm(beta_new - beta)
        dz = np.linalg.norm(Z_new - Z)
        beta = beta_new
        Z    = Z_new

        if db < tol and dz < tol:
            print(f"Converged at iteration {iteration}.")
            break

    return beta, Z

###############################################################################
#                 MAIN DEMO + Extended Output, With Comments                  #
###############################################################################
def main():
    """
    Main function that:
      1) Prompts user for parameters (with defaults).
      2) Optionally generates synthetic dictionary Xint & data Y_noisy, or
         (in real scenario) you'd load Xint from your own data file.
      3) Solves the above 'low-pass + optional sparse beta' decomposition.
      4) Compares to naive LS.
      5) Prints two sets of residual norms: 
         (A) wrt the full measured Y, ignoring the baseline
         (B) wrt the 'true known only' portion (only valid if synthetic).
    """

    # Helper function for user input with defaults
    def input_with_default(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            val = default
        return val

    # Prompt the user with default values
    n_total_str = input_with_default("Enter total # of random Lorentzian spectra (XPint size)", "30")
    n_total     = int(n_total_str)

    n_ref_str   = input_with_default("Enter how many known reference spectra in Xint", "10")
    n_ref       = int(n_ref_str)

    # Ask about sparse prior
    sparse_str = input_with_default("Activate sparse prior on beta? (Yes/No)", "no").lower()
    # We'll do low-pass baseline in either case, but optionally add Lasso
    sparse_beta = (sparse_str.startswith('y'))

    alpha_lasso_str = "0.0"
    if sparse_beta:
        alpha_lasso_str = input_with_default("Enter alpha_lasso (L1 penalty strength, e.g. 0.01~0.1)", "0.01")
    alpha_lasso = float(alpha_lasso_str)

    # Low-pass parameters
    cutoff_freq_str = input_with_default("Enter low-pass cutoff freq (0.0~0.5 typical)", "0.2")
    cutoff_freq     = float(cutoff_freq_str)
    filter_order_str= input_with_default("Enter filter order (2~8 typical)", "2")
    filter_order    = int(filter_order_str)

    # Noise
    noise_std_str   = input_with_default("Enter noise std (0.0~0.1 typically)", "0.05")
    noise_std       = float(noise_std_str)

    ############################################################################
    # In a real scenario, you would load Xint from your data, e.g.
    #
    # Xint = np.loadtxt("my_NMR_dictionary.csv", delimiter=",")
    #
    # 'n_ref' might be the # of dictionary columns you have, or part of the file.
    #
    # For demonstration, we generate synthetic columns:
    ############################################################################

    num_points = 1024  # # of data points in each spectrum
    XPint = np.zeros((num_points, n_total))
    for j in range(n_total):
        n_peaks = np.random.randint(1,4)
        XPint[:, j] = generate_lorentzian_spectrum(num_points=num_points, num_peaks=n_peaks)

    # We'll pick a random subset of columns as "known reference"
    ref_indices = np.random.choice(n_total, n_ref, replace=False)
    Xint        = XPint[:, ref_indices]

    # Generate synthetic "true" coefficients for those known references
    true_beta_known = np.random.uniform(0.5, 2.0, size=n_ref)

    # Build Y from known references
    Y = Xint @ true_beta_known

    # Also add "unknown" columns to the data
    unknown_indices = [idx for idx in range(n_total) if idx not in ref_indices]
    if len(unknown_indices) > 0:
        Xunknown = XPint[:, unknown_indices]
        true_beta_unknown = np.random.uniform(0.5, 2.0, size=len(unknown_indices))
        Y += Xunknown @ true_beta_unknown
    else:
        true_beta_unknown = np.array([])

    # Finally, add random noise
    Y_noisy = Y + np.random.normal(0, noise_std, size=num_points)

    # For demonstration, we have "true known only" portion:
    # If using real data, you won't have this.  => Comment it out or remove it.
    Y_true_known_only = Xint @ true_beta_known

    print("\n=== Summary ===")
    print("ref_indices (known):", ref_indices)
    print("unknown_indices    :", unknown_indices)
    print("true_beta_known   =", true_beta_known, 
          "(If real data, you won't have these or the next line).")
    print("true_beta_unknown =", true_beta_unknown)
    print(f"\nLow-pass baseline approach. sparse_beta={sparse_beta}, alpha_lasso={alpha_lasso}")
    print(f"cutoff_freq={cutoff_freq}, filter_order={filter_order}, noise_std={noise_std}")

    # Call the solver for low-pass + optional Lasso on beta
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint, cutoff_freq=cutoff_freq, filter_order=filter_order,
        alpha_lasso=alpha_lasso, sparse_beta=sparse_beta, max_iter=20
    )

    # Dictionary-only portion from the low-pass approach
    Y_fit_lp_dict = Xint @ beta_lp

    # Compare to naive LS (unconstrained)
    beta_ls, _, _, _ = np.linalg.lstsq(Xint, Y_noisy, rcond=None)
    Y_fit_ls = Xint @ beta_ls

    # ========== Residuals wrt full measured Y (dictionary only) ==========
    # ignoring the baseline Z in both methods
    resid_lp_dict_wrt_Y = Y_noisy - Y_fit_lp_dict
    resid_lp_dict_norm_wrt_Y = np.linalg.norm(resid_lp_dict_wrt_Y)

    resid_ls_wrt_Y = Y_noisy - Y_fit_ls
    resid_ls_norm_wrt_Y = np.linalg.norm(resid_ls_wrt_Y)

    # ========== Residuals wrt "True Known Only" ==========
    # This portion is only valid if you have synthetic data that includes
    # a known "true" portion.  For real data, you wouldn't do this.
    resid_lp_dict_wrt_known = Y_true_known_only - Y_fit_lp_dict
    resid_lp_dict_norm_wrt_known = np.linalg.norm(resid_lp_dict_wrt_known)

    resid_ls_wrt_known = Y_true_known_only - Y_fit_ls
    resid_ls_norm_wrt_known = np.linalg.norm(resid_ls_wrt_known)

    # RMSE for known Beta  (again, only for synthetic scenario)
    def rmse(a,b):
        return np.sqrt(np.mean((a-b)**2)) if len(a)==len(b) else np.nan

    rmse_lp = rmse(beta_lp, true_beta_known)
    rmse_ls = rmse(beta_ls, true_beta_known)

    # Decide whether to print "with Lasso" or "with standard LS" for the dictionary step
    if sparse_beta:
        mode_str = "(Low-pass + Lasso on beta)"
    else:
        mode_str = "(Low-pass, no sparse prior)"

    print("\n=== Final Results ===")
    print(f"beta_lp {mode_str}:", beta_lp)
    print("beta_ls (unconstrained LS):", beta_ls)

    # Print both sets of residual norms
    print(f"\n--- Residual Norms wrt FULL Y (dict portion only) ---")
    print(f"LP dictionary vs Y  : {resid_lp_dict_norm_wrt_Y:.3f}")
    print(f"LS dictionary vs Y  : {resid_ls_norm_wrt_Y:.3f}")

    print(f"\n--- Residual Norms wrt 'True Known Only' (synthetic only) ---")
    print(f"LP dict vs known    : {resid_lp_dict_norm_wrt_known:.3f}")
    print(f"LS dict vs known    : {resid_ls_norm_wrt_known:.3f}")

    print(f"\nRMSE(known beta) LP vs. True = {rmse_lp:.3f}")
    print(f"RMSE(known beta) LS vs. True = {rmse_ls:.3f}")

    # --- Plotting ---
    plt.figure(figsize=(10,8))

    # 1) Top subplot: measured vs. dictionary fits vs. "true known only"
    plt.subplot(3,1,1)
    plt.plot(Y_noisy, 'k', lw=1, label='Y (noisy)')
    plt.plot(Y_fit_ls, 'r', lw=1, label='LS Dict')
    plt.plot(Y_fit_lp_dict, 'b', lw=1, label=mode_str + " Dict Only")
    plt.plot(Y_true_known_only, 'g--', lw=1, label='True Known Only (synthetic)')

    plt.legend()
    plt.title(f"Measured vs. LS vs. LP Dict vs. True Known   {mode_str}")

    # 2) Middle subplot: residuals wrt Y (dictionary portion only)
    plt.subplot(3,1,2)
    plt.plot(resid_ls_wrt_Y, 'r', label='LS Resid (vs Y)')
    plt.plot(resid_lp_dict_wrt_Y, 'b', label='LP Resid (vs Y)')
    plt.legend()
    plt.title("Residuals wrt Y (Ignoring Baseline Z)")

    # 3) Bottom subplot: baseline
    plt.subplot(3,1,3)
    plt.plot(Z_lp, 'g', lw=1, label='Z (Low-pass baseline)')
    plt.legend()
    plt.title("Unmodeled / Baseline Z (Low-pass)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()