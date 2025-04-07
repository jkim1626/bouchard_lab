import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import Lasso
from scipy.optimize import nnls, minimize
from sklearn.linear_model import LinearRegression
from lmfit import Model, Parameters

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

# Post-optimization of beta values
def optimize_beta_post_process(Y, X, beta_initial, Z_initial, cutoff_freq=0.01, filter_order=4):
    """
    Refine beta values after alternating solver solution using constrained optimization.
    """
    def objective(beta):
        beta_reshaped = beta.reshape(-1)
        residual = Y - X @ beta_reshaped - Z_initial
        return np.sum(residual**2)
    
    bounds = [(0, None) for _ in range(len(beta_initial))]
    
    result = minimize(
        objective,
        beta_initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 100}
    )
    
    beta_optimized = result.x
    
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
        raise ValueError(f"Experimental data has length {len(Y_noisy)}, but Xint expects length {num_points}.")
    print(f"Loaded experimental data from {file_path}. Shape of Y_noisy: {Y_noisy.shape}")
    return Y_noisy

# Load true beta values
def load_Beta(path):
    true_beta = np.loadtxt(path, delimiter=',')
    return true_beta

# Calculate the RMSE and residual scores of different methods
def calc_scores(Y, Xint, beta_ls, beta_lp, beta_opt, true_beta, Z_lp, Z_opt):
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
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

    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"Optimized RMSE: {rmse_opt}")
    print(f"LS RMSE: {rmse_ls}")

    print(f"\n--- Residual Norms wrt Y ---")
    print(f"LP + baseline vs Y      : {resid_lp_norm:.25f}")
    print(f"Optimized + baseline vs Y: {resid_opt_norm:.25f}")
    print(f"LS vs Y                 : {resid_ls_norm:.25f}")

# Plotting Functions for a Single Spectrum
def plot_spectrum_fit_comparison(Y, Xint, beta_opt, Z_opt, beta_ls, ppm):
    # Reconstructed spectra
    recon_new = Xint @ beta_opt + Z_opt
    recon_ls  = Xint @ beta_ls

    # Compute residuals
    resid_new = Y - recon_new
    resid_ls  = Y - recon_ls

    # Figure 1: Overlay and Residuals
    plt.figure(figsize=(12, 10))

    # Panel A: Spectrum overlay
    plt.subplot(2,1,1)
    plt.plot(ppm, Y, 'k', label='Simulated Spectrum')
    plt.plot(ppm, recon_new, 'b', label='New Algorithm')
    plt.plot(ppm, recon_ls, 'r', label='Least Squares')
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Figure 1A: Spectrum Fit Comparison")
    plt.legend()

    # Panel B: Residuals
    plt.subplot(2,1,2)
    plt.plot(ppm, resid_new, 'b', label='Residual (New Algorithm)')
    plt.plot(ppm, resid_ls, 'r', label='Residual (LS)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Residual Intensity")
    plt.title("Figure 1B: Residuals")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_peak_identification(Y, Xint, beta_opt, Z_opt, beta_ls, ppm):
    # Reconstructed spectra (same as before)
    recon_new = Xint @ beta_opt + Z_opt
    recon_ls  = Xint @ beta_ls

    # For demonstration, we use the simulated spectrum Y as the "true" spectrum.
    # Find peaks in a zoomed-in region.
    # Let’s choose a zoom region from 4 to 6 ppm.
    idx_zoom = np.where((ppm >= 4) & (ppm <= 6))[0]
    ppm_zoom = ppm[idx_zoom]
    Y_zoom = Y[idx_zoom]
    recon_new_zoom = recon_new[idx_zoom]
    recon_ls_zoom = recon_ls[idx_zoom]

    # Use a simple peak finding (you can adjust parameters as needed)
    peaks_true, _ = find_peaks(Y_zoom, prominence=0.05)
    peaks_new, _ = find_peaks(recon_new_zoom, prominence=0.05)
    peaks_ls, _ = find_peaks(recon_ls_zoom, prominence=0.05)

    plt.figure(figsize=(10, 6))
    plt.plot(ppm_zoom, Y_zoom, 'k-', label='Simulated Spectrum')
    plt.plot(ppm_zoom, recon_new_zoom, 'b-', label='New Algorithm')
    plt.plot(ppm_zoom, recon_ls_zoom, 'r-', label='Least Squares')
    plt.plot(ppm_zoom[peaks_true], Y_zoom[peaks_true], 'ko', label='True Peaks')
    plt.plot(ppm_zoom[peaks_new], recon_new_zoom[peaks_new], 'bo', markersize=8, fillstyle='none', label='New Algo Peaks')
    plt.plot(ppm_zoom[peaks_ls], recon_ls_zoom[peaks_ls], 'r^', markersize=8, fillstyle='none', label='LS Peaks')
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Figure 2: Peak Identification in Zoomed Region (4-6 ppm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_concentration_accuracy(true_beta, beta_opt, beta_ls):
    # Scatter plot for predicted vs. true concentrations.
    # Compute R^2 values
    r_new = np.corrcoef(true_beta, beta_opt)[0,1]
    R2_new = r_new**2
    r_ls = np.corrcoef(true_beta, beta_ls)[0,1]
    R2_ls = r_ls**2

    plt.figure(figsize=(8, 6))
    plt.plot(true_beta, true_beta, 'k--', label="Identity Line")
    plt.scatter(true_beta, beta_opt, c='b', label=f"New Algorithm (R²={R2_new:.2f})")
    plt.scatter(true_beta, beta_ls, c='r', label=f"LS (R²={R2_ls:.2f})")
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title("Figure 3: Concentration Prediction Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residual_error(true_beta, beta_opt, beta_ls):
    # Compute absolute errors per metabolite
    error_new = np.abs(true_beta - beta_opt)
    error_ls = np.abs(true_beta - beta_ls)
    metabolites = np.arange(1, len(true_beta)+1)

    width = 0.35  # width of the bars
    plt.figure(figsize=(10, 6))
    plt.bar(metabolites - width/2, error_new, width, color='b', label="New Algorithm")
    plt.bar(metabolites + width/2, error_ls, width, color='r', label="LS")
    plt.xlabel("Metabolite ID")
    plt.ylabel("Absolute Error")
    plt.title("Figure 4: Residual/Error per Metabolite")
    plt.xticks(metabolites)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_performance_metrics(rmse_opt, rmse_ls, true_beta, beta_opt, beta_ls):
    # Compute R^2 values as in Figure 3
    r_new = np.corrcoef(true_beta, beta_opt)[0,1]
    R2_new = r_new**2
    r_ls = np.corrcoef(true_beta, beta_ls)[0,1]
    R2_ls = r_ls**2

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: RMSE Comparison
    axs[0].bar(["New Algorithm", "Least Squares"], [rmse_opt, rmse_ls], color=['b', 'r'])
    axs[0].set_ylabel("RMSE (Intensity Units)")
    axs[0].set_title("Figure 5A: Overall RMSE Comparison")
    
    # Panel B: R² Comparison
    axs[1].bar(["New Algorithm", "Least Squares"], [R2_new, R2_ls], color=['b', 'r'])
    axs[1].set_ylabel("R²")
    axs[1].set_title("Figure 5B: Overall R² Comparison")
    
    plt.tight_layout()
    plt.show()

def display_summary_tables(true_beta, beta_opt, beta_ls, rmse_opt, rmse_ls):
    # Table 1: Summary of Key Numerical Results
    r_new = np.corrcoef(true_beta, beta_opt)[0,1]
    R2_new = r_new**2
    r_ls = np.corrcoef(true_beta, beta_ls)[0,1]
    R2_ls = r_ls**2

    summary_data = {
        "Metric": ["RMSE", "R²"],
        "New Algorithm": [rmse_opt, R2_new],
        "Least Squares": [rmse_ls, R2_ls]
    }
    df_summary = pd.DataFrame(summary_data)
    print("\nTable 1: Summary of Key Numerical Results")
    print(df_summary.to_string(index=False))

    # Table 2: Metabolite-by-Metabolite Concentration Comparison with Errors
    metabolite_ids = np.arange(1, len(true_beta)+1)
    error_new_percent = np.array([100 * abs(t - p) / t if t != 0 else np.nan for t, p in zip(true_beta, beta_opt)])
    error_ls_percent = np.array([100 * abs(t - p) / t if t != 0 else np.nan for t, p in zip(true_beta, beta_ls)])
    data = {
        "Metabolite": metabolite_ids,
        "True Concentration": true_beta,
        "New Predicted": beta_opt,
        "Error New (%)": error_new_percent,
        "LS Predicted": beta_ls,
        "Error LS (%)": error_ls_percent
    }
    df_metabolites = pd.DataFrame(data)
    print("\nTable 2: Metabolite-by-Metabolite Concentration Comparison")
    print(df_metabolites.to_string(index=False))

# Plot RMSE comparison across all 50 samples
def plot_rmse_comparison_all_samples(rmse_lp_list, rmse_ls_list):
    samples = np.arange(1, len(rmse_lp_list)+1)
    plt.figure(figsize=(10, 6))
    plt.plot(samples, rmse_lp_list, marker='o', linestyle='-', color='blue', label='Low Pass')
    plt.plot(samples, rmse_ls_list, marker='s', linestyle='-', color='red', label='Least Squares')
    plt.xlabel("Sample")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison Across 50 Samples")
    plt.xlim(0, len(rmse_lp_list)+1)
    plt.legend()
    plt.grid(True)
    plt.show()

# Display a RMSE Table for each sample 
def display_rmse_table(rmse_lp_list, rmse_ls_list):
    samples = np.arange(1, len(rmse_lp_list)+1)
    df = pd.DataFrame({
        'Sample': samples,
        'Low Pass RMSE': np.round(rmse_lp_list, 3),
        'Least Squares RMSE': np.round(rmse_ls_list, 3)
    })
    df.set_index('Sample', inplace=True)
    print("\nRMSE Table (50 x 3):")
    print(df.to_string())

# Display Concentration Comparison Table
def display_concentration_comparison_table(true_beta, beta_lp, beta_ls, beta_opt):
    num_metabolites = len(true_beta)
    metabolites = [f"Metabolite {i+1}" for i in range(num_metabolites)]
    data = {
        "Exact Concentration": np.round(true_beta, 3),
        "Low Pass": np.round(beta_lp, 3),
        "Least Squares": np.round(beta_ls, 3),
        "Low Pass Optimized": np.round(beta_opt, 3)
    }
    df = pd.DataFrame(data, index=metabolites)
    print("\nConcentration Comparison Table (20 x 4):")
    print(df.to_string())

def run_NMRglue(Xint, Y):
    # Ensure the input spectra are numpy arrays
    Xint = np.asarray(Xint)
    Y = np.asarray(Y)
    
    # Check if the dimensions are correct
    if Xint.shape[0] != Y.shape[0]:
        raise ValueError("Number of rows in Xint must match the length of Y (120001).")
    
    if Xint.shape[1] != 20:
        raise ValueError("Xint must have 20 columns representing 20 metabolites.")
    
    # Normalize the spectra (optional but recommended)
    Xint_normalized = Xint / np.linalg.norm(Xint, axis=0)
    Y_normalized = Y / np.linalg.norm(Y)
    
    # Fit a linear regression model
    model = LinearRegression(fit_intercept=False)  # No intercept since spectra are zero-centered
    model.fit(Xint_normalized, Y_normalized)
    
    # Get the beta coefficients (metabolite coefficients)
    beta_values = model.coef_
    
    return beta_values

def run_lmfit(Xint, Y):
    # Determine the number of metabolites (columns in Xint)
    num_metabolites = Xint.shape[1]
    indices = list(range(num_metabolites))
    
    # Define the model function. Although lmfit requires an independent variable (here 'x'),
    # it is not used because our model is simply the dot product of Xint with the coefficients.
    def metabolite_model(x, **params):
        # Build an array of coefficients from the parameters
        coeffs = np.array([params[f'c{i}'] for i in indices])
        # Compute the predicted spectrum as the dot product of Xint and the coefficients
        return np.dot(Xint, coeffs)
    
    # Create an lmfit Model object using the metabolite_model function.
    # The independent variable is named 'x' (a dummy variable in this case).
    model = Model(metabolite_model, independent_vars=['x'])
    
    # Initialize the parameters with an initial guess (here, 1.0 for each coefficient)
    params = Parameters()
    for i in indices:
        # Optionally, set bounds (min=0 for non-negative coefficients)
        params.add(f'c{i}', value=1.0, min=0)
    
    # Create a dummy x-axis; it must be the same length as Y even though it's not used in the model.
    x = np.arange(len(Y))
    
    # Fit the model to the data Y using lmfit.
    result = model.fit(Y, params, x=x)
    
    # Extract the fitted coefficients from the result and store them in a numpy array.
    beta_values = np.array([result.params[f'c{i}'].value for i in indices])
    
    return beta_values

# Main test function 
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
                                        cutoff_freq=0.01, filter_order=4, max_iter=50, tol=1e-6)
    beta_ls = solve_beta_least_squares(Xint, Y)

    # Apply post-optimization to refine beta values
    beta_opt, Z_opt = optimize_beta_post_process(Y, Xint, beta_lp, Z_lp)

    rmse_lp, rmse_opt, rmse_ls, resid_lp_norm, resid_opt_norm, resid_ls_norm = calc_scores(
        Y, Xint, beta_ls, beta_lp, beta_opt, true_beta, Z_lp, Z_opt)
    
    output(beta_lp, beta_opt, beta_ls, true_beta, rmse_lp, rmse_opt, rmse_ls, 
           resid_lp_norm, resid_opt_norm, resid_ls_norm)
    
    if rmse_lp < rmse_ls:
        count_lp += 1
    if rmse_opt < rmse_ls:
        count_opt += 1
        
    if rmse_opt < rmse_lp:
        print(f"\nOptimization IMPROVED RMSE by {((rmse_lp - rmse_opt) / rmse_lp * 100):.2f}%")
    else:
        print(f"\nOptimization DID NOT IMPROVE RMSE (change: {((rmse_lp - rmse_opt) / rmse_lp * 100):.2f}%)")

    # Return detailed results for plotting
    results = {
        'rmse_lp': rmse_lp,
        'rmse_opt': rmse_opt,
        'rmse_ls': rmse_ls,
        'Y': Y,
        'Xint': Xint,
        'beta_ls': beta_ls,
        'beta_lp': beta_lp,
        'beta_opt': beta_opt,
        'Z_lp': Z_lp,
        'Z_opt': Z_opt,
        'true_beta': true_beta
    }
    return count_lp, count_opt, results


def main():
    data_folder = "normalized_data"
    test_folder = "synthetic_spectra"
    true_beta_folder = "synthetic_spectra"
    
    count_lp = 0      
    count_opt = 0     
    count_opt_vs_lp = 0  
    
    # Lists to store RMSE values for each of the 50 samples
    rmse_lp_list = []
    rmse_ls_list = []
    
    # Process test cases (for summary printout)
    for i in range(1, 51):
        print(f"\n\n========== Processing Test Case {i} ==========")
        test_file = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"
        count_lp, count_opt, test_results = test(data_folder, test_folder, test_file, true_beta_folder, true_beta_file, count_lp, count_opt)
        
        rmse_lp_list.append(test_results['rmse_lp'])
        rmse_ls_list.append(test_results['rmse_ls'])
        
        if test_results['rmse_opt'] < test_results['rmse_lp']:
            count_opt_vs_lp += 1
    
    print(f"\n========== Final Results ==========")
    print(f"Low-pass Baseline Filtering outperformed Least Squares in {count_lp} out of 50 cases.")
    print(f"Optimized Low-pass method outperformed Least Squares in {count_opt} out of 50 cases.")
    print(f"Optimized Low-pass method outperformed regular Low-pass in {count_opt_vs_lp} out of 50 cases.")
    
    if count_opt > count_lp:
        print(f"Post-optimization improved overall performance vs LS in {count_opt - count_lp} cases.")
    elif count_opt < count_lp:
        print(f"Post-optimization reduced overall performance vs LS in {count_lp - count_opt} cases.")
    else:
        print("Post-optimization did not change the number of wins vs Least Squares.")
    
    # Plot RMSE comparison across all 50 samples
    plot_rmse_comparison_all_samples(rmse_lp_list, rmse_ls_list)
    display_rmse_table(rmse_lp_list, rmse_ls_list)
    
    # For demonstration, use the results from the last test case for single spectrum plots and concentration table
    results = test_results
    Y = results['Y']
    Xint = results['Xint']
    beta_opt = results['beta_opt']
    beta_ls = results['beta_ls']
    beta_lp = results['beta_lp']
    true_beta = results['true_beta']
    
    # Generate ppm scale (assumes 120001 points spanning -2 to 10 ppm)
    ppm = np.linspace(10, -2, len(Y))
    
    # Generate Figures for a single spectrum
    plot_spectrum_fit_comparison(Y, Xint, beta_opt, results['Z_opt'], beta_ls, ppm)
    plot_peak_identification(Y, Xint, beta_opt, results['Z_opt'], beta_ls, ppm)
    plot_concentration_accuracy(true_beta, beta_opt, beta_ls)
    plot_residual_error(true_beta, beta_opt, beta_ls)
    plot_performance_metrics(results['rmse_opt'], results['rmse_ls'], true_beta, beta_opt, beta_ls)
    
    # Display summary tables (existing ones)
    display_summary_tables(true_beta, beta_opt, beta_ls, results['rmse_opt'], results['rmse_ls'])
    
    # Display the new concentration comparison table for the last spectrum (20 x 4)
    display_concentration_comparison_table(true_beta, beta_lp, beta_ls, beta_opt)

if __name__ == "__main__":
    main()
