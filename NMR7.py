import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

# Low-Pass Baseline Filtering
def butter_lowpass_filter(signal, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

# Alternating Solver: (Beta-step) + (Z-step)
def alternating_solver(Y, X, cutoff_freq=0.01, filter_order=4, max_iter=50, tol=1e-6):
    Y = Y.flatten()
    m, n = X.shape
    beta = np.zeros(n)
    Z = np.zeros(m)

    for iteration in range(1, max_iter + 1):
        # (1) Beta-step
        R = Y - Z  # Residual ignoring current baseline
        beta_new, _ = nnls(X, R)

        # (2) Z-step 
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
    return Xint

# Load experimental file into Y
def load_Y(num_points, file_path):
    Y_noisy = np.loadtxt(file_path, delimiter=',')
    if len(Y_noisy) != num_points: 
        raise ValueError(f"Experimental data has length {len(Y_noisy)}, but Xint expects length {num_points}.")
    return Y_noisy

# Calculate the RMSE and residual scores of the algorithms
def calc_scores(Y, Xint, beta_ls, beta_lp, Z_lp, true_beta):
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(true_beta, beta_lp)
    rmse_ls = rmse(true_beta, beta_ls)

    def resid(Y, X, beta, Z=None):
        if Z is None:
            return np.linalg.norm(Y - (X @ beta))
        else:
            return np.linalg.norm(Y - (X @ beta + Z))
            
    resid_lp_norm = resid(Y, Xint, beta_lp, Z_lp)
    resid_ls_norm = resid(Y, Xint, beta_ls)

    return rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm

# Helper function to output results
def output(beta_lp, beta_ls, true_beta, rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm):
    print("\n====== Final Results =====")
    print(f"beta_lp (Alternating Solver) vs beta_ls (Unconstrained LS):")
    col_width = 15
    print(f"{'beta_lp':<{col_width}} {'beta_ls':<{col_width}} {'true_beta':<{col_width}}")
    for beta_lp_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.6f} {beta_ls_val:<{col_width}.6f} {true_beta_val:<{col_width}.6f}")

    # Print RMSE values
    print("\n--- RMSE Values ---")
    print(f"LP RMSE: {rmse_lp}")
    print(f"LS RMSE: {rmse_ls}")

    # Print Residual Norms
    print(f"\n--- Residual Norms wrt Y ---")
    print(f"LP vs Y : {resid_lp_norm:.25f}")
    print(f"LS vs Y : {resid_ls_norm:.25f}")

# Calculate R^2 value
def calculate_r_squared(true_values, predicted_values):
    r = np.corrcoef(true_values, predicted_values)[0, 1]
    return r**2

# Calculate global R^2 value across all samples
def calculate_global_r_squared(all_true_values, all_predicted_values):
    # Flatten all samples into single arrays
    true_flat = np.concatenate(all_true_values)
    pred_flat = np.concatenate(all_predicted_values)
    
    r = calculate_r_squared(true_flat, pred_flat) 
    return r**2

# Plotting Functions for a Single Spectrum
def plot_spectrum_fit_comparison(Y, Xint, beta_lp, Z_lp, beta_ls, ppm):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp + Z_lp
    recon_ls = Xint @ beta_ls

    # Compute residuals
    resid_lp = Y - recon_lp
    resid_ls = Y - recon_ls

    # Figure 1: Overlay and Residuals
    plt.figure(figsize=(14, 12))

    # Panel A: Spectrum overlay
    plt.subplot(2, 1, 1)
    plt.plot(ppm, Y, 'k', label='Simulated Spectrum')
    plt.plot(ppm, recon_lp, 'b', label='Low Pass + Baseline')
    plt.plot(ppm, recon_ls, 'r', label='Least Squares')
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Figure 1A: Spectrum Fit Comparison")
    plt.legend()

    # Panel B: Residuals
    plt.subplot(2, 1, 2)
    plt.plot(ppm, resid_lp, 'b', label='Residual (Low Pass + Baseline)')
    plt.plot(ppm, resid_ls, 'r', label='Residual (LS)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Residual Intensity")
    plt.title("Figure 1B: Residuals")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting Functions for Peak Identification
def plot_peak_identification(Y, Xint, beta_lp, Z_lp, beta_ls, ppm):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp + Z_lp
    recon_ls = Xint @ beta_ls

    # For demonstration, we use the simulated spectrum Y as the "true" spectrum.
    # Find peaks in a zoomed-in region.
    # Let's choose a zoom region from 4 to 6 ppm.
    idx_zoom = np.where((ppm >= 4) & (ppm <= 6))[0]
    ppm_zoom = ppm[idx_zoom]
    Y_zoom = Y[idx_zoom]
    recon_lp_zoom = recon_lp[idx_zoom]
    recon_ls_zoom = recon_ls[idx_zoom]

    plt.figure(figsize=(12, 8))
    plt.plot(ppm_zoom, Y_zoom, 'k-', label='Simulated Spectrum')
    plt.plot(ppm_zoom, recon_lp_zoom, 'b-', label='Low Pass + Baseline')
    plt.plot(ppm_zoom, recon_ls_zoom, 'r-', label='Least Squares')
    
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Figure 2: Peak Identification in Zoomed Region (4-6 ppm)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()

# Plotting Functions for Concentration Accuracy
def plot_concentration_accuracy(true_beta, beta_lp, beta_ls):
    # Scatter plot for predicted vs. true concentrations.
    # Compute R^2 values
    R2_lp = calculate_r_squared(true_beta, beta_lp)
    R2_ls = calculate_r_squared(true_beta, beta_ls)

    plt.figure(figsize=(10, 8))
    plt.plot(true_beta, true_beta, 'k--', label="Identity Line")
    plt.scatter(true_beta, beta_lp, c='b', label=f"Alternating Solver (R²={R2_lp:.2f})")
    plt.scatter(true_beta, beta_ls, c='r', label=f"LS (R²={R2_ls:.2f})")
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title("Figure 3: Concentration Prediction Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting Functions for Residual/Error per Metabolite
def plot_residual_error(true_beta, beta_lp, beta_ls):
    # Compute absolute errors per metabolite
    error_lp = np.abs(true_beta - beta_lp)
    error_ls = np.abs(true_beta - beta_ls)
    metabolites = np.arange(1, len(true_beta)+1)

    width = 0.35  # width of the bars
    plt.figure(figsize=(12, 8))
    plt.bar(metabolites - width/2, error_lp, width, color='b', label="Alternating Solver")
    plt.bar(metabolites + width/2, error_ls, width, color='r', label="LS")
    plt.xlabel("Metabolite ID")
    plt.ylabel("Absolute Error")
    plt.title("Figure 4: Residual/Error per Metabolite")
    plt.xticks(metabolites)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting Functions for Performance Metrics
def plot_performance_metrics(rmse_lp, rmse_ls, true_beta, beta_lp, beta_ls):
    # Compute R^2 values
    R2_lp = calculate_r_squared(true_beta, beta_lp)
    R2_ls = calculate_r_squared(true_beta, beta_ls)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: RMSE Comparison
    methods = ["Alternating Solver", "Least Squares"]
    rmse_values = [rmse_lp, rmse_ls]
    colors = ['b', 'r']
    
    axs[0].bar(methods, rmse_values, color=colors)
    axs[0].set_ylabel("RMSE (Intensity Units)")
    axs[0].set_title("Figure 5A: RMSE Comparison")
    
    # Panel B: R² Comparison
    r2_values = [R2_lp, R2_ls]
    axs[1].bar(methods, r2_values, color=colors)
    axs[1].set_ylabel("R²")
    axs[1].set_title("Figure 5B: R² Comparison")
    
    plt.tight_layout()
    plt.show()

# Display summary tables with RMSE and R^2 values
def display_summary_tables(num_files, true_beta, beta_lp, beta_ls, rmse_lp, rmse_ls, 
                          rmse_lp_list=None, rmse_ls_list=None,
                          all_true_betas=None, all_beta_lps=None, all_beta_ls=None):
    # Table 1: Summary of Key Numerical Results
    
    # If we have lists of values from all samples, calculate averages
    if (rmse_lp_list is not None and rmse_ls_list is not None and
        all_true_betas is not None and all_beta_lps is not None and 
        all_beta_ls is not None):
        
        # Calculate average RMSE values across all samples
        avg_rmse_lp = np.mean(rmse_lp_list)
        avg_rmse_ls = np.mean(rmse_ls_list)
        
        # Calculate global R² values across all samples
        global_r2_lp = calculate_global_r_squared(all_true_betas, all_beta_lps)
        global_r2_ls = calculate_global_r_squared(all_true_betas, all_beta_ls)
        
        # Use the average values for the summary table
        lp_values = [avg_rmse_lp, global_r2_lp]
        ls_values = [avg_rmse_ls, global_r2_ls]
    else:
        # Use the single sample values if we don't have data from all samples
        R2_lp = calculate_r_squared(true_beta, beta_lp)
        R2_ls = calculate_r_squared(true_beta, beta_ls)
        
        lp_values = [rmse_lp, R2_lp]
        ls_values = [rmse_ls, R2_ls]
    
    # Print the summary table
    print(f"\nSummary of Key Numerical Results over {num_files} samples:")
    
    # Define column widths for formatting
    metric_width = 15
    value_width = 20
    
    # Print header
    header = f"{'Metric':<{metric_width}}{'Alternating Solver':<{value_width}}{'Least Squares':<{value_width}}"
    print(header)
    
    # Print rows
    metrics = ["RMSE", "R²"]
    for i, metric in enumerate(metrics):
        print(f"{metric:<{metric_width}}{lp_values[i]:<{value_width}.6f}{ls_values[i]:<{value_width}.6f}")
    
# Plot RMSE comparison across all samples
def plot_rmse_comparison_all_samples(rmse_lp_list, rmse_ls_list):
    samples = np.arange(1, len(rmse_lp_list)+1)
    plt.figure(figsize=(12, 8))
    plt.plot(samples, rmse_lp_list, marker='o', linestyle='-', color='blue', label='Alternating Solver')
    plt.plot(samples, rmse_ls_list, marker='s', linestyle='-', color='red', label='Least Squares')
    plt.xlabel("Sample")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison Across Samples")
    plt.xlim(0, len(rmse_lp_list)+1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Display a RMSE Table for each sample 
def display_rmse_table(rmse_lp_list, rmse_ls_list):
    print("\nRMSE Table:")
    print("Sample \t Low Pass RMSE \t Least Squares RMSE")
    for i in range(len(rmse_lp_list)):
        print(f"{i+1} \t {rmse_lp_list[i]:.3f} \t\t {rmse_ls_list[i]:.3f}")

# Display Concentration Comparison Table
def display_concentration_comparison_table(true_beta, beta_lp, beta_ls, num_files):
    num_metabolites = len(true_beta)
    print(f"\nConcentration Comparison Table for sample {num_files}:")
    print(f"Metabolite \t True Concentration \t Alternating Solver \t Least Squares")
    for i in range(num_metabolites):
        print(f"{i+1} \t\t {true_beta[i]:.3f} \t\t\t {beta_lp[i]:.3f} \t\t\t {beta_ls[i]:.3f}")

# Main test function 
def test(Xint, num_points, test_folder, test_file, true_beta_folder, true_beta_file, count_lp_vs_ls):    
    # Load Experimental Data
    Y_path = os.path.join(test_folder, test_file)
    Y = load_Y(num_points, Y_path)

    # Load True Beta Values
    beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = np.loadtxt(beta_path, delimiter=',')     

    # Run each algorithm
    beta_lp, Z_lp = alternating_solver(Y, Xint, cutoff_freq=0.0128, filter_order=2, max_iter=50, tol=1e-6)
    beta_ls, _ = nnls(Xint, Y)

    # Calculate scores for all methods
    rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm = calc_scores(
        Y, Xint, beta_ls, beta_lp, Z_lp, true_beta)
    
    # Output results
    output(beta_lp, beta_ls, true_beta, rmse_lp, rmse_ls, resid_lp_norm, resid_ls_norm)
    
    # Count wins for each method
    if rmse_lp < rmse_ls:
        count_lp_vs_ls += 1

    # Return detailed results for plotting
    results = {
        'rmse_lp': rmse_lp,
        'rmse_ls': rmse_ls,
        'Y': Y,
        'Xint': Xint,
        'beta_ls': beta_ls,
        'beta_lp': beta_lp,
        'Z_lp': Z_lp,
        'true_beta': true_beta
    }
    return count_lp_vs_ls, results

def main():
    data_folder = "normalized_data"
    test_folder = "synthetic_spectrum"
    true_beta_folder = "synthetic_spectrum"
    
    count_lp_vs_ls = 0      
    
    # Lists to store RMSE values for each sample
    rmse_lp_list = []
    rmse_ls_list = []
    
    # Lists to store beta values for each sample
    all_true_betas = []
    all_beta_lps = []
    all_beta_ls = []
    
    # Pre-defined dimension of metabolites and experimental spectra
    num_points = 120001

    # Load Metabolite Dictionary
    metabolite_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1,21)]
    Xint = load_Xint(num_points, metabolite_files)

    # Number of synthetic spectra to run script on 
    num_files = 5

    # Process test cases (for summary printout)
    for i in range(1, num_files + 1):
        print(f"\n\n========== Processing Test Case {i} ==========")
        test_file = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"
        count_lp_vs_ls, test_results = test(
            Xint, num_points, test_folder, test_file, true_beta_folder, true_beta_file, 
            count_lp_vs_ls)
        
        # Store RMSE values
        rmse_lp_list.append(test_results['rmse_lp'])
        rmse_ls_list.append(test_results['rmse_ls'])
        
        # Store beta values
        all_true_betas.append(test_results['true_beta'])
        all_beta_lps.append(test_results['beta_lp'])
        all_beta_ls.append(test_results['beta_ls'])
    
    # Print final results
    print(f"\n========== Final Results ==========")
    print(f"Alternating Solver outperformed Least Squares in {(count_lp_vs_ls/num_files) * 100}% of cases.")
    
    # Plot RMSE comparison across all samples
    plot_rmse_comparison_all_samples(rmse_lp_list, rmse_ls_list)
    display_rmse_table(rmse_lp_list, rmse_ls_list)
    
    # For demonstration, use the results from the last test case for single spectrum plots and concentration table
    results = test_results
    Y = results['Y']
    Xint = results['Xint']
    beta_lp = results['beta_lp']
    beta_ls = results['beta_ls']
    true_beta = results['true_beta']
    
    # Generate ppm scale (assumes 120001 points spanning -2 to 10 ppm)
    ppm = np.linspace(10, -2, len(Y))
    
    # Generate Figures for a single spectrum
    plot_spectrum_fit_comparison(Y, Xint, beta_lp, results['Z_lp'], beta_ls, ppm)
    plot_peak_identification(Y, Xint, beta_lp, results['Z_lp'], beta_ls, ppm)
    plot_concentration_accuracy(true_beta, beta_lp, beta_ls)
    plot_residual_error(true_beta, beta_lp, beta_ls)
    plot_performance_metrics(results['rmse_lp'], results['rmse_ls'], true_beta, beta_lp, beta_ls)
    
    # Display summary tables with global R² values across all samples
    display_summary_tables(num_files, true_beta, beta_lp, beta_ls, 
                          results['rmse_lp'], results['rmse_ls'],
                          rmse_lp_list, rmse_ls_list,
                          all_true_betas, all_beta_lps, all_beta_ls)
    
    # Display the concentration comparison table for the last spectrum
    display_concentration_comparison_table(true_beta, beta_lp, beta_ls, num_files)


if __name__ == "__main__":
    main()
