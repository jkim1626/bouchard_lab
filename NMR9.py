import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import nnls
from pathlib import Path

# Get the desktop path
DESKTOP_PATH = str(Path.home() / "Desktop")
FIGURES_PATH = os.path.join(DESKTOP_PATH, "nmr_figures")

# Create the figures directory if it doesn't exist
os.makedirs(FIGURES_PATH, exist_ok=True)

# Function to save figures
def save_figure(name):
    filename = os.path.join(FIGURES_PATH, f"{name}.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')

# Low-Pass Baseline Filtering
def butter_lowpass_filter(signal, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

# Alternating Solver: (Beta-step) + (Z-step)
def alternating_solver(Y, X, cutoff_freq=0.01, filter_order=2, max_iter=50, tol=1e-6):
    Y = Y.flatten()
    m, n = X.shape

    if len(Y) != m:
        raise ValueError(f"Dimension mismatch: Y has length {len(Y)}, X has {m} rows")

    beta = np.zeros(n)
    Z = np.zeros(m)

    if not 0 < cutoff_freq < 1:
        raise ValueError(f"Cutoff frequency must be between 0 and 1. Got {cutoff_freq}.")
    if filter_order < 1:
        raise ValueError(f"Filter order must be at least 1. Got {filter_order}.")
    if max_iter < 1:
        raise ValueError(f"Max iterations must be at least 1. Got {max_iter}.")

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
    if not metabolite_files:
        raise ValueError("No metabolite files provided.")
    
    Xint = np.zeros((num_points, len(metabolite_files)))
    for i, file_path in enumerate(metabolite_files):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metabolite file not found: {file_path}")
        data = np.loadtxt(file_path, delimiter=',')
        if data.shape[0] != num_points:
            raise ValueError(f"Dimension mismatch: {file_path} has {data.shape[0]} points, expected {num_points}")
        Xint[:, i] = data
    return Xint

# Load experimental file into Y
def load_Y(num_points, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Experimental data file not found: {file_path}")

    Y_noisy = np.loadtxt(file_path, delimiter=',')
    if len(Y_noisy) != num_points: 
        raise ValueError(f"Experimental data has length {len(Y_noisy)}, but Xint expects length {num_points}.")
    return Y_noisy

# Calculate the RMSE and residual scores of the algorithms
def calc_scores(Y, Xint, beta_ls, beta_lp, Z_lp, true_beta):
    def rmse(a, b):
        if len(a) != len(b):
            raise ValueError(f"Arrays must have same length. Got {len(a)} and {len(b)}.")
        return np.sqrt(np.mean((a - b)**2))
    
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
    print(f"LP RMSE : {rmse_lp}")
    print(f"LS RMSE : {rmse_ls}")

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
    if len(all_true_values) != len(all_predicted_values):
        raise ValueError(f"Arrays must have same length. Got {len(all_true_values)} and {len(all_predicted_values)}.")

    # Flatten all samples into single arrays
    true_flat = np.concatenate(all_true_values)
    pred_flat = np.concatenate(all_predicted_values)
    
    r = calculate_r_squared(true_flat, pred_flat) 
    return r**2

# Plotting Functions for a Single Spectrum
def plot_spectrum_fit_comparison(Y, Xint, beta_lp, Z_lp, beta_ls, ppm, num_files):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp + Z_lp
    recon_ls = Xint @ beta_ls

    # Compute residuals
    resid_lp = Y - recon_lp
    resid_ls = Y - recon_ls

    # Figure 1: Overlay and Residuals
    plt.figure(figsize=(14, 10))

    # Panel A: Spectrum overlay
    plt.subplot(2, 1, 1)
    plt.plot(ppm, Y, 'k', label='Simulated Spectrum')
    plt.plot(ppm, recon_lp, 'c', label='Low Pass + Baseline', alpha=0.8)
    plt.plot(ppm, recon_ls, 'r', label='Least Squares', alpha=0.5)
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum Overlay for Sample {num_files}")
    plt.legend()

    # Panel B: Residuals
    plt.subplot(2, 1, 2)
    plt.plot(ppm, resid_lp, 'c', label='Residual (Low Pass + Baseline)', alpha=0.8)
    plt.plot(ppm, resid_ls, 'r', label='Residual (LS)', alpha=0.5)
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Residual Intensity")
    plt.title("Residuals")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    save_figure(f"spectrum_fit_comparison_sample_{num_files}")
    
    plt.show()

# Plotting Functions for Peak Identification
def plot_peak_identification(Y, Xint, beta_lp, Z_lp, beta_ls, ppm):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp + Z_lp
    recon_ls = Xint @ beta_ls 

    # For demonstration, we use the simulated spectrum Y as the "true" spectrum.
    # Find peaks in a zoomed-in region.
    # Let's choose a zoom region from 4 to 6 ppm.
    idx_zoom = np.where((ppm >= 2) & (ppm <= 4))[0]
    ppm_zoom = ppm[idx_zoom]
    Y_zoom = Y[idx_zoom]
    recon_lp_zoom = recon_lp[idx_zoom]
    recon_ls_zoom = recon_ls[idx_zoom]

    plt.figure(figsize=(12, 8))
    plt.plot(ppm_zoom, Y_zoom, 'k-', label='Simulated Spectrum')
    plt.plot(ppm_zoom, recon_lp_zoom, 'c-', label='Low Pass + Baseline')
    plt.plot(ppm_zoom, recon_ls_zoom, 'r-', label='Least Squares', alpha=0.5)
    
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Peak Identification in Zoomed Region (2-4 ppm)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    
    # Save the figure
    save_figure("peak_identification_zoomed")
    
    plt.show()

# Plotting Functions for Concentration Accuracy
def plot_concentration_accuracy(true_beta, beta_lp, beta_ls, num_files):
    # Scatter plot for predicted vs. true concentrations.
    # Compute R^2 values
    R2_lp = calculate_r_squared(true_beta, beta_lp)
    R2_ls = calculate_r_squared(true_beta, beta_ls)

    plt.figure(figsize=(10, 8))
    plt.plot(true_beta, true_beta, 'k--', label="Identity Line")
    plt.scatter(true_beta, beta_lp, c='c', label=f"Alternating Solver (R²={R2_lp:.2f})", alpha=0.8)
    plt.scatter(true_beta, beta_ls, c='r', label=f"LS (R²={R2_ls:.2f})", alpha = 0.5)
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title(f"Concentration Prediction Accuracy for Sample {num_files}")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    save_figure(f"concentration_accuracy_sample_{num_files}")
    
    plt.show()

# Plotting Functions for Residual/Error per Metabolite
def plot_residual_error(true_beta, beta_lp, beta_ls, num_files):
    # Compute absolute errors per metabolite
    error_lp = np.abs(true_beta - beta_lp)
    error_ls = np.abs(true_beta - beta_ls)
    metabolites = np.arange(1, len(true_beta)+1)

    width = 0.35  # width of the bars
    plt.figure(figsize=(12, 8))
    plt.bar(metabolites - width/2, error_lp, width, color='c', label="Alternating Solver", alpha=0.8)
    plt.bar(metabolites + width/2, error_ls, width, color='r', label="LS", alpha=0.5)
    plt.xlabel("Metabolite ID")
    plt.ylabel("Absolute Error")
    plt.title(f"Residual/Error per Metabolite for Sample {num_files}")
    plt.xticks(metabolites)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    save_figure(f"residual_error_sample_{num_files}")
    
    plt.show()

# Plotting Functions for Performance Metrics
def plot_performance_metrics(rmse_lp, rmse_ls, true_beta, beta_lp, beta_ls, num_files):
    # Compute R^2 values
    R2_lp = calculate_r_squared(true_beta, beta_lp)
    R2_ls = calculate_r_squared(true_beta, beta_ls)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: RMSE Comparison
    methods = ["Alternating Solver", "Least Squares"]
    rmse_values = [rmse_lp, rmse_ls]
    
    axs[0].bar(methods, rmse_values, color=['c', (0.8, 0, 0, 0.5)])
    axs[0].set_ylabel("RMSE (Intensity Units)")
    axs[0].set_title(f"RMSE Comparison for Sample {num_files}")
    
    # Panel B: R^2 Comparison
    r2_values = [R2_lp, R2_ls]
    axs[1].bar(methods, r2_values, color=['c', (0.8, 0, 0, 0.5)])
    axs[1].set_ylabel("R²")
    axs[1].set_title(f"R² Comparison for Sample {num_files}")
    
    plt.tight_layout()
    
    # Save the figure
    save_figure(f"performance_metrics_sample_{num_files}")
    
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
    if not rmse_lp_list or not rmse_ls_list:
        print("Error: Empty RMSE lists provided")
        return
        
    if len(rmse_lp_list) != len(rmse_ls_list):
        print(f"Warning: RMSE lists have different lengths ({len(rmse_lp_list)} vs {len(rmse_ls_list)})")
    
    samples = np.arange(1, len(rmse_lp_list)+1)
    plt.figure(figsize=(12, 8))
    plt.plot(samples, rmse_lp_list, marker='o', linestyle='-', color='c', label='Alternating Solver', alpha=0.8)
    plt.plot(samples, rmse_ls_list, marker='s', linestyle='-', color='r', label='Least Squares', alpha=0.5)
    plt.xlabel("Sample")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison Across Samples")
    plt.xlim(0, len(rmse_lp_list)+1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_figure("rmse_comparison_all_samples")
    
    plt.show()

# Plot R^2 comparison across all samples
def plot_r2_comparison_all_samples(r2_lp_list, r2_ls_list):    
    if not r2_lp_list or not r2_ls_list:
        print("Error: Empty R² lists provided")
        return
        
    if len(r2_lp_list) != len(r2_ls_list):
        print(f"Warning: R² lists have different lengths ({len(r2_lp_list)} vs {len(r2_ls_list)})")
    
    samples = np.arange(1, len(r2_lp_list) + 1)
    
    # Calculate running averages
    running_avg_lp = np.zeros(len(r2_lp_list))
    running_avg_ls = np.zeros(len(r2_ls_list))
    
    # Calculate the running average at each point
    for i in range(len(r2_lp_list)):
        running_avg_lp[i] = np.mean(r2_lp_list[:i+1])
        running_avg_ls[i] = np.mean(r2_ls_list[:i+1])
    
    plt.figure(figsize=(12, 8))
    
    # Plot running averages
    plt.plot(samples, running_avg_lp, marker='o', linestyle='-', color='c', 
             label='Alternating Solver (Running Avg)', alpha=0.8)
    plt.plot(samples, running_avg_ls, marker='s', linestyle='-', color='r', 
             label='Least Squares (Running Avg)', alpha=0.5)
    
    # Add horizontal lines for final average values
    plt.axhline(y=running_avg_lp[-1], color='c', linestyle='--', 
                label=f'Final Avg AS: {running_avg_lp[-1]:.3f}', alpha=0.5)
    plt.axhline(y=running_avg_ls[-1], color='r', linestyle='--', 
                label=f'Final Avg LS: {running_avg_ls[-1]:.3f}', alpha=0.3)
    
    plt.xlabel("Sample")
    plt.ylabel("Running Average R²")
    plt.title("Running Average R² Comparison Across Samples")
    plt.xlim(0, len(r2_lp_list) + 1)
    plt.ylim(0, 1.05)  # R² is between 0 and 1
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_figure("r2_running_average_comparison")
    
    plt.show()

# Display a RMSE Table for each sample 
def display_rmse_table(rmse_lp_list, rmse_ls_list):
    if not rmse_lp_list or not rmse_ls_list:
        print("Error: Empty RMSE lists provided")
        return
        
    if len(rmse_lp_list) != len(rmse_ls_list):
        print(f"Warning: RMSE lists have different lengths ({len(rmse_lp_list)} vs {len(rmse_ls_list)})")
    
    print(f"\nRMSE Table over {len(rmse_lp_list)} samples:")
    print("Sample \t Low Pass RMSE \t Least Squares RMSE")
    for i in range(len(rmse_lp_list)):
        print(f"{i+1} \t {rmse_lp_list[i]:.3f} \t\t {rmse_ls_list[i]:.3f}")

# Display Concentration Comparison Table
def display_concentration_comparison_table(true_beta, beta_lp, beta_ls, num_files):
    if len(true_beta) != len(beta_lp) or len(true_beta) != len(beta_ls):
        print(f"Warning: Beta arrays have different lengths: true={len(true_beta)}, lp={len(beta_lp)}, ls={len(beta_ls)}")
    
    num_metabolites = len(true_beta)
    print(f"\nConcentration Comparison Table for sample {num_files}:")
    print(f"Metabolite \t True Concentration \t Alternating Solver \t Least Squares")
    for i in range(num_metabolites):
        print(f"{i+1} \t\t {true_beta[i]:.3f} \t\t\t {beta_lp[i]:.3f} \t\t\t {beta_ls[i]:.3f}")

# Main test function 
def test(Xint, num_points, test_folder, test_file, true_beta_folder, true_beta_file, count_lp_vs_ls):        
    # Load Experimental Data
    Y_path = os.path.join(test_folder, test_file)
    try:
        Y = load_Y(num_points, Y_path)
    except FileNotFoundError as e:
        print(f"Error: Test file not found: {Y_path}")
        return count_lp_vs_ls, None

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
    
    # Check if required folders exist
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        os.makedirs(data_folder, exist_ok=True)
        
    if not os.path.exists(test_folder):
        print(f"Error: Test folder not found: {test_folder}")
        os.makedirs(test_folder, exist_ok=True)
        
    if not os.path.exists(true_beta_folder):
        print(f"Error: True beta folder not found: {true_beta_folder}")
        os.makedirs(true_beta_folder, exist_ok=True)
    
    # Count sampels that alternating solver outperforms least squares
    count_lp_vs_ls = 0      
    
    # Lists to store RMSE values for each sample
    rmse_lp_list = []
    rmse_ls_list = []
    
    # Lists to store beta values for each sample
    all_true_betas = []
    all_beta_lps = []
    all_beta_ls = []

    # Lists to store R^2 values for all samples
    r2_lp_list = []
    r2_ls_list = []
    
    # Pre-defined dimension of metabolites and experimental spectra
    num_points = 120001
    num_metabolites = 20

    # Load Metabolite Dictionary
    metabolite_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1,num_metabolites+1)]
    Xint = load_Xint(num_points, metabolite_files)

    # Number of synthetic spectra to run script on 
    num_files = 50

    # Process test cases (for summary printout)
    for i in range(1, num_files + 1):
        print(f"\n\n========== Processing Test Case {i} ==========")
        test_file = f"synthetic_spectrum_{i}.txt"
        true_beta_file = f"spectrum_ratios_{i}.txt"

        # Check if test files exist
        if not os.path.exists(os.path.join(test_folder, test_file)):
            raise FileNotFoundError(f"Test file not found: {os.path.join(test_folder, test_file)}")

        if not os.path.exists(os.path.join(true_beta_folder, true_beta_file)):
            raise FileNotFoundError(f"True beta file not found: {os.path.join(true_beta_folder, true_beta_file)}")

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
    
        R2_lp = calculate_r_squared(test_results['true_beta'], test_results['beta_lp'])
        R2_ls = calculate_r_squared(test_results['true_beta'], test_results['beta_ls'])

        # Append R^2 values to the lists
        r2_lp_list.append(R2_lp)
        r2_ls_list.append(R2_ls)

    # Print final results
    print(f"\n========== Final Results ==========")
    print(f"Alternating Solver outperformed Least Squares in {(count_lp_vs_ls/num_files) * 100:.1f}% of cases.")

    # Plot RMSE comparison across all samples
    plot_rmse_comparison_all_samples(rmse_lp_list, rmse_ls_list)
    plot_r2_comparison_all_samples(r2_lp_list, r2_ls_list)
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
    plot_spectrum_fit_comparison(Y, Xint, beta_lp, results['Z_lp'], beta_ls, ppm, num_files)
    plot_peak_identification(Y, Xint, beta_lp, results['Z_lp'], beta_ls, ppm)
    plot_concentration_accuracy(true_beta, beta_lp, beta_ls, num_files)
    plot_residual_error(true_beta, beta_lp, beta_ls, num_files)
    plot_performance_metrics(results['rmse_lp'], results['rmse_ls'], true_beta, beta_lp, beta_ls, num_files)
    
    # Display summary tables with global R² values across all samples
    display_summary_tables(num_files, true_beta, beta_lp, beta_ls, 
                          results['rmse_lp'], results['rmse_ls'],
                          rmse_lp_list, rmse_ls_list,
                          all_true_betas, all_beta_lps, all_beta_ls)
    
    # Display the concentration comparison table for the last spectrum
    display_concentration_comparison_table(true_beta, beta_lp, beta_ls, num_files)


if __name__ == "__main__":
    main()