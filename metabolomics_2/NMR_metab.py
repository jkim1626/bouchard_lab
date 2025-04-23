# DICTIONARY MADE UP OF ALL 5 SPECTRA

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from pathlib import Path

# Get the desktop path
DESKTOP_PATH = str(Path.home() / "Desktop")
FIGURES_PATH = os.path.join(DESKTOP_PATH, "metab_samples_3")

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
def alternating_solver(
    Y, X, cutoff_freq=0.0128, filter_order=2,
    alpha_lasso=0.01, sparse_beta=False,
    max_iter=50, tol=1e-6
):
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
            beta_new,_ = nnls(X, R)

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
            break

    return beta, Z

# Load metabolite files to create Xint dictionary
def load_Xint(num_points, lipid_files):    
    if not lipid_files:
        raise ValueError("No lipid files provided.")
    
    Xint = np.zeros((num_points, len(lipid_files)))
    for i, file_path in enumerate(lipid_files):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Lipid file not found: {file_path}")
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

# Plotting Functions for a Single Spectrum
def plot_spectrum_fit_comparison(Y, Xint, beta_lp, Z_lp, beta_ls, ppm, num):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp 
    recon_ls = Xint @ beta_ls

    # Spectrum overlay
    plt.figure(figsize=(14, 10))
    plt.plot(ppm, Y, 'k', label='EV Sample')
    plt.plot(ppm, recon_ls, 'r', label='Least Squares', alpha=0.5)
    plt.plot(ppm, recon_lp, 'c', label='Low Pass', alpha=0.8)
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum Overlay for Sample {num}")
    plt.legend()
    
    # Save the figure
    save_figure(f"Sample_{num}_overlay")
    print(f"Figure saved as Sample_{num}_overlay.pdf")

    # plt.show()

# Plotting Functions for Peak Identification
def plot_peak_identification(Y, Xint, beta_lp, Z_lp, beta_ls, ppm, idx):
    # Reconstructed spectra
    recon_lp = Xint @ beta_lp 
    recon_ls = Xint @ beta_ls 

    # Find peaks in a zoomed-in region from 2 to 6 ppm.
    idx_zoom = np.where((ppm >= 2) & (ppm <= 6))[0]
    ppm_zoom = ppm[idx_zoom]
    Y_zoom = Y[idx_zoom]
    recon_lp_zoom = recon_lp[idx_zoom]
    recon_ls_zoom = recon_ls[idx_zoom]

    plt.figure(figsize=(12, 8))
    plt.plot(ppm_zoom, Y_zoom, 'k-', label='EV Sample')
    plt.plot(ppm_zoom, recon_ls_zoom, 'r-', label='Least Squares', alpha=0.5)
    plt.plot(ppm_zoom, recon_lp_zoom, 'c-', label='Low Pass')

    
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title("Peak Overlay in Zoomed Region (2-6 ppm)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    
    # Save the figure
    save_figure(f"Sample_{idx}_overlay_zoomed")
    print(f"Figure saved as Sample_{idx}_overlay_zoomed.pdf")
    
    # plt.show()

# Main test function 
def test(Xint, num_points, test_folder, test_file):        
    # Load Experimental Data
    Y_path = os.path.join(test_folder, test_file)
    try:
        Y = load_Y(num_points, Y_path)
    except FileNotFoundError as e:
        print(f"Error: Test file not found: {Y_path}")
        return None

    # Run each algorithm
    beta_lp, Z_lp = alternating_solver(
        Y, Xint, cutoff_freq=0.0128, filter_order=2, alpha_lasso=0.01,
        sparse_beta=False, max_iter=200, tol=1e-8)
    beta_ls,_ = nnls(Xint, Y)

    # Return detailed results for plotting
    results = {
        'Y': Y,
        'Xint': Xint,
        'beta_ls': beta_ls,
        'beta_lp': beta_lp,
        'Z_lp': Z_lp,
    }

    return results

def main():    
    data_folder = "dict"
    test_folder = "test"
    
    # Check if required folders exist
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        os.makedirs(data_folder, exist_ok=True)
        
    if not os.path.exists(test_folder):
        print(f"Error: Test folder not found: {test_folder}")
        os.makedirs(test_folder, exist_ok=True)
        
    # Pre-defined dimension of metabolites and experimental spectra
    num_points = 7163
    num_lipids = 5

    # Chemical shift axis (7163 points from 10 ppm to -2 ppm)
    ppm = np.linspace(10, -2, num_points)

    # Load Metabolite Dictionary
    metabolite_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, num_lipids + 1)]
    Xint = load_Xint(num_points, metabolite_files)

    # Number of synthetic spectra to run script on 
    num_files = 3
    
    # Process test cases (for summary printout)
    for i in range(1, num_files + 1):
        print(f"\n\n========== Processing Test Case {i} ==========")
        test_file = f"{i}.txt"

        # Check if test files exist
        if not os.path.exists(os.path.join(test_folder, test_file)):
            raise FileNotFoundError(f"Test file not found: {os.path.join(test_folder, test_file)}")

        test_results = test(Xint, num_points, test_folder, test_file)
            
        
        # Generate Figures for a single spectrum
        plot_spectrum_fit_comparison(
            test_results['Y'], test_results['Xint'], 
            test_results['beta_lp'], test_results['Z_lp'], 
            test_results['beta_ls'], ppm, i)
        
        plot_peak_identification(
            test_results['Y'], test_results['Xint'], 
            test_results['beta_lp'], test_results['Z_lp'], 
            test_results['beta_ls'], ppm, i)
        
        # Print beta coefficients from algorithm
        print(test_results['beta_lp'])
        print(test_results['beta_ls'])

if __name__ == "__main__":
    main()