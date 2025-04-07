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

# Load experimental file for Y
def load_Y(num_points, file_path):
    Y_noisy = np.loadtxt(file_path, delimiter=',')
    if len(Y_noisy) != num_points: 
        raise ValueError(f"Experimental data has length {len(Y_noisy)}, but Xint expects length {num_points}.")
    return Y_noisy

# Calculate RMSE between true beta and estimated beta
def calc_rmse(true_beta, beta):
    return np.sqrt(np.mean((true_beta - beta)**2))

# Main test function 
def test(Xint, data_folder, test_folder, test_file, true_beta_folder, true_beta_file, i, j):
    num_points = 120001
    
    # Load Experimental Data
    Y_path = os.path.join(test_folder, test_file)
    Y = load_Y(num_points, Y_path)

    # Load True Beta Values
    beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = np.loadtxt(beta_path, delimiter=',')     

    # Run Alternating Solver and Naive LS approach
    beta_lp,_ = alternating_solver(Y, Xint, cutoff_freq=i, filter_order=j, max_iter=50, tol=1e-6)

    score = calc_rmse(beta_lp, true_beta)

    return score


def main():
    data_folder = "normalized_data"
    test_folder = "synthetic_spectrum"
    true_beta_folder = "synthetic_spectrum"

    num_points = 120001  
    # Load Metabolite Dictionary
    metabolite_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1,21)]
    Xint = load_Xint(num_points, metabolite_files)

    scores = {}

    # Process test cases (for summary printout)
    for j in np.arange(0.010, 0.015, 0.001):
        for k in range(2,3):
            for i in range(1, 26):
                test_file = f"synthetic_spectrum_{i}.txt"
                true_beta_file = f"spectrum_ratios_{i}.txt"
                score = test(Xint, data_folder, test_folder, test_file, true_beta_folder, true_beta_file, j, k)
                scores[(j,k)] = score
            print(f"Cutoff Frequency: {j}, Filter Order: {k}, Score: {score}")

    # print scores in descending order of score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\n\n========== Sorted Scores ==========")
    for (j, k), score in sorted_scores:
        print(f"Cutoff Frequency: {j}, Filter Order: {k}, Score: {score}")




if __name__ == "__main__":
    main()
