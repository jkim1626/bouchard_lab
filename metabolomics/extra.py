import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import nnls

def butter_lowpass_filter(signal, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

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

def main():    
    b16 = "./B16_buffer.txt"
    raw = "./RAW_buffer.txt"
    blank_buffer = "./blank_buffer.txt"
    
    # Load data
    Xint = np.loadtxt(blank_buffer, delimiter=',').reshape(-1, 1) 
    Y_b16 = np.loadtxt(b16, delimiter=',')
    Y_raw = np.loadtxt(raw, delimiter=',')

    # Run each algorithm
    b_b16, Z_b16 = alternating_solver(Y_b16, Xint, cutoff_freq=0.0128, filter_order=2, max_iter=50, tol=1e-6)
    b_raw, Z_raw = alternating_solver(Y_raw, Xint, cutoff_freq=0.0128, filter_order=2, max_iter=50, tol=1e-6)

    # Save to txt file
    np.savetxt("Z_b16.txt", Z_b16.reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt("Z_raw.txt", Z_raw.reshape(1, -1), delimiter=',', fmt='%.10f')

    # Plot results for B16 buffer
    plt.figure(figsize=(12, 10))

    # Panel A: Original noisy spectrum for B16
    plt.subplot(3, 1, 1)
    plt.plot(Y_b16, label="Original Noisy Spectrum (Y)", color='k')
    plt.title("B16 Buffer: Original Noisy Spectrum (Y)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    # Panel B: Reconstructed spectrum for B16
    reconstructed_b16 = Xint @ b_b16
    plt.subplot(3, 1, 2)
    plt.plot(reconstructed_b16, label="Reconstructed Spectrum (Xint @ b)", color='b')
    plt.title("B16 Buffer: Reconstructed Spectrum (Xint @ b)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    # Panel C: Leftover baseline for B16
    plt.subplot(3, 1, 3)
    plt.plot(Z_b16, label="Leftover Baseline (Z)", color='r')
    plt.title("B16 Buffer: Leftover Baseline (Z)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot results for RAW buffer
    plt.figure(figsize=(12, 10))

    # Panel A: Original noisy spectrum for RAW
    plt.subplot(3, 1, 1)
    plt.plot(Y_raw, label="Original Noisy Spectrum (Y)", color='k')
    plt.title("RAW Buffer: Original Noisy Spectrum (Y)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    # Panel B: Reconstructed spectrum for RAW
    reconstructed_raw = Xint @ b_raw
    plt.subplot(3, 1, 2)
    plt.plot(reconstructed_raw, label="Reconstructed Spectrum (Xint @ b)", color='b')
    plt.title("RAW Buffer: Reconstructed Spectrum (Xint @ b)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    # Panel C: Leftover baseline for RAW
    plt.subplot(3, 1, 3)
    plt.plot(Z_raw, label="Leftover Baseline (Z)", color='r')
    plt.title("RAW Buffer: Leftover Baseline (Z)")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()