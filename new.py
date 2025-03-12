import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Lasso

###############################################################################
#                     Helper Functions & Signal Generation                    #
###############################################################################

def lorentzian(x, x0, gamma, amplitude=1.0):
    """
    Creates a single Lorentzian peak.
    L(x) = amplitude * ( (gamma/2)^2 / ( (x-x0)^2 + (gamma/2)^2 ) )
    """
    return amplitude * ((gamma/2.0)**2 / ((x - x0)**2 + (gamma/2.0)**2))

def generate_lorentzian_spectrum(num_points=1024, num_peaks=3, max_gamma=10.0):
    """
    Generate a random Lorentzian spectrum of length num_points.
    """
    x = np.arange(num_points, dtype=float)
    spectrum = np.zeros(num_points, dtype=float)
    for _ in range(num_peaks):
        center = np.random.uniform(0, num_points)
        gamma  = np.random.uniform(1.0, max_gamma)
        amp    = np.random.uniform(0.2, 1.0)
        spectrum += lorentzian(x, center, gamma, amplitude=amp)
    max_val = spectrum.max()
    if max_val > 0:
        spectrum /= max_val  # normalize to 1.0
    return spectrum

def butter_lowpass_filter(signal, cutoff_freq, order=4):
    """
    Applies a forward-backward Butterworth lowpass filter.
    """
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

###############################################################################
#                   Alternating Solver: Beta-Step and Z-Step                  #
###############################################################################
def solve_lowpass_with_sparse_beta(Y, X, cutoff_freq=0.01, filter_order=4,
                                   alpha_lasso=0.01, sparse_beta=False,
                                   max_iter=10, tol=1e-6):
    """
    Alternating optimization for: Y = X·beta + Z
      - Beta-step: Either standard LS or Lasso (for sparsity)
      - Z-step: Update baseline Z via low-pass filtering of (Y - X beta)
    """
    Y = Y.flatten()
    m, n = X.shape
    beta = np.zeros(n)
    Z    = np.zeros(m)

    for iteration in range(1, max_iter+1):
        beta_old = beta.copy()
        Z_old    = Z.copy()
        
        # (1) Beta-step: solve for beta with current baseline Z
        R = Y - Z  # residual ignoring baseline
        
        if sparse_beta:
            model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=5000, warm_start=True)
            model.fit(X, R)
            beta_new = model.coef_
        else:
            beta_new, _, _, _ = np.linalg.lstsq(X, R, rcond=None)
        
        # (2) Z-step: update baseline using low-pass filter on residual
        R2 = Y - X @ beta_new
        Z_new = butter_lowpass_filter(R2, cutoff_freq, order=filter_order)
        
        # Check convergence
        db = np.linalg.norm(beta_new - beta)
        dz = np.linalg.norm(Z_new - Z)
        beta = beta_new
        Z    = Z_new
        
        if db < tol and dz < tol:
            break

    return beta, Z

###############################################################################
#               Particle Swarm Optimization for Beta Refinement               #
###############################################################################
def objective(beta, Y, X, cutoff_freq, filter_order, alpha_lasso=0.0):
    """
    Objective function for PSO optimization.
    Given beta, compute baseline Z via low-pass filtering and return:
      f(beta) = || Y - X beta - Z ||^2 + alpha_lasso * ||beta||_1
    """
    R = Y - X @ beta
    Z = butter_lowpass_filter(R, cutoff_freq, order=filter_order)
    residual = Y - X @ beta - Z
    obj = np.linalg.norm(residual)**2 + alpha_lasso * np.sum(np.abs(beta))
    return obj

def pso_optimize_beta(Y, X, cutoff_freq, filter_order, alpha_lasso=0.0,
                      init_beta=None, num_particles=30, max_iter=50,
                      w=0.7, c1=1.5, c2=1.5, bounds=None):
    """
    A simple Particle Swarm Optimization (PSO) to refine beta.
    
    Arguments:
      Y, X, cutoff_freq, filter_order, alpha_lasso: parameters for the model.
      init_beta: initial guess (from the alternating solver)
      num_particles: number of particles in the swarm.
      max_iter: maximum number of iterations.
      w: inertia weight.
      c1: cognitive (particle) coefficient.
      c2: social (swarm) coefficient.
      bounds: tuple (lb, ub) for beta values (if None, use -10 to 10).
    Returns:
      best_beta: the beta vector with the lowest objective value found.
    """
    n = X.shape[1]
    if bounds is None:
        lb = -10 * np.ones(n)
        ub = 10 * np.ones(n)
    else:
        lb, ub = bounds

    # Initialize particles around init_beta if provided; else random initialization
    if init_beta is None:
        particles = np.random.uniform(lb, ub, (num_particles, n))
    else:
        particles = init_beta + np.random.uniform(-0.5, 0.5, (num_particles, n))
        particles = np.clip(particles, lb, ub)
    
    velocities = np.random.uniform(-1, 1, (num_particles, n))
    
    # Initialize personal best positions and scores
    pbest = particles.copy()
    pbest_scores = np.array([objective(p, Y, X, cutoff_freq, filter_order, alpha_lasso)
                             for p in particles])
    # Global best
    gbest_index = np.argmin(pbest_scores)
    gbest = pbest[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            # Update position
            particles[i] = particles[i] + velocities[i]
            # Enforce bounds
            particles[i] = np.clip(particles[i], lb, ub)
            # Evaluate objective
            score = objective(particles[i], Y, X, cutoff_freq, filter_order, alpha_lasso)
            # Update personal best
            if score < pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                # Update global best if needed
                if score < gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score

        # Optional: print progress every 10 iterations
        if (iteration+1) % 10 == 0:
            print(f"PSO iteration {iteration+1}/{max_iter}, best objective = {gbest_score:.6f}")

    return gbest

###############################################################################
#                                MAIN FUNCTION                                #
###############################################################################
def main():
    # ---- Simulation Parameters ----
    num_points = 1024      # number of data points in each spectrum
    n_total    = 30        # total number of synthetic dictionary spectra
    n_ref      = 10        # number of "known" reference spectra used in fitting

    # ---- Generate Synthetic Dictionary ----
    XPint = np.zeros((num_points, n_total))
    for j in range(n_total):
        n_peaks = np.random.randint(1, 4)
        XPint[:, j] = generate_lorentzian_spectrum(num_points=num_points, num_peaks=n_peaks)
    
    # Randomly select reference columns to form Xint (the dictionary used for fitting)
    ref_indices = np.random.choice(n_total, n_ref, replace=False)
    X = XPint[:, ref_indices]
    
    # ---- Generate Synthetic Signal Y ----
    # Generate "true" coefficients for the known references
    true_beta_known = np.random.uniform(0.5, 2.0, size=n_ref)
    Y = X @ true_beta_known

    # Optionally add contributions from "unknown" dictionary elements
    unknown_indices = [idx for idx in range(n_total) if idx not in ref_indices]
    if unknown_indices:
        Xunknown = XPint[:, unknown_indices]
        true_beta_unknown = np.random.uniform(0.5, 2.0, size=len(unknown_indices))
        Y += Xunknown @ true_beta_unknown

    # Add Gaussian noise to simulate measurement noise
    noise_std = 0.05
    Y_noisy = Y + np.random.normal(0, noise_std, size=num_points)
    
    # For synthetic validation (if available), compute the "true known" portion
    Y_true_known_only = X @ true_beta_known

    # ---- User Parameters for Solver ----
    cutoff_freq = 0.2       # low-pass filter cutoff (0.0 to 0.5 typical)
    filter_order = 2        # order of Butterworth filter
    alpha_lasso = 0.01      # L1 penalty strength (for Lasso and PSO objective)
    use_sparse  = False      # flag to use L1 regularization in alternating solver
    max_iter_alt = 50       # max iterations for alternating solver

    print("\n=== Starting Alternating Solver ===")
    beta_alt, Z_alt = solve_lowpass_with_sparse_beta(
        Y_noisy, X, cutoff_freq=cutoff_freq, filter_order=filter_order,
        alpha_lasso=alpha_lasso, sparse_beta=use_sparse, max_iter=max_iter_alt
    )
    Y_fit_alt = X @ beta_alt

    print("\nAlternating solver results (beta):")
    print(beta_alt)

    # ---- PSO Refinement Step ----
    print("\n=== Starting PSO Refinement on Beta ===")
    # Use the alternating solver beta as initial guess for PSO
    beta_refined = pso_optimize_beta(Y_noisy, X, cutoff_freq, filter_order,
                                     alpha_lasso=alpha_lasso, init_beta=beta_alt,
                                     num_particles=30, max_iter=50)
    Y_fit_refined = X @ beta_refined

    print("\nPSO refined beta:")
    print(beta_refined)

    # ---- Compare with Naive Least Squares ----
    beta_ls, _, _, _ = np.linalg.lstsq(X, Y_noisy, rcond=None)
    Y_fit_ls = X @ beta_ls

    # ---- Compute Residuals ----
    resid_alt = np.linalg.norm(Y_noisy - Y_fit_alt)
    resid_refined = np.linalg.norm(Y_noisy - Y_fit_refined)
    resid_ls = np.linalg.norm(Y_noisy - Y_fit_ls)
    
    print(f"\nResidual Norms (dictionary portion only):")
    print(f"Alternating Solver: {resid_alt:.3f}")
    print(f"PSO Refined:        {resid_refined:.3f}")
    print(f"Naive LS:           {resid_ls:.3f}")

    # ---- Plotting Results ----
    plt.figure(figsize=(12, 10))

    # Plot measured spectrum and dictionary fits
    plt.subplot(3,1,1)
    plt.plot(Y_noisy, 'k', lw=1, label='Y (Noisy)')
    plt.plot(Y_fit_ls, 'r', lw=1, label='LS Fit')
    plt.plot(Y_fit_alt, 'b', lw=1, label='Alternating Solver Fit')
    plt.plot(Y_fit_refined, 'g', lw=1, label='PSO Refined Fit')
    plt.plot(Y_true_known_only, 'm--', lw=1, label='True Known Only')
    plt.legend()
    plt.title("Measured Spectrum vs. Various Dictionary Fits")

    # Plot residuals for alternating and PSO refined methods
    plt.subplot(3,1,2)
    plt.plot(Y_noisy - Y_fit_alt, 'b', lw=1, label='Residual (Alt)')
    plt.plot(Y_noisy - Y_fit_refined, 'g', lw=1, label='Residual (PSO)')
    plt.legend()
    plt.title("Residuals (Y - Fit)")

    # Plot the estimated baseline from alternating solver
    plt.subplot(3,1,3)
    plt.plot(Z_alt, 'c', lw=1, label='Estimated Baseline (Z from Alt)')
    plt.legend()
    plt.title("Estimated Baseline (Low-pass Filtered)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
