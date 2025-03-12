import numpy as np
import os
import time
import multiprocessing as mp
from sko.PSO import PSO
from unit_test_NMR import solve_lowpass_with_sparse_beta, solve_beta_least_squares, get_parameters

# Global variables for PSO objective function
# These will be set before calling parallel_pso
GLOBAL_PARAMS = None
GLOBAL_Y_NOISY = None
GLOBAL_XINT = None

# Top-level function that can be pickled
def pso_objective(x):
    """
    Global objective function for PSO that can be pickled.
    Uses global variables that must be set before use.
    """
    return objective_function(x, GLOBAL_PARAMS, GLOBAL_Y_NOISY, GLOBAL_XINT)

def objective_function(beta, params, Y_noisy, Xint):
    """
    Define loss function.
    'params' represents the variables PSO is optimizing.
    """
    sparse_beta  = bool(round(params[0]))  # Ensure boolean for sparse_beta
    alpha_lasso  = params[1]
    cutoff_freq  = params[2]
    filter_order = int(round(params[3]))  # Ensure integer for filter order
    max_iter     = int(round(params[4]))  # Ensure integer for iterations

    # Run Deconvolution Algorithm
    beta_lp, Z_lp = solve_lowpass_with_sparse_beta(
        Y_noisy, Xint,
        sparse_beta=sparse_beta,
        alpha_lasso=alpha_lasso,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        max_iter=max_iter
    )

    # Compute Error as Residual
    Y_fit = Xint @ beta_lp + Z_lp
    error = np.linalg.norm(Y_noisy - Y_fit) ** 2
    
    return error  # Minimize this error

# Function to evaluate a batch of particles
def evaluate_batch(batch, params, Y_noisy, Xint):
    """
    Evaluate a batch of particles in parallel.
    
    Args:
        batch: List of particles to evaluate
        params: Algorithm parameters
        Y_noisy: Experimental data
        Xint: Dictionary matrix
    
    Returns:
        List of fitness values for each particle
    """
    return [objective_function(particle, params, Y_noisy, Xint) for particle in batch]

# Parallel PSO implementation
def parallel_pso(func, n_dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5, n_processes=None):
    """
    Runs PSO algorithm with parallel evaluation of particles.
    
    Args:
        func: The objective function to minimize
        n_dim: Number of dimensions
        pop: Population size
        max_iter: Maximum iterations
        lb, ub: Lower and upper bounds
        w, c1, c2: PSO parameters
        n_processes: Number of processes to use
        
    Returns:
        Tuple of (best_position, best_fitness)
    """
    # Create a regular PSO instance
    pso = PSO(func=func, n_dim=n_dim, pop=pop, max_iter=max_iter, 
              lb=lb, ub=ub, w=w, c1=c1, c2=c2)
    
    n_processes = n_processes if n_processes else mp.cpu_count()
    
    # We'll manually handle the iterations to add parallel evaluation
    for iter_num in range(max_iter):
        # Evaluate particles in batches to avoid creating too many processes
        chunk_size = max(1, pop // n_processes)
        results = []
        
        # Process particles in sequential chunks
        for i in range(0, pop, chunk_size):
            end = min(i + chunk_size, pop)
            batch = pso.X[i:end]
            
            # Create a pool for each batch to avoid pickling issues
            with mp.Pool(processes=min(n_processes, end-i)) as pool:
                batch_results = pool.map(func, batch)
                results.extend(batch_results)
        
        # Update fitness values
        for i, fitness in enumerate(results):
            pso.Y[i] = fitness
            if fitness < pso.pbest_y[i]:
                pso.pbest_y[i] = fitness
                pso.pbest_x[i] = pso.X[i].copy()
        
        # Update global best
        gbest_idx = pso.pbest_y.argmin()
        if pso.pbest_y[gbest_idx] < pso.gbest_y:
            pso.gbest_y = pso.pbest_y[gbest_idx]
            pso.gbest_x = pso.pbest_x[gbest_idx].copy()
        
        # Update velocities and positions
        pso.update_V()
        pso.update_X()
    
    return pso.gbest_x, pso.gbest_y

def main():
    # Start timing
    start_time = time.time()
    
    # Load data
    num_points = 120001  # Assumed number of points in each file
    data_folder = "New_Dict"
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 21)]
    
    print("Loading dictionary data...")
    Xint = np.zeros((num_points, len(all_files)))
    for j, fpath in enumerate(all_files):
        Xint[:, j] = np.loadtxt(fpath, delimiter=',')
    
    test_folder = "test_3"
    test_name = "synthetic_spectrum_1.txt"
    test_file = os.path.join(test_folder, test_name)
    print(f"Loading test data from {test_file}...")
    Y_noisy = np.loadtxt(test_file, delimiter=',')

    beta_initial = np.zeros((0,20))

    # Get parameters
    n_ref, sparse_beta, alpha_lasso, cutoff_freq, filter_order, num_iter = get_parameters()
    params = [sparse_beta, alpha_lasso, cutoff_freq, filter_order, num_iter]
    
    # Get number of processes for parallel computation
    cpu_count = mp.cpu_count()
    n_processes_str = input(f"Enter number of CPU processes to use (1-{cpu_count}, 'auto' for all) [auto]: ").strip()
    n_processes = cpu_count if (not n_processes_str or n_processes_str.lower() == 'auto') else int(n_processes_str)
    
    # Set global variables for the PSO objective function
    global GLOBAL_PARAMS, GLOBAL_Y_NOISY, GLOBAL_XINT
    GLOBAL_PARAMS = params
    GLOBAL_Y_NOISY = Y_noisy
    GLOBAL_XINT = Xint
    
    # PSO Optimization with parallel processing
    print(f"\nRunning PSO optimization with {n_processes} processes...")
    pso_start_time = time.time()
    
    beta_ls, best_y = parallel_pso(
        func=pso_objective,
        n_dim=20,
        pop=30,
        max_iter=50,
        lb=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ub=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
        w=0.7, c1=1.5, c2=1.5,
        n_processes=n_processes
    )
    
    pso_time = time.time() - pso_start_time
    print(f"PSO optimization completed in {pso_time:.2f} seconds")

    # Solve with least squares (for comparison)
    print("Solving with least squares...")
    beta_lp = solve_beta_least_squares(Xint, Y_noisy)

    # Load true beta values
    true_beta_folder = "test_3"
    true_beta_file = "spectrum_ratios_1.txt"
    true_beta_path = os.path.join(true_beta_folder, true_beta_file)
    true_beta = np.loadtxt(true_beta_path, delimiter=',')
    
    # Calculate RMSE values
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) if len(a) == len(b) else np.nan
    
    rmse_lp = rmse(beta_lp, true_beta)
    rmse_ls = rmse(beta_ls, true_beta)

    print(f"\nRMSE Comparison:")
    print(f"LS RMSE: {rmse_lp:.10f}")
    print(f"PSO RMSE: {rmse_ls:.10f}")
    
    if rmse_lp < rmse_ls:
        print("Least Squares performed better")
    else:
        print("PSO performed better")

    # Print detailed results
    print("\n====== Detailed Results =====")
    col_width = 20
    print(f"{'beta_lp (LS)':<{col_width}} {'beta_ls (PSO)':<{col_width}} {'true_beta':<{col_width}}")
    for beta_lp_val, beta_ls_val, true_beta_val in zip(beta_lp, beta_ls, true_beta):
        print(f"{beta_lp_val:<{col_width}.10f} {beta_ls_val:<{col_width}.10f} {true_beta_val:<{col_width}.10f}")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    # Set start method for multiprocessing
    # 'spawn' is more stable than 'fork' especially on some platforms
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
    main()