"""
Deconvolve the cell cluster data 
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from size import check_ppm_file

DESKTOP = Path.home() / "Desktop"
FIG_DIR = DESKTOP / "cell_clusters"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(name: str) -> None:
    plt.savefig(FIG_DIR / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def load_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")

def build_matrix(num_points: int, file_list: list[Path]) -> np.ndarray:
    X = np.zeros((num_points, len(file_list)))
    for j, fp in enumerate(file_list):
        if not fp.exists():
            raise FileNotFoundError(fp)
        v = load_vector(fp)
        if v.size != num_points:
            raise ValueError(f"{fp} length {v.size} ≠ expected {num_points}")
        X[:, j] = v
    return X

# ------------------------------------------------------------------ #
#  Baseline-aware alternating NNLS
# ------------------------------------------------------------------ #
def butter_lowpass_filter(signal, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype="low")
    return filtfilt(b, a, signal)

def alternating_solver(
    y, X,
    cutoff_freq=0.0128, filter_order=2,
    alpha_lasso=0.01, sparse_beta=False,
    max_iter=50, tol=1e-6):
    """
    Return β (non-negative) and smooth baseline z for   y ≈ Xβ + z
    """
    y = y.ravel()
    m, n = X.shape
    beta = np.zeros(n)
    z    = np.zeros(m)

    for _ in range(max_iter):
        # ---- β-step -------------------------------------------------
        r = y - z
        if sparse_beta:
            model = Lasso(alpha=alpha_lasso, fit_intercept=False,
                          max_iter=5000, warm_start=True, positive=True)
            model.fit(X, r)
            beta_new = model.coef_
        else:
            beta_new, _ = nnls(X, r)

        # ---- z-step -------------------------------------------------
        r2   = y - X @ beta_new
        z_new = butter_lowpass_filter(r2, cutoff_freq, order=filter_order)

        # ---- convergence -------------------------------------------
        if np.linalg.norm(beta_new - beta) < tol and np.linalg.norm(z_new - z) < tol:
            beta, z = beta_new, z_new
            break

        beta, z = beta_new, z_new

    return beta, z

# ------------------------------------------------------------------ #
#  Plot helpers 
# ------------------------------------------------------------------ #
def plot_overlay(y, recon_ls, recon_lp, ppm, tag):
    plt.figure(figsize=(11, 6))
    plt.plot(ppm, y, "k",  label="sample")
    plt.plot(ppm, recon_ls, "r", alpha=.5, label="LS")
    plt.plot(ppm, recon_lp, "c", alpha=.8, label="LP")
    plt.xlabel("ppm");  plt.ylabel("intensity")
    plt.title(f"Spectrum overlay – {tag}")
    plt.gca().invert_xaxis()
    plt.legend()
    save_figure(f"{tag}_overlay")

def plot_residual(residual, m2, ppm, name):
    plt.figure(figsize=(11,6))
    plt.plot(ppm, residual, "k", label="residual")
    plt.plot(ppm, m2, 'c', label="m2")
    plt.xlabel("ppm")
    plt.ylabel("intensity")
    plt.gca().invert_xaxis()
    plt.legend()
    save_figure(f"{name}_overlay")

# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    dict_folder  = Path("filter_buff")
    test_folder  = Path(".")

    num_points   = check_ppm_file(os.path.dirname(__file__)  )
    ppm          = np.linspace(10, -2, num_points)

    # ----- build dictionaries --------------------------------------
    buffer_path  = dict_folder / "1.txt"
    mix_paths    = [dict_folder / f"{i}.txt" for i in range(2, 6)]

    X_buffer     = build_matrix(num_points, [buffer_path])
    X_mix        = build_matrix(num_points, mix_paths)

    # ----- process every sample in *test* --------------------------
    for sample_file in sorted(test_folder.glob("*.txt")):
        tag = sample_file.stem
        print(f"\n=== {tag} ===")

        y = load_vector(sample_file)

        # -- Round 1 : fit + subtract buffer ------------------------
        beta_buf, z_buf = alternating_solver(y, X_buffer)
        y_no_buffer     = y - (X_buffer @ beta_buf + z_buf)

        # -- Round 2 : fit remaining spectra ------------------------
        beta_lp, z_lp   = alternating_solver(y_no_buffer, X_mix,
                                             sparse_beta=False, alpha_lasso=0.01,
                                             max_iter=200, tol=1e-8)
        beta_ls, _      = nnls(X_mix, y_no_buffer)

        # -- plotting ----------------------------------------------
        recon_lp = X_mix @ beta_lp 
        recon_ls = X_mix @ beta_ls                 
        plot_overlay(y, recon_ls, recon_lp, ppm, tag)

        # -- print coefficients ------------------------------------
        print("β  (baseline model):", beta_lp)
        print("β  (plain LS)     :", beta_ls)


    m2_file_path = Path("filter_buff/5.txt")
    m2 = np.loadtxt(m2_file_path, delimiter=',')
    dict_2 = build_matrix(num_points, [m2_file_path])

    residual_no_buffer = y_no_buffer - (X_mix @ beta_lp)
    residual_with_buffer = y - (X_mix @ beta_lp)

    plot_residual(residual_no_buffer, m2, ppm, "Residual_no_buffer")
    plot_residual(residual_with_buffer, m2, ppm, "Residual_with_buffer")

    beta_2, _ = alternating_solver(residual_no_buffer, dict_2,
                                sparse_beta=False, alpha_lasso=0.01,
                                max_iter=200, tol=1e-8)
    recon_2 = dict_2 @ beta_2
    plot_residual(residual_no_buffer, recon_2, ppm, "Residual_2")
    
if __name__ == "__main__":
    main()
