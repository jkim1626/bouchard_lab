import matplotlib.pyplot as plt
import numpy as np
import random 
import os

def load_metabolites():
    """
    Create file path to metabolites directory 
    Load metabolites into a matrix and return
    Each column of Xint is one metabolite (length = 120001). 
    """

    # Assume metabolites are in a subdirectory called "Dictionary" 
    data_folder = "../normalized_data"
    
    # Assume metabolite files are named/numbered 1,2,3,...20.txt 
    all_files = [os.path.join(data_folder, f"{i}.txt") for i in range(1, 21)]
    
    # Assume metabolite files have dimension 120001
    num_points = 120001
    Xint = np.zeros((num_points, len(all_files)))
    
    for i, file_path in enumerate(all_files):
        Xint[:, i] = np.loadtxt(file_path, delimiter=',')

    return Xint

def create_spectrum(Xint):
    """
    Create a random subset of 5-15 metabolites, pick random coefficients (0.5–3),
    and sum them up to form one synthetic spectrum.
    
    Returns:
        final_spectrum (np.ndarray): The linear combination of the chosen metabolites.
        metabolite_ratios (dict): A dict of {metabolite_ID : random_coefficient}.
    """
    # Pick random subset of metabolite list
    file_nums = list(range(1, 21))
    subset_size = random.randint(5, 15)
    metabolite_IDS = random.sample(file_nums, subset_size)

    # Initialize final spectrum
    Y = np.zeros(Xint.shape[0])

    # Pick random coefficient between (0.5,3) rounded to 10 decimals
    metabolite_ratios = {}
    for metabolite in metabolite_IDS:
        coeff =  round(random.uniform(0.5, 3), 10)
        metabolite_ratios[metabolite] = coeff
        
        # Get the metabolite column
        Xnorm = Xint[:, metabolite - 1] 

        # Multipy metabolite column with corresponding coefficient
        Y += Xnorm * coeff

    return Y, metabolite_ratios

def add_noise(Y, snr):
    """
    Adds Additive White Gaussian Noise (AWGN) to a given NMR signal.
    
    Parameters:
        Y (numpy array): The synthetic NMR spectrum.
        snr (float): Desired signal-to-noise ratio (SNR) in dB.

    Returns:
        numpy array: The noisy signal.
    """
    # Compute signal power
    signal_power = np.mean(Y ** 2)
    signal_power_db = 10 * np.log10(signal_power)
    
    # noise_power_db = signal_power_db - snr  --> difference in dB
    noise_power_db = signal_power_db - snr
    noise_power = 10 ** (noise_power_db / 10)  # this is the variance of the noise
    
    # Standard deviation = sqrt(variance)
    sigma = np.sqrt(noise_power)
    
    # Generate Gaussian noise with standard dev = sigma
    noise = np.random.normal(loc=0, scale=sigma, size=len(Y))
    
    # Add noise to signal
    Y_noisy = Y + noise

    return Y_noisy

def lorentzian(x, x0, gamma, amplitude):
    """
    Creates a single Lorentzian peak at position x0 with half-width-at-half-max gamma.
    amplitude is the peak's height.
    
    Lorentzian functional form:
        L(x) = amplitude * ( (gamma/2)^2 / ( (x - x0)^2 + (gamma/2)^2 ) )
    """
    return amplitude * ((gamma/2.0)**2 / ((x - x0)**2 + (gamma/2.0)**2))

def add_lorentzian_peaks(Y, lower_level, upper_level, num_points=120001, num_peaks=3, gamma_std=100.0):
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
        gamma  = np.random.normal(100, gamma_std)
        amp    = np.random.uniform(lower_level * np.max(Y), upper_level * np.max(Y))
        spectrum += lorentzian(x, center, gamma, amp)

    return spectrum

def save_file(file, name):
    filename = f"{name}.txt"

    # If data is an array, save as a single row vector
    if isinstance(file, (np.ndarray, list)):
        np.savetxt(filename, file, delimiter=',')

    # If data is a dictionary (metabolite -> ratio)
    elif isinstance(file, dict):
        with open(filename, 'w') as f:
            for i in range(1, 21):
                if i in file:
                    f.write(f"{file[i]}\n")
                else:
                    f.write(f"{0}\n")

    # Fallback, convert to string
    else:
        with open(filename, 'w') as f:
            f.write(str(file))

    print(f"Saved {filename}")

def plot_spectrum(ppm, original, noisy, peaks):
    plt.figure(figsize=(10, 8))

    # Subplot 1: Noisy Spectrum
    plt.subplot(3,1,1)
    plt.plot(ppm, original)
    plt.title("Original Spectrum")

    # Subplot 2: Noisy Spectrum with peaks
    plt.subplot(3,1,2)
    plt.plot(ppm, noisy)
    plt.title("Noisy Spectrum")

    # Subplot 3: Just peaks
    plt.subplot(3,1,3)
    plt.plot(ppm, peaks)
    plt.title("Isolated Peak")

    plt.show()

def main():
    # Load dictionary
    Xint = load_metabolites()
    num_points = 120001
    ppm = np.linspace(-2, 10, num_points)

    # Loop 50 times to create synthetic spectra with varying noise and number of peaks
    for i in range(1,51):
        Y, metabolite_ratios = create_spectrum(Xint)

        # Parameters
        num_peaks = random.randint(20,25)
        lower_level = 1
        upper_level = 3

        # Add noise to the spectrum with SNR = 10
        Y_noisy = add_noise(Y, 10)

        peaks = add_lorentzian_peaks(Y_noisy, lower_level, upper_level, num_points, num_peaks)
        Y_noisy_with_peaks = Y_noisy + peaks

        # Save original and noisy spectra 
        save_file(Y, f"original_{i}")
        save_file(Y_noisy_with_peaks, f"synthetic_spectrum_{i}")
        save_file(metabolite_ratios, f"spectrum_ratios_{i}")
        
        # Function to plot the spectra, its noisy version, and the isolated peaks
        # Uncomment the following line to visualize the spectra
        # plot_spectrum(ppm, Y, Y_noisy_with_peaks, peaks)

if __name__ == "__main__":
    main()