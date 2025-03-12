# Variation #2 of creating fake data
# Spectra are created as linear combinations of Dictionary entries
# Noise of varying degrees is added to the same algorithm of variation #1
    # Group 1 with little noise
    # Group 2 with medium noise
    # Group 3 with lot of noise
# Looking for metabolite identification and ratio concentration identification

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
    data_folder = "../new_data"
    
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

def save_file(file, name):
    """
    Save the given data (array or dictionary) in a local txt file
    If 'file' is final spectrum, then delimit using ','
    If 'file' is dictionary, then delimit using new line
    """
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

def plot_spectrum(ppm, original, noisy, snr):
    plt.figure(figsize=(10, 8))

    # Subplot 1: Original Spectrum
    plt.subplot(3,1,1)
    plt.plot(ppm, original)
    plt.title("Original Spectrum")

    # Subplot 2: Noisy Spectrum
    plt.subplot(3,1,2)
    plt.plot(ppm, noisy)
    plt.title("Noisy Spectrum")

    plt.legend()
    plt.show()

def main():
    # Load dictionary
    Xint = load_metabolites()

    # Loop 5 times to create 5 synthetic spectrum with low noise
    for i in range(1,6):
        Y, metabolite_ratios = create_spectrum(Xint)
        snr = 50
        Y_noisy = add_noise(Y, snr)
        
        # Save original and noisy spectra 
        save_file(Y, f"original_{i}")
        save_file(Y_noisy, f"synthetic_spectrum_{i}")
        save_file(metabolite_ratios, f"spectrum_ratios_{i}")
        # ppm = np.linspace(-2, 10, len(Y))
        # plot_spectrum(ppm, Y, Y_noisy, snr)

    # Loop 5 times to create 5 synthetic spectrum with medium noise
    for i in range(6,11):
        Y, metabolite_ratios = create_spectrum(Xint)
        snr = 30
        Y_noisy = add_noise(Y, snr)
        
        # Save original and noisy spectra 
        save_file(Y, f"original_{i}")
        save_file(Y_noisy, f"synthetic_spectrum_{i}")
        save_file(metabolite_ratios, f"spectrum_ratios_{i}")

    # Loop 5 times to create 5 synthetic spectrum with high noise
    for i in range(11,16):
        Y, metabolite_ratios = create_spectrum(Xint)
        snr = 10
        Y_noisy = add_noise(Y, snr)
        
        # Save original and noisy spectra 
        save_file(Y, f"original_{i}")
        save_file(Y_noisy, f"synthetic_spectrum_{i}")
        save_file(metabolite_ratios, f"spectrum_ratios_{i}")

if __name__ == "__main__":
    main()