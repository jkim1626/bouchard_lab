# Variation #1 of creating fake data
# Spectra are created as linear combinations of Dictionary entries
# No noise added onto data
# Looking for metabolite identification and ratio concentration identification

import nmrglue as ng
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
    data_folder = "New_Dict"
    
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

def plot_spectrum(ppm, original, snr):
    plt.figure(figsize=(10, 5))
    plt.plot(ppm, original, label="Original Spectrum")
    plt.xlabel("ppm")
    plt.ylabel("Intensity")
    plt.title("Synthetic NMR Spectrum with Noise")
    plt.legend()
    plt.show()

def main():
    # Loop 20 times to create 20 synthetic spectrum
    for i in range(1,16):
        # -- (1) Load dictionary of metabolites
        Xint = load_metabolites()

        # -- (2) Create synthetic spectrum by randomly combining a subset of them
        Y, metabolite_ratios = create_spectrum(Xint)

        # -- (3) Save the final synthetic spcetrum as "synthetic_spectrum.txt"
        save_file(Y, f"synthetic_spectrum_{i}")
    
        # -- (4) Save the metabolite ratios (and IDs) as "spectrum_ratios.txt"
        save_file(metabolite_ratios, f"spectrum_ratios_{i}")

if __name__ == "__main__":
    main()