# Script to normalize data to relative concentrations 

import os
import numpy as np
from scipy.integrate import simpson

def calculate_integrals(paths):
    # Set bounds for integration
    x = np.linspace(-2, 10, 120001)
    idx = (x >= -2) & (x <= 10)

    # Initialize a hashmap for integrals corresponding to each metabolite
    metabolite_integrals = {}

    for i in range(20):
        # Load intensities, calculate integral using Simpson's Rule, and add to hashmap
        y = np.loadtxt(paths[i], delimiter=',')
        area_simps = simpson(y=y[idx], x=x[idx])
        metabolite_integrals[i] = area_simps

    print("\nIntegrals using Simpson's Rule:")
    for key, val in metabolite_integrals.items():
        print(f"Metabolite {key + 1}  \t {val}")

    return metabolite_integrals 

def count_protons():
    # Manually input number of H's for each metabolite
    metabolite_protons = {}
    protons_list = [28, 4, 4, 3, 2, 3, 5, 5, 7, 4, 4, 7, 7, 19, 13, 4, 3, 4, 7, 3]
    for i in range(20):
        metabolite_protons[i] = protons_list[i]
    
    # Print number of protons 
    print("\nNumber of protons:")
    for key, val in metabolite_protons.items():
        print(f"Metabolite {key + 1}  \t {val}")
    
    return metabolite_protons

def find_reference_integral(metabolite_integrals, num_protons):
    # Pick reference metabolite (metabolite with least number of protons)
    ref_metabolite, lowest = next(iter(num_protons.items()))
    for key, val in num_protons.items():
        if val < lowest:
            ref_metabolite = key
            lowest = val
    
    return metabolite_integrals[ref_metabolite] / num_protons[ref_metabolite]

def calculate_scale_factors(metabolite_integrals, num_protons, ref_int):
    # Calculate scale factors directly
    scale_factors = {}
    for i in range(20):
        # Current integral per proton
        current_int_per_proton = metabolite_integrals[i] / num_protons[i]
        
        # Scale factor needed to match reference integral per proton
        scale_factor = ref_int / current_int_per_proton if current_int_per_proton != 0 else 1.0
        
        scale_factors[i] = round(scale_factor, 15)
    
    return scale_factors

def rescale_and_save(paths, scale_factors):
    new_folder = "normalized_data"
    
    # Create new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        
    for i, file_path in enumerate(paths):
        Y = np.loadtxt(file_path, delimiter=',')
        scaled_Y = Y * scale_factors[i]
        new_file_path = os.path.join(new_folder, f"{i+1}.txt")
        np.savetxt(new_file_path, scaled_Y, delimiter=',')
    
    return new_folder

def main():
    # Path to metabolite files
    data_folder = "original_data"
    paths = [f"{data_folder}/{i}.txt" for i in range(1, 21)]

    # Calculate integrals for each metabolite and store in hashmap
    metabolite_integrals = calculate_integrals(paths)

    # Manually input number of H's for each metabolite
    num_protons = count_protons()

    # Pick reference metabolite (metabolite with least number of protons)
    ref_int = find_reference_integral(metabolite_integrals, num_protons)
    print(f"\nReference integral per proton: {ref_int}")

    # Calculate scale factors directly
    scale_factors = calculate_scale_factors(metabolite_integrals, num_protons, ref_int)
    
    print("\nCalculated scale factors:")
    for i in range(20):
        print(f"Metabolite {i + 1}  \t {scale_factors[i]}")

    # Verify the scaling by checking the normalized integrals
    x = np.linspace(-2, 10, 120001)
    idx = (x >= -2) & (x <= 10)
    
    print("\nScaled Integrals vs Reference Integrals vs Difference:")
    for i in range(20):
        Y = np.loadtxt(paths[i], delimiter=',')
        Y_scaled = Y * scale_factors[i]
        integral = simpson(y=Y_scaled[idx], x=x[idx])
        integral_per_proton = integral / num_protons[i]
        print(f"Metabolite {i + 1}: \t {integral_per_proton:.6f} \t {ref_int:.6f} \t {integral_per_proton - ref_int:.6f}")

    # Scale and save new metabolite intensities into new directory
    new_folder = rescale_and_save(paths, scale_factors)
    print(f"\nNormalized data saved to {os.path.abspath(new_folder)} directory")

if __name__ == "__main__":
    main()