# Script to normalize data to relative concentrations 

import os
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from sko.PSO import PSO

def calculate_integrals(paths):
    # Set bounds for integration
    x = np.linspace(-2,10,120001)
    idx = (x >= -2) & (x <= 10)

    # Initialize a hashmap for integrals corresponding to each metabolite
    metabolite_integrals = {}

    for i in range(20):
        # Load intensities, calculate integral using Simpson's Rule, and add to hashmap
        y = np.loadtxt(paths[i], delimiter=',')
        area_simps = simpson(y[idx], x[idx])
        metabolite_integrals[i] = area_simps

    print("\nIntegrals using Simpson's Rule:")
    for key, val in metabolite_integrals.items():
        print(f"Metabolite {key + 1}  \t {val}")

    return metabolite_integrals 

def count_protons():
    # Manually input number of H's for each metabolite
    metabolite_protons = {}
    protons_list = [28,4,4,3,2,3,5,5,7,4,4,7,7,19,13,4,3,4,7,3]
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
    for key,val in num_protons.items():
        if val < lowest:
            ref_metabolite = key
            lowest = val
    
    return metabolite_integrals[ref_metabolite] / num_protons[ref_metabolite]

def objective_function(a, Y, x, idx, num_protons, ref_int):
    Y_scaled = a * Y
    integral = simpson(Y_scaled[idx], x[idx])
    integral /= num_protons
    error = (integral - ref_int) ** 2
    return error

def rescale_and_save(paths, scale_factors):
    new_folder = "New_Dict"
    
    # Create new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        
    for i, file_path in enumerate(paths):
        Y = np.loadtxt(file_path, delimiter=',')
        scaled_Y = Y * scale_factors[i]
        new_file_path = os.path.join(new_folder, f"{i+1}.txt")
        np.savetxt(new_file_path, scaled_Y, delimiter=',')
        
def main():
    # Path to metabolite files
    data_folder = "Dictionary"
    paths = [f"{data_folder}/{i}.txt" for i in range(1,21)]

    # Calculate integrals for each metabolite and store in hashmap
    metabolite_integrals = calculate_integrals(paths)

    # Manually input number of H's for each metabolite
    num_protons = count_protons()

    # Pick reference metabolite (metabolite with least number of protons)
    ref_int = find_reference_integral(metabolite_integrals, num_protons)

    # Calculate optimal scale factors for each metabolite
    x = np.linspace(-2,10,120001)
    idx = (x >= -2) & (x <= 10)
    scale_factors = {}
    for i in range(20):
        Y = np.loadtxt(paths[i], delimiter=',')

        pso = PSO(
            func=lambda a: objective_function(a, Y, x, idx, num_protons[i], ref_int),
            n_dim=1,
            pop=50,
            max_iter=200,
            lb=[0.1],
            ub=[2],
            w=0.7, 
            c1=1.5, 
            c2=1.5
        )

        best, _ = pso.run()
        scale_factors[i] = round(best[0], 15)

    # Check scale_factor values
    print("\nScaled Integrals vs Reference Integrals vs Difference:")
    for i in range(20):
        Y = np.loadtxt(paths[i], delimiter=',')
        Y *= scale_factors[i]
        integral = simpson(Y[idx], x[idx])
        integral /= num_protons[i]
        print(integral, '\t', ref_int, '\t', integral - ref_int)

    # Scale and save new metabolite intensities into new directory
    rescale_and_save(paths, scale_factors)

if __name__ == "__main__":
    main()