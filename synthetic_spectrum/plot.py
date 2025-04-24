import os
import numpy as np
import matplotlib.pyplot as plt

# Function to plot row vectors from .txt files (experimental EVs)
def plot_txt_files(folder_path):
    ppm = np.linspace(-2, 10, 120001)  
    for file_name in os.listdir(folder_path):
        if file_name.startswith('synthetic_spectrum_50') and file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the row vector from the file
                with open(file_path, 'r') as file:
                    # Read all lines and convert them to a NumPy array
                    lines = file.readlines()
                    if lines:
                        vector = np.array([float(line.strip()) for line in lines if line.strip()])
                        
                        # Plot the vector
                        plt.figure(figsize=(10, 6))
                        plt.plot(ppm, vector, label=file_name, color='b', linewidth=1.5)
                        plt.gca().invert_xaxis()
                        plt.title(f"Plot of {file_name}")
                        plt.xlabel("PPM")
                        plt.ylabel("Intensity")
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"{file_name}: Empty file, skipping plot.")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    folder_name = "."
    folder_path = os.path.join(".", folder_name)
    plot_txt_files(folder_path)