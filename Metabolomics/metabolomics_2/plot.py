import os
import numpy as np
import matplotlib.pyplot as plt

# Function to plot row vectors from .txt files (experimental EVs)
def plot_txt_files(folder_path):
    ppm = np.linspace(-2, 10, 7163)  
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the row vector from the file
                with open(file_path, 'r') as file:
                    first_line = file.readline().strip()
                    if first_line:
                        # Convert the comma-delimited row vector to a NumPy array
                        vector = np.array([float(x) for x in first_line.split(',')])
                        
                        # Plot the vector
                        plt.figure(figsize=(10, 6))
                        plt.plot(ppm,vector, label=file_name, color='b', linewidth=1.5)
                        plt.gca().invert_xaxis()
                        plt.title(f"Plot of {file_name}")
                        plt.xlabel("Index")
                        plt.ylabel("Value")
                        plt.legend()
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"{file_name}: Empty file, skipping plot.")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    folder_name = "normalized_data"
    folder_path = os.path.join(".", folder_name)
    plot_txt_files(folder_path)