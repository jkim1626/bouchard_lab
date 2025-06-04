import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_data(data):
    total = sum([abs(x) for x in data])
    for i in range(len(data)):
        data[i] = data[i] / total
    return data

def process_txt_files(input_directory, output_directory):
    ppm = np.linspace(-2, 10, 7163) 
    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)
            try:
                # Read the file content as a comma-separated string
                with open(input_filepath, 'r') as file:
                    content = file.read()
                # Parse numbers from comma-separated string
                data = np.array([float(x.strip()) for x in content.split(',') if x.strip() != ''])
                
                data = data.tolist()
                normalized_data = normalize_data(data)

                # Write normalized values back, comma-separated with 6 decimal precision
                with open(output_filepath, 'w') as file:
                    file.write(', '.join(f"{x:.10f}" for x in normalized_data))

                print(f"Normalized: {filename}")

                # Plot the vector
                plt.figure(figsize=(10, 6))
                plt.plot(ppm, normalized_data, label=filename, color='b', linewidth=1.5)
                plt.gca().invert_xaxis()
                plt.title(f"Plot of {filename}")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(current_directory, "original_data")
    output_directory = os.path.join(current_directory, "normalized_data")

    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' does not exist. Please create it and add .txt files.")
    else:
        process_txt_files(input_directory, output_directory)