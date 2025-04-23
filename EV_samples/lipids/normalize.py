import os
import numpy as np

def normalize_data(data):
    """Normalize data to a 0-1.0 scale."""
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        return np.zeros_like(data)  # Avoid division by zero
    return (data - min_val) / (max_val - min_val)

def process_txt_files(directory):
    """Normalize data in all .txt files in the given directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                # Read the file content as a comma-separated string
                with open(filepath, 'r') as file:
                    content = file.read()
                # Parse numbers from comma-separated string
                data = np.array([float(x.strip()) for x in content.split(',') if x.strip() != ''])

                # Normalize the data
                normalized_data = normalize_data(data)

                # Write normalized values back, comma-separated with 6 decimal precision
                with open(filepath, 'w') as file:
                    file.write(', '.join(f"{x:.6f}" for x in normalized_data))

                print(f"Normalized: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    process_txt_files(directory)
