import os

# Current directory
folder_path = '.'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the file
        with open(file_path, 'r') as file:
            line = file.readline().strip()
        
        # Process the row vector
        numbers = [float(num) for num in line.split(',')]
        processed_numbers = [num if num >= 0 else 0 for num in numbers]
        
        # Write the updated row vector back to the file
        with open(file_path, 'w') as file:
            file.write(','.join(map(str, processed_numbers)))

print("Processing complete. Negative values have been zeroed out.")