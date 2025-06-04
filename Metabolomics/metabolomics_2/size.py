import os

# Function that checks the size of row vectors in .txt files (experimental EVs)
def check_vector_sizes(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as file:
                    first_line = file.readline().strip()
                    if first_line:
                        vector = first_line.split(',')
                        print(f"{file_name}: {len(vector)} elements in the row vector")
                    else:
                        print(f"{file_name}: Empty file")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    folder_path = os.path.dirname(__file__)  
    check_vector_sizes(folder_path)
    print(f"Folder path: {folder_path}")

# Row vector of dimension: (1, 7163)