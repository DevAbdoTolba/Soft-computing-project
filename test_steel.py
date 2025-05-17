import pandas as pd
import os
import sys
import traceback

# Print Python version and current working directory
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    # Create test directory
    print("Creating test directory...")
    os.makedirs('test_dir', exist_ok=True)
    print("Test directory created")
    
    # Check if file exists
    file_path = "stell-faults.csv"
    if os.path.exists(file_path):
        print(f"File {file_path} exists with size: {os.path.getsize(file_path)} bytes")
    else:
        print(f"File {file_path} not found!")
        sys.exit(1)
    
    # Try to read dataset
    print("Reading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {data.shape}")
    print("First 2 rows:")
    print(data.head(2))
    
    # Write some test output
    with open("test_dir/test_output.txt", "w") as f:
        f.write(f"Successfully loaded {file_path}\n")
        f.write(f"Dataset shape: {data.shape}\n")
    
    print("Test completed successfully")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()
