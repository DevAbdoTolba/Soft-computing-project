import os
import sys
import traceback
import pandas as pd
import numpy as np

print("Script starting - Python version:", sys.version)

try:
    # Create output directories
    print("Creating output directories...")
    os.makedirs('out_steel/pure', exist_ok=True)
    os.makedirs('out_steel/G', exist_ok=True)
    os.makedirs('out_steel/P', exist_ok=True)
    
    print("\nStep 1: Loading dataset")
    # Check if file exists
    file_path = "stell-faults.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found!")
        sys.exit(1)
        
    print(f"File {file_path} exists with size: {os.path.getsize(file_path)} bytes")
    
    # Load dataset with error handling
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        print(f"First few rows:")
        print(data.head())
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\nStep 2: Checking columns")
    print("Columns:", data.columns.tolist())
    
    # Identify target columns (fault types)
    fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Check if fault columns exist in the data
    missing_columns = [col for col in fault_columns if col not in data.columns]
    if missing_columns:
        print(f"ERROR: Missing columns in dataset: {missing_columns}")
        sys.exit(1)
    else:
        print("All fault columns found in dataset")
    
    print("\nAnalysis completed successfully - output directories created.")

except Exception as e:
    print(f"Unexpected error: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
