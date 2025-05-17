"""
Minimal Steel Plate Faults Analysis - Just to verify basic functionality
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    """Basic analysis to verify functionality"""
    print("Starting minimal analysis...")
    
    # Create a simple output file
    with open("test_file.txt", "w") as f:
        f.write("Testing file writing\n")
    print("Created test file")
    
    # Try to read the dataset
    try:
        data = pd.read_csv("stell-faults.csv")
        print(f"Dataset loaded: {data.shape}")
        
        # Identify fault columns
        fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        
        # Create a single target column
        data['Fault_Type'] = 'Other'  # Default value
        for col in fault_columns:
            mask = data[col] == 1
            data.loc[mask, 'Fault_Type'] = col
        
        # Prepare data
        X = data.drop(columns=fault_columns + ['Fault_Type'])
        y = data['Fault_Type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train a simple model
        print("Training a simple model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Write results to file
        with open("minimal_results.txt", "w") as f:
            f.write(f"Dataset shape: {data.shape}\n")
            f.write(f"Model accuracy: {accuracy:.4f}\n")
        
        print("Analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
