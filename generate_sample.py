import pandas as pd
import os
import numpy as np

# Create synthetic dataset since we can't access the original
print("Creating a synthetic IoT intrusion detection dataset...")
try:
    # Define features
    n_samples = 2000
    n_features = 30
    
    # Generate random feature data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Scale some features to represent different network metrics
    X[:, :5] = X[:, :5] * 10 + 50  # e.g., packet size
    X[:, 5:10] = np.abs(X[:, 5:10]) * 100  # e.g., duration
    X[:, 10:15] = np.abs(X[:, 10:15])  # e.g., probability features
    X[:, 15:20] = np.random.randint(0, 100, size=(n_samples, 5))  # e.g., count features
    
    # Generate target: 0 = normal, 1 = intrusion
    # Let's create an imbalanced dataset with more normal than attack samples
    y = np.zeros(n_samples)
    attack_indices = np.random.choice(n_samples, size=int(n_samples*0.3), replace=False)
    y[attack_indices] = 1
    
    # Create multiple attack types: 1, 2, 3, 4 (different types of intrusions)
    attack_types = np.random.randint(1, 5, size=len(attack_indices))
    y[attack_indices] = attack_types
    
    # Create a DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['intrusion_type'] = y
    
    # Output file path
    output_file = "IoT_Intrusion_2000.csv"
    
    # Save the synthetic dataset
    print(f"Saving synthetic dataset to {output_file}...")
    data.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved successfully!")
    
    # Print some basic info
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Memory usage: {data.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Check for the target column (assuming it's the last column)
    target_column = data.columns[-1]
    print(f"\nTarget column: {target_column}")
    
    # Check class distribution in the sample
    class_counts = data[target_column].value_counts()
    print(f"\nClass distribution in the sample:")
    print(class_counts)
    
    print("\nSynthetic dataset generation completed successfully!")
    
except Exception as e:
    print(f"Error generating synthetic dataset: {e}")
