import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback

try:
    print("Starting Steel Plate Faults Analysis")
    
    # Create directory
    os.makedirs('out_steel', exist_ok=True)
    
    # Load dataset
    print("Loading steel faults dataset...")
    data = pd.read_csv("stell-faults.csv")
    print(f"Dataset shape: {data.shape}")
    
    # Quick look at the data
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Summary statistics
    print("\nSummary statistics:")
    data_description = data.describe()
    data_description.to_csv('out_steel/stats_summary.csv')
    print("Summary statistics saved to out_steel/stats_summary.csv")
    
    # Identify fault columns
    fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Count occurrences of each fault type
    fault_counts = {col: data[col].sum() for col in fault_columns}
    print("\nFault counts:")
    for fault, count in fault_counts.items():
        print(f"{fault}: {count} ({count/len(data)*100:.2f}%)")
        
    # Create a dataframe for the fault distribution
    fault_df = pd.DataFrame({
        'Fault Type': fault_counts.keys(),
        'Count': fault_counts.values()
    })
    fault_df.to_csv('out_steel/fault_distribution.csv', index=False)
    
    # Plot fault distribution
    plt.figure(figsize=(10, 6))
    plt.bar(fault_df['Fault Type'], fault_df['Count'])
    plt.title('Distribution of Fault Types')
    plt.ylabel('Count')
    plt.xlabel('Fault Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('out_steel/fault_distribution.png')
    plt.close()
    print("Fault distribution saved to out_steel/fault_distribution.png")
    
    # Create a report file with the analysis
    with open('out_steel/analysis_report.txt', 'w') as f:
        f.write("STEEL PLATE FAULTS DATASET ANALYSIS\n")
        f.write("===================================\n\n")
        f.write(f"Dataset Dimensions: {data.shape[0]} rows × {data.shape[1]} columns\n\n")
        f.write("Features:\n")
        for col in data.columns:
            if col not in fault_columns:
                f.write(f"- {col}\n")
        f.write(f"\nTotal features: {len(data.columns) - len(fault_columns)}\n\n")
        
        f.write("Fault Types (Target Classes):\n")
        for fault, count in fault_counts.items():
            f.write(f"- {fault}: {count} instances ({count/len(data)*100:.2f}%)\n")
        
        f.write("\nThis dataset contains measurements of steel plates with various types of faults.\n")
        f.write("The task is to classify the type of fault based on the feature measurements.\n")
        f.write("It's a multi-class classification problem with 7 different types of faults.\n")
    
    print("\nAnalysis completed. Results saved to out_steel/ directory.")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()
