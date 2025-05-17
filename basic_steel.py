import pandas as pd
import os

# Create output directory
os.makedirs('out_steel', exist_ok=True)

# Load dataset
print("Loading steel faults dataset...")
data = pd.read_csv("stell-faults.csv")
print(f"Dataset loaded with shape: {data.shape}")

# Save a small report
with open('out_steel/basic_report.txt', 'w') as f:
    f.write(f"Steel Faults Dataset Analysis\n")
    f.write(f"Dataset shape: {data.shape}\n")
    f.write(f"Columns: {', '.join(data.columns.tolist())}")

print("Analysis complete. Results saved to out_steel/basic_report.txt")
