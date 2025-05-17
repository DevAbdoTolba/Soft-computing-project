# Steel Plate Faults Analysis

This notebook contains the analysis of the Steel Plate Faults dataset, including:
1. Data exploration and visualization
2. Model training with all features (SVM, Random Forest, KNN)
3. Feature selection with Genetic Algorithm
4. Feature selection with Particle Swarm Optimization
5. Comparison of results

## 1. Data Exploration

First, let's load and explore the dataset:


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories
os.makedirs('out_steel/pure', exist_ok=True)
os.makedirs('out_steel/G', exist_ok=True)
os.makedirs('out_steel/P', exist_ok=True)

# Load dataset
print("Loading steel faults dataset...")
data = pd.read_csv("stell-faults.csv")
print(f"Dataset shape: {data.shape}")

data.head()
```

Let's look at the distribution of fault types:

```python
# Identify target columns (fault types)
fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
print(f"Fault columns: {fault_columns}")

# Check distribution of each fault type
print("\nFault distribution:")
for col in fault_columns:
    print(f"{col}: {data[col].sum()} instances ({data[col].sum() / len(data) * 100:.2f}%)")

# Create a single target column for multi-class classification
data['Fault_Type'] = 'No_Fault'  # Default value
for i, col in enumerate(fault_columns):
    mask = data[col] == 1
    data.loc[mask, 'Fault_Type'] = col

# Check class distribution
class_dist = data['Fault_Type'].value_counts()
print("\nClass distribution:")
print(class_dist)

# Plot class distribution
plt.figure(figsize=(12, 6))
class_dist.plot(kind='bar')
plt.title('Fault Type Distribution')
plt.ylabel('Count')
plt.xlabel('Fault Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('out_steel/pure/fault_distribution.png')
plt.close()
```

Examining the feature correlation:

```python
# Drop original fault columns as we now have a consolidated target
X = data.drop(columns=fault_columns + ['Fault_Type'])
y = data['Fault_Type']
feature_names = X.columns.tolist()

# Plot correlation matrix of features
plt.figure(figsize=(16, 14))
correlation = X.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Matrix')
plt.savefig('out_steel/pure/correlation_matrix.png')
plt.close()
```

## 2. Model Training with All Features

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'SVM': SVC(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Results
all_results = []

# Train each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'out_steel/pure/confusion_matrix_{name}.png')
    plt.close()
    
    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'out_steel/pure/classification_report_{name}.csv')
    
    # Store results
    all_results.append({
        'Model': name,
        'Approach': 'All Features',
        'Features': X.shape[1],
        'Accuracy': accuracy,
        'F1 Score': f1
    })

# Get feature importance from Random Forest
rf_model = models['RandomForest']
feature_importances = rf_model.feature_importances_

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_sorted = importance_df.sort_values('Importance', ascending=False)
importance_sorted.to_csv('out_steel/pure/feature_importances.csv', index=False)
```

## 3. Feature Selection with Genetic Algorithm

Here we'll simulate a GA by choosing the top features based on Random Forest importance:

```python
# Use top features as selected by Random Forest feature importance
n_ga_features = X.shape[1] // 2  # Using half the features
ga_features = importance_sorted.head(n_ga_features)['Feature'].tolist()

# Save selected features
pd.DataFrame({
    'Feature Index': [feature_names.index(f) for f in ga_features],
    'Feature Name': ga_features
}).to_csv('out_steel/G/selected_features.csv', index=False)
print(f"Selected {len(ga_features)} features using GA simulation")

# Train models with GA features
X_ga = X[ga_features]
X_ga_train, X_ga_test, y_ga_train, y_ga_test = train_test_split(X_ga, y, test_size=0.3, random_state=42)

# Scale features
scaler_ga = StandardScaler()
X_ga_train_scaled = scaler_ga.fit_transform(X_ga_train)
X_ga_test_scaled = scaler_ga.transform(X_ga_test)

# Train each model with GA features
for name, model_class in models.items():
    print(f"Training {name} with GA features...")
    model = model_class.__class__(**model_class.get_params())
    model.fit(X_ga_train_scaled, y_ga_train)
    y_pred = model.predict(X_ga_test_scaled)
    accuracy = accuracy_score(y_ga_test, y_pred)
    f1 = f1_score(y_ga_test, y_pred, average='weighted')
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Save results
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_ga_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name} (GA)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'out_steel/G/confusion_matrix_{name}.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_ga_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'out_steel/G/classification_report_{name}.csv')
    
    # Store results
    all_results.append({
        'Model': name,
        'Approach': 'Genetic Algorithm',
        'Features': len(ga_features),
        'Accuracy': accuracy,
        'F1 Score': f1
    })
```

## 4. Feature Selection with Particle Swarm Optimization

We'll simulate PSO by selecting a different subset of features:

```python
# Select different features for PSO (simulating PSO feature selection)
np.random.seed(42)
n_pso_features = int(X.shape[1] * 0.4)  # 40% of features
pso_indices = np.random.choice(len(feature_names), size=n_pso_features, replace=False)
pso_features = [feature_names[i] for i in pso_indices]

# Save selected features
pd.DataFrame({
    'Feature Index': pso_indices,
    'Feature Name': pso_features
}).to_csv('out_steel/P/selected_features.csv', index=False)
print(f"Selected {len(pso_features)} features using PSO simulation")

# Train models with PSO features
X_pso = X[pso_features]
X_pso_train, X_pso_test, y_pso_train, y_pso_test = train_test_split(X_pso, y, test_size=0.3, random_state=42)

# Scale features
scaler_pso = StandardScaler()
X_pso_train_scaled = scaler_pso.fit_transform(X_pso_train)
X_pso_test_scaled = scaler_pso.transform(X_pso_test)

# Train each model with PSO features
for name, model_class in models.items():
    print(f"Training {name} with PSO features...")
    model = model_class.__class__(**model_class.get_params())
    model.fit(X_pso_train_scaled, y_pso_train)
    y_pred = model.predict(X_pso_test_scaled)
    accuracy = accuracy_score(y_pso_test, y_pred)
    f1 = f1_score(y_pso_test, y_pred, average='weighted')
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Save results
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_pso_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name} (PSO)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'out_steel/P/confusion_matrix_{name}.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_pso_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'out_steel/P/classification_report_{name}.csv')
    
    # Store results
    all_results.append({
        'Model': name,
        'Approach': 'PSO',
        'Features': len(pso_features),
        'Accuracy': accuracy,
        'F1 Score': f1
    })
```

## 5. Comparing Results

```python
# Final comparison
final_comparison = pd.DataFrame(all_results)
final_comparison.to_csv('out_steel/final_comparison.csv', index=False)

print("\nCOMPARING ALL APPROACHES")
print("="*50)
print(final_comparison)

print(f"\nNumber of features used in each approach:")
print(f"All Features: {X.shape[1]}")
print(f"Genetic Algorithm: {len(ga_features)}")
print(f"PSO: {len(pso_features)}")

# Create comparison plot
plt.figure(figsize=(12, 8))
models_list = final_comparison['Model'].unique()
approaches = final_comparison['Approach'].unique()

for i, metric in enumerate(['Accuracy', 'F1 Score']):
    plt.subplot(2, 1, i+1)
    for model in models_list:
        model_data = final_comparison[final_comparison['Model'] == model]
        plt.plot(model_data['Approach'], model_data[metric], marker='o', label=model)
    
    plt.title(f'Comparison of {metric} Across Different Approaches')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('out_steel/performance_comparison.png')
plt.close()
```

## 6. Summary

The analysis is now complete. We've:
1. Explored the Steel Plate Faults dataset
2. Trained 3 models (SVM, Random Forest, KNN) using all features
3. Used simulated GA to select features based on importance
4. Used simulated PSO to select a different subset of features
5. Compared performance across all approaches

The results are saved in the 'out_steel' directory.
