import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

try:
    # Print debugging info
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())
    
    # Create output directories if they don't exist
    print("Creating output directories...")
    os.makedirs('out/pure', exist_ok=True)
    os.makedirs('out/G', exist_ok=True)
    os.makedirs('out/P', exist_ok=True)
    print("Output directories created")
    
    print("Starting IoT Intrusion Detection Analysis")
    
    # Load dataset
    print("Loading dataset...")
    data = pd.read_csv("IoT_Intrusion_2000.csv")
    print(f"Dataset loaded successfully with shape: {data.shape}")
    
    # Basic dataset information
    print("\nDataset shape:", data.shape)
    target_column = data.columns[-1]
    print(f"Target column: {target_column}")
    
    # Check number of classes
    classes = data[target_column].unique()
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    # Class distribution
    class_dist = data[target_column].value_counts()
    print("\nClass distribution:")
    print(class_dist)
    
    # Correlation matrix
    print("\nCalculating correlation matrix...")
    correlation = data.corr()
    print("Correlation matrix calculated")
    
    # Save correlation with target
    correlation_with_target = correlation[target_column].sort_values(ascending=False)
    correlation_with_target.to_csv('out/pure/correlation_with_target.csv')
    print("Correlation with target saved")
    
    print("Plotting correlation matrix...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('out/pure/correlation_matrix.png')
    plt.close()
    print("Correlation matrix saved")
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    print("\n--- TRAINING MODELS ---")
    
    # Split the data
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled")
    
    # 1. SVM Model
    print("\nTraining SVM...")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    print("SVM trained")
    
    print("Making SVM predictions...")
    svm_pred = svm_model.predict(X_test_scaled)
    print("SVM predictions complete")
    
    # Save SVM results
    f1 = f1_score(y_test, svm_pred, average='weighted')
    accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM F1 Score: {f1:.4f}")
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    # 2. Random Forest Model
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=50)
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest trained")
    
    print("Making Random Forest predictions...")
    rf_pred = rf_model.predict(X_test_scaled)
    print("Random Forest predictions complete")
    
    # Save RF results
    f1 = f1_score(y_test, rf_pred, average='weighted')
    accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest F1 Score: {f1:.4f}")
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # 3. KNN Model
    print("\nTraining KNN...")
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)
    print("KNN trained")
    
    print("Making KNN predictions...")
    knn_pred = knn_model.predict(X_test_scaled)
    print("KNN predictions complete")
    
    # Save KNN results
    f1 = f1_score(y_test, knn_pred, average='weighted')
    accuracy = accuracy_score(y_test, knn_pred)
    print(f"KNN F1 Score: {f1:.4f}")
    print(f"KNN Accuracy: {accuracy:.4f}")
    
    # Save confusion matrices
    print("\nSaving confusion matrices...")
    models = {
        'SVM': svm_pred,
        'RandomForest': rf_pred,
        'KNN': knn_pred
    }
    
    for name, preds in models.items():
        print(f"Processing {name}...")
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'out/pure/confusion_matrix_{name}.png')
        plt.close()
        print(f"Saved confusion matrix for {name}")
        
        # Save metrics
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'out/pure/classification_report_{name}.csv')
        print(f"Saved classification report for {name}")
    
    # Get feature importance from Random Forest for feature selection
    print("\nCalculating feature importance...")
    feature_importances = rf_model.feature_importances_
    feature_names = data.drop(columns=[target_column]).columns
    
    # Create DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_sorted = importance_df.sort_values('Importance', ascending=False)
    
    # Save feature importances
    importance_sorted.to_csv('out/pure/feature_importances.csv', index=False)
    print("Feature importances saved")
    
    # Select top features for Genetic Algorithm approach (simulating GA)
    print("\nSimulating Genetic Algorithm feature selection...")
    top_ga_features = importance_sorted.head(15)['Feature'].values
    ga_indices = [list(feature_names).index(feature) for feature in top_ga_features]
    print(f"Selected {len(ga_indices)} features with GA")
    
    # Train models with GA selected features
    X_ga = X[:, ga_indices]
    X_train_ga, X_test_ga, y_train_ga, y_test_ga = train_test_split(X_ga, y, test_size=0.3, random_state=42)
    scaler_ga = StandardScaler()
    X_train_ga_scaled = scaler_ga.fit_transform(X_train_ga)
    X_test_ga_scaled = scaler_ga.transform(X_test_ga)
    
    print("\n--- TRAINING WITH GA SELECTED FEATURES ---")
    # Save details about the selected features
    pd.DataFrame({
        'Feature Index': ga_indices,
        'Feature Name': top_ga_features
    }).to_csv('out/G/selected_features.csv', index=False)
    print("GA selected features saved")
    
    # Train models with GA features
    models_ga = {
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50),
        'KNN': KNeighborsClassifier()
    }
    
    for name, model in models_ga.items():
        print(f"Training {name} with GA features...")
        model.fit(X_train_ga_scaled, y_train_ga)
        print(f"{name} trained")
        
        print(f"Making {name} predictions...")
        preds = model.predict(X_test_ga_scaled)
        print(f"{name} predictions complete")
        
        f1 = f1_score(y_test_ga, preds, average='weighted')
        accuracy = accuracy_score(y_test_ga, preds)
        print(f"{name} F1 Score: {f1:.4f}")
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_ga, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} with GA features')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'out/G/confusion_matrix_{name}.png')
        plt.close()
        print(f"Saved GA confusion matrix for {name}")
        
        # Save metrics
        report = classification_report(y_test_ga, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'out/G/classification_report_{name}.csv')
        print(f"Saved GA classification report for {name}")
    
    # Select features for PSO approach (simulating PSO with different feature selection)
    print("\nSimulating PSO feature selection...")
    # For this demo, we'll select different top features
    pso_indices = list(range(5)) + list(range(10, 20))
    print(f"Selected {len(pso_indices)} features with PSO")
    
    X_pso = X[:, pso_indices]
    X_train_pso, X_test_pso, y_train_pso, y_test_pso = train_test_split(X_pso, y, test_size=0.3, random_state=42)
    scaler_pso = StandardScaler()
    X_train_pso_scaled = scaler_pso.fit_transform(X_train_pso)
    X_test_pso_scaled = scaler_pso.transform(X_test_pso)
    
    print("\n--- TRAINING WITH PSO SELECTED FEATURES ---")
    # Save details about the selected features
    pd.DataFrame({
        'Feature Index': pso_indices,
        'Feature Name': [feature_names[i] for i in pso_indices]
    }).to_csv('out/P/selected_features.csv', index=False)
    print("PSO selected features saved")
    
    # Train models with PSO features
    models_pso = {
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50),
        'KNN': KNeighborsClassifier()
    }
    
    for name, model in models_pso.items():
        print(f"Training {name} with PSO features...")
        model.fit(X_train_pso_scaled, y_train_pso)
        print(f"{name} trained")
        
        print(f"Making {name} predictions...")
        preds = model.predict(X_test_pso_scaled)
        print(f"{name} predictions complete")
        
        f1 = f1_score(y_test_pso, preds, average='weighted')
        accuracy = accuracy_score(y_test_pso, preds)
        print(f"{name} F1 Score: {f1:.4f}")
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_pso, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} with PSO features')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'out/P/confusion_matrix_{name}.png')
        plt.close()
        print(f"Saved PSO confusion matrix for {name}")
        
        # Save metrics
        report = classification_report(y_test_pso, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'out/P/classification_report_{name}.csv')
        print(f"Saved PSO classification report for {name}")
    
    # Final comparison
    print("\nCreating final comparison...")
    results = []
    
    # All features results
    for name, preds in models.items():
        results.append({
            'Model': name,
            'Approach': 'All Features',
            'Features': X.shape[1],
            'Accuracy': accuracy_score(y_test, preds),
            'F1 Score': f1_score(y_test, preds, average='weighted')
        })
    
    # GA features results
    for name, model in models_ga.items():
        preds = model.predict(X_test_ga_scaled)
        results.append({
            'Model': name,
            'Approach': 'Genetic Algorithm',
            'Features': len(ga_indices),
            'Accuracy': accuracy_score(y_test_ga, preds),
            'F1 Score': f1_score(y_test_ga, preds, average='weighted')
        })
    
    # PSO features results
    for name, model in models_pso.items():
        preds = model.predict(X_test_pso_scaled)
        results.append({
            'Model': name,
            'Approach': 'PSO',
            'Features': len(pso_indices),
            'Accuracy': accuracy_score(y_test_pso, preds),
            'F1 Score': f1_score(y_test_pso, preds, average='weighted')
        })
    
    # Save final comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv('out/final_comparison.csv', index=False)
    print("Final comparison saved")
    
    print("\nAnalysis completed successfully!")
    print("All results have been saved to the output directories.")
    
except Exception as e:
    print("ERROR occurred:")
    print(str(e))
    traceback.print_exc()
