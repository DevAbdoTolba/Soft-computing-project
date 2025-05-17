import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import time

# Create output directories if they don't exist
os.makedirs('out_steel/pure', exist_ok=True)
os.makedirs('out_steel/G', exist_ok=True)
os.makedirs('out_steel/P', exist_ok=True)

print("\n" + "="*50)
print("Starting Steel Plate Faults Analysis")
print("="*50 + "\n")

# Load dataset
print("Loading dataset...")
try:
    data = pd.read_csv("stell-faults.csv")
    print(f"Dataset loaded successfully with shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Basic dataset information
print("\nDataset shape:", data.shape)
print("\nDataset columns:", data.columns.tolist())

# Identify target columns (fault types)
fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
print(f"\nFault columns: {fault_columns}")

# Check distribution of each fault type
print("\nFault distribution:")
for col in fault_columns:
    print(f"{col}: {data[col].sum()} instances ({data[col].sum() / len(data) * 100:.2f}%)")

# Create a single target column for multi-class classification
# We'll convert the binary fault indicators into a single class label
data['Fault_Type'] = 'No_Fault'  # Default value
for i, col in enumerate(fault_columns):
    mask = data[col] == 1
    data.loc[mask, 'Fault_Type'] = col
print("\nCreated multi-class target column 'Fault_Type'")

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
print("Fault distribution plot saved")

# Drop original fault columns as we now have a consolidated target
X = data.drop(columns=fault_columns + ['Fault_Type'])
y = data['Fault_Type']
feature_names = X.columns.tolist()

# Plot correlation matrix of features
print("\nCalculating correlation matrix...")
plt.figure(figsize=(16, 14))
correlation = X.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Matrix')
plt.savefig('out_steel/pure/correlation_matrix.png')
plt.close()

# Save feature correlation data
correlation.to_csv('out_steel/pure/feature_correlations.csv')
print("Correlation matrix saved")

def train_and_evaluate_models(X, y, output_dir):
    """
    Train and evaluate three classification models and save results
    
    Args:
        X: Feature matrix
        y: Target vector
        output_dir: Directory to save results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        prediction_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print metrics
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
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
        plt.savefig(f'{output_dir}/confusion_matrix_{name}.png')
        plt.close()
        
        # Save detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{output_dir}/classification_report_{name}.csv')
        
        # Collect results for summary
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time': training_time,
            'Prediction Time': prediction_time
        })
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    print(f"\nResults saved to {output_dir}")
    
    return results_df, models

def genetic_algorithm_feature_selection(X, y, n_generations=5, population_size=15, crossover_rate=0.7, mutation_rate=0.1):
    """
    Implement Genetic Algorithm for feature selection
    
    Args:
        X: Feature matrix
        y: Target vector
        n_generations: Number of generations
        population_size: Size of population
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate
        
    Returns:
        best_features: Boolean array indicating selected features
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import random
    
    n_features = X.shape[1]
    
    # Initialize population randomly
    population = []
    for _ in range(population_size):
        # Each individual is a boolean array indicating which features are selected
        individual = [random.random() > 0.5 for _ in range(n_features)]
        # Ensure at least one feature is selected
        if sum(individual) == 0:
            individual[random.randint(0, n_features-1)] = True
        population.append(individual)
    
    # Fitness function - cross-validation score
    def fitness(individual):
        if sum(individual) == 0:  # No features selected
            return 0
        
        selected_features = [i for i, selected in enumerate(individual) if selected]
        X_selected = X.iloc[:, selected_features]
        
        try:
            # Use 3-fold cross-validation with Random Forest
            scores = cross_val_score(
                RandomForestClassifier(random_state=42, n_estimators=50), 
                X_selected, y, cv=3, scoring='f1_weighted'
            )
            return scores.mean()
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return 0
    
    best_fitness = -1
    best_individual = None
    
    print("\nRunning Genetic Algorithm for feature selection...")
    
    # Evolution process
    for generation in range(n_generations):
        # Evaluate fitness
        fitness_scores = [fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx]
            
        print(f"Generation {generation+1}/{n_generations}, Best fitness: {best_fitness:.4f}, Features selected: {sum(best_individual)}")
        
        # Create new population
        new_population = []
        
        # Elitism - keep the best individual
        new_population.append(population[max_fitness_idx])
        
        while len(new_population) < population_size:
            # Selection (tournament selection)
            def tournament_selection():
                idx1, idx2 = random.sample(range(population_size), 2)
                return population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
            
            parent1 = tournament_selection()
            parent2 = tournament_selection()
            
            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, n_features-1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1.copy()
            
            # Mutation
            for i in range(n_features):
                if random.random() < mutation_rate:
                    child[i] = not child[i]
            
            # Ensure at least one feature is selected
            if sum(child) == 0:
                child[random.randint(0, n_features-1)] = True
                
            new_population.append(child)
        
        # Replace population
        population = new_population
    
    # Get final best solution
    if best_individual is None:
        fitness_scores = [fitness(individual) for individual in population]
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        best_individual = population[max_fitness_idx]
        best_fitness = fitness_scores[max_fitness_idx]
    
    selected_features = [i for i, selected in enumerate(best_individual) if selected]
    print(f"\nGenetic Algorithm completed.")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {sum(best_individual)} out of {n_features}")
    print(f"Selected feature indices: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
    
    return best_individual

def particle_swarm_optimization_feature_selection(X, y, n_iterations=5, n_particles=15, w=0.6, c1=1.2, c2=1.2):
    """
    Implement Particle Swarm Optimization for feature selection
    
    Args:
        X: Feature matrix
        y: Target vector
        n_iterations: Number of iterations
        n_particles: Number of particles
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
        
    Returns:
        best_features: Boolean array indicating selected features
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    import random
    
    n_features = X.shape[1]
    
    # Fitness function - cross-validation score
    def fitness(position):
        # Convert continuous position to binary
        binary_position = [1 if p > 0.5 else 0 for p in position]
        
        if sum(binary_position) == 0:  # No features selected
            return 0
        
        selected_features = [i for i, selected in enumerate(binary_position) if selected]
        X_selected = X.iloc[:, selected_features]
        
        try:
            # Use 3-fold cross-validation with SVM
            scores = cross_val_score(
                SVC(random_state=42), 
                X_selected, y, cv=3, scoring='f1_weighted'
            )
            return scores.mean()
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return 0
    
    # Initialize particles
    positions = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    
    for _ in range(n_particles):
        # Initialize position with random values between 0 and 1
        position = [random.random() for _ in range(n_features)]
        positions.append(position)
        
        # Initialize velocity as small random values
        velocity = [(random.random() * 0.2) - 0.1 for _ in range(n_features)]
        velocities.append(velocity)
        
        # Initialize personal best
        personal_best_positions.append(position.copy())
        personal_best_scores.append(-1)  # Will be updated on first evaluation
    
    # Initialize global best
    global_best_position = None
    global_best_score = -1
    
    print("\nRunning Particle Swarm Optimization for feature selection...")
    
    # Main PSO loop
    for iteration in range(n_iterations):
        # Update personal and global bests
        for i in range(n_particles):
            score = fitness(positions[i])
            
            # Update personal best
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
            
            # Update global best
            if score > global_best_score:
                global_best_score = score
                global_best_position = positions[i].copy()
        
        # Count features in global best
        binary_global_best = [1 if p > 0.5 else 0 for p in global_best_position]
        n_selected_features = sum(binary_global_best)
        
        print(f"Iteration {iteration+1}/{n_iterations}, Best fitness: {global_best_score:.4f}, Features selected: {n_selected_features}")
        
        # Update velocities and positions
        for i in range(n_particles):
            for j in range(n_features):
                # Update velocity
                cognitive_component = c1 * random.random() * (personal_best_positions[i][j] - positions[i][j])
                social_component = c2 * random.random() * (global_best_position[j] - positions[i][j])
                
                velocities[i][j] = w * velocities[i][j] + cognitive_component + social_component
                
                # Update position
                positions[i][j] += velocities[i][j]
                
                # Clamp position to [0, 1]
                positions[i][j] = max(0, min(1, positions[i][j]))
    
    # Convert global best to binary
    best_features = [1 if p > 0.5 else 0 for p in global_best_position]
    selected_features = [i for i, selected in enumerate(best_features) if selected]
    
    print(f"\nPSO completed.")
    print(f"Best fitness: {global_best_score:.4f}")
    print(f"Number of selected features: {sum(best_features)} out of {n_features}")
    print(f"Selected feature indices: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
    
    return best_features

# Main execution
print(f"\nTotal number of features: {X.shape[1]}")
print(f"\n1. Training models with all features")
print("-"*50)

# 1. Train models using all features
results_all, models_all = train_and_evaluate_models(X, y, 'out_steel/pure')

# Get feature importance from RandomForest
rf_model = models_all['RandomForest']
feature_importances = rf_model.feature_importances_

# Create DataFrame of feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_sorted = importance_df.sort_values('Importance', ascending=False)

# Save feature importances
importance_sorted.to_csv('out_steel/pure/feature_importances.csv', index=False)
print("Feature importances saved")

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_sorted.head(15))
plt.title('Top 15 Features by Importance')
plt.tight_layout()
plt.savefig('out_steel/pure/feature_importances.png')
plt.close()
print("Feature importance plot saved")

print("\n\n2. Training models with Genetic Algorithm selected features")
print("-"*50)

# 2. Train models using Genetic Algorithm selected features
ga_selected_features = genetic_algorithm_feature_selection(X, y)

# Get the indices of selected features
ga_indices = [i for i, selected in enumerate(ga_selected_features) if selected]
X_ga = X.iloc[:, ga_indices]

# Save selected feature names
pd.DataFrame({
    'Feature Index': ga_indices,
    'Feature Name': [feature_names[i] for i in ga_indices]
}).to_csv('out_steel/G/selected_features.csv', index=False)

# Train models using GA features
results_ga, _ = train_and_evaluate_models(X_ga, y, 'out_steel/G')

print("\n\n3. Training models with PSO selected features")
print("-"*50)

# 3. Train models using PSO selected features
pso_selected_features = particle_swarm_optimization_feature_selection(X, y)

# Get the indices of selected features
pso_indices = [i for i, selected in enumerate(pso_selected_features) if selected]
X_pso = X.iloc[:, pso_indices]

# Save selected feature names
pd.DataFrame({
    'Feature Index': pso_indices,
    'Feature Name': [feature_names[i] for i in pso_indices]
}).to_csv('out_steel/P/selected_features.csv', index=False)

# Train models using PSO features
results_pso, _ = train_and_evaluate_models(X_pso, y, 'out_steel/P')

# Compare results from all three approaches
print("\n" + "="*50)
print("COMPARING ALL APPROACHES")
print("="*50)

results_all['Approach'] = 'All Features'
results_ga['Approach'] = 'Genetic Algorithm'
results_pso['Approach'] = 'PSO'

final_comparison = pd.concat([results_all, results_ga, results_pso])
final_comparison.to_csv('out_steel/final_comparison.csv', index=False)

print("\nFinal comparison saved to out_steel/final_comparison.csv")
print(f"\nNumber of features used in each approach:")
print(f"All Features: {X.shape[1]}")
print(f"Genetic Algorithm: {len(ga_indices)}")
print(f"PSO: {len(pso_indices)}")

print("\nSteel Plate Faults Analysis completed successfully!")
