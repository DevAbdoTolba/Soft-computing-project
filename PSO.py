import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

class ParticleSwarmOptimization:
    def __init__(self, n_particles=30, n_iterations=100, w=0.7, c1=1.5, c2=1.5, 
                 velocity_clamp=(-4.0, 4.0), binary=True):
        """
        Initialize PSO algorithm parameters
        
        Parameters:
        -----------
        n_particles : int
            Number of particles in the swarm
        n_iterations : int
            Maximum number of iterations
        w : float
            Inertia weight
        c1 : float
            Cognitive coefficient
        c2 : float
            Social coefficient
        velocity_clamp : tuple
            Min and max values for velocity
        binary : bool
            Whether to use binary PSO (for feature selection)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
        self.binary = binary
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []
        
    def sigmoid(self, x):
        """Apply sigmoid function to transform values to probability (0-1)"""
        return 1 / (1 + np.exp(-x))
    
    def initialize_swarm(self, n_features):
        """Initialize particles with random positions and velocities"""
        # Initialize particles with random binary positions
        positions = np.random.randint(0, 2, size=(self.n_particles, n_features))
        
        # Make sure each particle has at least one feature selected
        for i in range(self.n_particles):
            if np.sum(positions[i]) == 0:
                positions[i, np.random.randint(0, n_features)] = 1
        
        # Initialize velocities
        velocities = np.random.uniform(
            self.velocity_clamp[0], self.velocity_clamp[1], 
            size=(self.n_particles, n_features)
        )
        
        # Initialize personal best positions and fitnesses
        personal_best_positions = positions.copy()
        personal_best_fitnesses = np.zeros(self.n_particles)
        
        # Initialize global best
        global_best_position = None
        global_best_fitness = 0
        
        return {
            'positions': positions,
            'velocities': velocities,
            'personal_best_positions': personal_best_positions,
            'personal_best_fitnesses': personal_best_fitnesses,
            'global_best_position': global_best_position,
            'global_best_fitness': global_best_fitness
        }
    
    def calculate_fitness(self, position, X, y):
        """Calculate fitness of a particle position using model accuracy"""
        # If no features are selected, return 0 fitness
        if np.sum(position) == 0:
            return 0
        
        # Select only the features indicated by the position
        selected_features = X.iloc[:, position == 1]
        
        # If too few samples for stratification, don't stratify
        stratify = y if len(y) > 10 else None
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, y, test_size=0.2, random_state=42, stratify=stratify
            )
            
            # Train a classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Penalty for using too many features (to encourage feature reduction)
            n_features = np.sum(position)
            penalty_factor = 0.01 * (n_features / len(position))
            
            return accuracy - penalty_factor
            
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return 0
    
    def update_velocity(self, particle_idx, swarm):
        """Update velocity of a particle"""
        r1 = np.random.random(len(swarm['velocities'][particle_idx]))
        r2 = np.random.random(len(swarm['velocities'][particle_idx]))
        
        cognitive_velocity = self.c1 * r1 * (
            swarm['personal_best_positions'][particle_idx] - swarm['positions'][particle_idx]
        )
        social_velocity = self.c2 * r2 * (
            swarm['global_best_position'] - swarm['positions'][particle_idx]
        )
        
        new_velocity = (
            self.w * swarm['velocities'][particle_idx] +
            cognitive_velocity +
            social_velocity
        )
        
        # Apply velocity clamping
        new_velocity = np.clip(
            new_velocity, 
            self.velocity_clamp[0], 
            self.velocity_clamp[1]
        )
        
        return new_velocity
    
    def update_position(self, particle_idx, swarm):
        """Update position of a particle"""
        if self.binary:
            # For binary PSO, use sigmoid to get probability and then threshold
            probabilities = self.sigmoid(swarm['velocities'][particle_idx])
            new_position = np.zeros_like(swarm['positions'][particle_idx])
            new_position[probabilities > np.random.random(len(probabilities))] = 1
            
            # Ensure at least one feature is selected
            if np.sum(new_position) == 0:
                random_idx = np.random.randint(0, len(new_position))
                new_position[random_idx] = 1
                
            return new_position
        else:
            # For continuous PSO
            new_position = swarm['positions'][particle_idx] + swarm['velocities'][particle_idx]
            return new_position
    
    def optimize(self, X, y):
        """Run the PSO algorithm"""
        n_features = X.shape[1]
        
        # Initialize swarm
        swarm = self.initialize_swarm(n_features)
        
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # Calculate fitness for each particle
            fitnesses = []
            for i in range(self.n_particles):
                fitness = self.calculate_fitness(swarm['positions'][i], X, y)
                fitnesses.append(fitness)
                
                # Update personal best
                if fitness > swarm['personal_best_fitnesses'][i]:
                    swarm['personal_best_fitnesses'][i] = fitness
                    swarm['personal_best_positions'][i] = swarm['positions'][i].copy()
                    
                    # Update global best
                    if fitness > swarm['global_best_fitness']:
                        swarm['global_best_fitness'] = fitness
                        swarm['global_best_position'] = swarm['positions'][i].copy()
                        
                        # Update best solution overall
                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_solution = swarm['positions'][i].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                avg_fitness = np.mean(fitnesses)
                best_iter_fitness = np.max(fitnesses)
                selected_count = np.sum(self.best_solution) if self.best_solution is not None else 0
                elapsed = time.time() - start_time
                print(f"Iteration {iteration+1}/{self.n_iterations} | "
                      f"Best Fitness: {self.best_fitness:.4f} | "
                      f"Best Iteration Fitness: {best_iter_fitness:.4f} | "
                      f"Avg Fitness: {avg_fitness:.4f} | "
                      f"Selected Features: {selected_count}/{n_features} | "
                      f"Time: {elapsed:.2f}s")
            
            # Update velocities and positions
            for i in range(self.n_particles):
                # Skip updates if global best not yet found
                if swarm['global_best_position'] is None:
                    continue
                
                swarm['velocities'][i] = self.update_velocity(i, swarm)
                swarm['positions'][i] = self.update_position(i, swarm)
        
        # Return the best solution found
        return self.best_solution, self.best_fitness
    
    def plot_fitness_history(self):
        """Plot the fitness history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('PSO Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.savefig('pso_fitness_history.png')
        plt.show()

def feature_selection_with_pso(dataset_path, output_path=None, target_column=None):
    """Apply PSO for feature selection on the dataset"""
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Check if target column is provided, otherwise use the last column
    if target_column is None:
        target_column = df.columns[-1]
        print(f"No target column specified, using last column: {target_column}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    
    # Run the PSO algorithm
    print("\nStarting Particle Swarm Optimization for feature selection...")
    pso = ParticleSwarmOptimization()
    best_solution, best_fitness = pso.optimize(X, y)
    
    # Get selected feature names
    selected_features = X.columns[best_solution == 1].tolist()
    print(f"\nBest fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {len(selected_features)} out of {X.shape[1]}")
    print("Selected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Create new dataset with only selected features
    selected_df = df[selected_features + [target_column]]
    
    # Generate output path if not provided
    if output_path is None:
        output_path = dataset_path.rsplit('.', 1)[0] + '_pso_selected_features.csv'
    
    # Save the new dataset
    selected_df.to_csv(output_path, index=False)
    print(f"\nNew dataset with selected features saved to: {output_path}")
    
    # Plot fitness history
    pso.plot_fitness_history()
    
    return selected_features, best_fitness

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Particle Swarm Optimization for Feature Selection')
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset CSV file')
    parser.add_argument('--output', type=str, help='Path to save the output dataset with selected features')
    parser.add_argument('--target', type=str, help='Name of the target column')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run feature selection
    feature_selection_with_pso(args.input, args.output, args.target)
