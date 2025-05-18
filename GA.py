import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:
    def __init__(self, population_size=50, n_generations=100, crossover_rate=0.8, 
                 mutation_rate=0.1, tournament_size=3, elite_size=5):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []
        
    def initialize_population(self, n_features):
        """Initialize a population of chromosomes randomly"""
        # Each chromosome is a binary string where 1 means the feature is selected
        population = []
        for _ in range(self.population_size):
            # Make sure at least one feature is selected
            chromosome = np.random.choice([0, 1], size=n_features)
            while np.sum(chromosome) == 0:  # Ensure at least one feature is selected
                chromosome = np.random.choice([0, 1], size=n_features)
            population.append(chromosome)
        return population
    
    def calculate_fitness(self, chromosome, X, y):
        """Calculate fitness of a chromosome using model accuracy"""
        # If no features are selected, return 0 fitness
        if np.sum(chromosome) == 0:
            return 0
        
        # Select only the features indicated by the chromosome
        selected_features = X.iloc[:, chromosome == 1]
        
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
            n_features = np.sum(chromosome)
            penalty_factor = 0.01 * (n_features / len(chromosome))
            
            return accuracy - penalty_factor
            
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return 0
    
    def tournament_selection(self, population, fitnesses):
        """Select chromosomes using tournament selection"""
        selected = []
        for _ in range(len(population)):
            # Select random candidates for the tournament
            tournament_indices = np.random.choice(len(population), size=self.tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            # Select the winner
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx])
        return selected
    
    def crossover(self, parent1, parent2):
        """Apply crossover to create offspring"""
        if np.random.random() < self.crossover_rate:
            # One-point crossover
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            # Ensure at least one feature is selected in each child
            if np.sum(child1) == 0:
                random_idx = np.random.randint(0, len(child1))
                child1[random_idx] = 1
            if np.sum(child2) == 0:
                random_idx = np.random.randint(0, len(child2))
                child2[random_idx] = 1
            
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        """Apply mutation to a chromosome"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # Flip the bit
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, len(mutated))
            mutated[random_idx] = 1
            
        return mutated
    
    def evolve(self, X, y):
        """Run the genetic algorithm"""
        n_features = X.shape[1]
        population = self.initialize_population(n_features)
        
        start_time = time.time()
        
        for generation in range(self.n_generations):
            # Calculate fitness for each chromosome
            fitnesses = [self.calculate_fitness(chromosome, X, y) for chromosome in population]
            
            # Keep track of the best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = population[max_fitness_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Print progress
            if (generation + 1) % 10 == 0 or generation == 0:
                avg_fitness = np.mean(fitnesses)
                best_gen_fitness = np.max(fitnesses)
                selected_count = np.sum(self.best_solution)
                elapsed = time.time() - start_time
                print(f"Generation {generation+1}/{self.n_generations} | "
                      f"Best Fitness: {self.best_fitness:.4f} | "
                      f"Best Generation Fitness: {best_gen_fitness:.4f} | "
                      f"Avg Fitness: {avg_fitness:.4f} | "
                      f"Selected Features: {selected_count}/{n_features} | "
                      f"Time: {elapsed:.2f}s")
            
            # Elitism: keep the best solutions
            elites_idx = np.argsort(fitnesses)[-self.elite_size:]
            elites = [population[i].copy() for i in elites_idx]
            
            # Selection
            selected = self.tournament_selection(population, fitnesses)
            
            # Create the next generation
            next_population = elites.copy()
            
            # Crossover and mutation
            while len(next_population) < self.population_size:
                # Select parents
                parent1, parent2 = np.random.choice(selected, size=2)
                
                # Perform crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Perform mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to the next generation
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            # Update the population
            population = next_population
        
        # Return the best solution found
        return self.best_solution, self.best_fitness
    
    def plot_fitness_history(self):
        """Plot the fitness history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Fitness History')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.savefig('fitness_history.png')
        plt.show()

def feature_selection_with_ga(dataset_path, output_path=None, target_column=None):
    """Apply genetic algorithm for feature selection on the dataset"""
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
    
    # Run the genetic algorithm
    print("\nStarting Genetic Algorithm for feature selection...")
    ga = GeneticAlgorithm()
    best_solution, best_fitness = ga.evolve(X, y)
    
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
        output_path = dataset_path.rsplit('.', 1)[0] + '_selected_features.csv'
    
    # Save the new dataset
    selected_df.to_csv(output_path, index=False)
    print(f"\nNew dataset with selected features saved to: {output_path}")
    
    # Plot fitness history
    ga.plot_fitness_history()
    
    return selected_features, best_fitness

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Genetic Algorithm for Feature Selection')
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset CSV file')
    parser.add_argument('--output', type=str, help='Path to save the output dataset with selected features')
    parser.add_argument('--target', type=str, help='Name of the target column')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run feature selection
    feature_selection_with_ga(args.input, args.output, args.target)
