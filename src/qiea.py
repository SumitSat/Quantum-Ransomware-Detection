import numpy as np

class QIEA:
    """
    Quantum-Inspired Evolutionary Algorithm (QIEA)
    Used for intelligent feature selection across high-dimensional API call sequences.
    
    In QIEA, individuals are represented by Q-bit individuals, allowing the 
    superposition of states. This allows for better exploration of the feature space 
    without falling into local optima compared to classical Genetic Algorithms (GA).
    """
    def __init__(self, n_features, pop_size=20, max_gen=50):
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_gen = max_gen
        # Initialize quantum population (alpha and beta amplitude for each feature)
        # alpha^2 + beta^2 = 1. We start at 1/sqrt(2) for equal probability (0.5 for 0, 0.5 for 1)
        self.q_pop = np.ones((pop_size, n_features, 2)) / np.sqrt(2.0)
        
        self.best_solution = np.zeros(n_features)
        self.best_fitness = -1.0

    def observe(self, q_individual):
        """
        Collapses the quantum individual into a classical binary string based on alpha^2 probabilities.
        """
        prob_of_ones = q_individual[:, 1] ** 2 # beta^2 is probability of observing 1
        rand_vals = np.random.rand(self.n_features)
        return (rand_vals < prob_of_ones).astype(int)

    def update_q_gate(self, current_sol, fitness):
        """
        Quantum rotation gate update logic.
        Rotates the Q-bits of the population towards the best solution found so far.
        """
        # Theta logic (rotation angle) depends on fitness comparison
        # Simplified: if current_sol is better than best, we move Q-bits towards current_sol.
        # This is typically implemented via lookup tables in literature.
        theta = 0.05 * np.pi # rotation step
        
        new_q_pop = np.zeros_like(self.q_pop)
        for i in range(self.pop_size):
            for j in range(self.n_features):
                alpha = self.q_pop[i, j, 0]
                beta = self.q_pop[i, j, 1]
                
                # Check if we should rotate towards 1 or 0
                if self.best_solution[j] == 1 and current_sol[i, j] == 0:
                    delta_theta = theta
                elif self.best_solution[j] == 0 and current_sol[i, j] == 1:
                    delta_theta = -theta
                else:
                    delta_theta = 0
                
                # Rotation matrix application
                new_alpha = alpha * np.cos(delta_theta) - beta * np.sin(delta_theta)
                new_beta = alpha * np.sin(delta_theta) + beta * np.cos(delta_theta)
                
                new_q_pop[i, j, 0] = new_alpha
                new_q_pop[i, j, 1] = new_beta
                
        self.q_pop = new_q_pop

    def run(self, fitness_func):
        """
        Executes the QIEA search process.
        `fitness_func`: A callable that evaluates a classical binary array.
        """
        print(f"Starting QIEA for {self.max_gen} generations with {self.pop_size} indivs.")
        
        for gen in range(self.max_gen):
            current_pop = np.zeros((self.pop_size, self.n_features))
            fitnesses = np.zeros(self.pop_size)
            
            # 1. Observation
            for i in range(self.pop_size):
                current_pop[i] = self.observe(self.q_pop[i])
                
            # 2. Evaluation
            for i in range(self.pop_size):
                fitnesses[i] = fitness_func(current_pop[i])
                
                # Update best
                if fitnesses[i] > self.best_fitness:
                    self.best_fitness = fitnesses[i]
                    self.best_solution = current_pop[i].copy()
            
            # 3. Quantum Gate Update
            self.update_q_gate(current_pop, fitnesses)
            
            if (gen+1) % 10 == 0:
                print(f"Generation {gen+1} | Best Fitness: {self.best_fitness:.4f}")
                
        # Return best binary mask
        return self.best_solution, self.best_fitness

def test_qiea_dummy():
    # Simple dummy fitness to select the first 5 features
    def dummy_fitness(mask):
        target = np.array([1, 1, 1, 1, 1] + [0]*(len(mask)-5))
        return -np.sum(np.abs(mask - target)) # Maximize to 0
    
    qiea = QIEA(n_features=20, pop_size=10, max_gen=30)
    best_mask, score = qiea.run(dummy_fitness)
    print("Best Mask Found:", best_mask)

if __name__ == "__main__":
    test_qiea_dummy()
