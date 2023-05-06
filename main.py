from genetic_algorithm import GeneticAlgorithm

population_size = 100
mutation_rate = 0.1
num_generations = 50

# Create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(population_size, mutation_rate, num_generations, "Letter_Freq.txt", "Letter2_Freq.txt", "enc.txt")

# Run the genetic algorithm to find the solution
ga.evolve()
