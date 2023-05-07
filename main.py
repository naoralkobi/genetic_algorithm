from genetic_algorithm import GeneticAlgorithm
import time

population_size = 100
mutation_rate = 0.05
num_generations = 500
common_words_weight = 0.3

# Create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(population_size, mutation_rate, num_generations, "Letter_Freq.txt", "Letter2_Freq.txt",
                      "enc.txt", "dict.txt", common_words_weight)

# Run the genetic algorithm to find the solution
start_time = time.time()
ga.evolve()
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
