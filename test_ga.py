import string
import random
import time
import re
from genetic_algorithm import GeneticAlgorithm

population_size = 100
mutation_rate = 0.05
num_generations = 1000
common_words_weight = 0.3

with open("textFile.text", "r") as file:
    real_text = file.read().lower()

alphabet = string.ascii_lowercase
permutation = [char for char in alphabet]
random.shuffle(permutation)
permutation = {alphabet[i]: permutation[i] for i in range(len(alphabet))}

ciphertext = ""
for c in real_text:
    new_char = permutation.get(c, c)
    ciphertext += new_char

# with open("textFileEnc.txt", "w") as file:
#     file.write(ciphertext)

# Create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(population_size, mutation_rate, num_generations, "Letter_Freq.txt", "Letter2_Freq.txt",
                      "textFileEnc.txt", "dict.txt", common_words_weight)

# Run the genetic algorithm to find the solution
start_time = time.time()
ga.evolve(lamarckian=True)
end_time = time.time()
elapsed_time = (end_time - start_time) // 60
print("Elapsed time:", elapsed_time)

