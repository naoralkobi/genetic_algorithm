# genetic_algorithm


This program implements a genetic algorithm to solve a cryptogram. A cryptogram is a type of puzzle where a message is encrypted by replacing each letter with another letter. The goal is to find the original message by discovering the correct letter mappings.

The program uses a fitness function to evaluate the quality of each individual in the population. The fitness function is based on the frequency of letters and bigrams in the English language, as well as the presence of common English words in the decrypted message.

The genetic algorithm uses selection, crossover, and mutation operators to generate new individuals for the population. The algorithm runs for a fixed number of generations or until a stopping condition is met.

To use the program, you need to provide the following inputs:

A ciphertext file containing the encrypted message.
A letter frequency file containing the frequency of letters in the English language.
A bigram frequency file containing the frequency of bigrams in the English language.
A common words file containing a list of common English words.
The weight of common words in the fitness function.
The population size.
The mutation rate.
The number of generations.
The tournament size.
The number of parents.
The number of workers.
The Lamarckian steps.
The local maximum threshold.
The stop condition threshold.
Once you have provided these inputs, you can run the program and it will output the best individual found, along with the fitness score and the decrypted message.

Note: The program assumes that the input files are formatted correctly and that the ciphertext is in English. If the input files are not in the correct format or the ciphertext is in a different language, the program may not produce accurate results.
