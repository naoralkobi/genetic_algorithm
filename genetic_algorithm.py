import random
import re
import string


def read_frequencies(filename):
    """Load letter frequency data from a file."""
    with open(filename) as f:
        lines = f.readlines()
    frequencies = {}
    frequency = ""
    letter = ""
    for line in lines:
        m = re.match(r'''(\d\.\d+)\s+([a-zA-z]+)''', line, re.DOTALL|re.IGNORECASE)
        if m:
            frequency = float(m.group(1))
            letter = m.group(2)
        frequencies[letter.lower()] = float(frequency)
    return frequencies


def read_text(filename):
    with open(filename, 'r') as f:
        return f.read()


def find_missing_mapping(mapping_dict):
    all_letters = set(string.ascii_lowercase)
    mapped_letters = set(mapping_dict.keys())
    unmapped_letters = all_letters - mapped_letters
    used_values = set(mapping_dict.values())
    possible_values = all_letters - used_values
    for letter in unmapped_letters:
        random_value = random.choice(list(possible_values))
        mapping_dict[letter] = random_value
        possible_values -= set(random_value)
    return mapping_dict


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, letter_frequencies_file, bigram_frequencies_file,
                 ciphertext_file):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.bigram_frequencies = read_frequencies(bigram_frequencies_file)
        self.letter_frequencies = read_frequencies(letter_frequencies_file)
        self.ciphertext = read_text(ciphertext_file)
        # self.alphabet = sorted(list(set(self.ciphertext) - set(' .,;\n')))
        self.alphabet = [char for char in string.ascii_lowercase]
        self.population = self.generate_population()
        self.best_individual = None
        self.best_generation = 1
        self.best_fitness = float('-inf')
        self.steps = 0

    def generate_permutation(self):
        permutation = self.alphabet.copy()
        random.shuffle(permutation)
        permutation = {self.alphabet[i]: permutation[i] for i in range(len(self.alphabet))}
        # permutation[' '] = ' '
        # permutation['.'] = '.'
        # permutation[','] = ','
        # permutation[';'] = ';'
        # permutation['\n'] = '\n'
        return permutation

    def generate_population(self):
        return [self.generate_permutation() for _ in range(self.population_size)]

    def fitness(self, individual):
        # Generate decoded text using the individual's permutation table
        decoded_text = ''.join([individual.get(symbol, symbol) for symbol in self.ciphertext])

        # Calculate letter frequencies for the decoded text
        letter_frequencies = {letter: decoded_text.count(letter) / len(decoded_text) for letter in self.alphabet}

        # Calculate bigram frequencies for the decoded text
        bigram_frequencies = {}
        for i in range(len(decoded_text) - 1):
            bigram = decoded_text[i:i + 2]
            if bigram in self.bigram_frequencies:
                bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + 1
        total_bigrams = sum(bigram_frequencies.values())
        bigram_frequencies = {bigram: count / total_bigrams for bigram, count in bigram_frequencies.items()}

        # Calculate fitness as the sum of the squared differences between the observed and expected frequencies
        letter_fitness = sum(
            [(self.letter_frequencies[letter] - letter_frequencies.get(letter, 0)) ** 2 for letter in self.alphabet])
        bigram_fitness = sum([(self.bigram_frequencies[bigram] - bigram_frequencies.get(bigram, 0)) ** 2 for bigram in
                              self.bigram_frequencies])
        fitness_value = letter_fitness + bigram_fitness

        return fitness_value

    def select(self, fitness_scores):
        # Calculate total fitness score
        total_fitness = sum(fitness_scores)

        # Calculate selection probabilities
        selection_probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]

        # Perform roulette wheel selection
        selected_population = []
        for _ in range(self.population_size):
            spin = random.random()
            cumulative_probability = 0.0
            for i, probability in enumerate(selection_probabilities):
                cumulative_probability += probability
                if spin <= cumulative_probability:
                    selected_population.append(self.population[i])
                    # return selected_population
                    break

        unique_tuples = set(tuple(sorted(d.items())) for d in selected_population)
        unique_dicts = [dict(t) for t in unique_tuples]

        return unique_dicts
        # return selected_population

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)

        offspring1 = {}
        offspring2 = {}

        for i in range(crossover_point):
            key = list(parent1.keys())[i]
            offspring1[key] = parent1[key]

        for i in range(crossover_point, len(parent1)):
            key = list(parent2.keys())[i]
            value = parent2[key]
            if value not in offspring1.values():
                offspring1[key] = value

        for i in range(crossover_point):
            key = list(parent2.keys())[i]
            offspring2[key] = parent2[key]

        for i in range(crossover_point, len(parent1)):
            key = list(parent1.keys())[i]
            value = parent1[key]
            if value not in offspring2.values():
                offspring2[key] = value

        return find_missing_mapping(offspring1), find_missing_mapping(offspring2)

    def mutation(self, offspring):
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                # Choose two random positions in the individual
                idx1, idx2 = random.sample(range(len(self.alphabet)), 2)
                # Swap the characters at the two positions
                symbol1, symbol2 = self.alphabet[idx1], self.alphabet[idx2]
                offspring[symbol1], offspring[symbol2] = offspring[symbol2], offspring[symbol1]
        return offspring

    def decode_text(self, individual):
        """Decode the ciphertext using the given permutation table."""
        print(len(set(individual.values())))
        text = ""
        for c in self.ciphertext:
            new_char = individual.get(c, c)
            text += new_char
        return text

    def satisfactory_solution(self):
        """Check if a satisfactory solution has been found."""
        return self.best_fitness == 1

    def write_to_files(self, individual, generation):
        # Write the decoded text to plain.txt
        with open("plain.txt", "w") as f:
            f.write(self.decode_text(individual))

        # Write the permutation table to perm.txt
        values_list = list(individual.values())
        with open("perm.txt", "w") as f:
            for i, symbol in enumerate(self.alphabet):
                f.write(f"{symbol}: {values_list[i]}\n")

        # Print the number of steps and best fitness so far
        print(f"Generation {generation} - Steps: {self.steps}, Best Fitness: {self.best_fitness}")

    def evolve(self):
        for i in range(self.generations):
            # Evaluate fitness of each individual in population
            fitness_scores = [self.fitness(individual) for individual in self.population]

            # Update the best individual and best fitness
            best_index = fitness_scores.index(max(fitness_scores))
            if fitness_scores[best_index] > self.best_fitness:
                self.best_individual = self.population[best_index]
                self.best_generation = i
                self.best_fitness = fitness_scores[best_index]

            # Write current best individual to files
            self.write_to_files(self.best_individual, i+1)

            # Check for satisfactory solution
            if self.satisfactory_solution():
                break

            # Selection
            parents = self.select(fitness_scores)

            # Crossover
            offspring = []

            for j in range(0, self.population_size, 2):
                parent1 = parents[j % len(parents)]
                parent2 = parents[(j + 1) % len(parents)]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)

            # Mutation
            for j in range(self.population_size):
                offspring[j] = self.mutation(offspring[j])

            # Elitism
            if self.best_individual not in offspring:
                offspring[0] = self.best_individual

            # Update population
            self.population = offspring
            self.steps += 1

        # Write final best individual to files
        self.write_to_files(self.best_individual, self.best_generation + 1)

        # Print number of steps
        print("Total steps:", self.steps)




