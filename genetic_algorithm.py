import random
import re
import string
import time

STOP_CONDITION = 30
TOURNAMENT_SIZE = 10
NUM_PARENTS = 100
NUM_OF_WORKERS = 10
LAMARCKIAN_STEPS = 10
LOCAL_MAXIMUM = 10
IS_LOCAL_MAXIMUM = 1
LIMIT_RUN = 5

def read_frequencies(filename):
    """Load letter frequency data from a file."""
    with open(filename) as f:
        lines = f.readlines()
    frequencies = {}
    frequency = ""
    letter = ""
    for line in lines:
        m = re.match(r'''(\d\.\d+)\s+([a-zA-z]+)''', line, re.DOTALL | re.IGNORECASE)
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
    mapped_letters = set(mapping_dict.values())
    unmapped_letters = list(all_letters - mapped_letters)

    used_values = set()
    for key in mapping_dict:
        value = mapping_dict[key]
        if value in used_values:
            for i in range(len(unmapped_letters)):
                if unmapped_letters[i] not in used_values:
                    mapping_dict[key] = unmapped_letters[i]
                    used_values.add(unmapped_letters[i])
                    break
        else:
            used_values.add(value)
    return mapping_dict


def read_common_words(filename):
    with open(filename) as f:
        words = set(line.strip() for line in f)
    return words


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, letter_frequencies_file, bigram_frequencies_file,
                 ciphertext_file, common_words_file, common_words_weight):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.bigram_frequencies = read_frequencies(bigram_frequencies_file)
        self.letter_frequencies = read_frequencies(letter_frequencies_file)
        self.common_words = set(read_common_words(common_words_file))
        self.common_words_weight = common_words_weight
        self.ciphertext = read_text(ciphertext_file)
        # self.alphabet = sorted(list(set(self.ciphertext) - set(' .,;\n')))
        self.alphabet = [char for char in string.ascii_lowercase]
        self.population = self.generate_population()
        self.best_individual = None
        self.best_generation = 1
        self.best_fitness = float('-inf')
        self.steps = 0
        self.stop_condition = 0
        self.local_maximum = 0

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
        decoded_text = self.decode_text(individual)

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

        # Calculate fitness contribution of common words
        decoded_words = re.findall(r'''\b\w+\b''', decoded_text, re.DOTALL | re.IGNORECASE)
        common_words_count = sum([1 for word in decoded_words if word.lower() in self.common_words])
        common_words_fitness = (common_words_count * self.common_words_weight) ** 2

        fitness_value = letter_fitness + bigram_fitness + common_words_fitness

        return fitness_value

    # Selects a parent using tournament selection
    def tournament_selection(self, best_fitness_score):
        # Select a random subset of the population for the tournament
        tournament = random.sample(best_fitness_score, TOURNAMENT_SIZE)
        # Find the fittest individual in the tournament
        sorted_tournament = sorted(tournament, key=lambda x: -x[1])
        best_index, _ = sorted_tournament[0]
        return self.population[best_index]

    def crossover(self, parent1, parent2):
        cutoff = random.choice(list(parent1.keys()))
        child1 = {}
        child2 = {}
        for key in parent1:
            if key <= cutoff:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return find_missing_mapping(child1), find_missing_mapping(child2)

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
        text = ""
        for c in self.ciphertext:
            new_char = individual.get(c, c)
            text += new_char
        return text

    def write_to_files(self, individual, generation, best_fitness):
        # Write the decoded text to plain.txt
        with open("plain.txt", "w") as f:
            f.write(self.decode_text(individual))

        # Write the permutation table to perm.txt
        values_list = list(individual.values())
        with open("perm.txt", "w") as f:
            for i, symbol in enumerate(self.alphabet):
                f.write(f"{symbol}: {values_list[i]}\n")

        # Print the number of steps and best fitness so far
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

    def lamarckian_modification(self, individual):
        mutated_individual = individual.copy()
        max_fitness_score = int(self.fitness(mutated_individual))
        self.steps += 1

        # Lamarckian modification
        for _ in range(LAMARCKIAN_STEPS):
            # Choose two random keys from individual
            keys = random.sample(list(mutated_individual.keys()), 2)
            # Swap their values
            mutated_individual[keys[0]], mutated_individual[keys[1]] = mutated_individual[keys[1]], mutated_individual[
                keys[0]]
            # Check if the modified individual has a better fitness score than the original
            new_fitness_score = int(self.fitness(mutated_individual))
            self.steps += 1
            if new_fitness_score < max_fitness_score:
                # Swap back if the modification does not improve the fitness score
                mutated_individual[keys[1]], mutated_individual[keys[0]] = mutated_individual[keys[0]], \
                    mutated_individual[keys[1]]
            else:
                # Update max_fitness_score to reflect the latest best score
                max_fitness_score = new_fitness_score

        return mutated_individual

    def darwin_modification(self, offspring):
        mutated_individual = offspring.copy()
        individual = None
        generation = 0
        fitness_score = int(self.fitness(offspring[0]))
        self.steps += 1
        for permutaion in mutated_individual:
            current_fitness_score = int(self.fitness(permutaion))
            self.steps += 1
            for _ in range(2):
                # Choose two random keys from individual
                keys = random.sample(list(permutaion.keys()), 2)
                # Swap their values
                permutaion[keys[0]], permutaion[keys[1]] = permutaion[keys[1]], permutaion[keys[0]]
                # Check if the modified individual has a better fitness score than the original
                new_fitness_score = int(self.fitness(permutaion))
                self.steps += 1
                if new_fitness_score > fitness_score:
                    individual = permutaion
                    generation = self.generations
                    fitness_score = new_fitness_score
        return individual, generation, fitness_score

    def evolve(self, lamarckian=None, darwin=None):
        for i in range(self.generations):
            # Evaluate fitness of each individual in population
            fitness_scores = sorted([(i, int(self.fitness(individual))) for i, individual in
                                     enumerate(self.population)], key=lambda x: -x[1])

            self.steps += len(self.population)

            # Update the best individual and best fitness
            best_index, best_fitness_score = fitness_scores[0]
            if best_fitness_score > self.best_fitness:
                # print("-----------------------")
                self.best_individual = self.population[best_index]
                self.best_generation = i
                self.best_fitness = best_fitness_score
                self.stop_condition = 0
                self.local_maximum = 0

            # Write current best individual to files
            # self.write_to_files(self.best_individual, i + 1, self.best_fitness)

            # Selection
            parents = [self.tournament_selection(fitness_scores) for _ in range(NUM_PARENTS)]

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

            if lamarckian:
                # Lamarckian modification
                for j in range(self.population_size):
                    modified_individual = self.lamarckian_modification(offspring[j])
                    offspring[j] = modified_individual

            if darwin:
                individual, generation, fitness_score = self.darwin_modification(offspring)
                if fitness_score > self.best_fitness:
                    self.best_individual = individual
                    self.best_generation = generation
                    self.best_fitness = fitness_score
                    self.stop_condition = 0
                    self.local_maximum = 0


            # Elitism
            if self.best_individual not in offspring:
                offspring[0] = self.best_individual

            if self.local_maximum == LOCAL_MAXIMUM:
                print("move out from local maximum")
                return self.best_individual, self.best_generation, self.best_fitness, self.steps

            # Update population
            self.population = offspring
            self.stop_condition += 1
            self.local_maximum += 1

            if self.stop_condition == STOP_CONDITION and self.generations > 75:
                print("STOP DUO TO - No change after %s generation" % self.stop_condition)
                IS_LOCAL_MAXIMUM = 0
                break

        return self.best_individual, self.best_generation, self.best_fitness, self.steps


if __name__ == '__main__':
    ga = None
    population_size = 100
    mutation_rate = 0.05
    num_generations = 300
    common_words_weight = 0.3
    start_time = time.time()
    best_results = []
    index = 0
    while IS_LOCAL_MAXIMUM and LIMIT_RUN > 0:
        index += 1
        print("start running number: " + str(index))
        # Create an instance of the GeneticAlgorithm class
        ga = GeneticAlgorithm(population_size, mutation_rate, num_generations, "Letter_Freq.txt", "Letter2_Freq.txt",
                              "enc.txt", "dict.txt", common_words_weight)

        # Run the genetic algorithm to find the solution
        # best_results.append(ga.evolve())
        best_results.append(ga.evolve(lamarckian=True))
        # best_results.append(ga.evolve(darwin=True))
        ga.write_to_files(ga.best_individual, ga.best_generation + 1, ga.best_fitness)
        LIMIT_RUN -= 1

    best_score = 0
    best_individual = None
    best_steps = 0
    best_generation = 0
    for individual, generation, fitness_score, steps in best_results:
        if fitness_score > best_score:
            best_score = fitness_score
            best_individual = individual
            best_steps = steps
            best_generation = generation

    # Write final best individual to files
    ga.write_to_files(best_individual, best_generation, best_score)
    # Print number of steps
    print("Total number of calling to fitness:", best_steps)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Elapsed time:", elapsed_time)
