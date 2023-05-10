README

## Description
This program is a Python implementation of a genetic algorithm to solve a cryptogram. A cryptogram is a type of puzzle where a message is encrypted by replacing each letter with another letter. The goal is to find the original message by discovering the correct letter mappings. The program uses a fitness function based on the frequency of letters and bigrams in the English language, as well as the presence of common English words in the decrypted message.

## Features
- Uses a genetic algorithm with selection, crossover, and mutation operators to generate new individuals for the population
- The algorithm runs for a fixed number of generations or until a stopping condition is met
- Allows the user to specify the inputs such as ciphertext file, letter frequency file, bigram frequency file, common words file, weight of common words in the fitness function, population size, mutation rate, number of generations, tournament size, number of parents, number of workers, Lamarckian steps, and local maximum threshold.
- Outputs the best individual found, along with the fitness score and the decrypted message

## Requirements
- Python 3.7 or higher
- Required Python libraries are: 
  - pandas
  - numpy
  - joblib

## How to Use
1. Clone the repository or download the code as a zip file.
2. Install the required Python libraries (if not already installed).
3. Prepare the input files: 
    - A ciphertext file containing the encrypted message.
    - A letter frequency file containing the frequency of letters in the English language.
    - A bigram frequency file containing the frequency of bigrams in the English language.
    - A common words file containing a list of common English words.
4. Set the input parameters in the configuration file (config.json).
5. Run the program by executing the following command in the terminal:
   ```
   python main.py
   ```
6. The program will output the best individual found, along with the fitness score and the decrypted message.

**Note**: The program assumes that the input files are formatted correctly and that the ciphertext is in English. If the input files are not in the correct format or the ciphertext is in a different language, the program may not produce accurate results.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
