import csv
import sys
import numpy as np
import random


def extract_from_csv(path):
    # Extracts data for training and test sets from a csv file
    with open(f"{path}", newline='') as csvfile:
        spamreader = list(csv.reader(csvfile, delimiter=',', quotechar='#'))
        header = spamreader[0]
        independent_vars = []
        dependent_vars = [[]]
        for row in spamreader[1:]:
            independent_vars.append(row[:-1])
            dependent_vars[0].append(row[-1])
    return header, independent_vars, dependent_vars[0]


def main():
    # Sets a default value for elitism
    elitism = 0
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--train":
            training_set_path = sys.argv[i+1]
        elif sys.argv[i] == "--test":
            test_set_path = sys.argv[i+1]
        elif sys.argv[i] == "--nn":
            nn_architecture = sys.argv[i+1]
        elif sys.argv[i] == "--popsize":
            population_size = int(sys.argv[i+1])
        elif sys.argv[i] == "--elitism":
            elitism = int(sys.argv[i+1])
        elif sys.argv[i] == "--p":
            mutation_prob = float(sys.argv[i+1])
        elif sys.argv[i] == "--K":
            mutation_dev = float(sys.argv[i+1])
        elif sys.argv[i] == "--iter":
            iterations = int(sys.argv[i+1])

    # Extract from csv
    header_training, data_training, target_training = extract_from_csv(training_set_path)
    header_test, data_test, target_test = extract_from_csv(test_set_path)

    print(target_training)
    # Get dimensions of a network and run genetic algorithm
    dimensions = get_dimensions(nn_architecture, header_training)
    population = genetic_algorithm(population_size, elitism, mutation_prob, mutation_dev, data_training,
                                   target_training, iterations, dimensions)

    # Get test set results
    test(population, data_test, target_test)


def get_starting_weights(dimensions):
    # Sets initial values of all weights of a neural network
    # to random samples from normal distribution with standard deviation of 0.01
    m = []
    biases = []
    for i in range(len(dimensions) - 1):
        m2 = []
        for j in range(dimensions[i + 1]):
            weights_list = np.random.normal(loc=0.0, scale=0.01, size=dimensions[i]).tolist()
            m2.append(weights_list)
        m.append(m2)
        biases.append(np.random.normal(loc=0.0, scale=0.01, size=dimensions[i+1]).tolist())

    return m, biases


def get_dimensions(architecture, header):
    # Computes and returns dimensions of the neural network as a list
    dimensions = []
    for n in filter(None, architecture.split("s")):
        dimensions.append(int(n))
    dimensions.insert(0, len(header) - 1)
    dimensions.append(1)
    return dimensions


def build_nn(data, target, dimensions):
    # Builds a completely new neural network with randomly sampled weights
    # and calculates mean squared error
    weights_matrix, biases = get_starting_weights(dimensions)
    return forward_prop(data, target, weights_matrix, biases)


def mean_squared_error(arr1, arr2):
    # Computes mean squared error on two vectors
    sub_arr = np.subtract(arr1, arr2)
    squared_arr = np.square(sub_arr)
    return squared_arr.mean()


def sigmoid_function(x):
    # Calculates sigmoid function
    for i in range(len(x)):
        x[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return x


def forward_prop(data, target, weights_matrix, biases):
    # Performs a forward propagation for a network, returns mean squared
    # error as well as all weights and biases of the network
    target = np.array(target, dtype=float)
    weights_matrix = np.array(weights_matrix)
    biases = np.array(biases)
    results = []
    for d in data:
        current_input = np.array(d, dtype=float).T
        for layer_weights_index in range(len(weights_matrix)):
            current_weights = np.array(weights_matrix[layer_weights_index])
            current_biases = np.array(biases[layer_weights_index])
            dot_product = np.dot(current_weights, current_input)
            without_activation_function = np.add(dot_product, current_biases)
            if layer_weights_index < len(weights_matrix) - 1:
                current_input = sigmoid_function(without_activation_function)
            else:
                results.append(without_activation_function[0])
    return mean_squared_error(target, results), weights_matrix, biases


def crossover(population):
    # Performs a crossover
    # Probability of being picked as a parent corresponds to the fitness function defined as
    # 1 / mean_squared_error. Crossover operator for two parents is defined as the arithmetic
    # mean of their weights
    population_fitness = sum([1/i[0] for i in population])
    chromosome_probabilities = [(1/i[0])/population_fitness for i in population]
    number1, number2 = np.random.choice(a=[i for i in range(0, len(population))], p=chromosome_probabilities, size=2)
    nn1_weights = population[number1][1]
    nn1_biases = population[number1][2]
    nn2_weights = population[number2][1]
    nn2_biases = population[number2][2]
    new_nn_weights = []
    new_nn_biases = []
    for i in range(len(nn1_weights)):
        row = []
        for j in range(len(nn1_weights[i])):
            row2 = []
            for k in range(len(nn1_weights[i][j])):
                row2.append((nn1_weights[i][j][k] + nn2_weights[i][j][k]) / 2)
            row.append(row2)
        new_nn_weights.append(row)

        row_biases = []
        for j in range(len(nn1_biases[i])):
            row_biases.append((nn1_biases[i][j] + nn2_biases[i][j]) / 2)
        new_nn_biases.append(row_biases)

    return new_nn_weights, new_nn_biases


def mutate(m, p, k):
    # Performs mutation. Each weight is mutated with probability of p which is a parameter
    # given in program arguments (--p). Mutated weights are being added teh Gaussian noise
    # sampled from normal distribution with standard deviation k which is as well a parameter
    # given in program arguments (--K)
    for i in range(len(m)):
        for j in range(len(m[i][0])):
            for a in range(len(m[i][0][j])):
                for b in range(len(m[i][0][j][a])):
                    if random.random() < p:
                        br = np.random.normal(loc=0, scale=k, size=1)
                        m[i][0][j][a][b] += br[0]
        for j in range(len(m[i][1])):
            for a in range(len(m[i][1][j])):
                if random.random() < p:
                    br = np.random.normal(loc=0, scale=k, size=1)
                    m[i][1][j][a] += br[0]

    return m


def genetic_algorithm(population_size, elitism, mutation_prob, mutation_dev, data, target, iterations, dimensions):
    # Runs a genetic algorithm. First we make *population_size* number of networks and then iterate *iterations* times
    # using elitism, crossover and mutation to get networks with better results. Every 2000 iterations, prints
    # the mean squared error of the best neural network in a population.
    population = []
    for _ in range(population_size):
        err, weights, biases = build_nn(data, target, dimensions)
        population.append((err, weights, biases))

    for i in range(1, iterations+1):
        # print(i)
        next_gen = []
        if elitism > 0:
            population.sort(key=lambda x: x[0])
            elites = population[:elitism]
            for j in elites:
                next_gen.append((j[1], j[2]))
        if i % 2000 == 0:
            mini = min(population, key=lambda x: x[0])
            print(f"[Train error @{i}]:", mini[0])

        while len(next_gen) < population_size:
            next_gen.append(crossover(population))

        mutated_next_gen = mutate(next_gen, mutation_prob, mutation_dev)
        population = []
        for chromosome in mutated_next_gen:
            population.append(forward_prop(data, target, chromosome[0], chromosome[1]))
    return population


def test(population, data, target):
    # Uses the population we got in genetic_algorithm function and looks
    # for the neural network which has the least mean squared error on
    # test set data. Prints the error in question.
    all_errors = []
    for nn in population:
        all_errors.append(forward_prop(data, target, nn[1], nn[2]))
    print("[Test error]:", min(all_errors, key=lambda x: x[0])[0])


if __name__ == '__main__':
    main()