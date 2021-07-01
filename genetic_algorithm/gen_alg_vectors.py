import csv
import sys
import numpy as np
import random


def extract_from_csv(path):
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

    header_training, data_training, target_training = extract_from_csv(training_set_path)
    header_test, data_test, target_test = extract_from_csv(test_set_path)

    hidden_layer_size = hidden_layer_dimensions(nn_architecture, header_training)

    population = genetic_algorithm(population_size, elitism, mutation_prob, mutation_dev,
                                   data_training, target_training, iterations, hidden_layer_size)

    test(population, data_test, target_test, hidden_layer_size)


def number_of_random_weights(hidden_layer_size):
    suma = 0
    for i in range(len(hidden_layer_size)-1):
        suma += hidden_layer_size[i] * hidden_layer_size[i+1] + hidden_layer_size[i+1]
    return suma


def get_weights_list(hidden_layer_size):
    weights_np_array = np.random.normal(loc=0.0, scale=0.01, size=number_of_random_weights(hidden_layer_size))
    weights_list = weights_np_array.tolist()
    return weights_list


def hidden_layer_dimensions(architecture, header):
    hidden_layer_size = []
    for n in filter(None, architecture.split("s")):
        hidden_layer_size.append(int(n))
    hidden_layer_size.insert(0, len(header) - 1)
    hidden_layer_size.append(1)
    return hidden_layer_size


def build_nn(data, target, hidden_layer_size):
    weights_list = get_weights_list(hidden_layer_size)
    return get_mean_squared_error(hidden_layer_size, data, target, weights_list)


def mean_squared_error(arr1, arr2):
    sub_arr = np.subtract(arr1, arr2)
    squared_arr = np.square(sub_arr)
    return squared_arr.mean()


def get_mean_squared_error(hidden_layer_size, data, target, weights_list):
    target = np.array(target, dtype=float)
    results = np.array([])

    for d in data:
        weights_tmp = weights_list.copy()
        tmp_hidden_list = d.copy()
        for index in range(1, len(hidden_layer_size)):  # no need to loop through input layer
            d_np = np.array(tmp_hidden_list, dtype=float)
            d_np = np.append(d_np, 1.0)
            tmp_hidden_list = np.array([])
            k = 0
            while k < hidden_layer_size[index]:
                weights_np = np.array(weights_tmp[:(hidden_layer_size[index-1]+1)], dtype=float) # +1 for bias
                weights_tmp = weights_tmp[hidden_layer_size[index-1]:]
                dot_product = np.dot(d_np, weights_np)
                if index == len(hidden_layer_size) - 1:
                    end_value = dot_product
                else:
                    tmp_hidden_list = np.append(tmp_hidden_list, 1.0 / (1.0 + np.exp(-dot_product)))
                k += 1
        results = np.append(results, end_value)
        end_value = 0

    return mean_squared_error(target, results), weights_list


def crossover(population):
    population_fitness = sum([1/i[0] for i in population])
    chromosome_probabilities = [(1/i[0])/population_fitness for i in population]
    number1, number2 = np.random.choice(a=[i for i in range(0, len(population))], p=chromosome_probabilities, size=2)

    nn1 = population[number1][1]
    nn2 = population[number2][1]
    new_nn_weights = []

    for i in range(len(nn1)):
        sred = (nn1[i] + nn2[i]) / 2
        new_nn_weights.append(sred)

    return new_nn_weights


def mutate(m, p, k):
    for i in range(len(m)):
        for j in range(len(m[i])):
            if random.random() < p:
                br = np.random.normal(loc=0, scale=k, size=1)
                m[i][j] += br[0]
    return m.copy()


def genetic_algorithm(population_size, elitism, mutation_prob, mutation_dev, data, target, iterations, hidden_layer_size):
    population = []
    for _ in range(population_size):
        err, weights = build_nn(data, target, hidden_layer_size)
        population.append((err, weights))

    for i in range(1, iterations+1):
        # print(i)
        next_gen = []
        if elitism > 0:
            population.sort(key=lambda x: x[0])
            elites = population[:elitism]
            for j in elites:
                next_gen.append(j[1])
        if i % 2000 == 0:
            mini = min(population, key=lambda x: x[0])
            print(f"[Train error @{i}]:", mini[0])

        while len(next_gen) < population_size:
            next_gen.append(crossover(population))

        mutated_next_gen = mutate(next_gen, mutation_prob, mutation_dev)
        population = []
        for chromosome in mutated_next_gen:
            population.append(get_mean_squared_error(hidden_layer_size, data, target, chromosome))
    return population


def test(population, data, target, hidden_layer_size):
    all_errors = []
    for nn in population:
        all_errors.append(get_mean_squared_error(hidden_layer_size, data, target, nn[1]))
    print("[Test error]:", min(all_errors, key=lambda x: x[0])[0])


if __name__ == '__main__':
    main()