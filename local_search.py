import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fitness(state):
    fitness = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if state[i] != state[j] and abs(state[i] - state[j]) != abs(i - j):
                fitness += 1
    return float(fitness)


def is_goal(state):
    goal = (len(state) * (len(state) - 1)) / 2
    if fitness(state) == goal:
        return True
    else:
        return False


def fitness_probs(population):
    probabilities = []
    total_fitness = 0
    for pop in population:
        total_fitness += fitness(pop)
    for pop in population:
        probability = fitness(pop) / total_fitness
        probabilities.append(probability)
    return probabilities


def select_parents(population, probs):
    index = []
    for i in range(len(population)):
        index.append(i)

    n = np.random.choice(index, size=2, p=probs)
    return (population[n[0]], population[n[1]])


def reproduce(parent1, parent2):
    n = len(parent1)
    c = np.random.randint(0, n)
    return (parent1[0:c] + parent2[c:n])


def mutate(state, m_rate=0.1):
    f = np.random.uniform(0, 1)
    if f > m_rate:
        return state
    # else if less than or equal to m_rate
    else:
        n = len(state)
        first_sample = np.random.randint(0, n)
        second_sample = np.random.randint(0, n)
        state_list = list(state)
        state_list[first_sample] = second_sample
        state = tuple(state_list)
        return state

# returns the best individual in the population, and num of iterations executed.


def genetic_algorithm(population, m_rate=0.1, max_iters=5000):
    num_iters = 0
    contains_goal = False
    list = []
    while not contains_goal and num_iters < max_iters:
        probabilities = fitness_probs(population)
        population2 = []
        for i in range(len(population)):
            (parent1, parent2) = select_parents(population, probabilities)
            child = reproduce(parent1, parent2)
            mchild = mutate(child, m_rate)
            population2.append(mchild)
        for pop in population2:
            if is_goal(pop):
                contains_goal = True
                best_individual = pop
        population = population2
        num_iters += 1

    for p in population:
        list.append(fitness(p))
    i = np.argmax(list)
    return population[i], num_iters


def visualize_nqueens_solution(n_queens, file_name):
    n = len(n_queens)
    n_queens_array = []
    for row in range(n):
        rows = []
        for col in range(n):
            if row == n_queens[col]:
                rows.append(1)
            else:
                rows.append(0)
        n_queens_array.append(rows)
    plt.figure(figsize=(n, n))
    sns.heatmap(n_queens_array, cmap='Purples',
                linewidths=1.5, linecolor='k', cbar=False)
    plt.savefig(file_name, format='png')
    plt.close()
