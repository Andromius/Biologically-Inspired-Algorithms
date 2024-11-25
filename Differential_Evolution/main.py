from copy import deepcopy
import sys
from constants import *
import numpy as np
from animation_students import (
    render_graph,
    render_anim,
    make_surface,
    sphere,
    schwefel,
    rosenbrock,
    rastrigin,
    griewangk,
    levy,
    michalewicz,
    zakharov,
    ackley
)

def get_best_solution_from_population(population: list[tuple], function: callable):
    values = []
    for individual in population:
        values.append(function(individual))
    best_value = min(values)
    return population[values.index(best_value)], best_value

def printProgress(iteration, maxIterations):
    percent = (iteration / maxIterations) * 100
    sys.stdout.write(f'\r\033[1;38mRuning Generations:\033[0m \t\t\t\t\t\t\t\t[{percent:.2f}%]')
    if(percent == 100):
        sys.stdout.write("\n")
    sys.stdout.flush()

def generate_individuals(count: int, bounds: tuple, dimensions: int):
    individuals = []
    for _ in range(count):
        individuals.append(np.random.uniform(low=bounds[0], high=bounds[1], size=dimensions))
    return individuals
    
dimensions = 2
func = (SCHWEFEL, schwefel)
population = generate_individuals(count=50, bounds=func[0], dimensions=dimensions)

best_solutions = []
best_values = []
current_generation = 0
max_generations = 200
scaling_factor = 0.5
crossover_range = 0.9

best_solution, best_value = get_best_solution_from_population(population, func[1])
best_solutions.append(best_solution)
best_values.append(best_value)

printProgress(current_generation, max_generations)
while current_generation < max_generations:
    new_population = deepcopy(population)
    for i, individual in enumerate(population):
        r1, r2, r3 = np.random.choice([index for index in range(len(population)) if index != i], 3, replace=False)
        mutation_vector = np.clip(np.subtract(population[r1], population[r2]) * scaling_factor + population[r3], func[0][0], func[0][1])
        trial_vector = np.zeros(dimensions)
        j_rnd = np.random.randint(0, dimensions)

        for j in range(dimensions):
            if np.random.uniform() < crossover_range or j == j_rnd:
                trial_vector[j] = mutation_vector[j]
            else:
                trial_vector[j] = individual[j]

        trial_vector_value = func[1](trial_vector)

        if trial_vector_value <= func[1](individual):
            new_population[i] = trial_vector
        population = new_population
    
    best_solution, best_value = get_best_solution_from_population(population, func[1])
    if not any(np.array_equal(solution, best_solution) for solution in best_solutions):
        best_solutions.append(best_solution)
        best_values.append(best_value)
    
    current_generation += 1
    printProgress(current_generation, max_generations)
print(f'Number of solutions found: {len(best_solutions)}\nBest solution: {best_solutions[-1]} with value {best_values[-1]}')
best_solutions = np.array(best_solutions)
best_values = np.array(best_values)

best_solutions = [best_solutions[i].reshape(1, -1) for i in range(len(best_solutions))]
best_values = [np.array([best_values[i]]) for i in range(len(best_values))]

X_surt, Y_surf, Z_surf = make_surface(
    min=func[0][0], max=func[0][1], function=func[1], step=0.5
)

render_anim(X_surt, Y_surf, Z_surf, best_solutions, best_values)