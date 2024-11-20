from CV01 import animation_students
from CV01 import constants
import numpy as np
from numpy.typing import NDArray
from time import time
from copy import deepcopy

DIMENSIONS = 2
POPULATION_SIZE = 20
MAX_MIGRATIONS = 100
PRT_THRESHOLD = 0.4
STEP = 0.11 # (0, 1>
PATH_LENGTH = 3 # (0, 5>
BOUNDS = constants.SCHWEFEL
OBJECTIVE_FUNCTION = animation_students.schwefel

def generate_swarm():
    matrix = np.zeros((POPULATION_SIZE, DIMENSIONS))
    for i in range(DIMENSIONS):
        matrix[:, i] = np.random.uniform(BOUNDS[0], BOUNDS[1], POPULATION_SIZE)
    return matrix

def get_best_from_population(population):
    evaluations = np.apply_along_axis(OBJECTIVE_FUNCTION, axis=1, arr=population)
    min_idx = np.argmin(evaluations)
    return population[min_idx].copy()

population = generate_swarm()
global_best_solution = get_best_from_population(population)

migrations_xy_values = []
migrations_xy_values.append(np.vstack(deepcopy(population)))
migrations_z_values = []
start = time()

for _ in range(MAX_MIGRATIONS):
    new_pop = []
    for idx, individual in enumerate(population):
        start_individual = individual.copy()
        current_best = start_individual
        current_best_val = OBJECTIVE_FUNCTION(current_best)
        t = 0
        while t <= PATH_LENGTH:
            prt_vector = np.random.rand(DIMENSIONS) < PRT_THRESHOLD
            candidate_pos_individual = np.clip(individual + (global_best_solution - individual) * t * prt_vector, BOUNDS[0], BOUNDS[1])
            candidate_val = OBJECTIVE_FUNCTION(candidate_pos_individual)
            if candidate_val < current_best_val:
                current_best = candidate_pos_individual
                current_best_val = candidate_val 
            t += STEP
        new_pop.append(current_best)
    population = new_pop
    migrations_xy_values.append(np.vstack(deepcopy(population)))
    global_best_solution = get_best_from_population(population)

print(f'Elapsed: {time() - start}')
X, Y, Z = animation_students.make_surface(min=BOUNDS[0], max=BOUNDS[1], function=OBJECTIVE_FUNCTION, step=BOUNDS[1]*0.05)
for migration_xy_value in migrations_xy_values:
    migrations_z_values.append(np.apply_along_axis(OBJECTIVE_FUNCTION, axis=1, arr=migration_xy_value))

animation_students.render_anim(X, Y, Z, migrations_xy_values, migrations_z_values, False, 50)
