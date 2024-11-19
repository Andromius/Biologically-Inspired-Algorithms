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

def calculate_new_position(individual: NDArray[np.float64], leader: NDArray[np.float64], t: float):
    prt_vector = np.array([1 if np.random.uniform() < PRT_THRESHOLD else 0 for _ in range(DIMENSIONS)])
    candidate_pos_individual = np.clip(individual + (leader - individual) * t * prt_vector, BOUNDS[0], BOUNDS[1])
    if OBJECTIVE_FUNCTION(candidate_pos_individual) < OBJECTIVE_FUNCTION(individual):
        return candidate_pos_individual
    return individual

population = generate_swarm()
global_best_solution = get_best_from_population(population)

migrations_xy_values = []
migrations_xy_values.append(np.vstack(deepcopy(population)))
migrations_z_values = []
start = time()
for _ in range(MAX_MIGRATIONS):
    t = 0
    while t <= PATH_LENGTH:
        population = np.apply_along_axis(calculate_new_position, axis=1, arr=np.vstack(population), leader=global_best_solution, t=t)
        t += STEP
        migrations_xy_values.append(population)
    global_best_solution = get_best_from_population(population)

print(f'Elapsed: {time() - start}')

X, Y, Z = animation_students.make_surface(min=BOUNDS[0], max=BOUNDS[1], function=OBJECTIVE_FUNCTION, step=BOUNDS[1]*0.05)
for migration_xy_value in migrations_xy_values:
    migrations_z_values.append(np.apply_along_axis(OBJECTIVE_FUNCTION, axis=1, arr=migration_xy_value))

animation_students.render_anim(X, Y, Z, migrations_xy_values, migrations_z_values, False, 50)
