from CV01 import animation_students
from CV01 import constants
import numpy as np
from numpy.typing import NDArray
from time import time

DIMENSIONS = 2
SWARM_SIZE = 1000
MAX_MIGRATIONS = 200
PERSONAL_LEARNING_CONSTANT = 1.5
GLOBAL_LEARNING_CONSTANT = 2
INERTIA_WEIGHT_S = 0.9
INERTIA_WEIGHT_E = 0.4
MIN_VELOCITY = -10
MAX_VELOCITY = 10
bounds = constants.GRIEWANGK

def get_best_from_swarm(swarm, objective_function: callable):
    evaluations = np.apply_along_axis(objective_function, axis=1, arr=np.vstack(swarm[:,2]))
    min_value = np.min(evaluations)
    min_idx = np.argmin(evaluations)
    return swarm[min_idx, 2].copy(), min_value

def calculate_inertia_weight(current_migration: int):
    numerator = np.multiply(np.subtract(INERTIA_WEIGHT_S, INERTIA_WEIGHT_E), current_migration)
    term = np.divide(numerator, MAX_MIGRATIONS)
    return np.subtract(INERTIA_WEIGHT_S, term)

def calculate_new_velocity(particle: NDArray[np.float64], current_migration: int, global_best_solution: NDArray[np.float64]):
    r1 = np.random.uniform()
    inertia_weight = calculate_inertia_weight(current_migration)
    particle[1] *= inertia_weight
    particle[1] += r1 * PERSONAL_LEARNING_CONSTANT * (particle[2] - particle[0])
    particle[1] += r1 * GLOBAL_LEARNING_CONSTANT * (global_best_solution - particle[0])
    particle[1] = np.clip(particle[1], MIN_VELOCITY, MAX_VELOCITY)
    return particle

def calculate_new_position(particle: NDArray[np.float64]):
   particle[0] += particle[1]
   particle[0] = np.clip(particle[0], bounds[0], bounds[1])
   return particle

def evaluate_personal_best(personal_best: NDArray[np.float64], objective_function: callable):
    return objective_function(personal_best)

def evaluate_new_position(particle: NDArray[np.float64], objective_function: callable):
    value = objective_function(particle[0])
    if value < objective_function(particle[2]):
        particle[2] = particle[0].copy()
    return value

def update_particle(particle: NDArray[np.float64], current_migration: int, global_best_solution: NDArray[np.float64], objective_function: callable):
    calculate_new_velocity(particle, current_migration, global_best_solution)
    calculate_new_position(particle)
    return evaluate_new_position(particle, objective_function)

def generate_swarm(count: int, dimensions: int, bounds: tuple):
  matrix = np.zeros((count, 3), dtype=object)  # Use object dtype to store arrays

  for i in range(count):
    array1 = np.random.uniform(low=bounds[0], high=bounds[1], size=dimensions)
    array2 = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY, size=dimensions)
    matrix[i] = (array1, array2, array1.copy())

  return matrix

objective_function = animation_students.griewangk

#current_migration = 0
swarm = generate_swarm(SWARM_SIZE, DIMENSIONS, bounds)
global_best_solution, global_best_value = get_best_from_swarm(swarm, objective_function)

xy_values = []
z_values = []
start = time()
for current_migration in range(MAX_MIGRATIONS):
    new_position_evaluations = np.apply_along_axis(update_particle, 1, swarm, current_migration, global_best_solution, objective_function)
    candidate_global_best_value = np.min(new_position_evaluations)
    if candidate_global_best_value < global_best_value:
        global_best_value = candidate_global_best_value
        global_best_solution = swarm[np.argmin(new_position_evaluations), 2].copy()
    #current_migration += 1
    xy_values.append(np.vstack(swarm[:, 0]))
    z_values.append(new_position_evaluations)

print(f'Elapsed: {time() - start}')
X, Y, Z = animation_students.make_surface(min=bounds[0], max=bounds[1], function=objective_function, step=bounds[1]*0.05)
animation_students.render_anim(X, Y, Z, xy_values, z_values, False, 50)
