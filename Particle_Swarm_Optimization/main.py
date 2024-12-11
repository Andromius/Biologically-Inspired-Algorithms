from CV01 import animation_students
from CV01 import constants
import numpy as np
from numpy.typing import NDArray
from time import time

PERSONAL_LEARNING_CONSTANT = 1.5
GLOBAL_LEARNING_CONSTANT = 2
INERTIA_WEIGHT_S = 0.9
INERTIA_WEIGHT_E = 0.4
MIN_VELOCITY = -10
MAX_VELOCITY = 10

def generate_swarm(count: int, dimensions: int, bounds: tuple):
  matrix = np.zeros((count, 3), dtype=object)  # Use object dtype to store arrays

  for i in range(count):
    array1 = np.random.uniform(low=bounds[0], high=bounds[1], size=dimensions)
    array2 = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY, size=dimensions)
    matrix[i] = (array1, array2, array1.copy())

  return matrix

def get_best_from_swarm(swarm, objective_function: callable):
    evaluations = np.apply_along_axis(objective_function, axis=1, arr=np.vstack(swarm[:,2]))
    min_value = np.min(evaluations)
    min_idx = np.argmin(evaluations)
    return swarm[min_idx, 2].copy(), min_value

def calculate_inertia_weight(current_migration: int, max_migrations):
    numerator = np.multiply(np.subtract(INERTIA_WEIGHT_S, INERTIA_WEIGHT_E), current_migration)
    term = np.divide(numerator, max_migrations)
    return np.subtract(INERTIA_WEIGHT_S, term)

def calculate_new_velocity(particle: NDArray[np.float64], current_migration: int, global_best_solution, max_migrations):
    r1 = np.random.uniform()
    inertia_weight = calculate_inertia_weight(current_migration, max_migrations)
    particle[1] *= inertia_weight
    particle[1] += r1 * PERSONAL_LEARNING_CONSTANT * (particle[2] - particle[0])
    particle[1] += r1 * GLOBAL_LEARNING_CONSTANT * (global_best_solution - particle[0])
    particle[1] = np.clip(particle[1], MIN_VELOCITY, MAX_VELOCITY)
    return particle

def calculate_new_position(particle: NDArray[np.float64], bounds):
   particle[0] += particle[1]
   particle[0] = np.clip(particle[0], bounds[0], bounds[1])
   return particle

def evaluate_new_position(particle: NDArray[np.float64], objective_function: callable, num_ofe):
    value = objective_function(particle[0])
    if value < objective_function(particle[2]):
        particle[2] = particle[0].copy()
    num_ofe += 2
    return value, num_ofe

def do_PSO(dimensions, swarm_size, max_migrations, max_ofe, objective_function, bounds):
    swarm = generate_swarm(swarm_size, dimensions, bounds)
    global_best_solution, global_best_value = get_best_from_swarm(swarm, objective_function)

    num_ofe = swarm_size
    xy_values = []
    z_values = []
    # start = time()
    for current_migration in range(max_migrations):
        new_position_evaluations = []
        for i, particle in enumerate(swarm):
            calculate_new_velocity(particle, current_migration, global_best_solution, max_migrations)
            calculate_new_position(particle, bounds)
            value, num_ofe = evaluate_new_position(particle, objective_function, num_ofe)
            if num_ofe >= max_ofe:
                return global_best_value
            
            if value < global_best_value:
                global_best_value = value
                global_best_solution = particle[0].copy()
            new_position_evaluations.append(value)
        xy_values.append(np.vstack(swarm[:, 0]))
        z_values.append(new_position_evaluations)
    return global_best_value
    # print(f'Elapsed: {time() - start}')
    # X, Y, Z = animation_students.make_surface(min=bounds[0], max=bounds[1], function=objective_function, step=bounds[1]*0.05)
    # animation_students.render_anim(X, Y, Z, xy_values, z_values, False, 50)
