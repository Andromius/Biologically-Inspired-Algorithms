import Differential_Evolution.constants as cons
import CV01.animation_students as anim
from Differential_Evolution.main import generate_individuals
import numpy as np

LIGHT_ABSORPTION_COEFFICIENT = 2
NUM_GENERATIONS = 100
NUM_FIREFLIES = 50
DIMENSIONS = 2
OBJECTIVE_FUNCTION = anim.schwefel
BOUNDS = cons.SCHWEFEL
ATTRACTIVNESS = 1
RANDOMNESS_SCALING_PARAMETER = 0.3

def calculate_light_intensity(value: float, firefly, target_firefly):
    return value * np.pow(np.e, -LIGHT_ABSORPTION_COEFFICIENT * np.linalg.norm(firefly - target_firefly))

def calculate_attractiveness(firefly, target_firefly):
    return ATTRACTIVNESS / (1 + np.linalg.norm(firefly - target_firefly))

def move_towards_target(firefly, target_firefly):
    firefly += calculate_attractiveness(firefly, target_firefly) * (target_firefly - firefly)
    firefly += RANDOMNESS_SCALING_PARAMETER * np.random.normal(size=2)

def do_FA(dimensions, swarm_size, max_migrations, max_ofe, objective_function, bounds):
    NUM_GENERATIONS = max_migrations
    DIMENSIONS = dimensions
    NUM_FIREFLIES = swarm_size
    OBJECTIVE_FUNCTION = objective_function
    BOUNDS = bounds
    population = generate_individuals(NUM_FIREFLIES, BOUNDS, DIMENSIONS)
    best_solution = None
    best_value = float('inf')
    all_solutions = []
    all_values = []
    num_ofe = 0
    for _ in range(NUM_GENERATIONS):
        current_solutions = []
        current_values = []
        for current_firefly in population:
            value = OBJECTIVE_FUNCTION(current_firefly)
            num_ofe += 1
            if num_ofe >= max_ofe:
                return best_value
            for target_firefly in population:
                if current_firefly is target_firefly:
                    continue

                current_firefly_light_intensity = calculate_light_intensity(value, current_firefly, target_firefly)
                target_firefly_light_intensity = calculate_light_intensity(OBJECTIVE_FUNCTION(target_firefly), current_firefly, target_firefly)
                num_ofe += 1
                if num_ofe >= max_ofe:
                    return best_value
                if target_firefly_light_intensity < current_firefly_light_intensity:
                    move_towards_target(current_firefly, target_firefly)
                value = OBJECTIVE_FUNCTION(current_firefly)
                num_ofe += 1
                if num_ofe >= max_ofe:
                    return best_value
            current_values.append(value)
            current_solutions.append(np.copy(current_firefly))
        
        min_idx = np.argmin(current_values)
        if current_values[min_idx] < best_value:
            best_value = current_values[min_idx]
            best_solution = current_solutions[min_idx]
    return best_value

if __name__ == '__main__':
    population = generate_individuals(NUM_FIREFLIES, BOUNDS, DIMENSIONS)
    best_solution = None
    best_value = float('inf')
    all_solutions = []
    all_values = []
    for _ in range(NUM_GENERATIONS):
        current_solutions = []
        current_values = []
        for current_firefly in population:
            value = OBJECTIVE_FUNCTION(current_firefly)
            for target_firefly in population:
                if current_firefly is target_firefly:
                    continue

                current_firefly_light_intensity = calculate_light_intensity(value, current_firefly, target_firefly)
                target_firefly_light_intensity = calculate_light_intensity(OBJECTIVE_FUNCTION(target_firefly), current_firefly, target_firefly)
                if target_firefly_light_intensity < current_firefly_light_intensity:
                    move_towards_target(current_firefly, target_firefly)
                value = OBJECTIVE_FUNCTION(current_firefly)
            current_values.append(value)
            current_solutions.append(np.copy(current_firefly))
        
        all_solutions.append(np.vstack(current_solutions))
        all_values.append(current_values)
        min_idx = np.argmin(current_values)
        if current_values[min_idx] < best_value:
            best_value = current_values[min_idx]
            best_solution = current_solutions[min_idx]
    
    X, Y, Z = anim.make_surface(min=BOUNDS[0], max=BOUNDS[1], function=OBJECTIVE_FUNCTION, step=BOUNDS[1]*0.05)
    anim.render_anim(X, Y, Z, all_solutions, all_values, False, 100)
                
        