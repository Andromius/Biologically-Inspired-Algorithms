import numpy as np
import constants as cons
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


def blind_search(dimensions, iterations, search_range, xy_data: list, z_data: list, function):
    best_solution = None
    best_value = float('inf')

    for _ in range(iterations):
        candidate = np.random.uniform(low=search_range[0], high=search_range[1], size=dimensions)
        
        value = function(candidate)

        if value < best_value:
            xy_data.append(candidate)
            z_data.append(value)
            best_value = value
            best_solution = candidate

    xy_data = np.array(xy_data)
    z_data = np.array(z_data)
    return best_solution, best_value

def hill_climbing(dimensions, iterations, search_range, xy_data: list, z_data: list, function, neighbors, radius):
    def evaluate_candidates(candidates):
        local_best_solution = None
        local_best_value = float('inf')
        candidates = np.clip(candidates, search_range[0], search_range[1])
        for candidate in candidates:
            value = function(candidate)
            if value < local_best_value:
                local_best_solution = candidate
                local_best_value = value
        return local_best_solution, local_best_value

    best_solution = np.random.uniform(low=search_range[0], high=search_range[1], size=dimensions)
    best_value = function(best_solution)
    xy_data.append(best_solution)
    z_data.append(best_value)

    for _ in range(iterations):
        candidates = np.random.normal(size=(neighbors, dimensions), loc=best_solution, scale=radius)
        candidate, value = evaluate_candidates(candidates)
        if value < best_value:
            xy_data.append(candidate)
            z_data.append(value)
            best_value = value
            best_solution = candidate

    xy_data = np.array(xy_data)
    z_data = np.array(z_data)
    return best_solution, best_value


def simulated_annealing(dimensions, iterations, search_range, xy_data: list, z_data: list, function, neighbors, radius, rate, temp, min_temp):
    best_solution = np.random.uniform(low=search_range[0], high=search_range[1], size=dimensions)
    best_value = function(best_solution)
    xy_data.append(best_solution)
    z_data.append(best_value)
    current_temp = temp

    while current_temp > min_temp:
        candidate = np.random.normal(size=best_solution.shape, loc=best_solution, scale=radius)
        candidate = np.clip(candidate, search_range[0], search_range[1])
        value = function(candidate)
        if value < best_value:
            xy_data.append(candidate)
            z_data.append(value)
            best_value = value
            best_solution = candidate
        else:
            r = np.random.uniform()
            if r < np.exp(-(value - best_value) / current_temp):
                best_value = value
                best_solution = candidate
                xy_data.append(candidate)
                z_data.append(value)
        current_temp = current_temp * rate

    xy_data = np.array(xy_data)
    z_data = np.array(z_data)
    return best_solution, best_value

func = (cons.LEVY, levy)
dimensions = 2
iterations = 1000 
search_range = func[0]
xy_data, z_data = [], []
best_solution, best_value = simulated_annealing(dimensions, iterations, search_range, xy_data, z_data, func[1], neighbors=100, radius=1, rate=0.9, temp=100, min_temp=0.5)
print(f'Best value found: {best_value} with solution {best_solution}')

xy_data_anim = [xy_data[i].reshape(1, -1) for i in range(len(xy_data))]
z_data_anim = [np.array([z_data[i]]) for i in range(len(z_data))]

X_surt, Y_surf, Z_surf = make_surface(
    min=func[0][0], max=func[0][1], function=func[1], step=0.1
)

#render_graph(X_surt, Y_surf, Z_surf)

render_anim(X_surt, Y_surf, Z_surf, xy_data_anim, z_data_anim)
