from CV04.main import generate_cities, calculate_distance_matrix, compute_distance
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

NUM_CITIES = 20
NUM_ANTS = NUM_CITIES
PHEROMONE_IMPORTANCE = 1
DISTANCE_IMPORTANCE = 2
NUM_MIGRATIONS = 200
VAPORIZATION_COEFFICIENT = 0.5
Q = 1

def animate_connections(points, connection_orders, distances, best_idx, title="ACO TSP"):
    """
    Creates an animation of scatter points connected by lines based on given connection orders.

    Parameters:
        points (array-like): Array of shape (n, 2) containing the (x, y) coordinates of the points.
        connection_orders (array-like): Array of arrays specifying different ways to connect the points.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    scatter = ax.scatter(points[:, 0], points[:, 1], color='blue')

    # Function to update the animation for each frame
    def update(frame):
        ax.clear()
        if frame == len(connection_orders) - 1:
            ax.set_title(f"Best migration: {best_idx + 1} Distance: {distances[frame]:.2f}")
        else:
            ax.set_title(f"Migration: {frame + 1} Distance: {distances[frame]:.2f}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        
        # Get the current connection order
        order = connection_orders[frame]
        ordered_points = points[order]

        # Plot the points and the connecting lines
        scatter = ax.scatter(points[:, 0], points[:, 1], color='blue')
        ax.plot(ordered_points[:, 0], ordered_points[:, 1], color='orange', linestyle='-', linewidth=2)
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(connection_orders), repeat=False, interval=10)
    plt.show()

def calculate_visibility_matrix(distance_matrix: NDArray[np.float64]):
    shape = distance_matrix.shape[0]
    visibility_matrix = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            if i == j: 
                continue
            value = 1 / distance_matrix[i,j]
            visibility_matrix[i,j] = value
            visibility_matrix[j,i] = value
    return visibility_matrix

def calculate_probabilities(pheromone_matrix, visibility_matrix, current_city):
    arr = []
    for target_city in range(NUM_CITIES):
        pheromone = pheromone_matrix[current_city, target_city] ** PHEROMONE_IMPORTANCE
        distance = visibility_matrix[current_city, target_city] ** DISTANCE_IMPORTANCE
        arr.append(pheromone * distance)
    
    sum_weights = np.sum(arr)
    for i in range(NUM_CITIES):
        if arr[i] <= 0:
            continue
        arr[i] = arr[i] / sum_weights
    return arr

cities = generate_cities(NUM_CITIES)
distance_matrix = calculate_distance_matrix(cities)
visibility_matrix = calculate_visibility_matrix(distance_matrix)
pheromone_matrix = np.ones(distance_matrix.shape)

best_solution = None
best_distance = float('inf')
best_idx = -1

solutions = []
distances = []
start = time.time()
for migration in range(NUM_MIGRATIONS):
    current_solutions = []
    for ant in range(NUM_ANTS):
        current_ant_visibility_matrix = np.copy(visibility_matrix)
        solution = [ant]
        current_city = solution[-1]
        current_ant_visibility_matrix[:, current_city] = 0
        for _ in range(NUM_CITIES - 1):
            probabilities = calculate_probabilities(pheromone_matrix, current_ant_visibility_matrix, current_city)
            r = np.random.uniform()
            cumulative = np.cumsum(probabilities)
            city = np.where((r < cumulative) & (cumulative > r))[0][0]
            solution.append(city)
            current_city = city
            current_ant_visibility_matrix[:, current_city] = 0
        solution.append(ant)
        current_solutions.append(solution)
    
    pheromone_matrix *= VAPORIZATION_COEFFICIENT
    current_distances = []
    for solution in current_solutions:
        current_distances.append(compute_distance(solution, distance_matrix, compute_end_start=False))
        for i in range(len(solution) - 2):
            idx_1 = solution[i]
            idx_2 = solution[i + 1]
            pheromone_matrix[idx_1, idx_2] += Q / current_distances[-1]

    min_idx = np.argmin(current_distances)
    if current_distances[min_idx] < best_distance:
        best_solution = current_solutions[min_idx]
        best_distance = current_distances[min_idx]
        best_idx = migration
    solutions.append(current_solutions[min_idx])
    distances.append(current_distances[min_idx])

solutions.append(best_solution)
distances.append(best_distance)
print(f"Elapsed: {time.time() - start}")
animate_connections(np.array(cities), solutions, distances, best_idx)
