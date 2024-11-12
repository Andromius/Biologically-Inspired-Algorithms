import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def printProgress(iteration, maxIterations):
    percent = (iteration / maxIterations) * 100
    sys.stdout.write(f'\r\033[1;38mRuning Generations:\033[0m \t\t\t\t\t\t\t\t[{percent:.2f}%]')
    if(percent == 100):
        sys.stdout.write("\n")
    sys.stdout.flush()

def animate_connections(points, connection_orders, distance_matrix, title="Point Connections Animation"):
    """
    Creates an animation of scatter points connected by lines based on given connection orders.

    Parameters:
        points (array-like): Array of shape (n, 2) containing the (x, y) coordinates of the points.
        connection_orders (array-like): Array of arrays specifying different ways to connect the points.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 201)
    ax.set_ylim(-1, 201)
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    scatter = ax.scatter(points[:, 0], points[:, 1], color='blue')

    # Function to update the animation for each frame
    def update(frame):
        ax.clear()
        ax.set_xlim(-1, 201)
        ax.set_ylim(-1, 201)
        ax.set_title(f"Distance: {compute_distance(connection_orders[frame], distance_matrix, False):.2f}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        
        # Get the current connection order
        order = connection_orders[frame]
        ordered_points = points[order]

        # Plot the points and the connecting lines
        scatter = ax.scatter(points[:, 0], points[:, 1], color='blue')
        ax.plot(ordered_points[:, 0], ordered_points[:, 1], color='orange', linestyle='-', linewidth=2)
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(connection_orders), repeat=False, interval=200)
    plt.show()



def calculate_euclidean_distance(first, second):
    first_part = first[0] - second[0]
    second_part = first[1] - second[1]
    return np.sqrt((np.power(first_part, 2) + np.power(second_part, 2)))

def generate_cities(count):
    return [np.random.uniform(low=0, high=200, size=2) for _ in range(count)]

def calculate_distance_matrix(cities):
    count = len(cities)
    matrix = np.zeros((count, count))
    
    for i in range(count):
        for j in range(i, count):
            if i == j:
                continue
            distance = calculate_euclidean_distance(cities[i], cities[j])
            matrix[i,j] = distance
            matrix[j,i] = distance
    return matrix

def generate_individuals(individual_count, city_count):
    cities = [i for i in range(city_count)]
    individuals = []
    for _ in range(individual_count):
        np.random.shuffle(cities)
        individuals.append(cities.copy())
    return individuals

def compute_distance(individual, matrix, compute_end_start=True):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += matrix[individual[i], individual[i + 1]]
    if compute_end_start:
        total_distance += matrix[individual[0], individual[len(individual) - 1]]
    return total_distance

def choose_random_individual(population: list, individual_to_exclude):
    population_copy = population.copy()
    population_copy.remove(individual_to_exclude)
    return population_copy[np.random.choice([i for i in range(len(population_copy))])]

def crossover(A: list, B: list):
    offspring = [A[i] for i in range(int(len(A)/2))]
    B_copy = B.copy()
    for val in offspring:
        B_copy.remove(val)
    offspring.extend(B_copy)
    return offspring

def mutate(A: list):
    first_index = np.random.randint(0,len(A))
    arr = [i for i in range(len(A))]
    arr.remove(first_index)
    second_index = np.random.choice(arr)
    A[first_index], A[second_index] = A[second_index], A[first_index]

def evaluate_population(population, matrix):
    distances = []
    for individual in population:
        distances.append(compute_distance(individual, matrix))
    return distances

def get_best_solution_from_population(population: list[list], distances: list):
    best = population[distances.index(min(distances))].copy()
    best.append(best[0])
    return best

individual_count = 100
generation_count = 1000
city_count = 20


start = time.time()
cities = generate_cities(city_count)
distance_matrix = calculate_distance_matrix(cities)
population = generate_individuals(individual_count, city_count)

best_solutions = []
best_solutions.append(get_best_solution_from_population(population, evaluate_population(population, distance_matrix)))

printProgress(0, generation_count)
for i in range(generation_count):
    new_population = population.copy()
    for j in range(individual_count):
        parent_A = population[j]
        parent_B = choose_random_individual(population, parent_A)
        offspring_AB = crossover(parent_A, parent_B)
        if np.random.uniform() < 0.5:
            mutate(offspring_AB)
        
        if compute_distance(offspring_AB, distance_matrix) < compute_distance(parent_A, distance_matrix):
            new_population[j] = offspring_AB
    population = new_population

    best = get_best_solution_from_population(population, evaluate_population(population, distance_matrix))
    if best not in best_solutions:
        best_solutions.append(best)
    printProgress(i, generation_count)

printProgress(generation_count, generation_count)
print(f"Elapsed: {time.time() - start}")
input()
animate_connections(np.array(cities), best_solutions, distance_matrix)