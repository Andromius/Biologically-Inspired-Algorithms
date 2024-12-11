from Differential_Evolution.main import generate_individuals
import Differential_Evolution.constants as cons
import CV01.animation_students as anim
import numpy as np

NUM_GENERATIONS = 100
NUM_STUDENTS = 100
DIMENSIONS = 2
OBJECTIVE_FUNCTION = anim.schwefel
BOUNDS = cons.SCHWEFEL


def do_TLBO(dimensions, swarm_size, max_migrations, max_ofe, objective_function, bounds):
    NUM_GENERATIONS = max_migrations
    NUM_STUDENTS = swarm_size
    DIMENSIONS = dimensions
    num_ofe = NUM_STUDENTS
    OBJECTIVE_FUNCTION = objective_function
    BOUNDS = bounds
    students = generate_individuals(NUM_STUDENTS, BOUNDS, DIMENSIONS)
    fitnesses = [OBJECTIVE_FUNCTION(student) for student in students]
    indices = np.arange(0, NUM_STUDENTS)
    global_best_value = float('inf')

    for _ in range(NUM_GENERATIONS):
        teacher_idx = np.argmin(fitnesses)
        teacher = students[teacher_idx]
        students_mean_position = np.mean(students, axis=0)
        teaching_factor = np.random.randint(1, 3)

        candidate_teacher = teacher + np.random.uniform(size=DIMENSIONS) * (teacher - teaching_factor * students_mean_position)
        candidate_teacher = np.clip(candidate_teacher, BOUNDS[0], BOUNDS[1])

        candidate_fitness = OBJECTIVE_FUNCTION(candidate_teacher)
        num_ofe += 1
        if num_ofe >= max_ofe:
            return global_best_value
        if candidate_fitness < fitnesses[teacher_idx]:
            students[teacher_idx] = np.copy(candidate_teacher)
            fitnesses[teacher_idx] = candidate_fitness
        
        for idx, student in enumerate(students):
            candidate_student = None
            partner_idx = np.random.choice(indices[(indices != teacher_idx) & (indices != idx)])
            partner_student = students[partner_idx]
            if fitnesses[idx] < fitnesses[partner_idx]:
                candidate_student = student + np.random.uniform(size=DIMENSIONS) * (student - partner_student)
            else:
                candidate_student = student + np.random.uniform(size=DIMENSIONS) * (partner_student - student)
            
            candidate_student = np.clip(candidate_student, BOUNDS[0], BOUNDS[1])
            candidate_fitness = OBJECTIVE_FUNCTION(candidate_student)
            num_ofe += 1
            if num_ofe >= max_ofe:
                return global_best_value
            if candidate_fitness < fitnesses[idx]:
                students[idx] = np.copy(candidate_student)
                fitnesses[idx] = candidate_fitness
        
        best_fitness_idx = np.argmin(fitnesses)
        if fitnesses[best_fitness_idx] < global_best_value:
            global_best_value = fitnesses[best_fitness_idx]
           

if __name__ == '__main__':
    students = generate_individuals(NUM_STUDENTS, BOUNDS, DIMENSIONS)
    fitnesses = [OBJECTIVE_FUNCTION(student) for student in students]
    indices = np.arange(0, NUM_STUDENTS)
    all_solutions = []
    all_values = []
    for _ in range(NUM_GENERATIONS):
        teacher_idx = np.argmin(fitnesses)
        teacher = students[teacher_idx]
        students_mean_position = np.mean(students, axis=0)
        teaching_factor = np.random.randint(1, 3)

        candidate_teacher = teacher + np.random.uniform(size=DIMENSIONS) * (teacher - teaching_factor * students_mean_position)
        candidate_teacher = np.clip(candidate_teacher, BOUNDS[0], BOUNDS[1])

        candidate_fitness = OBJECTIVE_FUNCTION(candidate_teacher)
        if candidate_fitness < fitnesses[teacher_idx]:
            students[teacher_idx] = np.copy(candidate_teacher)
            fitnesses[teacher_idx] = candidate_fitness
        
        for idx, student in enumerate(students):
            candidate_student = None
            partner_idx = np.random.choice(indices[(indices != teacher_idx) & (indices != idx)])
            partner_student = students[partner_idx]
            if fitnesses[idx] < fitnesses[partner_idx]:
                candidate_student = student + np.random.uniform(size=DIMENSIONS) * (student - partner_student)
            else:
                candidate_student = student + np.random.uniform(size=DIMENSIONS) * (partner_student - student)
            

            candidate_student = np.clip(candidate_student, BOUNDS[0], BOUNDS[1])
            candidate_fitness = OBJECTIVE_FUNCTION(candidate_student)
            if candidate_fitness < fitnesses[idx]:
                students[idx] = np.copy(candidate_student)
                fitnesses[idx] = candidate_fitness
        all_solutions.append(np.copy(students))
        all_values.append(np.copy(fitnesses))
    X, Y, Z = anim.make_surface(min=BOUNDS[0], max=BOUNDS[1], function=OBJECTIVE_FUNCTION, step=BOUNDS[1]*0.05)
    anim.render_anim(X, Y, Z, all_solutions, all_values, False, 10)
        