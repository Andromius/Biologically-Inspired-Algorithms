from Particle_Swarm_Optimization.main import do_PSO
from Firefly_Algorithm.main import do_FA
from Teaching_Learning_Based_Optimization.main import do_TLBO
from SOMA_AllToOne.main import do_SOMA
from Differential_Evolution.main import do_DE
import pandas as pd
import Differential_Evolution.constants as cons
import CV01.animation_students as anim
from time import time


values = []
dimensions = 30
pop_size = 30
max_ofe = 3000
num_experiments = 30
objective_function = anim.schwefel
bounds = cons.SCHWEFEL
deltas = []

for algorithm in [do_PSO, do_FA, do_TLBO, do_SOMA, do_DE]:
    results = []
    start = time()
    for experiment in range(num_experiments):
        results.append(algorithm(dimensions, pop_size, 9999, max_ofe, objective_function, bounds))
    deltas.append(time() - start)
    values.append(results)
    print("Next..")

data = {
    'PSO': values[0],
    'FA': values[1],
    'TLBO': values[2],
    'SOMA': values[3],
    'DE': values[4]
}

df = pd.DataFrame(data)

means = df.mean()
std_devs = df.std()

df.loc['Mean'] = means
df.loc['Std_Dev'] = std_devs
df.loc['Delta'] = deltas

excel_file = "algo_comp_output.xlsx"
df.to_excel(excel_file, index=True)

# Display the DataFrame
print(df)

