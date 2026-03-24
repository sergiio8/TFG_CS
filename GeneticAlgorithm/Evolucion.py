# %%
#Imports generales
import math
import numpy as np
import random
import copy

#Imports gráficos
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rc
import re

#Imports AG
import sys
import deap
from deap import base, creator, tools

#Imports AC
import cellpylib as cpl

#Imports paralelización
#import multiprocessing
#from multiprocessing import Pool
from joblib import Parallel, delayed
import time
import os

#Metricas de similitd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score

from configuration import FrameworkConfig, CAConfig, GAConfig


# %%

def create_transition_rule(individual, ca_size_1, ca_size_2, ca_config):
    rule_table = np.array(individual, dtype=int)
    if ca_config.num_states == 2:
        if ca_config.ca_neighborhood == 'von Neumann':
            powers = np.array([16, 8, 4, 2, 1], dtype=int)
        elif ca_config.ca_neighborhood == 'Moore':
            powers = np.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=int)
        else:
            powers = np.array([2**i for i in range(2*int(ca_config.ca_neighborhood), -1, -1)], dtype=int)

    elif ca_config.num_states == 3:
        if ca_config.ca_neighborhood == 'von Neumann':
            powers = np.array([81, 27, 9, 3, 1], dtype=int)
        elif ca_config.ca_neighborhood == 'Moore':
            powers = np.array([6561, 2187, 729, 243, 81, 27, 9, 3, 1], dtype=int)
        else:
            powers = np.array([3**i for i in range(2*int(ca_config.ca_neighborhood), -1, -1)], dtype=int)

    elif ca_config.num_states == 4:
        if ca_config.ca_neighborhood == 'von Neumann':
            powers = np.array([256, 64, 16, 4, 1], dtype=int)
        elif ca_config.ca_neighborhood == 'Moore':
            powers = np.array([65536, 16384, 4096, 1024, 256, 64, 16, 4, 1], dtype=int)
        else:
            powers = np.array([4**i for i in range(2*int(ca_config.ca_neighborhood), -1, -1)], dtype=int)
    
    def my_rule(cells, r, t):
        if ca_config.ca_size[1] == 1:
            if r == 0:
                return ca_config.ca_row0_state[0]
            elif r == ca_config.ca_size[0] - 1:
                return ca_config.ca_row0_state[-1]
            
        else:

            if ca_config.ca_horizontal_boundary_conditions == 'customized':
                # Personalización de las condiciones de frontera horizontales
                pass

            elif ca_config.ca_horizontal_boundary_conditions == 'fixed':
                if r[0] == 0:
                    return ca_config.ca_row0_state[r[1]]
                elif r[0] == ca_config.ca_size[0] - 1:
                    return ca_config.ca_rowN_state[r[1]]
                

            if ca_config.ca_vertical_boundary_conditions == 'customized':
                # Personalización de las condiciones de frontera verticales
                pass

            elif ca_config.ca_vertical_boundary_conditions == 'fixed':
                if r[1] == 0:
                    return ca_config.ca_column0_state[r[0]]
                elif r[1] == ca_config.ca_size[1] - 1:
                    return ca_config.ca_columnN_state[r[0]]
                
        if ca_config.ca_neighborhood == 'von Neumann':
            neighbors = np.array([
                cells[1, 1], # Center
                cells[0, 1], # Up
                cells[1, 2], # Right
                cells[2, 1], # Down
                cells[1, 0]  # Left
            ], dtype=int)
        
        elif ca_config.ca_neighborhood == 'Moore':
            neighbors = np.array([
                cells[1, 1], # Center
                cells[0, 1], # North
                cells[0, 2], # Northeast
                cells[1, 2], # East
                cells[2, 2], # Southeast
                cells[2, 1], # South
                cells[2, 0], # Southwest
                cells[1, 0], # West
                cells[0, 0], # Northwest
            ], dtype=int)
        else:
            neighbors = cells.flatten().astype(int)

        '''Calculo del indice de la regla a partir de la configuración del vecindario'''
        idx = np.dot(neighbors, powers) 

        '''Devolvemos el nuevo estado de la célula central según el índice de la regla calculado previamente'''
        return rule_table[idx]
    
    return my_rule

'''Metrica accuracy'''
def accuracy_metric(ca_final, target, ca_config):
    return (np.sum(ca_final == target)/(ca_config.ca_size[0]*ca_config.ca_size[1]))

'''Metrica Jaccard'''
def jaccard_metric(ca_final, target, config):
    jaccard_states = []
    for state in range(config.ca.num_states):
        intersection = np.sum((ca_final == state) & (target == state))
        union = np.sum((target == state) | (ca_final == state))
        if union == 0:
            jaccard_states.append(1.0)  # Si no hay elementos en la unión, consideramos Jaccard como 1
        else:
            jaccard_states.append(intersection / union)
            
    if config.ca.num_states == 2:
        w0 = config.ga.weights_states_Jaccard[0]
        w1 = config.ga.weights_states_Jaccard[1]
        weighted_jaccard = (jaccard_states[0] * w0 + jaccard_states[1] * w1) / (w0 + w1)
        return weighted_jaccard  
    
    if config.ca.num_states == 3:
        w0 = config.ga.weights_states_Jaccard[0]
        w1 = config.ga.weights_states_Jaccard[1]
        w2 = config.ga.weights_states_Jaccard[2]
        weighted_jaccard = (jaccard_states[0] * w0 + jaccard_states[1] * w1 + jaccard_states[2] * w2) / (w0 + w1 + w2)
        return weighted_jaccard  
 
    else:
        return np.mean(jaccard_states)

'''Métrica SSIM'''
def ssim_metric(ca_final, target, ca_config):
    return ssim(ca_final, target, data_range = ca_config.num_states - 1, win_size=5) 

'''Metrica informacion mutua'''
def mutual_information_metric(ca_final, target, ca_config):
    N = ca_config.ca_size[0]*ca_config.ca_size[1]
    h_x = 0
    h_y = 0
    for i in range(ca_config.num_states):
        p_ca = np.sum(ca_final == i)/N
        p_target = np.sum(target == i)/N

        if p_ca != 0: log_p_ca = math.log(p_ca)
        else: log_p_ca = 0
        if p_target != 0: log_p_target = math.log(p_target)
        else: log_p_target = 0

        h_x = h_x - p_ca*log_p_ca
        h_y = h_y - p_target*log_p_target

    h_xy = 0

    for i in range(ca_config.num_states):
        for j in range(ca_config.num_states):
            p_ij = np.sum((ca_final == i) & (target == j))/N
            if (p_ij != 0): log_p_ij = math.log(p_ij)
            else: log_p_ij = 0
            h_xy = h_xy + p_ij*log_p_ij

    if (h_x + h_y) == 0:
        return 0
    return 2*(h_x + h_y + h_xy)/(h_x + h_y) 

'''Generación aleatoria de autómatas, teniendo en cuenta la frontera fija'''
def generate_CAs(ca_num, num_states, ca_size_1, ca_size_2, ca_config):
    CAs = []
    # Obtenemos la probabilidad de la configuración (si no la has definido aún, asume 0.0)
    prob_ruido = getattr(ca_config, 'random_probability', 0.0)
    
    if ca_size_2 == 1:
        for i in range(ca_num):
            ca = np.random.randint(0, num_states, size=ca_size_1)
            ca = np.expand_dims(ca, axis=0)
            CAs.append(ca)

    else:
        for i in range(ca_num): 
            if ca_config.random_initial_state:
                ca = np.copy(ca_config.target_state)
            else:
                ca = np.copy(ca_config.ca_initial_state)
                
            if prob_ruido > 0.0:
                # Generamos una matriz de booleanos: True donde cae el ruido, False donde se respeta la base
                mascara_ruido = np.random.rand(ca_size_1, ca_size_2) < prob_ruido
                
                # Generamos estados puros al azar
                valores_aleatorios = np.random.randint(0, num_states, size=(ca_size_1, ca_size_2))
                
                # Sustituimos: Donde haya True en la máscara, metemos el valor aleatorio; si no, dejamos la base
                ca = np.where(mascara_ruido, valores_aleatorios, ca)
                
            ca = np.expand_dims(ca, axis=0)
            CAs.append(ca)
            
    return CAs


'''Evolucion de los autómatas de la lista CAs mediante mi_regla'''
def evolved_CAs(CAs, mi_regla, ca_config):
    evolved_CAs_list = []
    if ca_config.ca_size[1] == 1:
        for i in range(len(CAs)):
            evolved_CA = cpl.evolve(CAs[i], timesteps=ca_config.ca_timesteps, apply_rule=mi_regla, r=int(ca_config.ca_neighborhood))
            evolved_CAs_list.append(evolved_CA)
    else:
        for i in range(len(CAs)):
            evolved_CA = cpl.evolve2d(CAs[i], timesteps=ca_config.ca_timesteps, neighbourhood = ca_config.ca_neighborhood, apply_rule=mi_regla)
            evolved_CAs_list.append(evolved_CA)

    return evolved_CAs_list

'''Funcion que genera la regla a partir de un individuo, evoluciona los autómatas con esa regla y devuelve el valor de fitness 
correspondiente.'''
def rule_and_evolve(ca_num, individual, CAs, ca_size_1, ca_size_2, ca_config):
    '''Creación de la regla de transición'''
    mi_regla = create_transition_rule(individual, ca_size_1, ca_size_2, ca_config)
    
    '''Evolución de los autómatas con la regla creada'''
    return evolved_CAs(CAs, mi_regla, ca_config)

def calculate_fitness(ca, config):
    '''Media ponderada entre SSIM, Jaccard y Accuracy'''
    fit = (config.ga.weight_SSIM * ssim_metric(ca[-1], config.ca.target_state, config.ca) + 
           config.ga.weight_Jaccard * jaccard_metric(ca[-1], config.ca.target_state, config) + 
           config.ga.weight_accuracy * accuracy_metric(ca[-1], config.ca.target_state, config.ca) + 
           config.ga.weight_mutual_information * mutual_information_metric(ca[-1], config.ca.target_state, config.ca))

    '''counts guarda el número de elementos de cada color en el estado final del autómata, y el objetivo es penalizar el fitness (reduciéndolo a la mitad)
    si alguno de los estados tiene menos del 10% de las células totales del autómata'''
    if (config.ga.penalized_fitness):
        counts = np.bincount(ca[-1].flatten(), minlength=config.ca.num_states)
        total_cells = ca[-1].size
        if np.any(counts < (config.ga.minimum_percentage * total_cells)):
            fit = fit * config.ga.penalization_factor 

    return fit


def evaluate(individual, config):

    if (config.ga.adaptative_fitness):

        CAs = generate_CAs(config.ga.ca_fitness//5, config.ca.num_states, config.ca.ca_size[0], config.ca.ca_size[1], config.ca)

        evolvedCAs = rule_and_evolve(config.ga.ca_fitness//5, individual, CAs, config.ca.ca_size[0], config.ca.ca_size[1], config.ca)

        '''Estudiar métricas para medir fitness, comparando el estado final del autómata con el estado objetivo'''
        fitness_values_1 = []
        for ca in evolvedCAs:
            fit = calculate_fitness(ca, config)
            fitness_values_1.append(fit)

        res1 = np.mean(fitness_values_1)
        if res1 < config.ga.adaptative_fitness_threshold:
            return (res1,)
            
        CAs = generate_CAs(config.ga.ca_fitness, config.ca.num_states, config.ca.ca_size[0], config.ca.ca_size[1], config.ca)
        
        fitness_values_2 = []
        for ca in evolvedCAs:
            fit = calculate_fitness(ca, config)
            fitness_values_2.append(fit)
        res2 = np.mean(fitness_values_2)
        return (res2,)

    else:
        CAs = generate_CAs(config.ga.ca_fitness, config.ca.num_states, config.ca.ca_size[0], config.ca.ca_size[1], config.ca)

        evolvedCAs = rule_and_evolve(config.ga.ca_fitness, individual, CAs, config.ca.ca_size[0], config.ca.ca_size[1], config.ca)

        '''Estudiar métricas para medir fitness, comparando el estado final del autómata con el estado objetivo'''
        fitness_values = []
        for ca in evolvedCAs:
            fit = calculate_fitness(ca, config)
            fitness_values.append(fit)

        res = np.mean(fitness_values)
        return (res,)
    

'''Funcion de mutacion personalizada: selecciona un gen al azar, lo cambia a uno distinto aleatoriamente y devuelve el individuo modificado'''
def mutChangeGen(individual, ca_config):
    gen = random.randint(0, ca_config.ind_size - 1)
    
    val = individual[gen]

    choices = [i for i in range(ca_config.num_states) if i != val]
    new_val = random.choice(choices)
    
    individual[gen] = new_val
    
    return individual,



# %%

def GeneticAlgorithm(config: FrameworkConfig, seed_rule: list = None):

    """
    Ejecuta el Algoritmo Genético usando DEAP.
    Recibe la configuración dictada por el CBR y, si es compatible, la regla semilla.
    """

    if config.ca.ca_neighborhood == 'von Neumann':
        config.ca.ind_size = config.ca.num_states ** 5
    elif config.ca.ca_neighborhood == 'Moore':
        config.ca.ind_size = config.ca.num_states ** 9
    else: # 1D
        config.ca.ind_size = config.ca.num_states ** (2 * int(config.ca.ca_neighborhood) + 1)
    print("\n[Evolución] Inicializando entorno DEAP...")
    
    # Limpiamos clases previas de DEAP por si se ejecuta varias veces
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    '''Creador de fitness y de individuo
    Fitness con único objetivo (weights = (1.0,) y máximo'''
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


    '''Especificación de genes, individuos y poblacion
    Individuos ternarios, attr_int entre 0 y num_states - 1, y en forma de lista
    '''
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, config.ca.num_states - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_int, config.ca.ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    '''Configuración del cruce y la selección (aunque no sea necesario), selecciona la mutacion personalizada (mutChangeGen) y
    registra la función de evaluacion personalizada del algoritmo '''
    if config.ga.cx_method == 'twoPoint':
        toolbox.register("mate", tools.cxTwoPoint)
    elif config.ga.cx_method == 'onePoint':
        toolbox.register("mate", tools.cxOnePoint)
    elif config.ga.cx_method == 'ordered':
        toolbox.register("mate", tools.cxOrdered)
    elif config.ga.cx_method == 'uniform':
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    elif config.ga.cx_method == 'partiallyMatched':
        toolbox.register("mate", tools.cxPartialyMatched)

    # --- SELECCIÓN ---
    if config.ga.sel_method == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=config.ga.tournament_size)
    elif config.ga.sel_method == 'roulette':
        toolbox.register("select", tools.selRoulette)
    elif config.ga.sel_method == 'best':
        toolbox.register("select", tools.selBest)
    elif config.ga.sel_method == 'random':
        toolbox.register("select", tools.selRandom)

    # --- MUTACIÓN Y EVALUACIÓN ---
    # Asumimos que mutChangeGen y evaluate ya están definidas en este script
    toolbox.register("mutate", mutChangeGen, ca_config=config.ca)
    
    # Es importante pasarle el config al evaluate para que sepa contra qué bandera medir (target_state)
    toolbox.register("evaluate", evaluate, config=config)

    # ---------------------------------------------------------
    # INYECCIÓN MEMÉTICA (Siembra guiada por el CBR)
    # ---------------------------------------------------------
    pop = toolbox.population(n=config.ga.pop_size)
    
    if seed_rule is not None and len(seed_rule) == config.ca.ind_size:
        num_clones = int(config.ga.pop_size * 0.05) # 2.5% clones exactos
        num_mutated = int(config.ga.pop_size * 0.05) # 2.5% clones mutados

        
        print(f"[Evolución] Sembrando población: {num_clones} clones y {num_mutated} mutados de la regla CBR.")
        print(f"Inyectamos la regla {seed_rule}")
        
        # 1. Clones exactos de la regla rotada
        for i in range(num_clones):
            pop[i] = creator.Individual(copy.deepcopy(seed_rule))
            
        # 2. Clones con ligera mutación
        for i in range(num_clones, num_clones + num_mutated):
            ind = creator.Individual(copy.deepcopy(seed_rule))
            # Aplicamos una mutación suave, 5% de los genes
            genes_a_mutar = random.sample(range(config.ca.ind_size), int(config.ca.ind_size*0.05))
            for g in genes_a_mutar:
                ind[g] = random.randint(0, config.ca.num_states - 1)
            pop[i] = ind


        
    print('Iniciando cálculo de fitness', flush = True)
    fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print('Cálculo de fitness terminado', flush = True)

    max_fitness_values = []
    
    # Pre-cargamos para evitar errores de referencia en las primeras iteraciones si adaptative_fitness no entra
    NUM_MUT = config.ga.num_mutations[0] if len(config.ga.num_mutations) > 0 else 1
    MUT_PROB = config.ga.mutation_probabilities[0] if len(config.ga.mutation_probabilities) > 0 else 0.1

    generaciones_sin_mejora = 0

    for g in range(config.ga.num_generations):
        print(f'--- Iniciando Generación {g} ---', flush=True)

        if (config.ga.classic_ga):
            
            '''Estrategia clásica'''
            elites = list(map(toolbox.clone, tools.selBest(pop, k= round(config.ga.elite_passing*config.ga.pop_size))))
            offspring = toolbox.select(pop, len(pop) - round(config.ga.elite_passing*config.ga.pop_size))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.ga.cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUT_PROB: 
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = elites + offspring

            if (config.ga.diversity > 0):
                pop_aux1 = tools.selBest(pop, k = round(config.ga.pop_size*(1 - config.ga.diversity)))
                pop_aux2 = toolbox.population(n=config.ga.pop_size - round(config.ga.pop_size*(1 - config.ga.diversity)))
                fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in pop_aux2)
                for ind, fit in zip(pop_aux2, fitnesses):
                    ind.fitness.values = fit
                    
                pop[:] = pop_aux1 + pop_aux2

            top = tools.selBest(pop, 3)

            if (config.ga.adaptative_mut_prob):
                current_fit = top[0].fitness.values[0]
                for i in range(len(config.ga.mutation_probability_interval) - 1):
                    if (current_fit >= config.ga.mutation_probability_interval[i] and current_fit < config.ga.mutation_probability_interval[i+1]):
                        MUT_PROB = config.ga.mutation_probabilities[i]
                        break


        elif config.ga.mu_lambda_ga:
            '''Estrategia mu + lambda'''
            pop.sort(key=lambda x: x.fitness.values[0], reverse=True)

            n_elites = int(round(config.ga.elite_passing * config.ga.pop_size))
            n_rest = config.ga.pop_size - n_elites
            elites = list(map(toolbox.clone, pop[:n_elites]))

            aux = list(map(toolbox.clone, pop[n_elites:]))
            offspring = list(map(toolbox.clone, pop[n_elites:]))

            for mutant in offspring:
                for i in range(NUM_MUT):
                    toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop_aux = aux + offspring
            pop_resto = tools.selBest(pop_aux, k = n_rest)
            pop[:] = elites + pop_resto
            
            if (config.ga.diversity > 0):
                pop_aux1 = tools.selBest(pop, k = round(config.ga.pop_size*(1 - config.ga.diversity)))
                pop_aux2 = toolbox.population(n=config.ga.pop_size - round(config.ga.pop_size*(1 - config.ga.diversity)))
                fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in pop_aux2)
                for ind, fit in zip(pop_aux2, fitnesses):
                    ind.fitness.values = fit
                pop[:] = pop_aux1 + pop_aux2

            top = tools.selBest(pop, 3)

            if (config.ga.adaptative_num_mut):
                current_fit = top[0].fitness.values[0]
                for i in range(len(config.ga.mutation_interval) - 1):
                    if (current_fit >= config.ga.mutation_interval[i] and current_fit < config.ga.mutation_interval[i+1]):
                        NUM_MUT = config.ga.num_mutations[i]
                        break
            
            if (config.ga.adaptative_mut_prob):
                current_fit = top[0].fitness.values[0]
                for i in range(len(config.ga.mutation_probability_interval) - 1):
                    if (current_fit >= config.ga.mutation_probability_interval[i] and current_fit < config.ga.mutation_probability_interval[i+1]):
                        MUT_PROB = config.ga.mutation_probabilities[i]
                        break

        print(f"--- Gen {g} Completada: Max Fitness (SSIM + Jaccard)= {top[0].fitness.values[0]:.2f}", flush=True)

        print(top[0]) 

        max_fitness_values.append(top[0].fitness.values[0])
        if len(max_fitness_values) > 1 and max_fitness_values[-1] == max_fitness_values[-2]: 
            generaciones_sin_mejora = generaciones_sin_mejora + 1
            if generaciones_sin_mejora > 75: return top, max_fitness_values 
        else: generaciones_sin_mejora = 0
        
        
        if top[0].fitness.values[0] >= config.ga.stop_condition: 
            print(f"Parado en la generación {g} con fitness {top[0].fitness.values[0]}")
            return top, max_fitness_values 

    return (tools.selBest(pop, 3), max_fitness_values)

# %%
'''Código de colores'''
AZUL_CLARO   = "\033[94m" # Para el estado azul
BLANCO_BRILLANTE = "\033[97m" # Para el estado blanco
ROJO_BRILLANTE   = "\033[91m" # Para el estado rojo
VERDE_BRILLANTE = "\033[92m" # Para el estado verde
RESET      = "\033[0m"  # Para resetear el color al final

def dibujar_regla_1dim(regla, ca_config):
    
    simbolo_0 = "■" # Azul
    simbolo_1 = "■" # Blanco
    simbolo_2 = "■" # Rojo
    
    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(ca_config.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = ca_config.colors[i].lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        
        # Creamos el símbolo coloreado para la terminal
        simbolos_map[str(i)] = f"\033[38;2;{r};{g};{b}m■\033[0m"
    
    print("--- Catálogo de Reglas del AC (Índice -> Configuración -> Resultado) ---")
    
    longitud_vecindad = 2 * int(ca_config.ca_neighborhood) + 1

    for indice in range(ca_config.ind_size):
        
        # 1. Obtener la configuración (en base 3)
        config_str = np.base_repr(indice, base=ca_config.num_states).zfill(longitud_vecindad)
        
        # 2. Convertir a símbolos visuales
        config_visual = "".join([simbolos_map[c] for c in config_str])
        
        # 3. Obtener el resultado
        resultado = regla[indice] # El resultado será 0, 1, o 2
        resultado_visual = simbolos_map[str(resultado)]
        
        # 4. Imprimir la línea (ajusta el padding '04d' si IND_SIZE es muy grande)
        print(f"Índice {indice:04d}:   {config_visual}   ->   {resultado_visual}")


'''Funcion que, dada una regla, la muestra gráfciamente con las configuraciones de vecinos y los colores correspondientes'''
def dibujar_regla_vonNeumann(regla, ca_config):
    simbolo = "■" 

    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(ca_config.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = ca_config.colors[i].lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        
        # Creamos el símbolo coloreado para la terminal
        simbolos_map[str(i)] = f"\033[38;2;{r};{g};{b}m■\033[0m"

    print(f"--- Catálogo de Reglas Von Neumann (5 Vecinos -> Resultado) ---")
    print("Formato visual:")
    print("  N  ")
    print("W C E  ->  Resultado")
    print("  S  ")
    print("-" * 40)
    
    # Von Neumann son 5 celdas
    longitud_vecindad = 5 

    for indice in range(len(regla)):
        
        # 1. Convertir índice a base 3 (o 2 si usas austriaca), rellenando a 5 dígitos
        config_str = np.base_repr(indice, base=ca_config.num_states).zfill(longitud_vecindad)
        
        # 2. Mapear cada posición según el orden [Centro, Norte, Este, Sur, Oeste]
        
        C = simbolos_map[config_str[0]] # Centro
        N = simbolos_map[config_str[1]] # Norte
        E = simbolos_map[config_str[2]] # Este
        S = simbolos_map[config_str[3]] # Sur
        W = simbolos_map[config_str[4]] # Oeste
        
        # 3. Resultado
        resultado = regla[indice]
        res_visual = simbolos_map[str(resultado)]
        
        # 4. Imprimir en formato bloque (Cross layout)
        # Usamos caracteres invisibles para alinear o espacios simples
        print(f"Índice {indice:03d}:")
        print(f"    {N}    ")       # Línea superior (Norte)
        print(f"  {W} {C} {E}  ->  {res_visual}") # Línea media (Oeste, Centro, Este) -> Resultado
        print(f"    {S}    ")       # Línea inferior (Sur)
        print("-" * 20)             # Separador

def dibujar_regla_Moore(regla, ca_config):
    simbolo = "■" 

    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(ca_config.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = ca_config.colors[i].lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        
        # Creamos el símbolo coloreado para la terminal
        simbolos_map[str(i)] = f"\033[38;2;{r};{g};{b}m■\033[0m"

    print(f"--- Catálogo de Reglas Moore (9 Vecinos -> Resultado) ---")
    print("Formato visual:")
    print("N1 N2 N3")
    print("N8 C N4  -> Resultado")
    print("N7 N6 N5")
    print("-" * 40)
    
    longitud_vecindad = 9 

    for indice in range(len(regla)):
        
        # 1. Convertir índice a base 3 (o 2 si usas austriaca), rellenando a 9 dígitos
        config_str = np.base_repr(indice, base=ca_config.num_states).zfill(longitud_vecindad)
        
        # 2. Mapear cada posición según el orden [Centro, N1, N2, N3, N4, N5, N6, N7, N8]
        
        C = simbolos_map[config_str[0]] # Centro
        N1 = simbolos_map[config_str[1]] # Norte 1
        N2 = simbolos_map[config_str[2]] # Norte 2
        N3 = simbolos_map[config_str[3]] # Norte 3
        N4 = simbolos_map[config_str[4]] # Este
        N5 = simbolos_map[config_str[5]] # Sur 3
        N6 = simbolos_map[config_str[6]] # Sur 2
        N7 = simbolos_map[config_str[7]] # Sur 1
        N8 = simbolos_map[config_str[8]] # Oeste
        
        # 3. Resultado
        resultado = regla[indice]
        res_visual = simbolos_map[str(resultado)]
        
        # 4. Imprimir en formato bloque (Cross layout)
        # Usamos caracteres invisibles para alinear o espacios simples
        print(f"Índice {indice:03d}:")
        print(f" {N1} {N2} {N3}    ")       # Línea superior (Norte 1)
        print(f" {N8} {C} {N4}  ->  {res_visual}") # Línea media (Norte 8, Centro, Norte 4) -> Resultado
        print(f" {N7} {N6} {N5}    ")       # Línea inferior (Sur 1)
        print("-" * 20)             # Separador

# ... (resto de funciones comentadas abajo como graphic_fitness_evolution y pruebas se mantienen igual, solo ten en cuenta de no ejecutarlas)

# %%
# '''Llama al algoritmo y hace un test con las 3 mejores reglas'''
# if __name__ == "__main__":
#     
#     top3, fitness_evolution = GeneticAlgorithm()
#     graphic_fitness_evolution(fitness_evolution)
#     for individual in top3:
#         regla = np.array(individual)
#         if (CA.ca_neighborhood == 'von Neumann'):
#             dibujar_regla_vonNeumann(regla)
#             CAs = generate_CAs(CA.num_ca_test, CA.num_states, CA.ca_size[0], CA.ca_size[1])
#         elif (CA.ca_neighborhood == 'Moore'):
#             dibujar_regla_Moore(regla)
#             CAs = generate_CAs(CA.num_ca_test, CA.num_states, CA.ca_size[0], CA.ca_size[1])
#         else:
#             dibujar_regla_1dim(regla)
#             
#         
#         CAs = generate_CAs(CA.num_ca_test, CA.num_states, CA.ca_size[0], CA.ca_size[1])
# 
#         '''rule_dict = gen_rule_dict(individual)'''
#         mi_regla = create_transition_rule(individual, CA.ca_size[0], CA.ca_size[1])
# 
#         '''Evolucion de los autómatas de prueba con la regla (ahora sí, en paralelo)'''
#         if CA.ca_neighborhood == 'von Neumann' or CA.ca_neighborhood == 'Moore':
#             CAs = Parallel(n_jobs=-1)(delayed(cpl.evolve2d)(CAs[i], timesteps=CA.ca_timesteps, neighbourhood = CA.ca_neighborhood, apply_rule=mi_regla) for i in range(len(CAs)))
#         else:
#             CAs = Parallel(n_jobs=-1)(delayed(cpl.evolve)(CAs[i], timesteps=CA.ca_timesteps, apply_rule=mi_regla, r=int(CA.ca_neighborhood)) for i in range(len(CAs)))
# 
#         '''Estudio del rendimiento de la regla'''
#         metrics_ssim = []
#         metrics_jaccard = []
#         for ca in CAs:
#             metrics_ssim.append(ssim_metric(ca[-1], CA.target_state))
#             metrics_jaccard.append(jaccard_metric(ca[-1], CA.target_state))
#                 
#         porcentaje_exito = (np.sum(metrics_ssim)*GA.weight_SSIM + np.sum(metrics_jaccard)*GA.weight_Jaccard) / CA.num_ca_test * 100
#         print("\n" + "*"*30)
#         print(f"  RENDIMIENTO DE LA REGLA:")
#         print(f"  FITNESS:  {porcentaje_exito:.2f}%")
#         print("*"*30 + "\n")
# 
#         
#         '''Graficación de la evolucion de los autómatas'''
#     
#         cmap_personal = ListedColormap(CA.colors)
#         
#         for i in range(len(CAs)):
#             plt.figure(figsize=(8, 4))
#             plt.imshow(CAs[i][-1], cmap=cmap_personal, interpolation='nearest', aspect='auto')
#             plt.xlabel("Celda")
#             plt.ylabel("Tiempo")
#             plt.title(f"Evolución del autómata CA {i}")
#             plt.show()
# 
# # %%

# %%
#Visualización de la evolución de un autómata aleatorio en 2 DIM con la mejor regla encontrada

#rc('animation', html='jshtml')
#mi_cmap = ListedColormap(CA.colors)
#regla_1 = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
#Fitness 0.85, 60% info mutua, 20% SSIM, 20% Jaccard, 30 timesteps

#regla_2 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
#Fitness 0.74, 60% info mutua, 20% SSIM, 20% Jaccard, 15 timesteps

# Creamos la función de la regla
#regla_ganadora = create_transition_rule(regla_1, CA.ca_size[0], CA.ca_size[1])


# 3. Inicializamos UN solo autómata para probar (con bordes fijoss)

#if (CA.random_initial_state):
#    inicial = np.random.randint(0, CA.num_states, size=(CA.ca_size[0], CA.ca_size[1]))
    
#else:
#    inicial = CA.ca_initial_state

#inicial = np.expand_dims(inicial, axis=0)


# 4. Evolucionamos (esto genera el historial completo que necesita la animación)
# cpl.evolve2d devuelve un array de forma (timesteps, alto, ancho)
#ca = cpl.evolve2d(
    #inicial, 
    #timesteps=CA.ca_timesteps, 
    #neighbourhood=CA.ca_neighborhood, 
    #apply_rule=regla_ganadora
#)

# 5. VISUALIZACIÓN DIRECTA con cpl
#cpl.plot2d_animate(ca, colormap=mi_cmap)




# %%



