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
#Imports paralelización
import multiprocessing
from joblib import Parallel, delayed 
import time
import os

#Metricas de similitd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.ndimage import gaussian_filter

#Metricas de similitd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
from scipy.ndimage import gaussian_filter

from configuration import FrameworkConfig, CAConfig, GAConfig


# %%

def evolve2d_vectorized(initial_state, rule_table, ca_config):
    H, W = ca_config.ca_size[0], ca_config.ca_size[1]
    timesteps = ca_config.ca_timesteps
    num_states = ca_config.num_states

    history = np.empty((timesteps + 1, H, W), dtype=int)
    history[0] = initial_state
    state = initial_state.copy()

    if ca_config.ca_neighborhood == 'Moore':
        offsets = [(0,0), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
        powers = num_states ** np.arange(8, -1, -1)
    else:  # von Neumann
        offsets = [(0,0), (-1,0), (0,1), (1,0), (0,-1)]
        powers = num_states ** np.arange(4, -1, -1)

    modo_v = 'wrap' if ca_config.ca_vertical_boundary_conditions == 'periodic' else 'edge'
    modo_h = 'wrap' if ca_config.ca_horizontal_boundary_conditions == 'periodic' else 'edge'

    for t in range(1, timesteps + 1):
        padded = np.pad(state, ((1, 1), (0, 0)), mode=modo_h)
        padded = np.pad(padded, ((0, 0), (1, 1)), mode=modo_v)

        idx = np.zeros((H, W), dtype=np.int64)
        for (dy, dx), p in zip(offsets, powers):
            vecino = padded[1+dy : 1+dy+H, 1+dx : 1+dx+W]
            idx += vecino * p

        nuevo_estado = rule_table[idx]

        if ca_config.ca_horizontal_boundary_conditions == 'fixed':
            nuevo_estado[0, :] = ca_config.ca_row0_state
            nuevo_estado[-1, :] = ca_config.ca_rowN_state
        if ca_config.ca_vertical_boundary_conditions == 'fixed':
            nuevo_estado[:, 0] = ca_config.ca_column0_state
            nuevo_estado[:, -1] = ca_config.ca_columnN_state

        state = nuevo_estado
        history[t] = state

    return history

def create_transition_rule(individual, ca_size_1, ca_size_2, ca_config):
    rule_table = np.array(individual, dtype=int)
    
    # Extraemos dimensiones de forma segura
    n = ca_config.ca_size[0]
    m = ca_config.ca_size[1] if len(ca_config.ca_size) > 1 else 1

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
        if m == 1:
            if ca_config.ca_horizontal_boundary_conditions == 'fixed':
                if r == 0:
                    # Usamos .item() o int() para extraer el número puro del array de NumPy
                    return int(ca_config.ca_row0_state[0])
                elif r == n - 1:
                    # Asegúrate de usar ca_rowN_state para el extremo final
                    return int(ca_config.ca_rowN_state[0])
        else:

            if ca_config.ca_horizontal_boundary_conditions == 'customized':
                # Personalización de las condiciones de frontera horizontales
                pass

            elif ca_config.ca_horizontal_boundary_conditions == 'fixed':
                if r[0] == 0:
                    return int(ca_config.ca_row0_state[r[1]])
                elif r[0] == n - 1:
                    return int(ca_config.ca_rowN_state[r[1]])
                

            if ca_config.ca_vertical_boundary_conditions == 'customized':
                # Personalización de las condiciones de frontera verticales
                pass

            elif ca_config.ca_vertical_boundary_conditions == 'fixed':
                if r[1] == 0:
                    return int(ca_config.ca_column0_state[r[0]])
                elif r[1] == m - 1:
                    return int(ca_config.ca_columnN_state[r[0]])
                
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
    n = ca_config.ca_size[0]
    m = ca_config.ca_size[1] if len(ca_config.ca_size) > 1 else 1
    return (np.sum(ca_final == target)/(n * m))

'''Metrica Jaccard'''
def jaccard_metric(ca_final, target, config):
    jaccard_states = []
    for state in range(config.ca.num_states):
        intersection = np.sum((ca_final == state) & (target == state))
        union = np.sum((target == state) | (ca_final == state))
        if union == 0:
            jaccard_states.append(1.0)
        else:
            jaccard_states.append(intersection / union)
            
    pesos = config.ga.weights_states_Jaccard
    
    if len(pesos) == config.ca.num_states:
        weighted_sum = sum(jaccard_states[i] * pesos[i] for i in range(config.ca.num_states))
        return weighted_sum / sum(pesos)
    
    return np.mean(jaccard_states)
    
def crear_paleta_rgb(colores):
    """
    Convierte una lista de colores HEX ["#FF0000", "#FFFFFF"] 
    en una paleta Numpy RGB de shape (N, 3) tipo uint8.
    """
    rgb_list = []
    for color_hex in colores:
        color_hex = color_hex.lstrip('#')
        # Partimos el string de 2 en 2 y lo pasamos de base 16 a base 10 (entero)
        rgb = [int(color_hex[i:i+2], 16) for i in (0, 2, 4)]
        rgb_list.append(rgb)
        
    return np.array(rgb_list, dtype=np.uint8)

'''Métrica SSIM'''
def ssim_metric(ca_final, target, ca_config):
    # SSIM no tiene sentido o falla en 1D con win_size=5 si el array es pequeño o plano
    if len(ca_config.ca_size) == 1 or ca_config.ca_size[1] == 1:
        return 0.0
        
    colores_rgb = crear_paleta_rgb(ca_config.colors)
    img_ca_rgb = colores_rgb[ca_final]
    img_target_rgb = colores_rgb[target]

    return ssim(img_ca_rgb, img_target_rgb, channel_axis=-1, data_range=255, win_size=5)

'''Metrica informacion mutua'''
def mutual_information_metric(ca_final, target, ca_config):
    return normalized_mutual_info_score(target.flatten(), ca_final.flatten(), average_method='arithmetic')

'''Generación aleatoria de autómatas, teniendo en cuenta la frontera fija'''
def generate_CAs(ca_num, num_states, ca_size_1, ca_size_2, ca_config):
    CAs = []
    # Obtenemos la probabilidad de la configuración (si no la has definido aún, asume 0.0)
    prob_ruido = getattr(ca_config, 'random_probability', 0.0)
    
    if ca_size_2 == 1:
        for i in range(ca_num):
            if ca_config.ca_initial_state is not None:
                # IMPORTANTE: Forzamos a que sea plano (N,) y luego CellPyLib 1D espera (1, N)
                ca = np.copy(ca_config.ca_initial_state).flatten()
            else:
                ca = np.random.randint(0, num_states, size=ca_size_1)
            
            ca = ca.reshape(1, -1) 
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
    m = ca_config.ca_size[1] if len(ca_config.ca_size) > 1 else 1

    if m == 1:
        for i in range(len(CAs)):
            # Limpiamos dimensiones extra de tamaño 1 para evitar el error de desempaquetado en cellpylib
            ca_input = CAs[i].squeeze().reshape(1, -1)
            evolved_CA = cpl.evolve(ca_input, timesteps=ca_config.ca_timesteps, apply_rule=mi_regla, r=int(ca_config.ca_neighborhood))
            evolved_CAs_list.append(evolved_CA)
    else:
        for i in range(len(CAs)):
            estado_inicial_2d = CAs[i][0]  # quitamos la dim de tiempo (expand_dims)
            evolved_CA = evolve2d_vectorized(estado_inicial_2d, rule_table=np.array(mi_regla), ca_config=ca_config)
            evolved_CAs_list.append(evolved_CA)

    return evolved_CAs_list

'''Funcion que genera la regla a partir de un individuo, evoluciona los autómatas con esa regla y devuelve el valor de fitness 
correspondiente.'''
def rule_and_evolve(ca_num, individual, CAs, ca_size_1, ca_size_2, ca_config):
    m = ca_config.ca_size[1] if len(ca_config.ca_size) > 1 else 1
    if m == 1:
        mi_regla = create_transition_rule(individual, ca_size_1, ca_size_2, ca_config)
        return evolved_CAs(CAs, mi_regla, ca_config)
    else:
        rule_table = np.array(individual, dtype=int)
        return evolved_CAs(CAs, rule_table, ca_config)

def calculate_fitness(ca, config):
    if (config.ga.gaussian_filter):
       
        ca_suavizado = gaussian_filter(ca[-1], sigma=config.ga.gaussian_sigma)
        target_suavizado = gaussian_filter(config.ca.target_state, sigma=config.ga.gaussian_sigma)
        
       
        
        # SSIM funciona perfecto con floats
        score_ssim = ssim(ca_suavizado, target_suavizado, data_range=1.0, win_size=5)
        
        # Para Jaccard y Accuracy, comparamos si las zonas de "alta densidad" coinciden
        # Binarizamos los gradientes suavizados
        ca_bin = (ca_suavizado > 0.5).astype(int)
        target_bin = (target_suavizado > 0.5).astype(int)
        
        score_jaccard = jaccard_metric(ca_bin, target_bin, config)
        score_accuracy = accuracy_metric(ca_bin, target_bin, config.ca)
        score_mi = mutual_information_metric(ca_bin, target_bin, config.ca)
        fit = (config.ga.weight_SSIM * score_ssim + 
               config.ga.weight_Jaccard * score_jaccard + 
               config.ga.weight_accuracy * score_accuracy + 
               config.ga.weight_mutual_information * score_mi)

        if (config.ga.penalized_fitness):
            counts = np.bincount(ca[-1].flatten(), minlength=3) 
            total_cells = ca[-1].size
            if np.any(counts[:2] < (config.ga.minimum_percentage * total_cells)):
                fit = fit * config.ga.penalization_factor 
    else:
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
    # --- Cambio clave: Extracción segura de dimensiones ---
    n = config.ca.ca_size[0]
    # Si es 1D, ca_size[1] no existe, así que asignamos 1
    m = config.ca.ca_size[1] if len(config.ca.ca_size) > 1 else 1

    if (config.ga.adaptative_fitness):
        # Usamos n y m en lugar de acceder a los índices de la tupla directamente
        CAs = generate_CAs(config.ga.ca_fitness//5, config.ca.num_states, n, m, config.ca)

        evolvedCAs = rule_and_evolve(config.ga.ca_fitness//5, individual, CAs, n, m, config.ca)

        '''Estudiar métricas para medir fitness...'''
        fitness_values_1 = []
        for ca in evolvedCAs:
            fit = calculate_fitness(ca, config)
            fitness_values_1.append(fit)

        res1 = np.mean(fitness_values_1)
        if res1 < config.ga.adaptative_fitness_threshold:
            return (res1,)
            
        CAs_full = generate_CAs(config.ga.ca_fitness, config.ca.num_states, n, m, config.ca)
      
        evolvedCAs_full = rule_and_evolve(config.ga.ca_fitness, individual, CAs_full, n, m, config.ca)
        
        fitness_values_2 = []
        for ca in evolvedCAs_full: # Usar el conjunto completo
            fit = calculate_fitness(ca, config)
            fitness_values_2.append(fit)
        res2 = np.mean(fitness_values_2)
        return (res2,)

    else:
        CAs = generate_CAs(config.ga.ca_fitness, config.ca.num_states, n, m, config.ca)

        evolvedCAs = rule_and_evolve(config.ga.ca_fitness, individual, CAs, n, m, config.ca)

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

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    #########################################################

    toolbox.register("attr_int", random.randint, 0, config.ca.num_states - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_int, config.ca.ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ... (Todo tu bloque de cruce, seleccion y mutacion sigue igual) ...
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

    if config.ga.sel_method == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=config.ga.tournament_size)
    elif config.ga.sel_method == 'roulette':
        toolbox.register("select", tools.selRoulette)
    elif config.ga.sel_method == 'best':
        toolbox.register("select", tools.selBest)
    elif config.ga.sel_method == 'random':
        toolbox.register("select", tools.selRandom)

    toolbox.register("mutate", mutChangeGen, ca_config=config.ca)
    toolbox.register("evaluate", evaluate, config=config)

    pop = toolbox.population(n=config.ga.pop_size)
    
    # ... (Tu bloque de inyección memética sigue igual) ...
    if seed_rule is not None and len(seed_rule) == config.ca.ind_size:
        num_clones = int(config.ga.pop_size * 0.05) 
        num_mutated = int(config.ga.pop_size * 0.05) 
        print(f"[Evolución] Sembrando población: {num_clones} clones y {num_mutated} mutados de la regla CBR.")
        for i in range(num_clones):
            pop[i] = creator.Individual(copy.deepcopy(seed_rule))
        for i in range(num_clones, num_clones + num_mutated):
            ind = creator.Individual(copy.deepcopy(seed_rule))
            genes_a_mutar = random.sample(range(config.ca.ind_size), int(config.ca.ind_size*0.05))
            for g in genes_a_mutar:
                ind[g] = random.randint(0, config.ca.num_states - 1)
            pop[i] = ind

    print('Iniciando cálculo de fitness', flush = True)
    
    ### MODIFICADO: Uso de toolbox.map en lugar de Parallel ###
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    ###########################################################
    
    print('Cálculo de fitness terminado', flush = True)

    max_fitness_values = []
    NUM_MUT = config.ga.num_mutations[0] if len(config.ga.num_mutations) > 0 else 1
    MUT_PROB = config.ga.mutation_probabilities[0] if len(config.ga.mutation_probabilities) > 0 else 0.1
    generaciones_sin_mejora = 0

    for g in range(config.ga.num_generations):
        print(f'--- Iniciando Generación {g} ---', flush=True)

        if (config.ga.classic_ga):
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
            
            ### MODIFICADO ###
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            ##################

            pop[:] = elites + offspring

            if (config.ga.diversity > 0):
                pop_aux1 = tools.selBest(pop, k = round(config.ga.pop_size*(1 - config.ga.diversity)))
                pop_aux2 = toolbox.population(n=config.ga.pop_size - round(config.ga.pop_size*(1 - config.ga.diversity)))
                
                ### MODIFICADO ###
                fitnesses = toolbox.map(toolbox.evaluate, pop_aux2)
                for ind, fit in zip(pop_aux2, fitnesses):
                    ind.fitness.values = fit
                ##################
                    
                pop[:] = pop_aux1 + pop_aux2

            top = tools.selBest(pop, 3)
            if (config.ga.adaptative_mut_prob):
                current_fit = top[0].fitness.values[0]
                for i in range(len(config.ga.mutation_probability_interval) - 1):
                    if (current_fit >= config.ga.mutation_probability_interval[i] and current_fit < config.ga.mutation_probability_interval[i+1]):
                        MUT_PROB = config.ga.mutation_probabilities[i]
                        break


        elif config.ga.mu_lambda_ga:
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
            
            ### MODIFICADO ###
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            ##################

            pop_aux = aux + offspring
            pop_resto = tools.selBest(pop_aux, k = n_rest)
            pop[:] = elites + pop_resto
            
            if (config.ga.diversity > 0):
                pop_aux1 = tools.selBest(pop, k = round(config.ga.pop_size*(1 - config.ga.diversity)))
                pop_aux2 = toolbox.population(n=config.ga.pop_size - round(config.ga.pop_size*(1 - config.ga.diversity)))
                
                ### MODIFICADO ###
                fitnesses = toolbox.map(toolbox.evaluate, pop_aux2)
                for ind, fit in zip(pop_aux2, fitnesses):
                    ind.fitness.values = fit
                ##################
                    
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

        max_fitness_values.append(top[0].fitness.values[0])
        if len(max_fitness_values) > 1 and max_fitness_values[-1] == max_fitness_values[-2]: 
            generaciones_sin_mejora = generaciones_sin_mejora + 1
            if generaciones_sin_mejora > 100: 
                pool.close()
                pool.join()
                return top, max_fitness_values 
        else: 
            generaciones_sin_mejora = 0
        
        if top[0].fitness.values[0] >= config.ga.stop_condition: 
            print(f"Parado en la generación {g} con fitness {top[0].fitness.values[0]}")
            pool.close()
            pool.join()
            return top, max_fitness_values 

    pool.close()
    pool.join()
    
    return (tools.selBest(pop, 3), max_fitness_values)

# %%

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



