import numpy as np
import random
import matplotlib.pyplot as plt

# Importa tus clases y tu algoritmo genético
from configuration import CAConfig, GAConfig, FrameworkConfig
from Evolucion import GeneticAlgorithm

# ==========================================
# 1. FUNCIÓN DE VISUALIZACIÓN
# ==========================================
def graphic_fitness_evolution(fitness_evolution, nombre_problema):
    generaciones = range(len(fitness_evolution))
    fitness_values = fitness_evolution

    plt.figure(figsize=(12, 6)) # Tamaño de la imagen
    plt.plot(generaciones, fitness_values, 
            linewidth=2, 
            color='#1f77b4', # Azul estándar
            label='Mejor Individuo')

    # Añadir puntos rojos en cada generación
    plt.scatter(generaciones, fitness_values, color='red', s=10, zorder=5)

    # Etiquetas y Estilo
    plt.title(f'Evolución del Fitness (Búsqueda Aleatoria) - {nombre_problema}', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Max Fitness', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # Anotación del valor máximo
    if len(fitness_values) > 0:
        max_val = max(fitness_values)
        max_gen = fitness_values.index(max_val)
        
        # Ajuste dinámico
        offset_x = -50 if max_gen > 50 else 20
    
        plt.annotate(f'Max: {max_val:.4f}', 
                    xy=(max_gen, max_val), 
                    xytext=(max_gen + offset_x, max_val),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    # Guardamos la imagen automáticamente por si acaso
    plt.savefig(f"grafica_random_{nombre_problema}.png", dpi=300)
    plt.close()

# ==========================================
# 2. GENERADOR DE CONFIGURACIÓN ALEATORIA
# ==========================================
def generate_random_config(ca_size, num_states, target_state, colors):
    # =================================================================
    # 1. AUTÓMATA CELULAR (CA) - Aleatoriedad Total
    # =================================================================
    if num_states >= 3:
        neighborhood = 'von Neumann'
    else:
        neighborhood = random.choice(['von Neumann', 'Moore'])
        
    top_edge = target_state[0, :]
    bottom_edge = target_state[-1, :]
    left_edge = target_state[:, 0]
    right_edge = target_state[:, -1]
    all_edges = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    borde_uniforme = len(np.unique(all_edges)) == 1

    if borde_uniforme:
        h_bound = 'periodic'
        v_bound = 'periodic'
    else:
        # El algoritmo aleatorio elige a ciegas UNA sola dimensión para fijar
        if random.choice([True, False]):
            h_bound = 'fixed'
            v_bound = 'periodic'
        else:
            h_bound = 'periodic'
            v_bound = 'fixed'
    
    # Asignamos las matrices de los bordes únicamente si la dimensión elegida ha sido 'fixed'
    ca_row0 = target_state[0, :].copy() if h_bound == 'fixed' else None
    ca_rowN = target_state[-1, :].copy() if h_bound == 'fixed' else None
    ca_col0 = target_state[:, 0].copy() if v_bound == 'fixed' else None
    ca_colN = target_state[:, -1].copy() if v_bound == 'fixed' else None
    
    ca_row0 = target_state[0, :].copy() if h_bound == 'fixed' else None
    ca_rowN = target_state[-1, :].copy() if h_bound == 'fixed' else None
    ca_col0 = target_state[:, 0].copy() if v_bound == 'fixed' else None
    ca_colN = target_state[:, -1].copy() if v_bound == 'fixed' else None

    is_random_initial = random.choice([True, False])
    init_state = None
    if not is_random_initial:
        init_state = np.zeros(ca_size, dtype=int)
        init_state[ca_size[0]//2, ca_size[1]//2] = 1 # Semilla en el centro

    ca_cfg = CAConfig(
        ca_size=ca_size,
        num_states=num_states,
        target_state=target_state,
        colors=colors,
        ca_timesteps=random.randint(20, 100),
        ca_neighborhood=neighborhood,
        ca_horizontal_boundary_conditions=h_bound,
        ca_vertical_boundary_conditions=v_bound,
        ca_row0_state=ca_row0,
        ca_rowN_state=ca_rowN,
        ca_column0_state=ca_col0,
        ca_columnN_state=ca_colN,
        random_probability=random.uniform(0.01, 0.1),
        random_initial_state=is_random_initial,
        ca_initial_state=init_state,
        num_ca_test=random.randint(10, 50) # Añadido
    )

    # =================================================================
    # 2. ALGORITMO GENÉTICO (GA) - Aleatoriedad Total
    # =================================================================
    is_classic = random.choice([True, False])
    
    # Pesos Globales
    raw_weights = [random.random() for _ in range(4)]
    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]
    
    # Pesos Jaccard
    jaccard_weights = [random.random() for _ in range(num_states)]
    jaccard_sum = sum(jaccard_weights)
    jaccard_weights_norm = np.array([w / jaccard_sum for w in jaccard_weights])

    # Arrays Mutaciones
    num_tramos_prob = random.randint(3, 6)
    puntos_corte_prob = sorted([random.uniform(0.1, 0.9) for _ in range(num_tramos_prob - 2)])
    array_intervalos_prob = np.array([0.0] + puntos_corte_prob + [1.0])
    array_probs = np.array([random.uniform(0.01, 0.5) for _ in range(num_tramos_prob - 1)])

    num_tramos_mut = random.randint(3, 6)
    puntos_corte_mut = sorted([random.uniform(0.1, 0.9) for _ in range(num_tramos_mut - 2)])
    array_intervalos_mut = np.array([0.0] + puntos_corte_mut + [1.0])
    array_num_muts = np.array([random.randint(1, 25) for _ in range(num_tramos_mut - 1)])

    ga_cfg = GAConfig(
        pop_size=random.randint(50, 400), 
        num_generations=random.randint(100, 400), 
        stop_condition=random.uniform(0.7, 0.95), 
        
        cx_prob=random.uniform(0.5, 0.95),
        mut_prob=random.uniform(0.1, 1.0),
        
        adaptative_mut_prob=random.choice([True, False]),
        mutation_probability_interval=array_intervalos_prob,
        mutation_probabilities=array_probs,
        
        adaptative_num_mut=random.choice([True, False]),
        mutation_interval=array_intervalos_mut,
        num_mutations=array_num_muts,

        # Fitness adaptativo randomizado al completo
        adaptative_fitness=random.choice([True, False]),
        ca_fitness=random.randint(10, 50), # Añadido
        adaptative_fitness_threshold=random.uniform(0.3, 0.8), # Añadido
        
        # Penalizaciones randomizadas al completo
        penalized_fitness=random.choice([True, False]),
        minimum_percentage=random.uniform(0.05, 0.3), 
        penalization_factor=random.uniform(0.1, 0.9), 
        
        # Filtro Gaussiano randomizado
        gaussian_filter=False, 
        gaussian_sigma=random.uniform(1.0, 5.0), 
        
        weight_SSIM=norm_weights[0],
        weight_Jaccard=norm_weights[1],
        weight_accuracy=norm_weights[2],
        weight_mutual_information=norm_weights[3],
        weights_states_Jaccard=jaccard_weights_norm,
        
        classic_ga=is_classic,
        mu_lambda_ga=not is_classic,
        
        # Operadores genéticos al completo
        sel_method=random.choice(['tournament', 'roulette']),
        tournament_size=random.randint(2, 5),
        cx_method=random.choice(['onePoint', 'twoPoint', 'uniform']), # Añadido
        elite_passing=random.uniform(0.05, 0.15) if is_classic else 0.0,
        diversity=random.uniform(0.0, 0.2) if not is_classic else 0.0 # Añadido
    )

    return FrameworkConfig(ca=ca_cfg, ga=ga_cfg)

if __name__ == "__main__":
    print("=========================================================")
    print(" EJECUCIÓN LÍNEA BASE: CONFIGURACIÓN ALEATORIA (RANDOM)")
    print("=========================================================")

    # Define tus matrices target aquí
    target_polonia = np.zeros((18, 30), dtype=int)
    target_polonia[0:9, :] = 0; target_polonia[9:18, :] = 1
    colors_polonia = ["#FFFFFF", "#FF0000"]
    
    target_italia = np.zeros((18, 30), dtype=int)
    target_italia[:, 0:10] = 0; target_italia[:, 10:20] = 1; target_italia[:, 20:30] = 2
    colors_italia = ["#008000", "#FFFFFF", "#FF0000"]

    target_benin = np.zeros((18, 30), dtype=int)
    target_benin[:, 0:12] = 0          # Verde (columna izquierda)
    target_benin[0:9, 12:30] = 1       # Amarillo (mitad superior derecha)
    target_benin[9:18, 12:30] = 2      # Rojo (mitad inferior derecha)
    colors_benin = ["#008000", "#FFFF00", "#FF0000"] # 0=Verde, 1=Amarillo, 2=Rojo


    target_suiza = np.zeros((18, 18), dtype=int) # 0 = Rojo
    # Cruz blanca en el centro
    target_suiza[7:11, 3:15] = 1  # Brazo horizontal
    target_suiza[3:15, 7:11] = 1  # Brazo vertical
    colors_suiza = ["#FF0000", "#FFFFFF"] # 0=Rojo, 1=Blanco

    target_austria = np.zeros((18, 30), dtype=int)
    target_austria[0:6, :] = 0   # Rojo
    target_austria[6:12, :] = 1  # Blanco
    target_austria[12:18, :] = 0 # Rojo
    colors_austria = ["#FF0000", "#FFFFFF"] # 0=Rojo, 1=Blanco

    Y_AUX, X_AUX = np.ogrid[:30, :45]

    # 7. BANGLADESH (Fondo verde, círculo rojo desplazado a la izquierda)
    target_bangladesh = np.zeros((30, 45), dtype=int) # 0 = Verde
    # Y centrado (15), X desplazado a la izquierda (20)
    centro_y_bd, centro_x_bd = 15, 20 
    # Radio ajustado a la proporción oficial (1/5 del ancho)
    radio_bd = 9 
    mask_bd = ((X_AUX - centro_x_bd)**2 + (Y_AUX - centro_y_bd)**2) <= radio_bd**2
    target_bangladesh[mask_bd] = 1 # 1 = Rojo
    colors_bangladesh = ["#008000", "#FF0000"] # 0=Verde, 1=Rojo

    target_colombia = np.zeros((18, 30), dtype=int)
    target_colombia[0:9, :] = 0    # Amarillo (mitad superior, 9 filas)
    target_colombia[9:14, :] = 1   # Azul (siguientes 5 filas)
    target_colombia[14:18, :] = 2  # Rojo (últimas 4 filas)
    colors_colombia = ["#FFFF00", "#0000FF", "#FF0000"] # 0=Amarillo, 1=Azul, 2=Rojo

    target_checa = np.zeros((18, 30), dtype=int)
    target_checa[0:9, :] = 0   # Blanco (mitad superior)
    target_checa[9:18, :] = 1  # Rojo (mitad inferior)

    # Dibujamos el Triángulo azul en el lado izquierdo (hoist)
    # El vértice central llega hasta la mitad del ancho (x = 15)
    for y in range(18):
        for x in range(30):
            # Mitad superior del triángulo
            if y < 9: 
                if x <= y * (15.0 / 8.0):
                    target_checa[y, x] = 2 # Azul
            # Mitad inferior del triángulo
            else:     
                if x <= (17 - y) * (15.0 / 8.0):
                    target_checa[y, x] = 2 # Azul
                    
    colors_checa = ["#FFFFFF", "#FF0000", "#0000FF"] # 0=Blanco, 1=Rojo, 2=Azul


    # Diccionario de problemas a probar
    problemas = [
        {"nombre": "Benin", "target": target_benin, "estados": 3, "colores": colors_benin},
    ]

    resultados_random = []

    for prob in problemas:
        print(f"\n---> Evaluando problema: {prob['nombre'].upper()} con Random Search <---")
        
        random_conf = generate_random_config(
            ca_size=prob['target'].shape, 
            num_states=prob['estados'], 
            target_state=prob['target'],
            colors=prob['colores']
        )
        
        print(f"[*] Lanzando DEAP ({random_conf.ga.num_generations} generaciones, Pop: {random_conf.ga.pop_size})...")
        
        try:
            top3, fitness_evolution = GeneticAlgorithm(config=random_conf, seed_rule=None)
            
            mejor_fitness = fitness_evolution[-1]
            generaciones_usadas = len(fitness_evolution)
            
            print(f"[!] {prob['nombre']} -> Fitness final: {mejor_fitness:.4f}")
            
            # Guardar resultados
            resultados_random.append({
                "problema": prob['nombre'],
                "fitness": mejor_fitness,
                "generaciones": generaciones_usadas,
                "evolucion": fitness_evolution  # Por si quisieras hacer algo con todo el array después
            })
            
            # ¡MOSTRAR Y GUARDAR LA GRÁFICA!
            graphic_fitness_evolution(fitness_evolution, prob['nombre'])
            
        except Exception as e:
            print(f"[ERROR] Fallo en la ejecución de {prob['nombre']}: {e}")
            resultados_random.append({
                "problema": prob['nombre'],
                "fitness": 0.0,
                "generaciones": "ERROR",
                "evolucion": []
            })

    # Resumen Final en consola
    print("\n=========================================================")
    print(" RESUMEN DE RESULTADOS LÍNEA BASE (RANDOM)")
    print("=========================================================")
    print(f"{'Problema':<15} | {'Fitness Alcanzado':<18} | {'Generaciones'}")
    print("-" * 50)
    for r in resultados_random:
        if r['generaciones'] != "ERROR":
            print(f"{r['problema']:<15} | {r['fitness']:<18.4f} | {r['generaciones']}")
        else:
            print(f"{r['problema']:<15} | {'ERROR':<18} | {'ERROR'}")