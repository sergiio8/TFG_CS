import numpy as np
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

    plt.figure(figsize=(12, 6))
    plt.plot(generaciones, fitness_values, linewidth=2, color='#ff7f0e', label='Heurística')
    plt.scatter(generaciones, fitness_values, color='black', s=10, zorder=5)

    plt.title(f'Evolución del Fitness - {nombre_problema}', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Max Fitness', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    if len(fitness_values) > 0:
        max_val = max(fitness_values)
        max_gen = fitness_values.index(max_val)
        offset_x = -50 if max_gen > 50 else 20
        plt.annotate(f'Max: {max_val:.4f}', xy=(max_gen, max_val), xytext=(max_gen + offset_x, max_val),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(f"Grafica_HeuristicaFuerte_{nombre_problema}.png", dpi=300)
    plt.close()


def generate_competent_heuristic_config(ca_size, num_states, target_state, colors, h_bound, v_bound, use_random_init):

    
    ca_row0 = target_state[0, :].copy() if h_bound == 'fixed' else None
    ca_rowN = target_state[-1, :].copy() if h_bound == 'fixed' else None
    ca_col0 = target_state[:, 0].copy() if v_bound == 'fixed' else None
    ca_colN = target_state[:, -1].copy() if v_bound == 'fixed' else None

    # Configuración del estado inicial (Ruido vs Semilla Central)
    init_state = None
    if not use_random_init:
        init_state = np.zeros(ca_size, dtype=int)
        # Colocamos la semilla viva en el centro exacto del lienzo
        init_state[ca_size[0]//2, ca_size[1]//2] = 1 

    neighborhood = 'von Neumann' if num_states >= 3 else 'Moore'

    prob_ruido = 1.0 if use_random_init else 0.0

    ca_cfg = CAConfig(
        ca_size=ca_size,
        num_states=num_states,
        target_state=target_state,
        colors=colors,
        ca_timesteps=60, 
        ca_neighborhood=neighborhood,
        ca_horizontal_boundary_conditions=h_bound,
        ca_vertical_boundary_conditions=v_bound,
        ca_row0_state=ca_row0,
        ca_rowN_state=ca_rowN,
        ca_column0_state=ca_col0,
        ca_columnN_state=ca_colN,
        random_probability=prob_ruido,        
        random_initial_state=use_random_init, 
        ca_initial_state=init_state           
    )

    # =================================================================
    # 2. ALGORITMO GENÉTICO (GA) - Determinista con 2 Etapas
    # =================================================================
    
    w_jaccard = 0.35
    w_ssim = 0.35
    w_mutual = 0.30
    jaccard_weights_norm = np.ones(num_states) / num_states

    array_intervalos_mut = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    array_num_muts = np.array([15,10,6, 3])

    ga_cfg = GAConfig(
        pop_size=200,        
        num_generations=400, 
        stop_condition=0.9,
        
        classic_ga=False,
        mu_lambda_ga=True,
        cx_prob=0.0,
        mut_prob=1.0,
        
        adaptative_mut_prob=False,
        adaptative_num_mut=True,
        mutation_interval=array_intervalos_mut,  # <-- Inyectamos los 2 tramos
        num_mutations=array_num_muts,            # <-- Inyectamos los 2 tramos

        adaptative_fitness=False,
        ca_fitness=10,
        adaptative_fitness_threshold=0.5,
        
        penalized_fitness=True,
        minimum_percentage=0.1,
        penalization_factor=0.5,
        
        weight_accuracy=0.0,
        weight_Jaccard=w_jaccard,
        weight_SSIM=w_ssim,
        weight_mutual_information=w_mutual,
        weights_states_Jaccard=jaccard_weights_norm,
        
        sel_method='tournament',
        tournament_size=3,
        elite_passing=0.1,
        diversity=0.05
    )

    return FrameworkConfig(ca=ca_cfg, ga=ga_cfg)

if __name__ == "__main__":
    print("=========================================================")
    print(" EJECUCIÓN LÍNEA BASE: HEURÍSTICA COMPETENTE (ABLACIÓN)")
    print("=========================================================")

    # Define tus matrices target aquí
    target_polonia = np.zeros((18, 30), dtype=int)
    target_polonia[0:9, :] = 0; target_polonia[9:18, :] = 1
    
    target_italia = np.zeros((18, 30), dtype=int)
    target_italia[:, 0:10] = 0; target_italia[:, 10:20] = 1; target_italia[:, 20:30] = 2

    target_benin = np.zeros((18, 30), dtype=int)
    target_benin[:, 0:12] = 0          
    target_benin[0:9, 12:30] = 1       
    target_benin[9:18, 12:30] = 2      
    colors_benin = ["#008000", "#FFFF00", "#FF0000"] 

    target_suiza = np.zeros((18, 18), dtype=int) 
    target_suiza[7:11, 3:15] = 1  
    target_suiza[3:15, 7:11] = 1  
    colors_suiza = ["#FF0000", "#FFFFFF"] 

    Y_AUX, X_AUX = np.ogrid[:18, :30]
    target_bangladesh = np.zeros((18, 30), dtype=int) 
    centro_y_bd, centro_x_bd = 9, 15
    radio_bd = 6
    mask_bd = ((X_AUX - centro_x_bd)**2 + (Y_AUX - centro_y_bd)**2) <= radio_bd**2
    target_bangladesh[mask_bd] = 1 
    colors_bangladesh = ["#008000", "#FF0000"] 

    target_colombia = np.zeros((18, 30), dtype=int)
    target_colombia[0:9, :] = 0    
    target_colombia[9:14, :] = 1   
    target_colombia[14:18, :] = 2  
    colors_colombia = ["#FFFF00", "#0000FF", "#FF0000"]

    # =========================================================
    # DICCIONARIO: BORDES Y ESTADOS INICIALES INYECTADOS
    # =========================================================
    # =========================================================
    # DICCIONARIO: BORDES Y ESTADOS INICIALES INYECTADOS
    # =========================================================
    problemas = [
        {
            "nombre": "Colombia", "target": target_colombia, "estados": 3, "colores": colors_colombia,
            # Franjas horizontales (asimétricas): anclamos arriba/abajo
            "h_bound": "fixed", "v_bound": "periodic", "random_init": True
        },
        {
            "nombre": "Benin", "target": target_benin, "estados": 3, "colores": colors_benin,
            # Patrón mixto: necesitamos anclar todos los bordes para que las 3 áreas tengan referencia
            "h_bound": "periodic", "v_bound": "fixed", "random_init": True
        }
    ]
        

    resultados_heuristica = []

    for prob in problemas:
        estado_texto = "Ruido (100%)" if prob['random_init'] else "Semilla Central (0% Ruido)"
        print(f"\n---> Evaluando: {prob['nombre'].upper()} (H:{prob['h_bound']}, V:{prob['v_bound']} | Inicio: {estado_texto}) <---")
        
        heuristica_conf = generate_competent_heuristic_config(
            ca_size=prob['target'].shape, 
            num_states=prob['estados'], 
            target_state=prob['target'],
            colors=prob['colores'],
            h_bound=prob['h_bound'], 
            v_bound=prob['v_bound'],
            use_random_init=prob['random_init'] 
        )
        
        print(f"[*] Lanzando DEAP (Ablación: Mismo entorno físico, 2 Etapas Mutación, SIN Transfer Learning)...")
        
        try:
            top3, fitness_evolution = GeneticAlgorithm(config=heuristica_conf, seed_rule=None)
            
            mejor_fitness = fitness_evolution[-1]
            generaciones_usadas = len(fitness_evolution)
            
            print(f"[!] {prob['nombre']} -> Fitness final: {mejor_fitness:.4f}")
            
            resultados_heuristica.append({
                "problema": prob['nombre'],
                "fitness": mejor_fitness,
                "generaciones": generaciones_usadas
            })
            
            graphic_fitness_evolution(fitness_evolution, prob['nombre'])
            
        except Exception as e:
            print(f"[ERROR] Fallo en la ejecución de {prob['nombre']}: {e}")
            resultados_heuristica.append({
                "problema": prob['nombre'],
                "fitness": 0.0,
                "generaciones": "ERROR"
            })

    # Resumen Final en consola
    print("\n=========================================================")
    print(" RESUMEN DE RESULTADOS LÍNEA BASE (HEURÍSTICA ABLACIÓN)")
    print("=========================================================")
    print(f"{'Problema':<15} | {'Fitness Alcanzado':<18} | {'Generaciones'}")
    print("-" * 50)
    for r in resultados_heuristica:
        if r['generaciones'] != "ERROR":
            print(f"{r['problema']:<15} | {r['fitness']:<18.4f} | {r['generaciones']}")
        else:
            print(f"{r['problema']:<15} | {'ERROR':<18} | {'ERROR'}")