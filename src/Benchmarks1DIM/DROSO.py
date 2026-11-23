import numpy as np
from configuration import (
    CAConfig, GAConfig, FrameworkConfig, 
    CaracteristicasProblema, SolucionCBR, CasoCBR
)

def obtener_caso_drosophila() -> CasoCBR:
    """Instancia y devuelve el caso resuelto de la Drosophila en 1D."""
    
    # 1. Definición del tamaño y el estado objetivo (Target)
    ca_size = (70, 1)
    
    '''Patrón final deseado (drosophila)'''
    dim_drosophila = 70
    target_droso = np.zeros(dim_drosophila, dtype=int)
        
    ancho_franja = 5
    for i in range(7):
        # Calculamos el inicio y fin de cada una de las 7 franjas activas
        inicio_activa = i * 2 * ancho_franja
        fin_activa = inicio_activa + ancho_franja
        target_droso[inicio_activa:fin_activa] = 1
    
    # 2. Configuración del Autómata Celular (CAConfig)
    ca_conf = CAConfig(
        ca_size=ca_size,
        num_states=2,
        colors = ["#333333", "#00FFFF"], 
        ca_timesteps=90,
        ca_neighborhood='3', # Radio de vecindad de 3 (3 celdas)
        random_initial_state=True,
        ca_horizontal_boundary_conditions='fixed',
        ca_vertical_boundary_conditions='fixed',
        ca_row0_state=np.array([0, 0], dtype=int), # Bordes fijos (0 a la izquierda, 0 a la derecha)
        num_ca_test=50,
        random_probability = 1.0,
        target_state=target_droso
    )
    
    # 3. Configuración del Algoritmo Genético (GAConfig)
    ga_conf = GAConfig(
        pop_size=250,
        cx_prob=0.9,
        mut_prob=1.0,
        adaptative_mut_prob=False,
        mutation_probability_interval=np.array([0, 0.25, 0.45, 0.6, 1]),
        mutation_probabilities=np.array([0.2, 0.1, 0.05, 0.025]),
        adaptative_num_mut=False, # Mutación no adaptativa
        mutation_interval=np.array([0, 0.25, 0.45, 0.6, 0.75, 1]),
        num_mutations=np.array([3]), # Mutación estática a 3 genes
        num_generations=1000,
        stop_condition=0.97,
        adaptative_fitness=True,
        ca_fitness=50,
        adaptative_fitness_threshold=0.5,
        weight_SSIM=0.0,
        weight_Jaccard=0.0,
        weights_states_Jaccard = [1.0, 1.0, 1.0],
        weight_accuracy=1.0, 
        weight_mutual_information=0.0,
        penalized_fitness=False,
        minimum_percentage=0.1,
        penalization_factor=0.5,
        gaussian_filter= False,
        gaussian_sigma= 3.0,
        classic_ga=False,
        sel_method='tournament',
        tournament_size=3,
        cx_method='twoPoint',
        elite_passing=0.15,
        mu_lambda_ga=True,
        diversity=0.05
    )
    
    # Empaquetamos la configuración
    framework_conf = FrameworkConfig(ca=ca_conf, ga=ga_conf)
    
    # 4. Características del problema
    problema = CaracteristicasProblema(
        dimensiones=1,
        num_estados=2,
        borde_uniforme=True,
        patrones_detectados={"franjas", "periodicidad"},
        franjas_horizontales=False, 
        franjas_verticales=False,
        momentos_hu=[],
        componentes_conexas=14
    )
    
    # 5. La Solución (Regla genómica ganadora y fitness alcanzado)
    mejor_regla = [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    
    solucion = SolucionCBR(
        configuracion=framework_conf,
        mejor_regla=mejor_regla,
        fitness_alcanzado=0.71
    )
    
    # 6. Empaquetamos todo en el Caso Final
    return CasoCBR(
        id_caso='Drosophila1DIM', 
        problema=problema, 
        solucion=solucion
    )
