import numpy as np
from configuration import (
    CAConfig, GAConfig, FrameworkConfig, 
    CaracteristicasProblema, SolucionCBR, CasoCBR
)

def obtener_caso_francia() -> CasoCBR:
    """Instancia y devuelve el caso resuelto de la Bandera de Francia en 1D."""
    
    # 1. Definición del tamaño y el estado objetivo (Target)
    ca_size = (30, 1)
    
    # En 1D, el target state es una lista: Azul(0), Blanco(1), Rojo(2)
    tercio = ca_size[0] // 3
    target_francia = np.array([0] * tercio + [1] * tercio + [2] * tercio, dtype=int)
    
    # 2. Configuración del Autómata Celular (CAConfig)
    ca_conf = CAConfig(
        ca_size=ca_size,
        num_states=3,
        colors=["#310FF4", "#FFFFFF", "#EF1515"], # Azul, Blanco, Rojo
        ca_timesteps=90,
        ca_neighborhood='1', # Radio de vecindad de 1 (3 celdas)
        random_initial_state=True,
        ca_horizontal_boundary_conditions='periodic',
        ca_vertical_boundary_conditions='fixed',
        ca_row0_state=np.array([0, 2], dtype=int), # Bordes fijos (0 a la izquierda, 2 a la derecha)
        num_ca_test=100,
        random_probability = 1.0,
        target_state=target_francia
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
        num_mutations=np.array([1]), # Mutación estática a 1 gen
        num_generations=2000,
        stop_condition=0.97,
        adaptative_fitness=True,
        ca_fitness=100,
        adaptative_fitness_threshold=0.7,
        weight_SSIM=0.0,
        weight_Jaccard=0.0,
        weights_states_Jaccard = [1.0, 1.0, 1.0],
        weight_accuracy=1.0, 
        weight_mutual_information=0.0,
        penalized_fitness=True,
        minimum_percentage=0.1,
        penalization_factor=0.5,
        classic_ga=False,
        sel_method='tournament',
        tournament_size=3,
        cx_method='twoPoint',
        elite_passing=0.2,
        mu_lambda_ga=True,
        diversity=0.00
    )
    
    # Empaquetamos la configuración
    framework_conf = FrameworkConfig(ca=ca_conf, ga=ga_conf)
    
    # 4. Características del problema
    problema = CaracteristicasProblema(
        dimensiones=1,
        num_estados=3,
        borde_uniforme=False,
        patrones_detectados={"franjas"},
        franjas_horizontales=False, 
        franjas_verticales=False,
        momentos_hu=[],
        componentes_conexas=3
    )
    
    # 5. La Solución (Regla genómica ganadora y fitness alcanzado)
    mejor_regla = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 0, 1, 1, 0, 1, 2, 0, 2, 0, 0, 2, 2, 1, 0, 2
    ]
    
    solucion = SolucionCBR(
        configuracion=framework_conf,
        mejor_regla=mejor_regla,
        fitness_alcanzado=0.95
    )
    
    # 6. Empaquetamos todo en el Caso Final
    return CasoCBR(
        id_caso='BanderaFrancia1DIM', 
        problema=problema, 
        solucion=solucion
    )
