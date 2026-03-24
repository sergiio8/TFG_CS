import numpy as np
import cv2
import math
from configuration import (
    CAConfig, GAConfig, FrameworkConfig, 
    CaracteristicasProblema, SolucionCBR, CasoCBR
)

def calcular_momentos_hu(target_matrix: np.ndarray):
    """Función auxiliar para calcular los momentos de Hu de la bandera."""
    im_binaria = np.where(target_matrix > 0, 255, 0).astype(np.uint8)
    moments = cv2.moments(im_binaria)
    huMoments = cv2.HuMoments(moments)
    
    momentos_log = []
    for i in range(7):
        hu = huMoments[i][0]
        if hu != 0:
            transformado = -1 * math.copysign(1.0, hu) * math.log10(abs(hu))
        else:
            transformado = 0.0
        momentos_log.append(float(transformado))
    return momentos_log

def obtener_caso_hungria() -> CasoCBR:
    """Instancia y devuelve el caso resuelto de la Bandera de Hungría en 2D."""
    
    # 1. Definición del tamaño y el estado objetivo (Target)
    ca_size = (18, 30)
    target_hungria = np.zeros(ca_size, dtype=int)
    h = ca_size[0] // 3
    target_hungria[0:h, :] = 0     # Franja superior (Rojo)
    target_hungria[h:2*h, :] = 1   # Franja central (Blanco)
    target_hungria[2*h:, :] = 2    # Franja inferior (Verde)
    
    # 2. Configuración del Autómata Celular (CAConfig)
    ca_conf = CAConfig(
        ca_size=ca_size,
        num_states=3,
        colors=["#F40F0F", "#FFFFFF", "#18AB27"], # Rojo, Blanco, Verde
        ca_timesteps=75,
        ca_neighborhood='von Neumann',
        random_initial_state=True,
        ca_horizontal_boundary_conditions='fixed',
        ca_vertical_boundary_conditions='periodic',
        ca_row0_state=np.zeros(ca_size[1], dtype=int),         # Borde superior fijo a 0
        ca_rowN_state=np.full((ca_size[1],), 2, dtype=int),    # Borde inferior fijo a 2
        num_ca_test=30,
        random_probability = 1.0,
        target_state=target_hungria
    )
    
    # 3. Configuración del Algoritmo Genético (GAConfig)
    ga_conf = GAConfig(
        pop_size=300,
        cx_prob=0.9,
        mut_prob=1.0,
        adaptative_mut_prob=False,
        mutation_probability_interval=np.array([0, 0.25, 0.45, 0.6, 1]),
        mutation_probabilities=np.array([0.2, 0.1, 0.05, 0.025]),
        adaptative_num_mut=True,
        mutation_interval=np.array([0, 0.25, 0.45, 0.6, 0.75, 1]),
        num_mutations=np.array([10, 8, 6, 4, 2]),
        num_generations=2000,
        stop_condition=0.9,
        adaptative_fitness=True,
        ca_fitness=30,
        adaptative_fitness_threshold=0.5,
        weight_SSIM=0.8,
        weight_Jaccard=0.2,
        weights_states_Jaccard=[1.0, 1.0, 1.0],
        weight_accuracy=0.0,
        weight_mutual_information=0.0,
        penalized_fitness=True,
        minimum_percentage=0.1,
        penalization_factor=0.5,
        classic_ga=False,
        sel_method='tournament',
        tournament_size=3,
        cx_method='twoPoint',
        elite_passing=0.1,
        mu_lambda_ga=True,
        diversity=0.00
    )
    
    # Empaquetamos la configuración
    framework_conf = FrameworkConfig(ca=ca_conf, ga=ga_conf)
    
    # 4. Características del problema (Lo que leerá el CBR en la fase Retrieve)
    # Calculamos los momentos de Hu de la bandera de Hungría en este momento
    momentos_hungria = calcular_momentos_hu(target_hungria)
    
    problema = CaracteristicasProblema(
        dimensiones=2,
        num_estados=3,
        borde_uniforme=False, # Arriba es rojo, abajo es verde
        patrones_detectados={"franjas"},
        franjas_horizontales=True, 
        franjas_verticales=False,
        momentos_hu=momentos_hungria,
        componentes_conexas=3,
    )
    
    # 5. La Solución (Regla genómica ganadora y fitness alcanzado)
    mejor_regla = [
        0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 1, 1, 2, 2, 2, 0, 0, 2, 1, 0, 0, 0, 0, 2, 
        1, 2, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 2, 2, 1, 0, 2, 2, 
        1, 0, 0, 2, 1, 1, 1, 1, 0, 2, 1, 2, 1, 1, 2, 1, 1, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 
        1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0, 2, 0, 2, 2, 0, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 2, 1, 0, 1, 0, 2, 1, 2, 
        0, 2, 0, 0, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 0, 2, 0, 2, 2, 2, 1, 1, 2, 1, 1, 2, 
        1, 2, 1, 1, 1, 1, 1, 0, 2, 1, 1, 2, 1, 0, 2, 2, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2, 
        0, 1, 1, 2, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 1, 2, 2, 0, 1, 1, 2, 2, 
        0, 1, 0, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 2, 1, 1, 2, 0, 2
    ]
    
    solucion = SolucionCBR(
        configuracion=framework_conf,
        mejor_regla=mejor_regla,
        fitness_alcanzado=0.78
    )
    
    # 6. Empaquetamos todo en el Caso Final
    return CasoCBR(
        id_caso='BanderaHungria2DIM', 
        problema=problema, 
        solucion=solucion
    )
