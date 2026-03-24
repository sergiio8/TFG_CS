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

def obtener_caso_japon() -> CasoCBR:
    """Instancia y devuelve el caso resuelto de la Bandera de Japón en 2D."""
    
    # 1. Definición del tamaño y el estado objetivo (Target)
    ca_size = (30, 45)
    target_japon = np.zeros(ca_size, dtype=int)
    
    h = ca_size[0]
    w = ca_size[1]
    
    # Encontrar el centro exacto
    center_y = h // 2
    center_x = w // 2
    
    # Radio del círculo en la bandera de Japón es 3/5 del alto total. r = (h * 3/5) / 2 = h * 0.3
    radio = h * 0.3
    
    # Crear la cuadrícula de coordenadas
    y, x = np.ogrid[:h, :w]
    
    # Aplicar la fórmula del círculo
    mascara_circulo = ((x - center_x)**2 + (y - center_y)**2) <= radio**2
    target_japon[mascara_circulo] = 1 # Rojo
    
    # Semilla central (Morfogénesis)
    semilla_inicial = np.fromfunction(lambda i, j: (i == center_y) & (j == center_x), ca_size).astype(int)

    # 2. Configuración del Autómata Celular (CAConfig)
    ca_conf = CAConfig(
        ca_size=ca_size,
        num_states=2,
        colors=["#FFFFFF", "#D31414"], # Blanco, Rojo
        ca_timesteps=20,
        ca_neighborhood='Moore',
        random_initial_state=False,
        ca_initial_state=semilla_inicial,
        ca_horizontal_boundary_conditions='periodic',
        ca_vertical_boundary_conditions='periodic',
        ca_row0_state=np.array([0] + [1] * (ca_size[1] - 1), dtype=int),
        ca_rowN_state=np.array([0] + [2] * (ca_size[1] - 1), dtype=int),
        ca_column0_state=np.full((ca_size[0],), 0, dtype=int),
        ca_columnN_state=np.array([1]*(ca_size[0] // 2) + [2]*(ca_size[0] - (ca_size[0] // 2)), dtype=int),
        num_ca_test=1,
        random_probability = 0.0,
        target_state=target_japon
    )
    
    # 3. Configuración del Algoritmo Genético (GAConfig)
    ga_conf = GAConfig(
        pop_size=1000,
        cx_prob=0.8,
        mut_prob=1.0,
        adaptative_mut_prob=False,
        mutation_probability_interval=np.array([0, 0.25, 0.45, 0.6, 1]),
        mutation_probabilities=np.array([0.2, 0.1, 0.05, 0.025]),
        adaptative_num_mut=True,
        mutation_interval=np.array([0, 0.25, 0.45, 0.6, 0.75, 1]),
        num_mutations=np.array([25, 18, 12, 8, 4]),
        num_generations=2000,
        stop_condition=0.95,
        adaptative_fitness=False,
        ca_fitness=1,
        adaptative_fitness_threshold=0.5,
        weight_SSIM=0.2,
        weight_Jaccard=0.2,
        weights_states_Jaccard = [1.0, 3.0],
        weight_accuracy=0.0,
        weight_mutual_information=0.6,
        penalized_fitness=False,
        minimum_percentage=0.08,
        penalization_factor=0.5,
        classic_ga=False,
        sel_method='tournament',
        tournament_size=3,
        cx_method='twoPoint',
        elite_passing=0.1,
        mu_lambda_ga=True,
        diversity=0.02
    )
    
    # Empaquetamos la configuración
    framework_conf = FrameworkConfig(ca=ca_conf, ga=ga_conf)
    
    # 4. Características del problema
    momentos_japon = calcular_momentos_hu(target_japon)
    
    problema = CaracteristicasProblema(
        dimensiones=2,
        num_estados=2,
        borde_uniforme=True, # Todo el borde es blanco
        patrones_detectados={"circulo"},
        franjas_horizontales=False, 
        franjas_verticales=False,
        momentos_hu=momentos_japon,
        componentes_conexas=2
    )
    
    # 5. La Solución (Regla genómica ganadora y fitness alcanzado)
    mejor_regla = [
        0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
    
    solucion = SolucionCBR(
        configuracion=framework_conf,
        mejor_regla=mejor_regla,
        fitness_alcanzado=0.85
    )
    
    # 6. Empaquetamos todo en el Caso Final
    return CasoCBR(
        id_caso='BanderaJapon2DIM', 
        problema=problema, 
        solucion=solucion
    )
