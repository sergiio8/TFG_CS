import numpy as np
import math
import sys
import subprocess

import cv2
import copy
from typing import List, Tuple

# Importamos nuestros modelos de datos
from configuration import (
    CAConfig, GAConfig, FrameworkConfig, 
    CaracteristicasProblema, SolucionCBR, CasoCBR
)

import matplotlib.pyplot as plt

from Evolucion import GeneticAlgorithm

# ==========================================
# FUNCIONES AUXILIARES DE VISIÓN Y SIMILITUD
# ==========================================

'''Código de colores'''
AZUL_CLARO   = "\033[94m" # Para el estado azul
BLANCO_BRILLANTE = "\033[97m" # Para el estado blanco
ROJO_BRILLANTE   = "\033[91m" # Para el estado rojo
VERDE_BRILLANTE = "\033[92m" # Para el estado verde
RESET      = "\033[0m"  # Para resetear el color al final

def graphic_fitness_evolution(fitness_evolution):

    generaciones = range(len(fitness_evolution))
    fitness_values = fitness_evolution

    plt.figure(figsize=(12, 6)) # Tamaño de la imagen
    plt.plot(generaciones, fitness_values, 
            linewidth=2, 
            color='#1f77b4', # Azul estándar
            label='Mejor Individuo')

    # Añadir puntos rojos en cada generación
    plt.scatter(generaciones, fitness_values, color='red', s=10, zorder=5)

    # 3. Etiquetas y Estilo
    plt.title('Evolución del Fitness (Algoritmo Genético)', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Max Fitness', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # 4. Anotación del valor máximo
    if len(fitness_values) > 0:
        max_val = max(fitness_values)
        max_gen = fitness_values.index(max_val)
        
        # Ajuste dinámico: si la generación es baja, pone el texto a la derecha, si es alta, a la izquierda
        offset_x = -50 if max_gen > 50 else 20
    
        plt.annotate(f'Max: {max_val:.4f}', 
                    xy=(max_gen, max_val), 
                    xytext=(max_gen + offset_x, max_val),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    # 5. Mostrar
    plt.tight_layout()
    plt.show()

def dibujar_regla_1dim(regla, CA):
    
    simbolo_0 = "■" # Azul
    simbolo_1 = "■" # Blanco
    simbolo_2 = "■" # Rojo
    
    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(CA.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = CA.colors[i].lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        
        # Creamos el símbolo coloreado para la terminal
        simbolos_map[str(i)] = f"\033[38;2;{r};{g};{b}m■\033[0m"
    
    print("--- Catálogo de Reglas del AC (Índice -> Configuración -> Resultado) ---")
    
    longitud_vecindad = 2 * int(CA.ca_neighborhood) + 1

    for indice in range(CA.ind_size):
        
        # 1. Obtener la configuración (en base 3)
        config_str = np.base_repr(indice, base=CA.num_states).zfill(longitud_vecindad)
        
        # 2. Convertir a símbolos visuales
        config_visual = "".join([simbolos_map[c] for c in config_str])
        
        # 3. Obtener el resultado
        resultado = regla[indice] # El resultado será 0, 1, o 2
        resultado_visual = simbolos_map[str(resultado)]
        
        # 4. Imprimir la línea (ajusta el padding '04d' si IND_SIZE es muy grande)
        print(f"Índice {indice:04d}:   {config_visual}   ->   {resultado_visual}")


'''Funcion que, dada una regla, la muestra gráficamente con las configuraciones de vecinos y los colores correspondientes'''
def dibujar_regla_vonNeumann(regla, CA):
    simbolo = "■" 

    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(CA.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = CA.colors[i].lstrip('#')
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
        config_str = np.base_repr(indice, base=CA.num_states).zfill(longitud_vecindad)
        
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
        print(f"Índice {indice:03d}:")
        print(f"    {N}    ")       # Línea superior (Norte)
        print(f"  {W} {C} {E}  ->  {res_visual}") # Línea media (Oeste, Centro, Este) -> Resultado
        print(f"    {S}    ")       # Línea inferior (Sur)
        print("-" * 20)             # Separador

def dibujar_regla_Moore(regla, CA):
    simbolo = "■" 

    # Ajusta los colores si usas otra bandera
    simbolos_map = {}
    for i in range(CA.num_states):
        # Quitamos el '#' y sacamos los valores RGB del color Hexadecimal
        hex_c = CA.colors[i].lstrip('#')
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
        config_str = np.base_repr(indice, base=CA.num_states).zfill(longitud_vecindad)
        
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
        print(f"Índice {indice:03d}:")
        print(f" {N1} {N2} {N3}    ")       # Línea superior (Norte 1)
        print(f" {N8} {C} {N4}  ->  {res_visual}") # Línea media (Norte 8, Centro, Norte 4) -> Resultado
        print(f" {N7} {N6} {N5}    ")       # Línea inferior (Sur 1)
        print("-" * 20)             # Separador


def momentos_hu(target: np.ndarray) -> List[float]:
    """Calcula los momentos de Hu para matrices 2D."""
    # Si es 1D, no aplicamos Momentos de Hu
    if len(target.shape) < 2 or target.shape[1] == 1 or target.shape[0] == 1:
        return []
        
    im_binaria = np.where(target > 0, 255, 0).astype(np.uint8)
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

def similitud_hu(momentos_A: List[float], momentos_B: List[float]) -> float:
    if not momentos_A or not momentos_B:
        return 0.0 # Si no hay momentos (ej. 1D), similitud 0
        
    vec_A = np.array(momentos_A)
    vec_B = np.array(momentos_B)
    distancia = np.linalg.norm(vec_A - vec_B)
    return 1.0 / (1.0 + distancia)

def porcentaje_color(target: np.ndarray) -> List[float]:
    num_pixels = target.size
    
    _, conteos = np.unique(target, return_counts=True)
    
    porcentajes = [c / num_pixels for c in conteos]
    
    porcentajes.sort(reverse=True)
    
    return porcentajes

def contar_componentes_conexas(target: np.ndarray) -> int:
    if len(target.shape) < 2 or target.shape[0] == 1 or target.shape[1] == 1:
        target_flat = target.flatten()
        if len(target_flat) == 0: return 0
        # Cuenta cuántas veces cambia el color al siguiente píxel + 1
        cambios = np.sum(target_flat[:-1] != target_flat[1:])
        return int(cambios + 1)
        
    total_componentes = 0
    estados = np.unique(target)
    
    for estado in estados:
        mascara = np.uint8(target == estado)
        
        # 2. cv2.connectedComponents cuenta los bloques conectados.
        num_labels, labels = cv2.connectedComponents(mascara, connectivity=4)
        
        # 3. num_labels siempre cuenta el fondo (los 0s) como una componente.
        total_componentes += (num_labels - 1)
        
    return total_componentes


def franjas(target: np.ndarray) -> Tuple[bool, bool]:
    """Detecta si hay franjas horizontales o verticales."""
    if len(target.shape) < 2 or target.shape[1] == 1 or target.shape[0] == 1:
        return False, False # En 1D lo manejamos diferente
        
    es_vertical = np.all(target == target[0, :], axis=0).all()
    es_horizontal = np.all(target.T == target.T[0, :], axis=0).all()

    return es_horizontal, es_vertical

def tiene_circulo_central(target: np.ndarray) -> bool:
    """
    Detecta si hay una masa de color en el centro que no toca los bordes
    y tiene proporciones similares de ancho y alto (como un círculo o cuadrado).
    """
    if len(target.shape) < 2 or target.shape[0] < 3 or target.shape[1] < 3:
        return False

    h, w = target.shape
    centro_y, centro_x = h // 2, w // 2
    
    # 1. Miramos qué color hay exactamente en el píxel central
    color_centro = target[centro_y, centro_x]
    
    # 2. Aislamos ese color
    mascara = np.uint8(target == color_centro)
    
    # 3. Extraemos las islas y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, connectivity=4)
    
    # 4. Vemos a qué isla pertenece el píxel central
    label_central = labels[centro_y, centro_x]
    
    # Extraemos las propiedades de esa isla específica
    x = stats[label_central, cv2.CC_STAT_LEFT]
    y = stats[label_central, cv2.CC_STAT_TOP]
    w_comp = stats[label_central, cv2.CC_STAT_WIDTH]
    h_comp = stats[label_central, cv2.CC_STAT_HEIGHT]
    
    # Condición A: La forma NO puede tocar los bordes del lienzo
    toca_bordes = (x == 0) or (y == 0) or (x + w_comp == w) or (y + h_comp == h)
    if toca_bordes:
        return False
        
    # Condición B: Debe ser aproximadamente simétrica
    aspect_ratio = w_comp / float(h_comp)
    if 0.70 <= aspect_ratio <= 1.40:
        return True
        
    return False

def mutual_information_metric(target_A: np.ndarray, target_B: np.ndarray) -> float:
    """Calcula la Información Mutua Normalizada (NMI)."""
    if target_A.shape != target_B.shape:
        try:
            target_B_resized = cv2.resize(
                target_B.astype(np.uint8), 
                (target_A.shape[1], target_A.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        except:
            target_B_resized = target_B # Fallback por si es 1D
    else:
        target_B_resized = target_B

    A = target_A.flatten()
    B = target_B_resized.flatten()
    
    min_len = min(len(A), len(B))
    A = A[:min_len]
    B = B[:min_len]
    N = len(A)
    
    estados_A = np.unique(A)
    estados_B = np.unique(B)

    h_a = sum(- (np.sum(A == i)/N) * math.log(np.sum(A == i)/N) for i in estados_A if np.sum(A == i) > 0)
    h_b = sum(- (np.sum(B == j)/N) * math.log(np.sum(B == j)/N) for j in estados_B if np.sum(B == j) > 0)

    h_ab = 0.0
    for i in estados_A:
        for j in estados_B:
            p_ij = np.sum((A == i) & (B == j)) / N
            if p_ij > 0: 
                h_ab += p_ij * math.log(p_ij) 

    if (h_a + h_b) == 0:
        return 1.0

    return 2 * (h_a + h_b + h_ab) / (h_a + h_b)


# ==========================================
# MOTOR CBR
# ==========================================

class CBREngine:
    def __init__(self, instanciador):
        self.instanciador = instanciador

    def extract_features(self, target_matrix: np.ndarray) -> CaracteristicasProblema:
        print("\n--- FASE 1: EXTRACT (Extracción de características) ---")
        
        dimension = 1 if len(target_matrix.shape) == 1 or target_matrix.shape[1] == 1 else 2
        num_estados = len(np.unique(target_matrix))
        print(f"[*] Dimensión: {dimension}D | Estados detectados: {num_estados}")
        patrones = set()

        # Análisis de bordes
        if dimension == 2:
            top_edge = target_matrix[0, :]
            bottom_edge = target_matrix[-1, :]
            left_edge = target_matrix[:, 0]
            right_edge = target_matrix[:, -1]
            all_edges = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
            borde_uniforme = len(np.unique(all_edges)) == 1
            f_h, f_v = franjas(target_matrix)
            if f_h or f_v:
                patrones.add("franjas")
            if tiene_circulo_central(target_matrix):
                patrones.add("circulo")
        else:
            borde_uniforme = len(np.unique(target_matrix[[0, -1]])) == 1
            f_h, f_v = False, False

        print(f"[*] Borde uniforme: {borde_uniforme} | Franjas H: {f_h} | Franjas V: {f_v}")
        
        momentos = momentos_hu(target_matrix)

        comp_conexas = contar_componentes_conexas(target_matrix)
        print(f"[*] Componentes conexas detectadas: {comp_conexas}")
       

        problema_actual = CaracteristicasProblema(
            dimensiones=dimension,
            num_estados=num_estados,
            borde_uniforme=borde_uniforme,
            franjas_horizontales=f_h,
            franjas_verticales=f_v,
            target_state=target_matrix,
            patrones_detectados= patrones,
            componentes_conexas=comp_conexas
        )
        problema_actual.momentos_hu = momentos
        
        return problema_actual

    def retrieve(self, target_matrix: np.ndarray, target_features: CaracteristicasProblema) -> List[Tuple[CasoCBR, float]]:
        print("\n--- FASE 2: RETRIEVE (Recuperación de casos) ---")
        casos_disponibles = self.instanciador.obtener_todos_los_casos()
        casos_ordenados = []
        threshold_similarity = 0.55

        for caso in casos_disponibles:
            # 1. Similitud Topológica (20%)
            topo_score = 0.0
            s_h = 1 if caso.problema.franjas_verticales and target_features.franjas_verticales else 0
            s_v = 1 if caso.problema.franjas_horizontales and target_features.franjas_horizontales else 0
            b_u = 1 if caso.problema.borde_uniforme == target_features.borde_uniforme else 0
            franjas = 1 if "franjas" in caso.problema.patrones_detectados and "franjas" in target_features.patrones_detectados else 0
            circulo = 1 if "circulo" in caso.problema.patrones_detectados and "circulo" in target_features.patrones_detectados else 0
            
            topo_score = (s_h + s_v + b_u + franjas + circulo) / 5.0

            # 2. Similitud Estructural (50%)
            target_caso = caso.solucion.configuracion.ca.target_state
            
            # NMI Directo (sin evaluar rotaciones)
            sim_mutual_info = mutual_information_metric(target_matrix, target_caso)

            # Porcentaje de cada color (Histograma)
            porc_target = porcentaje_color(target_matrix)
            porc_caso = porcentaje_color(caso.solucion.configuracion.ca.target_state)

            max_len = max(len(porc_target), len(porc_caso))
            porc_target += [0.0] * (max_len - len(porc_target))
            porc_caso += [0.0] * (max_len - len(porc_caso))

            distancia = np.linalg.norm(np.array(porc_target) - np.array(porc_caso))
            sim_histograma = 1.0 / (1.0 + distancia)

            # Componentes conexas
            comp_target = target_features.componentes_conexas
            comp_caso = caso.problema.componentes_conexas
            sim_componentes = 1.0 / (1.0 + abs(comp_target - comp_caso))

            forma_score = (sim_mutual_info + sim_histograma + sim_componentes) / 3.0

            # 3. Compatibilidad de Estados (10%)
            diferencia_estados = abs(target_features.num_estados - caso.problema.num_estados)
            if diferencia_estados == 0:
                estado_score = 1.0   
            elif diferencia_estados == 1:
                estado_score = 0.5   
            else:
                estado_score = 0.0   

            # 4 Compatibilidad de Dimensión (20%)
            score_dim = 1.0 if target_features.dimensiones == caso.problema.dimensiones else 0.0

            
            # Puntuación final ponderada
            score_total = (topo_score * 0.20) + (forma_score * 0.50) + (estado_score * 0.10) + (score_dim * 0.20)
            
            print(f"[*] Analizando: {caso.id_caso}")
            print(f"    |-- Topológica  (20%): {topo_score:.2f}  [SimH:{s_h} | SimV:{s_v} | BordeU:{b_u} | Franjas:{franjas} | Circulo:{circulo}]")
            print(f"    |-- Estructural (50%): {forma_score:.2f}  [NMI:{sim_mutual_info:.2f} | Hist:{sim_histograma:.2f} | Comp:{sim_componentes:.2f}]")
            print(f"    |-- Estados     (10%): {estado_score:.2f}  [Diff:{diferencia_estados}]")
            print(f"    |-- Dimensiones (20%): {score_dim:.2f}")
            print(f"    └──> SIMILITUD FINAL:  {score_total:.2f}\n")

            casos_ordenados.append((caso, score_total))

        casos_ordenados.sort(key=lambda x: x[1], reverse=True)
        resultados_finales = [c for c in casos_ordenados[:1] if c[1] >= threshold_similarity]

        if len(resultados_finales) == 0:
            resultados_finales = [casos_ordenados[0]]

        return resultados_finales

    def reuse(self, retrieved_cases: List[Tuple[CasoCBR, float]], target_features: CaracteristicasProblema, target_matrix: np.ndarray, colors: np.ndarray):
        print("\n--- FASE 3: REUSE (Adaptación y Promedio) ---")
        
        total_score = sum(score for _, score in retrieved_cases)
        pesos = [score / total_score for _, score in retrieved_cases]
        top1_case = retrieved_cases[0][0]
        
        print(f"[*] Mezclando hiperparámetros. Caso dominante: {top1_case.id_caso} (Peso: {pesos[0]:.2f})")
        
        nueva_ca = copy.deepcopy(top1_case.solucion.configuracion.ca)
        nueva_ga = copy.deepcopy(top1_case.solucion.configuracion.ga)
        
        nueva_ca.target_state = target_matrix.copy() 
        nueva_ca.num_states = target_features.num_estados

        if target_features.dimensiones == 1:
            nueva_ca.ca_size = (target_matrix.shape[0],) 
        else:
            nueva_ca.ca_size = tuple(target_matrix.shape)
            nueva_ca.target_state = target_matrix.copy() 

        nueva_ca.colors = colors

        # --- 1. Inicialización de la semilla (si aplica) ---
        if hasattr(nueva_ca, 'ca_initial_state') and nueva_ca.ca_initial_state is not None:
            nueva_ca.ca_initial_state = np.zeros(nueva_ca.ca_size, dtype=int)
            
            if len(nueva_ca.ca_size) > 1 and nueva_ca.ca_size[1] > 1:
                centro_y = nueva_ca.ca_size[0] // 2
                centro_x = nueva_ca.ca_size[1] // 2
                nueva_ca.ca_initial_state[centro_y, centro_x] = 1 
                print(f"[*] Estado inicial 2D (semilla central) creado en {nueva_ca.ca_size}")
            else:
                centro = nueva_ca.ca_size[0] // 2
                if len(nueva_ca.ca_size) > 1:
                    nueva_ca.ca_initial_state[centro, 0] = 1
                else:
                    nueva_ca.ca_initial_state[centro] = 1
                    
                print(f"[*] Estado inicial 1D (semilla central) creado en posición {centro}")

        # --- 2. Adaptación Dinámica de los Timesteps ---
        old_shape = top1_case.solucion.configuracion.ca.ca_size
        new_shape = nueva_ca.ca_size

        old_iterable = old_shape if isinstance(old_shape, (tuple, list)) else [old_shape]
        new_iterable = new_shape if isinstance(new_shape, (tuple, list)) else [new_shape]

        max_dim_old = max(old_iterable)
        max_dim_new = max(new_iterable)

        ratio_crecimiento = max_dim_new / max_dim_old
        old_timesteps = top1_case.solucion.configuracion.ca.ca_timesteps

        nueva_ca.ca_timesteps = math.ceil(old_timesteps * ratio_crecimiento)

        if target_features.num_estados == 3 and target_features.dimensiones == 2:
            nueva_ca.ca_neighborhood = 'von Neumann'
        elif target_features.dimensiones == 1 and target_features.num_estados == 3:
            nueva_ca.ca_neighborhood = '1' 
        
        if target_features.dimensiones == 1:
            nueva_ca.random_initial_state = False 
            
            porcentajes = porcentaje_color(target_matrix)
            color_dominante = porcentajes[0]
            
            if color_dominante > 0.80:
                print("[*] 1D Detectado (Alta pureza): Inyectando estado inicial biológico (baja densidad de ruido).")
                estado_bio = np.random.choice([0, 1], size=target_matrix.shape, p=[0.95, 0.05])
                nueva_ca.ca_initial_state = estado_bio
            else:
                print("[*] 1D Detectado (Equilibrado): Inyectando estado inicial de ruido aleatorio puro.")
                nueva_ca.random_initial_state = True
                nueva_ca.ca_initial_state = None
                
        else:
            nueva_ca.random_initial_state = not target_features.borde_uniforme
            
            if hasattr(nueva_ca, 'ca_initial_state') and not nueva_ca.random_initial_state:
                nueva_ca.ca_initial_state = np.zeros(nueva_ca.ca_size, dtype=int)
                centro_y = nueva_ca.ca_size[0] // 2
                centro_x = nueva_ca.ca_size[1] // 2
                nueva_ca.ca_initial_state[centro_y, centro_x] = 1 
                print(f"[*] Estado inicial 2D (semilla central) creado en {nueva_ca.ca_size}")

        # Configuración de fronteras según la dimensión
        if target_features.dimensiones == 1:
            nueva_ca.ca_horizontal_boundary_conditions = 'fixed'
            nueva_ca.ca_vertical_boundary_conditions = 'fixed'
            
            nueva_ca.ca_row0_state = np.array([target_matrix[0]])
            nueva_ca.ca_rowN_state = np.array([target_matrix[-1]])
            
            nueva_ca.ca_column0_state = np.array([target_matrix[0]])
            nueva_ca.ca_columnN_state = np.array([target_matrix[-1]])
            
        else:
            # Heredamos directamente las condiciones sin rotación
            nueva_ca.ca_horizontal_boundary_conditions = top1_case.solucion.configuracion.ca.ca_horizontal_boundary_conditions
            nueva_ca.ca_vertical_boundary_conditions = top1_case.solucion.configuracion.ca.ca_vertical_boundary_conditions

            if nueva_ca.ca_horizontal_boundary_conditions == 'fixed':
                nueva_ca.ca_row0_state = target_matrix[0, :].copy()   
                nueva_ca.ca_rowN_state = target_matrix[-1, :].copy()  
            else:
                nueva_ca.ca_row0_state = None
                nueva_ca.ca_rowN_state = None
                
            if nueva_ca.ca_vertical_boundary_conditions == 'fixed':
                nueva_ca.ca_column0_state = target_matrix[:, 0].copy()  
                nueva_ca.ca_columnN_state = target_matrix[:, -1].copy() 
            else:
                nueva_ca.ca_column0_state = None
                nueva_ca.ca_columnN_state = None

        # Media Ponderada de GAConfig
        casos_ga = [c[0].solucion.configuracion.ga for c in retrieved_cases]
        
        nueva_ga.cx_prob = sum(ga.cx_prob * p for ga, p in zip(casos_ga, pesos))
        nueva_ga.mut_prob = sum(ga.mut_prob * p for ga, p in zip(casos_ga, pesos))
        nueva_ga.pop_size = int(round(sum(ga.pop_size * p for ga, p in zip(casos_ga, pesos))))
        nueva_ga.num_generations = int(round(sum(ga.num_generations * p for ga, p in zip(casos_ga, pesos))))
        
        w_ssim = sum(ga.weight_SSIM * p for ga, p in zip(casos_ga, pesos))
        w_jacc = sum(ga.weight_Jaccard * p for ga, p in zip(casos_ga, pesos))
        
        if target_features.num_estados == top1_case.problema.num_estados:
            w_states_jacc = copy.deepcopy(top1_case.solucion.configuracion.ga.weights_states_Jaccard)
            print(f"[*] Heredando pesos de estado para Jaccard de {top1_case.id_caso}.")
            
        else:
            print("[*] Adaptando pesos de estado para Jaccard (diferencia de dimensiones).")
            _, conteos = np.unique(target_matrix, return_counts=True)
            num_pixels = target_matrix.size
            
            w_states_jacc = []
            for c in conteos:
                proporcion = c / num_pixels
                peso = 1.0 - proporcion 
                w_states_jacc.append(peso)
            
            suma_pesos = sum(w_states_jacc)
            w_states_jacc = [w / suma_pesos for w in w_states_jacc]

        nueva_ga.weights_states_Jaccard = w_states_jacc
        w_acc = sum(ga.weight_accuracy * p for ga, p in zip(casos_ga, pesos))
        w_nmi = sum(ga.weight_mutual_information * p for ga, p in zip(casos_ga, pesos))
        
        total_weights = w_ssim + w_jacc + w_acc + w_nmi
        if total_weights > 0:
            nueva_ga.weight_SSIM = w_ssim / total_weights
            nueva_ga.weight_Jaccard = w_jacc / total_weights
            nueva_ga.weight_accuracy = w_acc / total_weights
            nueva_ga.weight_mutual_information = w_nmi / total_weights

        def promediar_arrays(lista_arrays, pesos, is_integer=False):
            min_len = min(len(arr) for arr in lista_arrays) 
            arrays_recortados = [arr[:min_len] for arr in lista_arrays]
            promedio = np.average(arrays_recortados, axis=0, weights=pesos)
            return np.round(promedio).astype(int) if is_integer else promedio

        if nueva_ga.adaptative_mut_prob:
            nueva_ga.mutation_probability_interval = promediar_arrays([ga.mutation_probability_interval for ga in casos_ga], pesos)
            nueva_ga.mutation_probabilities = promediar_arrays([ga.mutation_probabilities for ga in casos_ga], pesos)
            
        if nueva_ga.adaptative_num_mut:
            nueva_ga.mutation_interval = promediar_arrays([ga.mutation_interval for ga in casos_ga], pesos)
            
            tasas_mutacion_casos = []
            for (caso_tupla, score), ga in zip(retrieved_cases, casos_ga):
                caso_original = caso_tupla 
                longitud_historica = caso_original.solucion.configuracion.ca.ind_size
                tasa_historica = np.array(ga.num_mutations) / longitud_historica
                tasas_mutacion_casos.append(tasa_historica)

            tasa_promedio = promediar_arrays(tasas_mutacion_casos, pesos, is_integer=False)
            nueva_longitud = nueva_ca.ind_size
            
            mutaciones_adaptadas = np.round(tasa_promedio * nueva_longitud).astype(int)
            mutaciones_adaptadas = np.maximum(mutaciones_adaptadas, 1)
            nueva_ga.num_mutations = mutaciones_adaptadas

        # Inyección Memética (sin alterar por rotación)
        i = 0
        encontrado = False
        regla_semilla = None
        while i < len(retrieved_cases) and not encontrado:
            vecindario_coincide = retrieved_cases[i][0].solucion.configuracion.ca.ca_neighborhood == nueva_ca.ca_neighborhood
            estados_coinciden = retrieved_cases[i][0].problema.num_estados == target_features.num_estados
            
            if vecindario_coincide and estados_coinciden:
                encontrado = True
                regla_semilla = retrieved_cases[i][0].solucion.mejor_regla
                print(f"[*] Inyección Memética HABILITADA. Se usará la regla de {retrieved_cases[i][0].id_caso}")
            else:
                print("[*] Inyección Memética DESCARTADA (incompatibilidad de cromosoma).")
            i += 1

        print("\n" + "="*50)
        print("  RESUMEN DE LA NUEVA CONFIGURACIÓN HEREDADA")
        print("="*50)
        
        print("\n[1] AUTÓMATA CELULAR (CAConfig)")
        print(f"  |-- Dimensiones: {nueva_ca.ca_size}")
        print(f"  |-- Num Estados: {nueva_ca.num_states}")
        print(f"  |-- Vecindario:  {nueva_ca.ca_neighborhood}")
        print(f"  |-- Estado inicial aleatorio (Ruido): {nueva_ca.random_initial_state}")
        
        print("\n  [Condiciones de Contorno]")
        print(f"  |-- Horizontales: {nueva_ca.ca_horizontal_boundary_conditions}")
        print(f"      > Fila Sup (0) fijada:  {nueva_ca.ca_row0_state is not None}")
        print(f"      > Fila Inf (N) fijada:  {nueva_ca.ca_rowN_state is not None}")
        print(f"  |-- Verticales:   {nueva_ca.ca_vertical_boundary_conditions}")
        print(f"      > Col Izq (0) fijada:   {nueva_ca.ca_column0_state is not None}")
        print(f"      > Col Der (N) fijada:   {nueva_ca.ca_columnN_state is not None}")

        print("\n[2] ALGORITMO GENÉTICO (GAConfig)")
        print(f"  |-- Población:    {nueva_ga.pop_size}")
        print(f"  |-- Generaciones: {nueva_ga.num_generations}")
        print(f"  |-- Prob. Cruce:  {nueva_ga.cx_prob:.3f}")
        print(f"  |-- Prob. Mut:    {nueva_ga.mut_prob:.3f}")
        
        print("\n  [Métricas de Fitness Ponderadas]")
        print(f"  |-- SSIM:         {nueva_ga.weight_SSIM:.3f}")
        print(f"  |-- Jaccard:      {nueva_ga.weight_Jaccard:.3f}")
        print(f"  |-- NMI:          {nueva_ga.weight_mutual_information:.3f}")
        print(f"  |-- Accuracy:     {nueva_ga.weight_accuracy:.3f}")
        
        pesos_jacc_str = "[" + ", ".join([f"{w:.2f}" for w in nueva_ga.weights_states_Jaccard]) + "]"
        print(f"  |-- Pesos por Estado (Jaccard): {pesos_jacc_str}")
        
        print("\n[3] INYECCIÓN MEMÉTICA")
        if regla_semilla is not None:
            print("  |-- ESTADO: ACTIVA")
            print(f"  |-- Regla semilla inyectada correctamente en la Gen 0: {regla_semilla}")
        else:
            print("  |-- ESTADO: DESCARTADA (Evolución desde cero).")
        print("="*50 + "\n")
        config_final = FrameworkConfig(ca=nueva_ca, ga=nueva_ga)

        return config_final, regla_semilla

    def revise(self, config: FrameworkConfig, regla_semilla: List[int]):
        print("\n--- FASE 4: REVISE (Ejecución del Algoritmo) ---")
        print("[*] Simulando la ejecución evolutiva en DEAP...")

        top3, fitness_evolution = GeneticAlgorithm(
            config=config, 
            seed_rule=regla_semilla
        )
        graphic_fitness_evolution(fitness_evolution)
        for individual in top3:
            regla = np.array(individual)
            if config.ca.ca_neighborhood == 'von Neumann':
                dibujar_regla_vonNeumann(regla, config.ca)
            elif config.ca.ca_neighborhood == 'Moore':
                dibujar_regla_Moore(regla, config.ca)
            else:
                dibujar_regla_1dim(regla, config.ca)

        print(f"[*] Algoritmo finalizado. Fitness máximo alcanzado: {fitness_evolution[-1]:.4f}")
        return fitness_evolution[-1], top3[0]

    def retain(self, problem_id: str, features: CaracteristicasProblema, best_rule: List[int], config_final: FrameworkConfig, fitness: float):
        print("\n--- FASE 5: RETAIN (Guardado del conocimiento) ---")
        fitness_threshold = 0.5
        num_casos = len(self.instanciador.obtener_todos_los_casos())
        if num_casos > 15: fitness_threshold = 0.7
        elif num_casos > 10: fitness_threshold = 0.6

        if fitness > fitness_threshold:
            solucion = SolucionCBR(configuracion=config_final, mejor_regla=best_rule, fitness_alcanzado=fitness)
            nuevo_caso = CasoCBR(id_caso=problem_id, problema=features, solucion=solucion)
            
            self.instanciador.anadir_nuevo_caso(nuevo_caso)
        else:
            print("[*] Fitness insuficiente para memoria a largo plazo. Descartando caso.")
