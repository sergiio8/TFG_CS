import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rc
import cellpylib as cpl

# 1. Importa tus configuraciones para poder instanciar la física correcta
from configuration import CAConfig

# Para que las animaciones funcionen bien en Jupyter/VS Code interactivo
rc('animation', html='jshtml')

# ==========================================
# CONFIGURACIÓN DEL TEST
# ==========================================

# A. LA REGLA A PROBAR 
DIMENSIONES = (30, 30) # ¡Ajusta esto! (60,) para 1D, o (30,30) para 2D
NUM_ESTADOS = 2
TIMESTEPS = 15
VECINDARIO = 'Moore' # '1' para 1D, 'Moore' o 'von Neumann' para 2D

REGLA_PRUEBA = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# C. CONDICIONES DE CONTORNO
BORDES_HORIZONTALES = 'periodic'
BORDES_VERTICALES = 'periodic'

# Opcional: Si quieres probar fixed, define la matriz, si no, déjalo en None
target_matrix = None 
colors_matrix = ["#000000", "#FFFFFF"] # Fondo negro, figura blanca

# Instanciamos la física base
ca_config_test = CAConfig(
    ca_size=DIMENSIONES,
    num_states=NUM_ESTADOS,
    ca_neighborhood=VECINDARIO,
    ca_timesteps=TIMESTEPS,
    ca_horizontal_boundary_conditions=BORDES_HORIZONTALES,
    ca_vertical_boundary_conditions=BORDES_VERTICALES
)

# Fijamos los bordes matemáticos SOLO si target_matrix existe
if target_matrix is not None:
    if BORDES_HORIZONTALES == 'fixed':
        ca_config_test.ca_row0_state = target_matrix[0, :].copy()
        ca_config_test.ca_rowN_state = target_matrix[-1, :].copy()

    if BORDES_VERTICALES == 'fixed':
        ca_config_test.ca_column0_state = target_matrix[:, 0].copy()
        ca_config_test.ca_columnN_state = target_matrix[:, -1].copy()

# ==========================================
# IMPORTACIÓN DE LA FUNCIÓN DE TRANSICIÓN
# ==========================================
from Evolucion import create_transition_rule

# Dimensiones seguras por si es 1D
dim_1 = DIMENSIONES[0]
dim_2 = DIMENSIONES[1] if len(DIMENSIONES) > 1 else 1

mi_regla = create_transition_rule(
    individual=REGLA_PRUEBA, 
    ca_size_1=dim_1, 
    ca_size_2=dim_2, 
    ca_config=ca_config_test
)

# ==========================================
# INICIALIZACIÓN DEL LIENZO
# ==========================================

# 1. Ruido total inicial (CORREGIDO)
estado_inicial = np.random.randint(0, NUM_ESTADOS, DIMENSIONES)

# 2. Respetamos los anclajes de las fronteras fijadas
if target_matrix is not None:
    if BORDES_HORIZONTALES == 'fixed' and len(DIMENSIONES) > 1:
        estado_inicial[0, :] = target_matrix[0, :]
        estado_inicial[-1, :] = target_matrix[-1, :]

    if BORDES_VERTICALES == 'fixed' and len(DIMENSIONES) > 1:
        estado_inicial[:, 0] = target_matrix[:, 0]
        estado_inicial[:, -1] = target_matrix[:, -1]

# Expandimos dimensiones para cellpylib
estado_inicial = np.expand_dims(estado_inicial, axis=0)

# ==========================================
# SIMULACIÓN Y VISUALIZACIÓN
# ==========================================
print(f"[*] Iniciando evolución de {TIMESTEPS} timesteps...")

is_2d = VECINDARIO in ['von Neumann', 'Moore']

if is_2d:
    ca_evolucionado = cpl.evolve2d(
        estado_inicial, 
        timesteps=TIMESTEPS, 
        neighbourhood=VECINDARIO, 
        apply_rule=mi_regla
    )
else:
    # Para 1D
    ca_evolucionado = cpl.evolve(
        estado_inicial, 
        timesteps=TIMESTEPS, 
        apply_rule=mi_regla, 
        r=int(VECINDARIO)
    )

cmap_personal = ListedColormap(colors_matrix[:NUM_ESTADOS])

# Renderizado condicional según si es 1D o 2D
if is_2d:
    print("[*] Renderizando animación 2D...")
    # cpl.plot2d_animate a veces falla si el array es (T, Alto, 1). 
    cpl.plot2d_animate(ca_evolucionado, colormap=cmap_personal)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(ca_evolucionado[-1], cmap=cmap_personal, interpolation='nearest', aspect='auto')
    plt.title(f"Estado Final 2D (Timestep {TIMESTEPS})")
    plt.axis('off')
    plt.show()
else:
    print("[*] Renderizando gráfica 1D...")
    cpl.plot(ca_evolucionado, colormap=cmap_personal)