import numpy as np
from Memoria import InstanciadorCasos
from MotorCBR import CBREngine
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    print("=========================================================")
    print(" MOTOR CBR + AG: EVOLUCIÓN DE AUTÓMATAS CELULARES")
    print("=========================================================")
    
    instanciador = InstanciadorCasos()
    cbr = CBREngine(instanciador)

    # ==========================================
    # 1. POLONIA (Blanco y Rojo, horizontal)
    # ==========================================
    target_polonia = np.zeros((18, 30), dtype=int)
    target_polonia[0:9, :] = 0   # Blanco
    target_polonia[9:18, :] = 1  # Rojo
    colors_polonia = ["#FFFFFF", "#FF0000"] # 0=Blanco, 1=Rojo

    # ==========================================
    # 2. AUSTRIA (Rojo, Blanco, Rojo, horizontal) 
    # ==========================================
    target_austria = np.zeros((18, 30), dtype=int)
    target_austria[0:6, :] = 0   # Rojo
    target_austria[6:12, :] = 1  # Blanco
    target_austria[12:18, :] = 0 # Rojo
    colors_austria = ["#FF0000", "#FFFFFF"] # 0=Rojo, 1=Blanco

    # ==========================================
    # 3. PERÚ (Rojo, Blanco, Rojo, vertical)
    # ==========================================
    target_peru = np.zeros((18, 30), dtype=int)
    target_peru[:, 0:10] = 0   # Rojo
    target_peru[:, 10:20] = 1  # Blanco
    target_peru[:, 20:30] = 0  # Rojo
    colors_peru = ["#FF0000", "#FFFFFF"] # 0=Rojo, 1=Blanco

    # ==========================================
    # 4. BENÍN (Franja vertical verde, horizontales amarilla y roja)
    # ==========================================
    target_benin = np.zeros((18, 30), dtype=int)
    target_benin[:, 0:12] = 0          # Verde (columna izquierda)
    target_benin[0:9, 12:30] = 1       # Amarillo (mitad superior derecha)
    target_benin[9:18, 12:30] = 2      # Rojo (mitad inferior derecha)
    colors_benin = ["#008000", "#FFFF00", "#FF0000"] # 0=Verde, 1=Amarillo, 2=Rojo


    # ==========================================
    # 6. EMIRATOS ÁRABES UNIDOS
    # ==========================================
    target_eau = np.zeros((18, 30), dtype=int)
    target_eau[:, 0:8] = 0          # Rojo (franja vertical izquierda)
    target_eau[0:6, 8:30] = 1       # Verde (franja superior)
    target_eau[6:12, 8:30] = 2      # Blanco (franja central)
    target_eau[12:18, 8:30] = 3     # Negro (franja inferior)
    colors_eau = ["#FF0000", "#008000", "#FFFFFF", "#000000"] # 0=Rojo, 1=Verde, 2=Blanco, 3=Negro

    # ==========================================
    # BANDERAS CON CÍRCULOS (Adaptado a 18x30)
    # ==========================================
    Y_AUX, X_AUX = np.ogrid[:30, :45]

    # 7. BANGLADESH (Fondo verde, círculo rojo desplazado a la izquierda)
    target_bangladesh = np.zeros((30, 45), dtype=int) # 0 = Verde
    centro_y_bd, centro_x_bd = 15, 20 
    radio_bd = 9 
    mask_bd = ((X_AUX - centro_x_bd)**2 + (Y_AUX - centro_y_bd)**2) <= radio_bd**2
    target_bangladesh[mask_bd] = 1 # 1 = Rojo
    colors_bangladesh = ["#008000", "#FF0000"] # 0=Verde, 1=Rojo

    Y, X = np.ogrid[:18, :30]

    # INGLATERRA (Fondo blanco, cruz roja centrada)
    target_inglaterra = np.zeros((18, 30), dtype=int) # 0 = Blanco
    target_inglaterra[7:11, :] = 1      # Franja horizontal roja (centro vertical)
    target_inglaterra[:, 13:17] = 1     # Franja vertical roja (centro horizontal)
    colors_inglaterra = ["#FFFFFF", "#FF0000"] # 0=Blanco, 1=Rojo

    # ==========================================
    # BANDERAS NÓRDICAS (Adaptadas a 18x30)
    # ==========================================

    # ==========================================
    # 12. SUIZA (Proporción cuadrada -> 18x18 para ser rápido)
    # ==========================================
    target_suiza = np.zeros((18, 18), dtype=int) # 0 = Rojo
    target_suiza[7:11, 3:15] = 1  # Brazo horizontal
    target_suiza[3:15, 7:11] = 1  # Brazo vertical
    colors_suiza = ["#FF0000", "#FFFFFF"] # 0=Rojo, 1=Blanco
        
    # ==========================================
    # 13. ITALIA (Vertical, 18x30)
    # ==========================================
    target_italia = np.zeros((18, 30), dtype=int)
    target_italia[:, 0:10] = 0   # Verde
    target_italia[:, 10:20] = 1  # Blanco
    target_italia[:, 20:30] = 2  # Rojo
    colors_italia = ["#008000", "#FFFFFF", "#FF0000"] # 0=Verde, 1=Blanco, 2=Rojo

    # ==========================================
    # 14. COLOMBIA (Amarillo 50%, Azul 25%, Rojo 25%, horizontal)
    # ==========================================
    target_colombia = np.zeros((18, 30), dtype=int)
    target_colombia[0:9, :] = 0    # Amarillo (mitad superior, 9 filas)
    target_colombia[9:14, :] = 1   # Azul (siguientes 5 filas)
    target_colombia[14:18, :] = 2  # Rojo (últimas 4 filas)
    colors_colombia = ["#FFFF00", "#0000FF", "#FF0000"] # 0=Amarillo, 1=Azul, 2=Rojo

    # ==========================================
    # 15. REPÚBLICA CHECA (Blanco, Rojo y Triángulo Azul)
    # ==========================================
    target_checa = np.zeros((18, 30), dtype=int)
    target_checa[0:9, :] = 0   # Blanco (mitad superior)
    target_checa[9:18, :] = 1  # Rojo (mitad inferior)
    for y in range(18):
        for x in range(30):
            if y < 9: 
                if x <= y * (15.0 / 8.0):
                    target_checa[y, x] = 2 # Azul
            else:     
                if x <= (17 - y) * (15.0 / 8.0):
                    target_checa[y, x] = 2 # Azul
    colors_checa = ["#FFFFFF", "#FF0000", "#0000FF"] # 0=Blanco, 1=Rojo, 2=Azul

    # ==========================================
    # 16. PANAMÁ (Cuadrantes alternos)
    # ==========================================
    target_panama = np.zeros((18, 30), dtype=int)
    target_panama[0:9, 0:15] = 0   # Blanco
    target_panama[0:9, 15:30] = 1  # Rojo
    target_panama[9:18, 0:15] = 2  # Azul
    target_panama[9:18, 15:30] = 0 # Blanco
    colors_panama = ["#FFFFFF", "#FF0000", "#0000FF"]


    # ==========================================
    # PATRONES ORGÁNICOS (Animales)
    # ==========================================
    filas, columnas = 20, 30
    np.random.seed(42)
    ruido = np.random.rand(filas, columnas)

    # 1. PIEL DE VACA
    ruido_suave_vaca = gaussian_filter(ruido, sigma=5.0)
    target_vaca = np.where(ruido_suave_vaca > 0.65, 1, 0).astype(int)
    colors_vaca = ["#FFFFFF", "#000000"]

    # 2. PATRÓN DE CEBRA
    x_c, y_c = np.meshgrid(np.linspace(0, 10*np.pi, filas), np.linspace(0, 10*np.pi, columnas), indexing='ij')
    np.random.seed(101)
    ruido_distorsion = gaussian_filter(np.random.rand(filas, columnas), sigma=2.0) * 5.0
    ondas = np.sin(x_c + y_c + ruido_distorsion)
    target_cebra = np.where(ondas > 0.0, 1, 0).astype(int)
    colors_cebra = ["#FFFFFF", "#000000"]

    # 3. PIEL DE GUEPARDO
    np.random.seed(123)
    ruido_guepardo = np.random.rand(filas, columnas)
    ruido_suave_guepardo = gaussian_filter(ruido_guepardo, sigma=1.0)
    target_guepardo = np.where(ruido_suave_guepardo > 0.5, 1, 0).astype(int)
    colors_guepardo = ["#E6AA54", "#000000"]


    # ==========================================
    # FORMAS GEOMÉTRICAS PURAS (30x30)
    # ==========================================
    cy, cx = 15, 15 # Nuevo centro exacto del lienzo de 30x30

    # A. CUADRADO (Reto para vecindario de Moore)
    target_cuadrado = np.zeros((30, 30), dtype=int)
    half = 8 # Aumentado (antes 5) para mantener la proporción visual
    target_cuadrado[cy-half:cy+half, cx-half:cx+half] = 1
    colors_cuadrado = ["#000000", "#FFFFFF"] # Fondo negro, figura blanca

    # B. ROMBO / DIAMANTE (Reto para vecindario de von Neumann)
    target_rombo = np.zeros((30, 30), dtype=int)
    radius = 10 # Aumentado (antes 7)
    for y in range(30): # Iteramos sobre toda la altura (30)
        for x in range(30): # Iteramos sobre toda la anchura (30)
            if abs(y - cy) + abs(x - cx) <= radius:
                target_rombo[y, x] = 1
    colors_rombo = ["#000000", "#FFFFFF"]

    # C. TRIÁNGULO (Reto de Ruptura de Simetría / Direccionalidad)
    target_triangulo = np.zeros((30, 30), dtype=int)
    height = 16 # Aumentado (antes 12)
    for y in range(30): # Iteramos sobre toda la altura (30)
        for x in range(30):
            # Dibujamos una cuña apuntando hacia arriba
            if (y > cy - height//2) and (y <= cy + height//2):
                relative_y = y - (cy - height//2)
                if abs(x - cx) <= relative_y: 
                    target_triangulo[y, x] = 1
    colors_triangulo = ["#000000", "#FFFFFF"]

    # ==========================================
    # PATRONES DE RAYAS PERIÓDICAS (30x30)
    # ==========================================
    dim_rayas = 30
    grosor = 5

    # 1. RAYAS VERTICALES (Alternando cada 5 columnas)
    target_rayas_v = np.zeros((dim_rayas, dim_rayas), dtype=int)
    for x in range(dim_rayas):
        if (x // grosor) % 2 == 1:
            target_rayas_v[:, x] = 1
    colors_rayas_v = ["#000000", "#FFFFFF"] # Negro y Blanco

    # 2. RAYAS HORIZONTALES (Alternando cada 5 filas)
    target_rayas_h = np.zeros((dim_rayas, dim_rayas), dtype=int)
    for y in range(dim_rayas):
        if (y // grosor) % 2 == 1:
            target_rayas_h[y, :] = 1
    colors_rayas_h = ["#000000", "#FFFFFF"]

    # 3. RAYAS DIAGONALES (Inclinación de 45 grados)
    target_rayas_d = np.zeros((dim_rayas, dim_rayas), dtype=int)
    for y in range(dim_rayas):
        for x in range(dim_rayas):
            # Sumamos x + y para obtener bandas diagonales constantes
            if ((x + y) // grosor) % 2 == 1:
                target_rayas_d[y, x] = 1
    colors_rayas_d = ["#000000", "#FFFFFF"]

    # ==========================================
    # 3. SELECCIÓN DE OBJETIVO Y EJECUCIÓN CBR
    # ==========================================
    
    # Cambia estas variables para evaluar la figura que desees:
    objetivo_actual = target_rombo
    colores_actuales = colors_rombo
    nombre_ejecucion = "Rombos_2D"

    print(f"\n[*] Evaluando: {nombre_ejecucion}...")
    
    features = cbr.extract_features(objetivo_actual)
    top_casos, exito_rotacion = cbr.retrieve(objetivo_actual, features)
    
    if len(top_casos) > 0:
        # Adaptamos hiperparámetros y extraemos semilla
        nueva_conf, regla_semilla = cbr.reuse(top_casos, features, objetivo_actual, colores_actuales, exito_rotacion)
        
        # Ejecutamos DEAP
        final_fitness, final_rule = cbr.revise(nueva_conf, regla_semilla)
        
        # Si el AG logró computación emergente exitosa, el CBR aprende la regla
        cbr.retain(nombre_ejecucion, features, final_rule, nueva_conf, final_fitness)
    else:
        print("[!] No hay casos base compatibles para guiar la evolución.")