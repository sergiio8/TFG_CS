"""
run_experiments.py
===================
Pipeline de validacion estadistica + ablation study para el motor CBR + AG.

Ejecuta cada uno de los casos de bandera/patron bajo distintas condiciones
(Random, Heuristica, CBR completo, y variantes de ablacion) con N semillas
cada una, y guarda fitness final, generaciones, tiempo de ejecucion Y LA MEJOR
REGLA ENCONTRADA en un CSV.

IMPORTANTE antes de correrlo:
- Borra o renombra 'casos_aprendidos_backup.pkl' si existe, para partir de una
  base de casos limpia y fija durante todo el experimento (el pipeline ya lo
  hace automaticamente al arrancar).
- Este pipeline NUNCA llama a cbr_engine.retain(), asi que la base de casos
  no crece entre ejecuciones -> todas las semillas parten de la misma memoria.
- Usa matplotlib en modo 'Agg' para evitar que se abran ventanas graficas
  durante las ejecuciones.
- La condicion "Random" usa generate_random_config() de main_random.py, que
  sortea TODO (pop_size, generaciones, pesos de fitness, vecindario,
  fronteras, mutacion adaptativa, tramos de mutacion...). Es un baseline
  "ciego" real, no una aproximacion simplificada -> espera mucha varianza
  entre semillas, es lo esperado y lo que se quiere mostrar.

NUEVO EN ESTA VERSION:
- 4 casos de test nuevos (ver build_test_cases), pensados para cubrir huecos
  del corpus original: simetria adversarial (Chequia, Austria), generalizacion
  a bordes diagonales (RDCongo), y un proxy "biologico" sintetico
  (CebraSintetica) antes de dar el salto a imagenes reales.
  El caso de colonia bacteriana (crecimiento desde semilla + validacion por
  trayectoria) se trabaja en un script aparte, no en este batch.
- Las funciones run_random/run_heuristica/run_cbr ya NO descartan el mejor
  individuo devuelto por GeneticAlgorithm: se guarda en la columna
  'mejor_regla' del CSV como una cadena de enteros separados por ';',
  exactamente igual que ya se hacia con 'fitness_curve'.
"""

import time
import random
import traceback
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # evita popups de graficas durante el batch

# The project keeps its modules in separate source directories rather than
# Python packages. Add those directories so this script can be run from the
# repository root with `python run_experiments.py`.
ROOT_DIR = Path(__file__).resolve().parent
for source_dir in ("Configuration", "Memory", "CBR", "GeneticAlgorithm", "BaseCases2DIM", "Benchmarks1DIM"):
    sys.path.insert(0, str(ROOT_DIR / source_dir))

from Memoria import InstanciadorCasos
from MotorCBR import CBREngine
from Evolucion import GeneticAlgorithm
from main_heuristica import generate_competent_heuristic_config
from main_random import generate_random_config

# =========================================================
# CONFIGURACION GENERAL
# =========================================================
N_SEEDS = 5  # sube a 8-10 si el tiempo de computo lo permite
OUTPUT_CSV = "resultados_experimentos_nuevos_casos.csv"
MEMORIA_PKL = "casos_aprendidos_backup.pkl"
EXPERIMENTAL_BASE_CASE_IDS = {"BanderaJapon2DIM", "BanderaHungria2DIM"}


# =========================================================
# HELPERS DE GENERACION DE PATRONES SINTETICOS
# =========================================================
def _rayas_diagonales(shape, ancho_franja=3, angulo_factor=1.0):
    """Patron de rayas diagonales tipo Turing (cebra/tigre), como proxy
    sintetico de una textura de morfogenesis real antes de usar una imagen
    fotografica de verdad."""
    H, W = shape
    Y, X = np.indices(shape)
    diag = (X + angulo_factor * Y) // ancho_franja
    return (diag % 2).astype(int)


# =========================================================
# 1. DEFINICION DE LOS CASOS DE TEST
# =========================================================
def build_test_cases():
    cases = {}

    # --- Casos ya resueltos en la tanda anterior (no tocar) ---
    '''t = np.zeros((18, 30), dtype=int)
    t[0:9, :] = 0
    t[9:18, :] = 1
    cases["Polonia"] = dict(
        target=t, colores=["#FFFFFF", "#FF0000"], estados=2,
        h_bound="fixed", v_bound="periodic", random_init=True,
    )

    t = np.zeros((18, 30), dtype=int)
    t[:, 0:10] = 0
    t[:, 10:20] = 1
    t[:, 20:30] = 2
    cases["Italia"] = dict(
        target=t, colores=["#008000", "#FFFFFF", "#FF0000"], estados=3,
        h_bound="periodic", v_bound="fixed", random_init=True,
    )

    t = np.zeros((18, 30), dtype=int)
    t[:, 0:12] = 0
    t[0:9, 12:30] = 1
    t[9:18, 12:30] = 2
    cases["Benin"] = dict(
        target=t, colores=["#008000", "#FFFF00", "#FF0000"], estados=3,
        h_bound="periodic", v_bound="fixed", random_init=True,
    )

    t = np.zeros((18, 18), dtype=int)
    t[7:11, 3:15] = 1
    t[3:15, 7:11] = 1
    cases["Suiza"] = dict(
        target=t, colores=["#FF0000", "#FFFFFF"], estados=2,
        h_bound="fixed", v_bound="fixed", random_init=False,
    )

    Y, X = np.ogrid[:18, :30]
    t = np.zeros((18, 30), dtype=int)
    t[((X - 15) ** 2 + (Y - 9) ** 2) <= 36] = 1
    cases["Bangladesh"] = dict(
        target=t, colores=["#008000", "#FF0000"], estados=2,
        h_bound="fixed", v_bound="fixed", random_init=False,
    )

    t = np.zeros((18, 30), dtype=int)
    t[0:9, :] = 0
    t[9:14, :] = 1
    t[14:18, :] = 2
    cases["Colombia"] = dict(
        target=t, colores=["#FFFF00", "#0000FF", "#FF0000"], estados=3,
        h_bound="fixed", v_bound="periodic", random_init=True,
    )'''

    # --- Casos NUEVOS ---

    # 1) Chequia: triangulo azul desde el hoist sobre blanco/rojo horizontal.
    #    Caso ADVERSARIAL A PROPOSITO: el propio paper cita formas simetricas
    #    (Austria/Chequia) como limite conocido del framework -> se espera
    #    fitness bajo, y ESO es justo lo que hay que reportar y analizar.
    H, W = 18, 30
    t = np.zeros((H, W), dtype=int)  # blanco arriba
    t[H // 2:, :] = 1  # rojo abajo
    apex_col = W // 2
    for row in range(H):
        if row <= H // 2:
            ancho = int(apex_col * (row / (H // 2))) if H // 2 > 0 else 0
        else:
            ancho = int(apex_col * ((H - row) / (H // 2)))
        t[row, 0:ancho] = 2  # triangulo azul
    cases["Chequia"] = dict(
        target=t, colores=["#FFFFFF", "#FF0000", "#11457E"], estados=3,
        h_bound="fixed", v_bound="fixed", random_init=True,
    )

    # 2) Austria: franjas horizontales rojo-blanco-rojo (mismo color arriba y
    #    abajo -> simetria especular vertical). Segundo caso ADVERSARIAL
    #    citado explicitamente en el paper como limitacion conocida.
    t = np.zeros((18, 30), dtype=int)
    t[0:6, :] = 0   # rojo
    t[6:12, :] = 1  # blanco
    t[12:18, :] = 0  # rojo (misma etiqueta que la franja superior)
    cases["Austria"] = dict(
        target=t, colores=["#ED2939", "#FFFFFF"], estados=2,
        h_bound="fixed", v_bound="periodic", random_init=True,
    )

    # 3) Republica Democratica del Congo (simplificado): banda diagonal roja
    #    con borde amarillo sobre fondo azul. Ninguno de los casos anteriores
    #    tiene un borde diagonal -> prueba de generalizacion real de la
    #    metrica de similitud (ec. 7 del paper).
    '''H, W = 18, 30
    Y, X = np.indices((H, W))
    diag = Y / (H - 1) + X / (W - 1)  # 0 (arriba-izq) .. 2 (abajo-der)
    ancho_banda, ancho_borde = 0.30, 0.08
    t = np.zeros((H, W), dtype=int)  # azul de fondo
    t[(diag > 1 - ancho_banda) & (diag < 1 + ancho_banda)] = 2  # rojo
    borde_mask = (
        ((diag > 1 - ancho_banda - ancho_borde) & (diag <= 1 - ancho_banda))
        | ((diag >= 1 + ancho_banda) & (diag < 1 + ancho_banda + ancho_borde))
    )
    t[borde_mask] = 1  # amarillo
    cases["RDCongo"] = dict(
        target=t, colores=["#007FFF", "#FFD100", "#CE1021"], estados=3,
        h_bound="fixed", v_bound="fixed", random_init=True,
    )

    # 4) Rayas diagonales sinteticas tipo Turing (proxy de textura biologica
    #    real -- cebra/tigre -- antes de dar el salto a una imagen real).
    t = _rayas_diagonales((18, 30), ancho_franja=3, angulo_factor=1.0)
    cases["CebraSintetica"] = dict(
        target=t, colores=["#F5DEB3", "#3B2A1A"], estados=2,
        h_bound="periodic", v_bound="periodic", random_init=True,
    )'''

    # Nota: el caso de colonia bacteriana (blob irregular creciendo desde
    # semilla) se trabaja aparte, en su propio script, porque requiere
    # ademas la validacion por trayectoria de crecimiento (area/generacion
    # vs. curva real) y no encaja bien en este batch de comparacion de
    # patrones estaticos. Ver run_experiment_colonia.py.

    return cases


# =========================================================
# 2. CONDICIONES EXPERIMENTALES
# =========================================================
def run_random(caso_data, seed):
    """Baseline 'Random' real del paper: configuracion totalmente aleatoria
    (poblacion, generaciones, pesos de fitness, vecindario, fronteras,
    mutacion adaptativa o no, todo sorteado) via generate_random_config."""
    np.random.seed(seed)
    random.seed(seed)

    config = generate_random_config(
        ca_size=caso_data["target"].shape,
        num_states=caso_data["estados"],
        target_state=caso_data["target"],
        colors=caso_data["colores"],
    )

    t0 = time.perf_counter()
    top3, fitness_evo = GeneticAlgorithm(config=config, seed_rule=None)
    t1 = time.perf_counter()
    mejor_regla = top3[0]  # GeneticAlgorithm devuelve el top-3; nos quedamos con el mejor
    return fitness_evo, t1 - t0, mejor_regla


def run_heuristica(caso_data, seed):
    """Baseline 'Expert Heuristic' del paper: misma config fisica, mutacion
    adaptativa y pesos afinados, pero SIN transfer learning ni CBR."""
    np.random.seed(seed)
    random.seed(seed)

    config = generate_competent_heuristic_config(
        ca_size=caso_data["target"].shape,
        num_states=caso_data["estados"],
        target_state=caso_data["target"],
        colors=caso_data["colores"],
        h_bound=caso_data["h_bound"],
        v_bound=caso_data["v_bound"],
        use_random_init=caso_data["random_init"],
    )

    t0 = time.perf_counter()
    top3, fitness_evo = GeneticAlgorithm(config=config, seed_rule=None)
    t1 = time.perf_counter()
    mejor_regla = top3[0]
    return fitness_evo, t1 - t0, mejor_regla


def run_cbr(caso_data, seed, cbr_engine, ablation_flags=None):
    """
    Motor CBR completo (o con componentes desactivados via ablation_flags).

    ablation_flags admite:
        "use_transfer_learning": False -> desactiva la inyeccion memetica

    NOTA: la invarianza rotacional ya no existe en MotorCBR.py (se descarto
    porque empeoraba los resultados), asi que retrieve()/reuse() solo
    manejan (casos, features, target, colores) -- sin flag de rotacion.
    """
    ablation_flags = ablation_flags or {}
    np.random.seed(seed)
    random.seed(seed)

    features = cbr_engine.extract_features(caso_data["target"])
    top_casos = cbr_engine.retrieve(caso_data["target"], features)  # 1 solo valor de retorno

    nueva_conf, regla_semilla = cbr_engine.reuse(
        top_casos, features, caso_data["target"], caso_data["colores"]
    )  # 4 argumentos, sin exito_rotacion

    if not ablation_flags.get("use_transfer_learning", True):
        regla_semilla = None

    # Llamamos directamente a GeneticAlgorithm (en vez de cbr_engine.revise)
    # para obtener la curva completa de fitness sin los prints/plots internos
    # de revise(), que no son adecuados para un batch de muchas ejecuciones.
    t0 = time.perf_counter()
    top3, fitness_evo = GeneticAlgorithm(config=nueva_conf, seed_rule=regla_semilla)
    t1 = time.perf_counter()
    mejor_regla = top3[0]
    return fitness_evo, t1 - t0, mejor_regla


def _regla_a_string(mejor_regla):
    """Convierte la mejor regla (lista/array/Individual de DEAP) en una
    cadena de enteros separados por ';', igual que ya se hace con
    fitness_curve. Devuelve None si no hay regla disponible."""
    if mejor_regla is None:
        return None
    try:
        return ";".join(str(int(g)) for g in mejor_regla)
    except TypeError:
        # por si mejor_regla no es iterable (defensivo)
        return str(mejor_regla)


# =========================================================
# 3. BUCLE PRINCIPAL
# =========================================================
def main():
    # Base de casos limpia y FIJA durante todo el experimento
    pkl_path = Path(MEMORIA_PKL)
    if pkl_path.exists():
        pkl_path.unlink()
        print(f"[*] Eliminado '{MEMORIA_PKL}' previo para partir de memoria limpia.")
        # Nota: esto es seguro aunque estemos retomando, porque el pipeline
        # nunca llama a retain(), así que este pkl no se usa durante el batch.

    instanciador = InstanciadorCasos()
    instanciador.casos_base = [
        caso for caso in instanciador.casos_base
        if caso.id_caso in EXPERIMENTAL_BASE_CASE_IDS
    ]
    print(
        "[*] Memoria experimental fija: "
        f"{', '.join(sorted(EXPERIMENTAL_BASE_CASE_IDS))}"
    )
    cbr_engine = CBREngine(instanciador)

    test_cases = build_test_cases()

    condiciones = {
        #"Random":            lambda c, s: run_random(c, s),
        #"Heuristica":        lambda c, s: run_heuristica(c, s),
        "CBR_sin_TL":        lambda c, s: run_cbr(c, s, cbr_engine, {"use_transfer_learning": False}),
        "CBR_Full":          lambda c, s: run_cbr(c, s, cbr_engine),
        
    }

    # --- cargar resultados previos si existen ---
    output_path = Path(OUTPUT_CSV)
    if output_path.exists():
        df_prev = pd.read_csv(output_path)
        filas = df_prev.to_dict("records")
        completados = set(
            zip(df_prev["caso"], df_prev["condicion"], df_prev["seed"])
        )
        print(f"[*] CSV previo encontrado con {len(filas)} filas. "
              f"Se saltearán las combinaciones ya completadas.")
    else:
        filas = []
        completados = set()

    total = len(test_cases) * len(condiciones) * N_SEEDS
    contador = 0

    for nombre_caso, data in test_cases.items():
        for nombre_cond, funcion in condiciones.items():
            for seed in range(N_SEEDS):
                contador += 1

                if (nombre_caso, nombre_cond, seed) in completados:
                    print(f">>> [{contador}/{total}] {nombre_caso} | {nombre_cond} | "
                          f"seed={seed} -> YA HECHO, se saltea")
                    continue

                print(f"\n>>> [{contador}/{total}] {nombre_caso} | {nombre_cond} | seed={seed}")

                try:
                    fitness_evo, tiempo, mejor_regla = funcion(data, seed)
                    filas.append({
                        "caso": nombre_caso,
                        "condicion": nombre_cond,
                        "seed": seed,
                        "fitness_final": fitness_evo[-1],
                        "generaciones": len(fitness_evo),
                        "tiempo_seg": round(tiempo, 2),
                        "fitness_curve": ";".join(f"{v:.4f}" for v in fitness_evo),
                        "mejor_regla": _regla_a_string(mejor_regla),
                    })
                    print(f"    Fitness final: {fitness_evo[-1]:.4f} | "
                          f"Generaciones: {len(fitness_evo)} | Tiempo: {tiempo:.1f}s")

                except Exception as e:
                    print(f"[ERROR] {nombre_caso}/{nombre_cond}/seed{seed}: {e}")
                    traceback.print_exc()
                    filas.append({
                        "caso": nombre_caso, "condicion": nombre_cond, "seed": seed,
                        "fitness_final": None, "generaciones": None,
                        "tiempo_seg": None, "fitness_curve": None,
                        "mejor_regla": None,
                    })

                # Guardado incremental por si la ejecucion se corta a mitad
                pd.DataFrame(filas).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(filas)
    print("\n" + "=" * 60)
    print(" RESUMEN FINAL (media +/- std de fitness por caso/condicion)")
    print("=" * 60)
    resumen = df.groupby(["caso", "condicion"])["fitness_final"].agg(["mean", "std", "count"])
    print(resumen)
    resumen.to_csv("resumen_estadistico.csv")


if __name__ == "__main__":
    main()