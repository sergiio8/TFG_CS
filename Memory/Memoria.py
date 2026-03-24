# %%
from typing import List
import pickle
import os

# Importamos las funciones que generan nuestros casos base
from HUN import obtener_caso_hungria
from JPN import obtener_caso_japon
from FRA import obtener_caso_francia

# Importamos el modelo para el tipado de datos
from configuration import CasoCBR

class InstanciadorCasos:
    """
    Gestor de la memoria del sistema CBR.
    Se encarga de cargar los casos base (estáticos) y cargar/guardar 
    los nuevos casos aprendidos (dinámicos) en el disco duro.
    """
    def __init__(self, ruta_memoria="casos_aprendidos.pkl"):
        self.ruta_memoria = ruta_memoria
        self.casos_base: List[CasoCBR] = []
        self.casos_aprendidos: List[CasoCBR] = []
        
        # 1. Cargamos las semillas estáticas
        self._cargar_casos_base()
        
        # 2. Cargamos lo que el sistema haya aprendido en ejecuciones anteriores
        self._cargar_memoria_disco()

    def _cargar_casos_base(self):
        """Carga los casos estáticos estudiados previamente en el TFG."""
        print("[*] Inicializando la Base de Casos del motor CBR...")
        
        self.casos_base.append(obtener_caso_hungria())
        self.casos_base.append(obtener_caso_japon())
        self.casos_base.append(obtener_caso_francia())
        
        print(f"[*] Casos base (semillas) cargados: {len(self.casos_base)}")

    def _cargar_memoria_disco(self):
        """Lee el archivo .pkl del disco y reconstruye los casos aprendidos."""
        if os.path.exists(self.ruta_memoria):
            try:
                with open(self.ruta_memoria, 'rb') as archivo:
                    self.casos_aprendidos = pickle.load(archivo)
                    print(f"[*] Memoria a largo plazo cargada: {len(self.casos_aprendidos)} casos aprendidos previamente.")
            except Exception as e:
                print(f"[!] Error al cargar la memoria CBR desde disco: {e}")
        else:
            print("[*] No hay memoria de casos aprendidos previa. Iniciando desde cero.")
            
        print(f"[*] TOTAL CASOS DISPONIBLES: {len(self.obtener_todos_los_casos())}\n")

    def obtener_todos_los_casos(self) -> List[CasoCBR]:
        """Devuelve la lista completa (Base + Aprendidos) para la fase RETRIEVE."""
        return self.casos_base + self.casos_aprendidos

    def anadir_nuevo_caso(self, nuevo_caso: CasoCBR):
        """Fase RETAIN: Añade un nuevo caso resuelto y lo guarda en disco."""
        self.casos_aprendidos.append(nuevo_caso)
        print(f"[*] Nuevo caso guardado en RAM: '{nuevo_caso.id_caso}'.")
        
        # Guardamos automáticamente en disco
        self._guardar_memoria_disco()

    def _guardar_memoria_disco(self):
        """Sobrescribe el archivo .pkl solo con la lista de casos aprendidos."""
        try:
            with open(self.ruta_memoria, 'wb') as archivo:
                pickle.dump(self.casos_aprendidos, archivo)
            print(f"[*] ÉXITO: Conocimiento guardado permanentemente en '{self.ruta_memoria}'.")
        except Exception as e:
            print(f"[!] Error al intentar guardar en disco: {e}")