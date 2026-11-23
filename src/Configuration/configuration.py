from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Optional, Set
import numpy as np

@dataclass
class CAConfig:
    """Configuración del Autómata Celular."""
    ca_size: Tuple[int, int] = (18, 30)
    num_states: int = 3 
    colors: List[str] = field(default_factory=lambda: ["#000000", "#FFFFFF", "#FF0000"])
    ca_timesteps: int = 75
    ca_neighborhood: str = 'von Neumann'
    random_initial_state: bool = True
    
    ca_initial_state: Optional[np.ndarray] = None
    ca_horizontal_boundary_conditions: str = 'fixed'
    ca_vertical_boundary_conditions: str = 'periodic'
    ca_row0_state: Optional[np.ndarray] = None 
    ca_rowN_state: Optional[np.ndarray] = None 
    ca_column0_state: Optional[np.ndarray] = None 
    ca_columnN_state: Optional[np.ndarray] = None 
    
    num_ca_test: int = 30

    # Ahora target_state se debe pasar al instanciar la clase, ya no está hardcodeado.
    target_state: Optional[np.ndarray] = None

    random_probability: float = 0.02
    
    ind_size: int = field(init=False)
    
    def __post_init__(self):
        # Calcular ind_size dinámicamente según el vecindario y los estados
        if self.ca_neighborhood == 'von Neumann':
            self.ind_size = self.num_states ** 5 
        elif self.ca_neighborhood == 'Moore':
            self.ind_size = self.num_states ** 9  
        else:
            self.ind_size = self.num_states ** (2 * int(self.ca_neighborhood) + 1)


@dataclass
class GAConfig:
    """Configuración del Algoritmo Genético."""
    pop_size: int = 300
    cx_prob: float = 0.9
    mut_prob: float = 1.0
    
    adaptative_mut_prob: bool = False
    mutation_probability_interval: np.ndarray = field(default_factory=lambda: np.array([0, 0.25, 0.45, 0.6, 1]))
    mutation_probabilities: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.1, 0.05, 0.025]))
    
    adaptative_num_mut: bool = True
    mutation_interval: np.ndarray = field(default_factory=lambda: np.array([0, 0.25, 0.45, 0.6, 0.75, 1]))
    num_mutations: np.ndarray = field(default_factory=lambda: np.array([10, 8, 6, 4, 2]))
    
    num_generations: int = 2000
    stop_condition: float = 0.9
    
    adaptative_fitness: bool = True
    ca_fitness: int = 30
    adaptative_fitness_threshold: float = 0.5
    
    weight_SSIM: float = 0.8
    weight_Jaccard: float = 0.2
    weights_states_Jaccard: np.ndarray = field(default_factory=lambda: np.array([1.0, 4.0])),
    weight_accuracy: float = 0.0
    weight_mutual_information: float = 0.0
    
    penalized_fitness: bool = True
    minimum_percentage: float = 0.1
    penalization_factor: float = 0.5
    
    classic_ga: bool = False
    sel_method: str = 'tournament'
    tournament_size: int = 3
    cx_method: str = 'twoPoint'
    elite_passing: float = 0.1
    
    mu_lambda_ga: bool = True
    diversity: float = 0.00


@dataclass
class FrameworkConfig:
    """Agrupa la configuración del Autómata y del Algoritmo Genético."""
    ca: CAConfig
    ga: GAConfig


@dataclass
class CaracteristicasProblema:
    """Características que describen el objetivo a alcanzar (ej. la bandera)."""
    dimensiones: int = 2
    num_estados: int = 3
    borde_uniforme: bool = False
    patrones_detectados: Set[str] = field(default_factory=set)
    franjas_horizontales: bool = False
    franjas_verticales: bool = False
    target_state: [np.ndarray] = None
    componentes_conexas: int = 2
    
    # Momentos de Hu: se pueden inyectar directamente (desde la base de casos) 
    # o calcularlos al vuelo (al meter un problema nuevo).
    momentos_hu: List[float] = field(default_factory=list)


@dataclass
class SolucionCBR:
    """Solución que el sistema encontró para un problema dado."""
    configuracion: FrameworkConfig  
    mejor_regla: List[int] = field(default_factory=list)
    fitness_alcanzado: float = 0.0


@dataclass
class CasoCBR:
    """El caso completo que se guarda en la memoria del sistema."""
    id_caso: str
    problema: CaracteristicasProblema
    solucion: SolucionCBR
