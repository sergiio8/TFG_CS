# Módulo de Configuración y Estructura de Datos (CBR Caso)

Este directorio define las estructuras de datos fuertemente tipadas y los esquemas de configuración globales del sistema. Utiliza `dataclasses` de Python para garantizar un manejo de datos consistente, escalable y fácilmente serializable entre el motor CBR, el framework evolutivo y el módulo de memoria.

---

## Estructura de un Caso CBR

Cada instancia se modela formalmente siguiendo la tupla matemática del problema, desacoplada en los siguientes objetos:

*   **`CasoCBR`:** El contenedor raíz. Vincula un identificador único (`id`) con las características del problema y su solución asociada.
*   **`CaracteristicasProblema`:** Descriptor topológico y estructural del estado objetivo (matriz de la imagen, dimensiones espaciales, número de estados, presencia de franjas, componentes conexas y uniformidad de bordes).
*   **`SolucionCBR`:** Almacena la configuración del framework utilizada, la mejor regla local encontrada ($\varphi^*$) y el valor de *fitness* final alcanzado ($f^*$).
*   **`FrameworkConfig`:** Clase aglutinadora que unifica los dos entornos de ejecución:
    *   **`CAConfig`:** Parámetros del Autómata Celular en *CellPyLib* (tamaño, vecindario de Moore/von Neumann, tipo de frontera y pasos de tiempo).
    *   **`GAConfig`:** Hiperparámetros del Algoritmo Evolutivo en *DEAP* (tamaño de población, generaciones, métodos de selección, operadores genéticos y pesos de las métricas de similitud).

---

## Características de Diseño

*   **Validación Automatizada (`__post_init__`):** Calcula dinámicamente propiedades del sistema en tiempo de ejecución, como el tamaño del cromosoma (`ind_size`) en función del vecindario escogido ($n^5$ para von Neumann y $n^9$ para Moore).
*   **Modularidad Completa:** Actúa como el contrato de datos único del repositorio, permitiendo que cualquier cambio en la parametrización se propague automáticamente por todo el pipeline del TFG.
