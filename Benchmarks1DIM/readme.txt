# Casos de Estudio 1-Dimensionales (1D Base Cases)

Este directorio contiene la modelización, configuración y scripts de experimentación para los problemas implementados en una sola dimensión. Estos escenarios sirvieron para validar la primera versión del framework evolutivo y actúan como casos base estáticos en la memoria del sistema CBR.

---

## Problemas Implementados

### 1. Density Classification Problem (DCP)
Benchmark clásico cuyo objetivo es encontrar una regla local determinista que clasifique la densidad inicial del autómata. Si el número de unos supera el 50%, todo el sistema debe converger a un estado uniforme de unos; en caso contrario, a todo ceros.
*   **Configuración:** Espacio binario, vecindario de 7 células y fronteras periódicas (circulares).

### 2. French Flag Problem (1D)
Problema que conceptualiza la morfogénesis biológica de bajo nivel. El autómata debe autoorganizarse a partir de un estado inicial completamente aleatorio hasta estabilizar un patrón axial dividido en tres franjas perfectas de igual tamaño (azul, blanco y rojo).
*   **Configuración:** Espacio ternario (3 estados), vecindario de tamaño 3 y fronteras fijas (constantes) para guiar la simetría.

### 3. Embriogénesis de la Drosophila
Simulación bioinspirada basada en el desarrollo temprano de la mosca de la fruta. A partir de una fase de inestabilidad y ruido inicial (expresión génica), las células deben comunicarse localmente hasta estabilizar un patrón periódico de siete franjas transversales.
*   **Configuración:** Espacio binario y vecindario ampliado a 3 vecinos a cada lado (7 células en total).

---

## Utilidad en el Sistema

Cada uno de estos archivos exporta su respectiva `CAConfig` y `GAConfig` optimizada. El motor CBR los utiliza como conocimiento de arranque para transferir hiperparámetros de búsqueda y reglas de empaquetado lineal a nuevos problemas con dinámicas de crecimiento o clasificación similares.
