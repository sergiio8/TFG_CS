# Módulo de Evolución (Evolutionary Framework)

Este directorio aloja el núcleo de optimización encargado de buscar y refinar las reglas de transición locales de los Autómatas Celulares (AC) mediante Computación Evolutiva. 

Conecta de forma directa la simulación física de **CellPyLib** con los algoritmos de búsqueda de **DEAP**.

---

## Estrategias Soportadas

El módulo está diseñado de forma híbrida y conmuta entre dos enfoques según la configuración del problema:

*   **Algoritmo Genético Clásico (AG):** Implementa el ciclo estándar con selección por torneo, cruce de dos puntos y mutación de inversión de bits. Ideal para problemas homogéneos como *Density Classification*.
*   **Estrategia Evolutiva $(\mu + \lambda)$-ES:** Opera exclusivamente mediante operadores de mutación y una intensa presión selectiva basada en elitismo. Incluye tasas de mutación dinámicas (exploración vs. explotación) críticas para resolver problemas de morfogénesis compleja (*French Flag*, Japón, Hungría).

---

## Funcionalidades Especiales

*   **Fitness Adaptativo:** Ejecuta un filtrado rápido evaluando las reglas en una subpoblación pequeña de autómatas; si superan el umbral inicial, realiza la evaluación completa en el entorno real, ahorrando hasta un 80% de coste computacional.
*   **Métricas Multicriterio 2D:** Evalúa la calidad de la imagen generada combinando simultáneamente el Índice de Similitud Estructural (SSIM), el Índice de Jaccard por estados y la Información Mutua.
*   **Inyección de Conocimiento (Transfer Learning):** Capaz de recibir reglas externas (recuperadas por el motor CBR), clonarlas y mutarlas ligeramente para inicializar la población con una ventaja competitiva.
