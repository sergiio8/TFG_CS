# Módulo del Motor CBR (CBR Core Engine)

Este directorio aloja el núcleo inteligente del sistema. Su función es orquestar y ejecutar el ciclo de vida de Razonamiento Basado en Casos (Retrieve, Reuse, Revise y Retain) para automatizar la configuración y acelerar la optimización de las reglas de los autómatas celulares.

---

## Funcionalidades de las Fases

### 1. Retrieve (Recuperar)
Dado un nuevo problema (matriz objetivo), extrae sus propiedades y calcula su similitud frente a la base de conocimiento usando una función experta:
$$Sim = 0.2 \cdot Sim_{topo} + 0.5 \cdot Sim_{struct} + 0.1 \cdot Sim_{states} + 0.2 \cdot Sim_{dims}$$
*   **Invariancia Espacial:** Compara el objetivo original y también **rotado 90 grados**. Si empareja con la versión rotada, recupera el caso aplicando un factor de penalización de $0.8$.

### 2. Reuse (Reutilizar)
Adapta los parámetros de los casos recuperados que superan el umbral ($0.5$). Los cualitativos se heredan del mejor caso y los cuantitativos se calculan mediante medias ponderadas.
*   **Transfer Learning:** Si las dimensiones y estados coinciden (o tras revertir geométricamente la rotación de 90° en los vecindarios de Moore/von Neumann), **inyecta la regla ganadora previa** en la población inicial del algoritmo evolutivo (5% de clones y 5% sutilmente mutados).

### 3. Revise (Revisar)
Dispara el framework de evolución utilizando la población inicial optimizada en la fase previa, refinando y adaptando la regla local para el nuevo problema específico de forma mucho más rápida que una ejecución desde cero.

### 4. Retain (Retener)
Evalúa el éxito de la nueva regla mediante un umbral de fitness adaptativo (que se vuelve más exigente conforme crece la base de datos). Si es apta, solicita al módulo de memoria su almacenamiento permanente como un nuevo caso aprendido.
