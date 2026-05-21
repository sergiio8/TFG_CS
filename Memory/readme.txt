# Módulo de Memoria y Persistencia (CBR Memory)

Este directorio gestiona el almacenamiento y la persistencia de los casos del sistema CBR, divididos en dos categorías:

*   **Casos Base:** Conocimiento inicial estático calibrado por experto (*Density Classification, French Flag, Hungría y Japón*).
*   **Casos Aprendidos:** Nuevas soluciones optimizadas por el algoritmo evolutivo que se almacenan de forma incremental.

---

## Componentes Clave

*   **Clase de Memoria:** Orquestador encargado de cargar los casos base al inicio, recuperar el histórico y guardar las nuevas soluciones en el disco.
*   **Serialización (`pickle`):** Almacenamiento binario que preserva la estructura exacta de los objetos de Python (matrices `numpy` de los estados objetivo y vectores de las reglas), garantizando una lectura y escritura ultrarrápida.

---

## Flujo de Trabajo

1. **Retrieve:** El motor solicita a este módulo la colección de casos para calcular las similitudes.
2. **Retain:** Si una nueva regla supera el umbral de fitness adaptativo, este módulo le asigna un identificador único (`id`) y la serializa de forma permanente.
