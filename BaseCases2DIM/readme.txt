# Casos de Estudio 2-Dimensionales (2D Base Cases)

Este directorio contiene la modelización, configuración y scripts de experimentación para los problemas implementados en dos dimensiones. Estos escenarios evalúan dinámicas de interacción celular más complejas y sirven como base de conocimiento avanzada para el motor CBR en problemas de morfogénesis plana.

---

## Problemas Implementados

### 1. Bandera de Hungría (Formación de Patrones Horizontales)
Problema de morfogénesis artificial que exige al autómata segmentar el lienzo bidimensional en tres franjas horizontales perfectas (rojo, blanco y verde). 
*   **Configuración:** Espacio ternario (3 estados), vecindario de von Neumann y un esquema híbrido de fronteras (fijas en horizontal para separar el rojo/verde y periódicas en vertical para mantener la continuidad).

### 2. Bandera de Japón (Morfogénesis Geométrica desde Semilla)
Desafío que introduce una tipología de problema radicalmente nueva: generar una forma geométrica precisa (un círculo centrado) a partir de una única célula roja inicial (semilla central) en un entorno blanco.
*   **Configuración:** Espacio binario, vecindario de Moore (indispensable para procesar diagonales y suavizar la curvatura) y fronteras periódicas. Evalúa el comportamiento emergente temporal del autómata frente a variaciones en los pasos de tiempo (*timesteps*).

---

## Innovaciones Técnicas del Entorno 2D

*   **Métricas de Fitness Avanzadas:** La evaluación del rendimiento en estos casos requirió abandonar el *accuracy* lineal en favor de la combinación ponderada del Índice de Similitud Estructural (SSIM) multicanal, el Índice de Jaccard pesado por importancia de estado e Información Mutua.
*   **Mutación Adaptativa Dinámica:** Implementa un régimen estricto de exploración vs. explotación que reduce de forma escalonada el número de genes mutados por cromosoma conforme el *fitness* supera los umbrales críticos de convergencia.
