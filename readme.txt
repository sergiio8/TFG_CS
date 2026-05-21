### Motor CBR para la Optimización de Reglas de Autómatas Celulares mediante Estrategias Evolutivas

Este repositorio contiene el framework de desarrollo y experimentación de mi Trabajo de Fin de Grado (TFG). El sistema combina el paradigma de Razonamiento Basado en Casos (CBR) con Algoritmos Evolutivos para automatizar y optimizar la búsqueda de reglas de transición locales en Autómatas Celulares (AC), permitiendo la emergencia de patrones (morfogénesis artificial) a partir de desorden inicial o semillas puntuales.

---

## Características Principales

*   **Motor CBR Completo:** Implementación del ciclo de vida de 4 fases (Retrieve, Reuse, Revise y Retain).
*   **Optimización evolutiva:** implementación de Estrategias Evolutivas ($\mu + \lambda$)-ES con tasas de mutación adaptativas y dinámicas.
*   **Métricas de Similitud 2D Avanzadas:** Evaluación de fitness basada en el Índice de Similitud Estructural (SSIM), Índice de Jaccard ponderado por estados e Información Mutua.

---

## Requisitos e Instalación

El proyecto está desarrollado en Python 3.12 y se apoya principalmente en las librerías DEAP (computación evolutiva) y cellpylib (simulación de autómatas celulares).

1. Clona este repositorio:
   
```bash
   git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
   cd tu-repositorio

2. Instala las dependencias necesarias:
pip install -r requirements.txt

## Estructura del Repositorio
El código está completamente desacoplado de forma modular. Cada directorio cuenta con su propio archivo README.md interno que detalla la responsabilidad de sus scripts

## Ejecución y Puntos de Entrada
El repositorio dispone de tres scripts principales en la raíz para ejecutar y contrastar las diferentes aproximaciones del estudio:

# 1. Motor CBR (Aproximación Propuesta)
Ejecuta el ciclo de vida completo del CBR. Toma un nuevo problema objetivo, recupera los casos más similares de la memoria (evaluando también rotaciones), inyecta el conocimiento adaptado en la población inicial y refina la solución mediante la estrategia evolutiva.

python main.py

Nota: El nuevo problema o imagen objetivo que se desea resolver debe configurarse directamente editando los parámetros editables dentro de este archivo.

# 2. Resolución Heurística (Línea de Base Experta)
Lanza de manera estática una batería de experimentos predeterminados utilizando configuraciones de hiperparámetros óptimas fijadas manualmente mediante conocimiento experto.

python main_heuristica.py

# 3. Resolución Aleatoria (Control de Entrada)
Lanza experimentos de control configurando los parámetros del autómata y del algoritmo de búsqueda a ciegas (dentro de rangos válidos), sirviendo como línea de base inferior para evaluar la ganancia real del CBR y la heurística.

python main_random.py
