# Motor CBR para la Optimización de Reglas de Autómatas Celulares mediante Estrategias Evolutivas

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
