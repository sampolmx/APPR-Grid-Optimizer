# APPR-Grid-Optimizer
Motor de optimización basado en Reinforcement Learning (DQN) para la gestión dinámica de la capacidad de transmisión eléctrica. Reduce el desperdicio de energía solar (*curtailment*) al gestionar proactivamente el almacenamiento de baterías
# APPR-Grid-Optimizer

![Badge de estado de desarrollo](https://img.shields.io/badge/Fase%20Actual-Entrenamiento%20DQN-blue)
![Badge de lenguaje](https://img.shields.io/badge/Lenguaje-Python-yellow.svg)
![Badge de framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)

## 🌟 Visión General

Este repositorio alberga el prototipo del **Agente de Planificación Predictiva de Red (APPR)**. El APPR es un sistema basado en **Aprendizaje por Refuerzo (Deep Q-Network - DQN)** diseñado para resolver uno de los cuellos de botella más críticos en la transición energética: la gestión de la intermitencia renovable.

El agente aprende a despachar dinámicamente recursos de almacenamiento (baterías) para **minimizar el *curtailment*** (desperdicio de energía solar) mientras se adhiere estrictamente a un **límite de capacidad de transmisión fijo** (simulando un cuello de botella).

### 🎯 Objetivo Estratégico

Convertirse en un **Optimizador de la Transición**, reduciendo la fricción técnica y económica que ralentiza la adopción masiva de energías limpias.

## 🚀 Estado del Proyecto (MVP)

El prototipo MVP se centra en una simulación controlada:

*   **Sistema:** 100 MW de capacidad solar instalada.
*   **Cuello de Botella:** Límite de transmisión estricto de **80 MW**.
*   **Recurso de Mitigación:** Batería de 60 MWh con tasa de 20 MW.
*   **Fase:** Entrenamiento del agente DQN completado, comparando la política aprendida contra una gestión ingenua (*Baseline*).

## 🛠️ Cómo Ejecutar el Prototipo

Este proyecto está diseñado para ejecutarse en un entorno Jupyter Notebook.

### 1. Prerrequisitos

Asegúrese de tener Python y Jupyter instalados.

### 2. Instalación de Dependencias

Instale todas las librerías necesarias a partir del archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Ejecución de los Notebooks

Ejecute los siguientes notebooks en orden secuencial dentro de la carpeta `/notebooks`:

1.  **`01_Data_Prep_Baseline.ipynb`**: Define el entorno simulado, genera el conjunto de datos de estrés y establece la métrica de comparación (*Baseline*).
2.  **`02_APPR_DQN_Training.ipynb`**: Contiene la implementación del entorno RL, el modelo DQN (TensorFlow/Keras) y el bucle de entrenamiento.

---

## 📁 Estructura del Repositorio

```
/APPR_Grid_Optimizer
├── .gitignore             # Archivos ignorados (cachés, datos brutos grandes)
├── requirements.txt       # Lista de dependencias para replicación
├── README.md              # Documentación actual
│
├── data/                  # (Se puede usar para datos reales si son necesarios)
│
└── notebooks/
    ├── 01_Data_Prep_Baseline.ipynb
    └── 02_APPR_DQN_Training.ipynb
```

## ⚙️ Próximos Pasos (Hoja de Ruta)

1.  **Refactorización y Validación:** Mejorar la Fase 3 para obtener una comparación visual y cuantitativa directa entre Baseline y APPR.
2.  **Integración de Predicción:** Migrar el estado del agente para incluir modelos de pronóstico de energía (usando LSTMs o Transformers) en lugar de solo datos instantáneos.
3.  **Escalabilidad a GCP:** Migrar la lógica del entorno y el entrenamiento a un servicio gestionado (ej. Vertex AI Training) para simular escenarios más grandes y complejos.

---
*Desarrollado con el objetivo de acelerar la adopción de energía limpia mediante optimización inteligente de sistemas.*
