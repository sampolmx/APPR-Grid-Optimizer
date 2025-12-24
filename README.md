# APPR-Grid-Optimizer

[![License](https://img.shields.io/github/license/sampolmx/APPR-Grid-Optimizer)](https://github.com/sampolmx/APPR-Grid-Optimizer/blob/main/LICENSE) [![Repo size](https://img.shields.io/github/repo-size/sampolmx/APPR-Grid-Optimizer)](https://github.com/sampolmx/APPR-Grid-Optimizer) [![Top language](https://img.shields.io/github/languages/top/sampolmx/APPR-Grid-Optimizer)](https://github.com/sampolmx/APPR-Grid-Optimizer) [![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

Motor de optimización basado en Reinforcement Learning (DQN) para la gestión dinámica de la capacidad de transmisión eléctrica. Reduce el desperdicio de energía solar (curtailment) al gestion[...]

Resumen corto / Short summary
- Español: Motor de optimización (DQN) para reducir el curtailment solar mediante la gestión proactiva de baterías y la capacidad de transmisión.
- English: Optimization engine (DQN) to reduce solar curtailment by proactively managing battery storage and transmission capacity.

Características / Features
- Implementación de agentes DQN para decisiones de carga/descarga de baterías.
- Simulaciones y notebooks para entrenamiento y evaluación.
- Scripts y utilidades para preprocesado de datos y visualización de resultados.

Quick start
1. Clona el repositorio:
   git clone https://github.com/sampolmx/APPR-Grid-Optimizer.git
2. Crea un entorno virtual (recomendado) e instala dependencias:
   python -m venv .venv
   source .venv/bin/activate  # o .venv\Scripts\activate en Windows
   pip install -r requirements.txt
3. Abre los notebooks en el directorio notebooks/ con Jupyter:
   jupyter lab

Installation
- Recomendado: Python 3.8+.
- Instala las dependencias:
  pip install -r requirements.txt

Usage
- Notebooks: revisa el directorio `notebooks/` para los flujos de trabajo principales (entrenamiento, evaluación, análisis de resultados).
- Scripts: Si el proyecto contiene un paquete `src/` o `app/`, ejecutar los scripts desde el entorno virtual.
- Reproducibilidad: fija seeds en los notebooks y revisa `experiments/` si existe para reproducir corridas.

Repository structure (suggested)
- notebooks/          # Jupyter notebooks (entrenamiento, evaluación, análisis)
- src/ or app/         # Código fuente de los agentes y entorno
- data/                # Datos (o instrucciones para obtenerlos)
- experiments/         # Resultados, checkpoints, logs
- requirements.txt     # Dependencias
- LICENSE
- README.md

Notebooks index (example)
- notebooks/01_data_preparation.ipynb
- notebooks/02_environment_and_agent.ipynb
- notebooks/03_training.ipynb
- notebooks/04_evaluation.ipynb

Model card / Reproducibility
- Describe el modelo (DQN): arquitectura, observaciones, acciones, recompensa.
- Guarda hiperparámetros y seeds en `experiments/` para reproducibilidad.

Contributing
- Si deseas contribuir, abre un issue o PR.
- Incluye descripciones claras y pasos para reproducir bugs.

License
- Este proyecto está licenciado bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

Citación / Citation
Si usas este repositorio en investigación, por favor cita: sampolmx/APPR-Grid-Optimizer (GitHub).

Contacto
- Autor: Sam Polanco Iñigo 
