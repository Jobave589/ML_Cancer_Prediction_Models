# ML_Modelos_Predictivos_Cáncer

Descripción del proyecto
Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de predecir si un paciente tiene cáncer basándose en datos clínicos y demográficos. Para ello, se han utilizado tres datasets distintos que contienen información relevante sobre los pacientes. El modelo busca proporcionar una herramienta de apoyo para la detección temprana del cáncer, mejorando así la toma de decisiones médicas.

Datasets utilizados
Se han empleado tres datasets diferentes para entrenar y evaluar el modelo. A continuación, se describen brevemente:

Dataset 1:

Descripción: Contiene información sobre antecedentes familiares, hábitos de vida y factores de riesgo.
Acceso: Público. Disponible en [[enlace al dataset]](https://github.com/Jobave589/ML_Cancer_Prediction_Models/blob/main/src/data_sample/cancer%20patient%20data%20sets.csv).

Dataset 2:

Descripción: Incluye datos clínicos como resultados de análisis de sangre y pruebas de laboratorio.
Acceso: Público. Disponible en [[enlace al dataset]](https://github.com/Jobave589/ML_Cancer_Prediction_Models/blob/main/src/data_sample/lung_cancer_prediction_dataset.csv).

Dataset 3:

Descripción: Proporciona datos demográficos y antecedentes médicos de los pacientes.
Acceso: Público. Disponible en [enlace al dataset].

Solución adoptada
Se ha implementado un pipeline de Machine Learning que incluye las siguientes etapas:

Preprocesamiento de datos:

Limpieza de datos, imputación de valores faltantes y codificación de variables categóricas.
Balanceo de clases para abordar posibles desbalances en los datos.
Entrenamiento del modelo:

Se probaron varios algoritmos de clasificación, como Random Forest, XGBoost y Gradient Boosting.
Se seleccionó el modelo con mejor rendimiento basado en métricas como precisión, recall y F1-score.
Evaluación:

Los modelos fueron evaluados utilizando validación cruzada y un conjunto de prueba independiente.
Interpretación:

Se analizaron las características más importantes para entender los factores que contribuyen al diagnóstico.

Estructura del repositorio
La estructura del repositorio es la siguiente:

ML_Cancer_Prediction/
│
├── src/  
│   ├── data_sample/       # Archivos de datos de muestra utilizados en el proyecto  
│   ├── img/               # Imágenes necesarias para el proyecto  
│   ├── notebooks/         # Notebooks usados para pruebas y análisis  
│   ├── results_notebook/  # Notebook final con el paso a paso del proyecto  
│   ├── models/            # Modelos guardados al ejecutar el código del proyecto  
│   ├── utils/             # Módulos, funciones auxiliares o clases creadas para el proyecto  
│
├── README.md              # Descripción del proyecto  




ML_Cancer_Prediction_Models

Project Description
This project aims to develop a Machine Learning model capable of predicting whether a patient has cancer based on clinical and demographic data. Three different datasets were used to train and evaluate the model. The goal is to provide a tool to support early cancer detection, improving medical decision-making.

Datasets Used
Three datasets were used in this project:

Dataset 1:

Description: Contains information about family history, lifestyle habits, and risk factors.
Access: Public. Available at [[dataset link]](https://github.com/Jobave589/ML_Cancer_Prediction_Models/blob/main/src/data_sample/cancer%20patient%20data%20sets.csv).

Dataset 2:

Description: Includes clinical data such as blood test results and laboratory analyses.
Access: Public. Available at [[dataset link]](https://github.com/Jobave589/ML_Cancer_Prediction_Models/blob/main/src/data_sample/lung_cancer_prediction_dataset.csv).

Dataset 3:

Description: Provides demographic and medical history data of patients.
Access: Public. Available at [dataset link].

Solution Adopted
The project implements a Machine Learning pipeline with the following steps:

Data Preprocessing:

Data cleaning, missing value imputation, and categorical variable encoding.
Class balancing to address potential imbalances in the data.
Model Training:

Several classification algorithms were tested, including Random Forest, XGBoost, and Gradient Boosting.
The best-performing model was selected based on metrics such as precision, recall, and F1-score.
Evaluation:

Models were evaluated using cross-validation and an independent test set.
Interpretation:

Feature importance analysis was conducted to understand the factors contributing to the diagnosis.


Repository Structure
The repository is organized as follows:

ML_Cancer_Prediction/
│
├── data/  
│   ├── raw/               # Original raw data  
│   ├── processed/         # Processed data ready for modeling  
│
├── notebooks/             # Jupyter notebooks with analysis and experiments  
│
├── src/                   # Project source code  
│   ├── preprocessing/     # Scripts for data preprocessing  
│   ├── models/            # Scripts for model training and evaluation  
│
├── reports/               # Results and visualizations  
│
├── README.md              # Project description    
