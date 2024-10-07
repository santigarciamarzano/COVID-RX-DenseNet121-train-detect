# Clasificación COVID-19 vs Normal usando DenseNet121

## Introducción

Este proyecto utiliza una red neuronal convolucional (CNN) basada en **DenseNet121** para clasificar imágenes de radiografías de tórax (RX) en dos categorías: **COVID-19** y **Normal**. El modelo ha sido entrenado usando un dataset que contiene estas dos clases, y el objetivo principal es detectar casos positivos de COVID-19.

Se ha configurado el proyecto para que cargue automáticamente los parámetros de entrenamiento desde un archivo de configuración `.yaml` y se ejecute en entornos controlados con un archivo `.env` para variables del sistema.

---

## Requisitos

Puedes instalar todas las dependencias con:

pip install -r requirements.txt

---

## Estructura del proyecto

Descarga el dataset https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database y utiliza las carpetas unicamente las carpetas "covid" y "normal". Luego estructura el dataset de la siguiente manera, con aproximadamente un ratio de imagenes de 70%train, 15%val y 15%test

.
├── src
│   ├── load_data.py       # Script para cargar los datos
│   ├── model.py           # Script para crear y entrenar el modelo
├── dataset
│   ├── train/             # Carpeta de imágenes de entrenamiento (contiene subcarpetas 'COVID' y 'Normal')
│   ├── val/               # Carpeta de imágenes de validación (contiene subcarpetas 'COVID' y 'Normal')
│   ├── test/              # Carpeta de imágenes de prueba (contiene subcarpetas 'COVID' y 'Normal')
├── config.yaml            # Archivo de configuración
├── .env                   # Variables de entorno
├── notebook.ipynb         # Jupyter Notebook para la ejecución de bloques de código
├── README.md              # Documento explicativo del proyecto (este archivo)



## Configuración del entorno .env

Este archivo incluye variables necesarias para el entorno de trabajo. Asegurate de copiar la ruta al dataset:

DATASET_DIR= ruta/dataset

---

## Archivo de configuración config.yaml

El archivo config.yaml contiene los parámetros clave para entrenar el modelo. Se puede configurar desde allí.

---

## Entrenamiento del modelo

Cómo entrenar el modelo

El entrenamiento del modelo está definido en src/model.py. Este script:

    Carga los datos de entrenamiento y validación desde las carpetas especificadas.
    Define la arquitectura del modelo utilizando DenseNet121.
    Compila el modelo con el optimizador Adam.
    Entrena el modelo utilizando los generadores de datos.
    Guarda el mejor modelo encontrado durante el entrenamiento.

Para ejecutar el entrenamiento, asegúrate de tener los datos organizados en las carpetas adecuadas y luego corre el script.

