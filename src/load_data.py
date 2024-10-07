# src/load_data.py

import os
import yaml
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar variables de entorno
load_dotenv()

# Obtener la ruta del dataset desde la variable de entorno
DATASET_DIR = os.getenv('DATASET_DIR')

# Cargar parámetros desde el archivo YAML
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Aquí puedes continuar con la carga de datos
def load_data():
    # Ejemplo de uso de ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, val_generator, test_generator
