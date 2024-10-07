from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

def train_covid_model(train_generator, val_generator, yaml_path='config.yaml'):
    # Cargar los parámetros del YAML
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Crear el modelo base DenseNet121 con pesos preentrenados de ImageNet
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Añadir capas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Para clasificación binaria (COVID vs. Normal)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Congelar las capas base del modelo preentrenado
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compilar el modelo usando los parámetros del archivo YAML
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    
    # Entrenar el modelo
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'],
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history



# Cargar variables de entorno
load_dotenv()

# Ruta de la carpeta de imágenes para inferencia desde el archivo .env
IMAGES_DIR = os.getenv('IMAGES_DIR')

def load_and_preprocess_image(img_path, image_size=(224, 224)):
    """ Carga y preprocesa una imagen para hacer inferencia. """
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Escalar los valores de la imagen
    return img_array

def predict_image(model, img_array):
    """ Realiza la predicción sobre la imagen preprocesada. """
    prediction = model.predict(img_array)
    return (prediction > 0.5).astype(int)  # Clasificación binaria

def run_inference_on_images(model_path, inference_path=IMAGES_DIR):
    """ Realiza inferencia en todas las imágenes de la carpeta dada. """
    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Verificar que la carpeta exista
    if not os.path.exists(inference_path):
        print(f"La carpeta {inference_path} no existe.")
        return

    # Obtener todas las imágenes de la carpeta
    image_files = [f for f in os.listdir(inference_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Realizar predicciones en cada imagen
    for img_file in image_files:
        img_path = os.path.join(inference_path, img_file)
        img_array = load_and_preprocess_image(img_path)

        prediction = predict_image(model, img_array)

        # Mostrar el resultado de la predicción
        label = 'COVID' if prediction == 1 else 'Normal'
        print(f"Imagen: {img_file} -> Predicción: {label}")

