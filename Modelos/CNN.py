import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,SGD,RMSprop, Adagrad, Adadelta
from eigenfaces_utils import load_images
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


#data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/augmediapipe2.npz")
#train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/casiafull.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
database_path = "C:/Users/Unax/Desktop/LegacyTFG/Database/CasiaCropped"
image_size = (130, 130) #Alto ancho en PIL, ancho alto numpy
training_ratio=0.8
input_shape = (image_size[0], image_size[1], 1)  # Canal único (escala de grises)
#train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
# Cargar imágenes y etiquetas
train_images = np.array(train_images)  
test_images = np.array(test_images) 
train_images = train_images.astype('float32') / 255.0
test_images =test_images.astype('float32') / 255.0
train_images = train_images.reshape(-1, 130, 130, 1)  # (n_samples, height, width, channels)
test_images = test_images.reshape(-1, 130, 130, 1)
# Codificar etiquetas de texto a números enteros
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)  # Usa el mismo encoder
# Obtener el número de clases después del filtrado
num_classes = len(np.unique(train_labels))  # Asegura el número correcto de clases

# Convertir etiquetas a one-hot encoding con el número correcto de clases
train_labels = keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes=num_classes)

print("carga completada, compilando CNN...")

model = keras.models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  
])
"""
##Adjunto otros modelos candidatos
model = keras.models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model = keras.models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),  # Dropout después de la primera capa convolucional
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),  # Dropout después de la segunda capa convolucional
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),  # Dropout después de la tercera capa convolucional
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Dropout en la capa densa
    layers.Dense(num_classes, activation='softmax')
])
model = keras.models.Sequential([
keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu",input_shape=input_shape),
keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
keras.layers.MaxPool2D(),
keras.layers.Flatten(),
keras.layers.Dropout(0.25),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(num_classes, activation="softmax")
])


"""
# Compilar el modelo
optimizer = Adam(learning_rate=0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Resumen del modelo
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Métrica a monitorear (pérdida en el conjunto de validación)
    patience=8,         # Número de épocas sin mejora antes de detener el entrenamiento
    restore_best_weights=True,
    mode='min'  # Restaurar los mejores pesos encontrados
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Métrica a monitorear
    factor=0.1,          # Factor por el cual se reduce la tasa de aprendizaje
    patience=5,         # Número de épocas sin mejora antes de reducir la tasa de aprendizaje
    min_lr=0.00001,
    mode='min'      
)

# Entrenar el modelo con callbacks
history = model.fit(
    train_images, train_labels,
    batch_size=64,
    epochs=80,  # Número máximo de épocas (puede detenerse antes debido al early stopping)
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, reduce_lr]  # Añadir los callbacks aquí
)
# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Guardar el modelo entrenado
model.save("C:/Users/Unax/Desktop/LegacyTFG/models/modelo_CASIA.keras")
#modelo_cargado = keras.models.load_model("modelo_reconocimiento_facial.h5")
#predicciones = modelo_cargado.predict(test_images)

#Plot the result
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()