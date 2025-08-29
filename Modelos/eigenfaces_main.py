import numpy as np
from eigenfaces_utils import load_images, compute_eigenfaces, project_images, predict_single_image
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

training_ratio = 0.8
image_size = (192, 168) #yaleB
num_components = 150
database_path="C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB"
# Cargar imágenes
train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
#np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz",train_images=train_images,train_labels=train_labels,test_images=test_images,test_labels=test_labels)
#data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz")
#train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

print("Carga finalizada") 

# Calcular eigenfaces
eigenfaces, mean= compute_eigenfaces(train_images, image_size)
#np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_yaleB.npz", eigenfaces=eigenfaces, mean=mean)
#data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_yaleB.npz")
#eigenfaces,mean=data["eigenfaces"], data["mean"]
print("Entrenamiento finalizado") 
# Proyectar imágenes de entrenamiento
trainE = project_images(train_images, mean, eigenfaces, num_components, image_size)
print("Imagenes proyectadas") 
# Evaluación del modelo
correct, incorrect = 0, 0
predicted_labels,actual_labels = [],[]

for img, actual_label in zip(test_images, test_labels):
    predicted_label = predict_single_image(img, mean, eigenfaces, trainE, train_labels, 
                                         image_size, num_components)
    predicted_labels.append(predicted_label)
    actual_labels.append(actual_label)
    print(f" Real: {actual_label}, Predicho: {predicted_label}")
test_accuracy = accuracy_score(actual_labels, predicted_labels)
print(test_accuracy)

# Calcular métricas
accuracy = np.mean(np.array(predicted_labels) == np.array(actual_labels))
print(f"\nExactitud: {accuracy * 100:.2f}%")
precision = precision_score(actual_labels, predicted_labels, zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted')

print(f"Precisión: {precision * 100:.2f}%")
print(f"Sensibilidad: {recall * 100:.2f}%")
print(f"Valor F!: {f1 * 100:.2f}%")

