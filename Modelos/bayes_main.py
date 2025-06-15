import numpy as np
from eigenfaces_utils import load_images, compute_eigenfaces, project_images, calculate_class_statistics, predict_single_image_bayes
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
database_path = "C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB"
image_size = (192, 168)
num_components = 30 
training_ratio=0.8
# Cargar imágenes
#train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
#np.savez("D:/Universidad/TFG IE/Legacy code/Database/ytf_sinnorm.npz",train_images=train_images,train_labels=train_labels,test_images=test_images,test_labels=test_labels)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

print("Carga finalizada") 
 
 #Calcular eigenfaces
#eigenfaces, mean = compute_eigenfaces(train_images,image_size)
#np.savez("D:/Universidad/TFG IE/Legacy code/Database/eigenfaces_PCAfull.npz", eigenfaces=eigenfaces, mean=mean)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_yaleB.npz")
eigenfaces,mean=data["eigenfaces"], data["mean"]
print("Entrenamiento finalizado") 
# Proyectar imágenes de entrenamiento
trainE = project_images(train_images, mean, eigenfaces, num_components,image_size)

# Calcular estadísticas de las clases
class_means, class_covariances = calculate_class_statistics(trainE, train_labels)
print("Proyeccion y stats finalizado") 
# Evaluación del modelo
correct= 0
incorrect=0
predicted_labels,actual_labels = [],[]
predicted_probabilities = []
class_labels = list(class_means.keys())
for img, actual_label in zip(test_images, test_labels):
    predicted_label, probs = predict_single_image_bayes(img, mean, eigenfaces, class_means, class_covariances, image_size, num_components)
    predicted_labels.append(predicted_label)
    actual_labels.append(actual_label)
    predicted_probabilities.append([probs[label] for label in class_labels])
    print(f"Actual: {actual_label}, Predicted: {predicted_label}")

# Calcular métricas
predicted_probabilities = np.array(predicted_probabilities)
accuracy = np.mean(np.array(predicted_labels) == np.array(actual_labels))

# Cálculo de TOP-3 y TOP-5
sorted_indices = np.argsort(predicted_probabilities, axis=1)[:, ::-1]
sorted_labels = np.array(class_labels)[sorted_indices]

top3_hits = sum(actual_labels[i] in sorted_labels[i][:3] for i in range(len(actual_labels)))
top5_hits = sum(actual_labels[i] in sorted_labels[i][:5] for i in range(len(actual_labels)))

top3_accuracy = top3_hits / len(actual_labels)
top5_accuracy = top5_hits / len(actual_labels)

# Imprimir métricas
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"TOP-3 Accuracy: {top3_accuracy * 100:.2f}%")
print(f"TOP-5 Accuracy: {top5_accuracy * 100:.2f}%")


# Generar reporte de clasificación (precision, recall, f1-score)
print("\nReporte de clasificación:")
print(classification_report(actual_labels, predicted_labels))
