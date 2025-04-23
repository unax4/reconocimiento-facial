import numpy as np
from eigenfaces_bayes_utils import load_images, compute_eigenfaces, project_images, calculate_class_statistics, predict_single_image_bayes
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
database_path = "C:/Users/Unax/Desktop/LegacyTFG/Database/AugmentedMediapipe_filtered"
image_size = (192, 168) #Alto ancho en PIL, ancho alto numpy
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
# Generar matriz de confusión
cm = confusion_matrix(actual_labels, predicted_labels)
print("\nMatriz de confusión:")
print(cm)

# Generar reporte de clasificación (incluye precision, recall, f1-score)
print("\nReporte de clasificación:")
print(classification_report(actual_labels, predicted_labels))

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.show()

# Calcular métricas adicionales
def calculate_metrics(cm):
    if cm.shape[0] == 2:  # Para clasificación binaria
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nMétricas adicionales:")
        print(f"Sensibilidad (Recall): {sensitivity:.4f}")
        print(f"Especificidad: {specificity:.4f}")
        print(f"Tasa de Falsos Positivos: {false_positive_rate:.4f}")
    else:
        print("\nMétricas detalladas disponibles en el reporte de clasificación")

calculate_metrics(cm)

'''# Inicializar listas para almacenar las precisiones
train_accuracies = []
test_accuracies = []

# Rango de valores para num_components
num_components_range = range(1, 80,10)  # Ajusta el rango según lo que necesites

# Evaluar el modelo para cada número de componentes
for num_components in num_components_range:
    # Proyectar las imágenes de entrenamiento
    trainE = project_images(train_images, mean, eigenfaces, num_components, image_size)
    
    # Calcular estadísticas de las clases (medias y covarianzas)
    class_means, class_covariances = calculate_class_statistics(trainE, train_labels)
    
    # Predicciones en el conjunto de entrenamiento
    predicted_train_labels = []
    for img in train_images:
        predicted_label = predict_single_image_bayes(img, mean, eigenfaces, class_means, class_covariances, image_size, num_components)
        predicted_train_labels.append(predicted_label)

    # Calcular la precisión en el conjunto de entrenamiento
    train_accuracy = accuracy_score(train_labels, predicted_train_labels)
    train_accuracies.append(train_accuracy)

    # Predicciones en el conjunto de prueba
    predicted_test_labels = []
    for img in test_images:
        predicted_label = predict_single_image_bayes(img, mean, eigenfaces, class_means, class_covariances, image_size, num_components)
        predicted_test_labels.append(predicted_label)

    # Calcular la precisión en el conjunto de prueba
    test_accuracy = accuracy_score(test_labels, predicted_test_labels)
    test_accuracies.append(test_accuracy)

    print(f"num_components: {num_components} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(num_components_range, train_accuracies, label='Train Accuracy', color='blue', marker='o')
plt.plot(num_components_range, test_accuracies, label='Test Accuracy', color='red', marker='o')

plt.title('Accuracy vs Number of Components (Bayes Eigenfaces)')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()'''