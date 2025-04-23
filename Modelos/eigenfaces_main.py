import numpy as np
from eigenfaces_utils import load_images, compute_eigenfaces, project_images, predict_single_image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

training_ratio = 0.8
#image_size = (130, 130) #Alto ancho en PIL, ancho alto numpy
image_size = (192, 168) #yaleB
num_components = 150 
#database_path="C:/Users/Unax/Desktop/LegacyTFG/Database/CasiaCropped"
database_path="C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB"
# Cargar imágenes
#train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
#np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz",train_images=train_images,train_labels=train_labels,test_images=test_images,test_labels=test_labels)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

print("Carga finalizada") 

# Calcular eigenfaces
#eigenfaces, mean= compute_eigenfaces(train_images, image_size)
#np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_yaleB.npz", eigenfaces=eigenfaces, mean=mean)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_yaleB.npz")
eigenfaces,mean=data["eigenfaces"], data["mean"]
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
precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted')

print(f"Precisión: {precision * 100:.2f}%")
print(f"Sensibilidad: {recall * 100:.2f}%")
print(f"Valor F!: {f1 * 100:.2f}%")

"""
# Generar matriz de confusión
cm = confusion_matrix(actual_labels, predicted_labels)
print("\nMatriz de confusión:")
print(cm)

# Generar reporte de clasificación (incluye precision, recall, f1-score)
print("\nReporte de clasificación:")
print(classification_report(actual_labels, predicted_labels))
"""
"""# Store results
train_accuracies = []
test_accuracies = []

# Test for different numbers of components
num_components_range = range(10, 250,40)  # You can change the range based on your requirements

for num_components in num_components_range:
    # Project training images
    trainE = project_images(train_images, mean, eigenfaces, num_components, (192, 168))

    # Predict labels for the training set
    predicted_train_labels = []
    for img in train_images:
        predicted_label = predict_single_image(img, mean, eigenfaces, trainE, train_labels, (192, 168), num_components)
        predicted_train_labels.append(predicted_label)
    
    # Calculate training accuracy
    train_accuracy = accuracy_score(train_labels, predicted_train_labels)
    train_accuracies.append(train_accuracy)

    # Project test images
    predicted_test_labels = []
    for img in test_images:
        predicted_label = predict_single_image(img, mean, eigenfaces, trainE, train_labels, (192, 168), num_components)
        predicted_test_labels.append(predicted_label)
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(test_labels, predicted_test_labels)
    test_accuracies.append(test_accuracy)

    print(f"num_components: {num_components} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_components_range, train_accuracies, label='Train Accuracy', color='blue', marker='o')
plt.plot(num_components_range, test_accuracies, label='Test Accuracy', color='red', marker='x')

plt.title('Accuracy vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()"""