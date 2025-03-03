import numpy as np
import os
from PIL import Image
from eigenfaces_bayes_utils import load_images, compute_eigenfaces, project_images, calculate_class_statistics, predict_single_image_bayes


# Configuración
database_path = "C:/Users/Unax/Desktop/LegacyTFG/Database/AugmentedMediapipe_filtered"
image_size = (100, 100) #Alto ancho en PIL, ancho alto numpy
num_components = 30 # 90% con 20!!!
training_ratio=0.8
# Cargar imágenes
#train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
#np.savez("D:/Universidad/TFG IE/Legacy code/Database/ytf_sinnorm.npz",train_images=train_images,train_labels=train_labels,test_images=test_images,test_labels=test_labels)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/mediapipeNEW.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

print("Carga finalizada") 
 
 #Calcular eigenfaces
#eigenfaces, mean = compute_eigenfaces(train_images,image_size)
#np.savez("D:/Universidad/TFG IE/Legacy code/Database/eigenfaces_PCAfull.npz", eigenfaces=eigenfaces, mean=mean)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_NEW.npz")
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

for img, actual_label in zip(test_images, test_labels):
    predicted_label = predict_single_image_bayes(img, mean, eigenfaces, class_means, class_covariances, image_size, num_components)
    if predicted_label == actual_label:
        correct += 1
        print("epiko")
    else:
        incorrect += 1
        print("basuraaaa")
    #print(f"Actual: {actual_label}, Predicted: {predicted_label}")
    
accuracy = correct / (correct + incorrect)
print(f"Accuracy: {accuracy * 100:.2f}%")
