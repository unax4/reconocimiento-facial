import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from eigenfaces_utils import load_images, compute_eigenfaces, project_images, predict_single_image

def cross_validation(train_images, train_labels, image_size, num_components_values, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {fold: {} for fold in range(1, k+1)}
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_images), start=1):
        train_subset = [train_images[i] for i in train_index]
        val_subset = [train_images[i] for i in val_index]
        train_labels_subset = [train_labels[i] for i in train_index]
        val_labels_subset = [train_labels[i] for i in val_index]
        eigenfaces, mean = compute_eigenfaces(train_subset, image_size)
        for num_components in num_components_values:
            
            trainE = project_images(train_subset, mean, eigenfaces, num_components, image_size)

            
            correct = 0
            for img, true_label in zip(val_subset, val_labels_subset):
                pred_label = predict_single_image(img, mean, eigenfaces, trainE, train_labels_subset, image_size, num_components)
                if pred_label == true_label:
                    correct += 1
            
            accuracy = correct / len(val_subset)
            results[fold][num_components] = accuracy
    
    return results

# Parámetros
num_components_values = [10,50,80,130,160,250]

# Cargar imágenes 
image_size = (192, 168)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/yaleB.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

results = cross_validation(train_images, train_labels, (192, 168), num_components_values)

# Graficar resultados
plt.figure(figsize=(10, 6))
for fold, accuracies in results.items():
    plt.plot(num_components_values, [accuracies[n] for n in num_components_values], marker='o', label=f'Fold {fold}')
plt.xlabel('Número de Componentes')
plt.ylabel('Exactitud')
plt.title('Resultados Validación Cruzada')
plt.legend()
plt.grid(True)
plt.show()
