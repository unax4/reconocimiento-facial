import numpy as np
import os
import random
from PIL import Image
from sklearn.decomposition import IncrementalPCA

def load_images(database_path, training_ratio,image_size):
    train_images, train_labels, test_images, test_labels = [], [], [], []
    
    for folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, folder)
        if os.path.isdir(folder_path):
            person_id = folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(".pgm") or f.endswith(".jpg") or f.endswith(".png")]
            random.shuffle(image_files)
            num_train = round(len(image_files) * training_ratio)

            for i, img_file in enumerate(image_files):
                img = np.array(Image.open(os.path.join(folder_path, img_file)).convert("L").resize(image_size)) 
                if i < num_train:
                    train_images.append(img)
                    train_labels.append(person_id)
                else:
                    test_images.append(img)
                    test_labels.append(person_id)
    
    return train_images, train_labels, test_images, test_labels

"""

def compute_eigenfaces(train_images, image_size):
    #Calcula las eigenfaces a partir de imágenes de entrenamiento.
    X = np.resize(train_images, (len(train_images), image_size[0] * image_size[1]))
    mean = np.mean(X, axis=0)
    X_std = X - mean

    cov_mat = X_std @ X_std.T #multipiclación de matrices
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    eigenvector_converted = X_std.T @ eigenvectors 
    eigenfaces = eigenvector_converted.T / np.sqrt((eigenvector_converted.T ** 2).sum(axis=1, keepdims=True))

    return eigenfaces, mean
    


def compute_eigenfaces(train_images, image_size,batch_size, n_components):
    N, h, w = len(train_images), image_size[0], image_size[1]
    X = np.reshape(train_images, (N, h * w))  # Convertimos a matriz 2D (N, d)

    mean = np.mean(X, axis=0)  # Calculamos la media
    X_std = X - mean  # Centramos los datos

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(X_std)  # Ajustamos PCA en lotes pequeños

    eigenfaces = ipca.components_  # Eigenfaces obtenidas

    return eigenfaces, mean
"""

def compute_eigenfaces(train_images, image_size):
    X = np.reshape(train_images, (len(train_images), image_size[0] * image_size[1]))
    mean = np.mean(X, axis=0)
    X_std = (X - mean).astype(np.float32)


    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    eigenfaces = Vt

    return eigenfaces, mean


def project_images(images, mean, eigenfaces, num_components, image_size):
    #Proyecta imágenes en el espacio de eigenfaces.
    projections = []
    for img in images:
        img_mean_centered = img.reshape(image_size[0] * image_size[1]) - mean #Aseguramos que la imagen zea point cero y de nuestra dimension
        projection = eigenfaces[:num_components].dot(img_mean_centered)
        projections.append(projection)
    return projections

def predict_single_image(image, mean, eigenfaces, trainE, train_labels, shape, num_components):
    #Predice la identidad de una imagen basada en eigenfaces.
    image = image.reshape(shape[0] * shape[1]) - mean
    img_projection = eigenfaces[:num_components] @ image

    smallest_distance = float('inf')
    predicted_label = None

    for i, train_proj in enumerate(trainE):
        distance = np.linalg.norm(img_projection - train_proj)
        if distance < smallest_distance:
            smallest_distance = distance
            predicted_label = train_labels[i]
    
    return predicted_label

def reconstruction_error(image, mean, eigenfaces, num_components, shape):
    
    # Proyectar la imagen en el espacio de eigenfaces
    img_mean_centered = image.reshape(shape[0] * shape[1]) - mean
    img_projection = eigenfaces[:num_components].dot(img_mean_centered)
    
    # Reconstruir la imagen
    reconstructed = mean + eigenfaces[:num_components].T.dot(img_projection)
    reconstructed_image = reconstructed.reshape((shape[0], shape[1]))  
    
    # Convertir a imagen PIL y redimensionar
    reconstructed_pil = Image.fromarray(reconstructed_image).resize((image.shape[1], image.shape[0]))
    reconstructed_array = np.array(reconstructed_pil)
    
    # Calcular el error de reconstrucción (norma L2)
    error = np.linalg.norm(image - reconstructed_array)
    return error

"""def compute_eigenfaces(train_images, num_components):
    num_samples, h, w = train_images.shape
    X = train_images.reshape(num_samples, h * w)

    pca = PCA(n_components=num_components, whiten=True)
    projections = pca.fit_transform(X)
    eigenfaces = pca.components_
    mean_face = pca.mean_

    return pca, eigenfaces, mean_face, projections

def project_images(pca, images):
    num_samples, h, w = images.shape
    X = images.reshape(num_samples, h * w)
    return pca.transform(X)

def predict_image(image, pca, train_projections, train_labels):
    img_projection = pca.transform(image.reshape(1, -1))

    # Usar Nearest Neighbors en lugar de cálculo manual de distancia
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train_projections)
    _, indices = nn.kneighbors(img_projection)
    
    return train_labels[indices[0][0]]
    
    # Carga de imágenes
train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)

# Computar eigenfaces
pca, eigenfaces, mean_face, train_projections = compute_eigenfaces(train_images, num_components)

# Proyectar imágenes de prueba
test_projections = project_images(pca, test_images)

# Evaluación
correct = 0
for img, true_label in zip(test_images, test_labels):
    predicted_label = predict_image(img, pca, train_projections, train_labels)
    correct += (predicted_label == true_label)

accuracy = correct / len(test_labels)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
"""
