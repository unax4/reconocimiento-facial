import os
import random
import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal

#Esta es la que he usado para ytf_traintest
def load_images(database_path, training_ratio,image_size):
    train_images, train_labels, test_images, test_labels = [], [], [], []
    
    for folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, folder)
        if os.path.isdir(folder_path):
            person_id = folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
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



def compute_eigenfaces(train_images, image_size):
    X = np.reshape(train_images, (len(train_images), image_size[0] * image_size[1]))
    mean = np.mean(X, axis=0)
    X_std = X - mean  # Centramos los datos

    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    eigenfaces = Vt

    return eigenfaces, mean

def project_images(images, mean, eigenfaces, num_components,image_size):
    
    #Proyecta imágenes en el espacio de eigenfaces.
    projections = []
    for img in images:
        img_mean_centered = img.reshape(image_size[0] * image_size[1]) - mean #Aseguramos que la imagen zea point cero y de nuestra dimension
        projection = eigenfaces[:num_components].dot(img_mean_centered)
        projections.append(projection)
    return projections

def calculate_class_statistics(trainE, train_labels):
    #Calcula la media y la covarianza de las proyecciones para cada clase.
    class_means = {}
    class_covariances = {}
    unique_labels = np.unique(train_labels)

    for label in unique_labels:
        class_projections = np.array([proj for proj, lbl in zip(trainE, train_labels) if lbl == label])
        class_means[label] = np.mean(class_projections, axis=0)
        class_covariances[label] = np.cov(class_projections, rowvar=False)

    return class_means, class_covariances

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


def predict_single_image_bayes(image, mean, eigenfaces, class_means, class_covariances, shape, num_components):
    image = Image.fromarray(image).resize(shape)
    img_mean_centered = np.array(image).reshape(shape[0] * shape[1]) - mean
    img_projection = eigenfaces[:num_components].dot(img_mean_centered)

    probabilities = {}
    for label in class_means.keys():
        probabilities[label] = multivariate_normal.logpdf(img_projection, mean=class_means[label], cov=class_covariances[label], allow_singular=True)

    return max(probabilities, key=probabilities.get)
