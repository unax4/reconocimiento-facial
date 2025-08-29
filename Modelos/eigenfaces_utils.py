import numpy as np
import os
import random
from PIL import Image
from scipy.stats import multivariate_normal


def load_images(database_path, training_ratio,image_size):
    """
    Carga imágenes desde una ruta dada y las divide en conjuntos de entrenamiento y prueba.

    Args:
        database_path (str): Ruta a la carpeta que contiene los conjuntos de imágenes.
        training_ratio (float): Proporción de imágenes que se utilizarán para entrenamiento.
        image_size (tuple): Dimensiones (ancho, alto) de las imágenes.

    Returns:
        tuple: Cuatro listas con imágenes de entrenamiento, etiquetas de entrenamiento,
               imágenes de prueba y etiquetas de prueba.
    """

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


def compute_eigenfaces(train_images, image_size):
    """
    Calcula las eigenfaces a partir de las imágenes de entrenamiento.

    Args:
        train_images (list): Lista de imágenes de entrenamiento.
        image_size (tuple): Dimensiones (ancho, alto) de las imágenes.

    Returns:
        tuple: Eigenfaces como un arreglo y la imagen promedio.
    """
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


def project_images(images, mean, eigenfaces, num_components, image_size):
    """
    Proyecta imágenes en el espacio de las eigenfaces.

    Args:
        images (list): Lista de imágenes a proyectar.
        mean (numpy.ndarray): Imagen promedio utilizada para centrar los datos.
        eigenfaces (numpy.ndarray): Arreglo de eigenfaces.
        num_components (int): Número de eigenfaces a utilizar para la proyección.
        image_size (tuple): Dimensiones (ancho, alto) de las imágenes.

    Returns:
        list: Proyecciones de las imágenes en el espacio de eigenfaces.
    """
    projections = []
    for img in images:
        img_mean_centered = img.reshape(image_size[0] * image_size[1]) - mean
        projection = eigenfaces[:num_components] @ img_mean_centered
        projections.append(projection)
    return projections


def predict_single_image(image, mean, eigenfaces, trainE, train_labels, shape, num_components):
    """
    Predice la etiqueta de una imagen en función de su proyección en el espacio de eigenfaces.

    Args:
        image (numpy.ndarray): Imagen a predecir.
        mean (numpy.ndarray): Imagen promedio utilizada para centrar los datos.
        eigenfaces (numpy.ndarray): Arreglo de eigenfaces.
        trainE (list): Proyecciones de las imágenes de entrenamiento.
        train_labels (list): Etiquetas de las imágenes de entrenamiento.
        shape (tuple): Dimensiones (ancho, alto) de las imágenes.
        num_components (int): Número de eigenfaces a utilizar para la proyección.

    Returns:
        str: Etiqueta predicha para la imagen.
    """
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


def calculate_class_statistics(trainE, train_labels):
    """
    Calcula la media y la covarianza de las proyecciones para cada clase.

    Args:
        trainE (list): Proyecciones de las imágenes de entrenamiento en el espacio de eigenfaces.
        train_labels (list): Etiquetas de las imágenes de entrenamiento.

    Returns:
        tuple: Diccionario de medias por clase y diccionario de covarianzas por clase.
    """
    class_means = {}
    class_covariances = {}
    unique_labels = np.unique(train_labels)

    for label in unique_labels:
        class_projections = np.array([proj for proj, lbl in zip(trainE, train_labels) if lbl == label])
        class_means[label] = np.mean(class_projections, axis=0)
        class_covariances[label] = np.cov(class_projections, rowvar=False)

    return class_means, class_covariances


def predict_single_image_bayes(image, mean, eigenfaces, class_means, class_covariances, shape, num_components):
    """
    Predice la clase de una imagen utilizando clasificación bayesiana en el espacio de eigenfaces.

    Args:
        image (numpy.ndarray): Imagen a clasificar.
        mean (numpy.ndarray): Imagen promedio utilizada para centrar los datos.
        eigenfaces (numpy.ndarray): Arreglo de eigenfaces usado para reducción de dimensionalidad.
        class_means (dict): Diccionario con las medias de proyección por clase.
        class_covariances (dict): Diccionario con las matrices de covarianza por clase.
        shape (tuple): Dimensiones (ancho, alto) de la imagen.
        num_components (int): Número de eigenfaces a utilizar para la proyección.

    Returns:
        tuple: Etiqueta de clase predicha y diccionario de probabilidades para cada clase.
    """
    image = Image.fromarray(image).resize(shape)
    img_mean_centered = np.array(image).reshape(shape[0] * shape[1]) - mean
    img_projection = eigenfaces[:num_components].dot(img_mean_centered)

    probabilities = {}
    for label in class_means.keys():
        probabilities[label] = multivariate_normal.logpdf(img_projection, mean=class_means[label], cov=class_covariances[label], allow_singular=True)

    return max(probabilities, key=probabilities.get),probabilities

