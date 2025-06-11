import numpy as np
import os
import random
from PIL import Image
from scipy.stats import multivariate_normal


def load_images(database_path, training_ratio,image_size):
    """
    Loads images from a given database path and splits them into training and testing datasets.

    Args:
        database_path (str): Path to the folder containing image datasets.
        training_ratio (float): Ratio of images to be used as training data.
        image_size (tuple): Dimensions (width, height) of images.

    Returns:
        tuple: Four lists containing training images, training labels, testing images, and testing labels.
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
    Computes eigenfaces using from training images.

    Args:
        train_images (list): List of training images.
        image_size (tuple): Dimensions (width, height) of images.

    Returns:
        tuple: Eigenfaces as an array and the mean image.
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
    Projects images into the eigenfaces space.

    Args:
        images (list): List of images to project.
        mean (numpy.ndarray): Mean image used for centering.
        eigenfaces (numpy.ndarray): Array of eigenfaces.
        num_components (int): Number of eigenfaces to use for projection.
        image_size (tuple): Dimensions (width, height) of images.

    Returns:
        list: Projections of the images in the eigenfaces space.
    """
    projections = []
    for img in images:
        img_mean_centered = img.reshape(image_size[0] * image_size[1]) - mean
        projection = eigenfaces[:num_components].dot(img_mean_centered)
        projections.append(projection)
    return projections


def predict_single_image(image, mean, eigenfaces, trainE, train_labels, shape, num_components):
    """
    Predicts the label of an image based on its projection in the eigenfaces space.

    Args:
        image (numpy.ndarray): Image to predict.
        mean (numpy.ndarray): Mean image used for centering.
        eigenfaces (numpy.ndarray): Array of eigenfaces.
        trainE (list): Projections of training images.
        train_labels (list): Labels for training images.
        shape (tuple): Dimensions (width, height) of images.
        num_components (int): Number of eigenfaces to use for projection.

    Returns:
        str: Predicted label for the image.
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
    Calculates the mean and covariance of projections for each class.

    Args:
        trainE (list): Projections of training images in eigenfaces space.
        train_labels (list): Labels for training images.

    Returns:
        tuple: Dictionary of class means and class covariances.
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
    Predicts the class of a given image using Bayesian classification in the eigenfaces space.

    Args:
        image (numpy.ndarray): The input image to classify.
        mean (numpy.ndarray): The mean image used for centering the data.
        eigenfaces (numpy.ndarray): Array of eigenfaces used for dimensionality reduction.
        class_means (dict): Dictionary containing the mean projections for each class.
        class_covariances (dict): Dictionary containing the covariance matrices for each class.
        shape (tuple): Dimensions (width, height) of the image.
        num_components (int): Number of eigenfaces to use for projection.

    Returns:
        tuple: Predicted class label and a dictionary of probabilities for each class.
    """
    image = Image.fromarray(image).resize(shape)
    img_mean_centered = np.array(image).reshape(shape[0] * shape[1]) - mean
    img_projection = eigenfaces[:num_components].dot(img_mean_centered)

    probabilities = {}
    for label in class_means.keys():
        probabilities[label] = multivariate_normal.logpdf(img_projection, mean=class_means[label], cov=class_covariances[label], allow_singular=True)

    return max(probabilities, key=probabilities.get),probabilities