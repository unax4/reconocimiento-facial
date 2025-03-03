import numpy as np
from eigenfaces_utils import load_images, compute_eigenfaces, project_images, predict_single_image

training_ratio = 0.8
image_size = (130, 130) #Alto ancho en PIL, ancho alto numpy
num_components = 200 # 25% con 100
database_path="C:/Users/Unax/Desktop/LegacyTFG/Database/CasiaCropped"
# Cargar imágenes
#train_images, train_labels, test_images, test_labels = load_images(database_path, training_ratio, image_size)
#np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/casiafull.npz",train_images=train_images,train_labels=train_labels,test_images=test_images,test_labels=test_labels)
data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/casiafull.npz")
train_images, test_images, train_labels, test_labels = data["train_images"], data["test_images"],data["train_labels"],data["test_labels"]
train_images, test_images = np.array(train_images), np.array(test_images)

print("Carga finalizada") 

# Calcular eigenfaces
eigenfaces, mean= compute_eigenfaces(train_images, image_size,batch_size=500, n_components=200)
np.savez_compressed("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_CASIA.npz", eigenfaces=eigenfaces, mean=mean)
#data = np.load("C:/Users/Unax/Desktop/LegacyTFG/Database/eigenfaces_NEW.npz")
#eigenfaces,mean=data["eigenfaces"], data["mean"]
print("Entrenamiento finalizado") 
# Proyectar imágenes de entrenamiento
trainE = project_images(train_images, mean, eigenfaces, num_components, image_size)
print("Imagenes proyectadas") 
# Evaluación del modelo
correct, incorrect = 0, 0
i=0
for img, actual_label in zip(test_images, test_labels):
    predicted_label = predict_single_image(img, mean, eigenfaces, trainE, train_labels, image_size, num_components)
    i=+1
    if i>100:
        break
    if predicted_label == actual_label:
        correct += 1
        print("correct")
    else:
        incorrect += 1
        print("basuraaaaaaaaaa")
    #print(f"Actual: yaleB{actual_label}, Predicted: yaleB{predicted_label}")

accuracy = correct / (correct + incorrect)
print(f"Accuracy: {accuracy * 100:.2f}%")

