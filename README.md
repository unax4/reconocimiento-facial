# Modelos de aprendizaje automático para tareas de reconocimiento facial

Este repositorio complementa mi Trabajo de Fin de Grado, que ha consistido en entrenar varios **modelos de aprendizaje automático** para que sean capaces de clasificar imágenes de caras. Se han utilizado dos bases de datos, la de YaleB, con solo 37 clases  y 2480 fotos, siendo la más pequeña, y una versión adaptada de VGGFaces2, con un total de 534 clases y 94600 muestras en total, siendo además esta última base de datos desequilibrada.
Para dar una aplicación práctica a los modelos ademas se ha desarrollado un programa con una interfaz gráfica capaz de hacer reconocimiento facial en vivo mediante el uso de una camara. La interfaz además permite editar la base de datos, añadir nuevas clases y elegir entre los dos modelos de PCA para ver cual funciona mejor. Como ejemplo se ha añadido una base de datos para uso exclusivo de la cámara donde ya hay añadidas unas 40 clases, por lo que el usuario puede añadir a esta fotos suyas para ver si los modelos son capaz de reconocerle.

## Uso

En el directorio raiz se encuentra el archivo 'mediapipe_facedetector', que es con el cual se han preprocesado las caras de la base de datos de VGGFaces2.

El directorio `Databases` solo está la base de datos usada en la cámara, debido a que las que se usan en el trabajo son de tamaño demasiado grande. Estas en cambio se pueden descargar en los siguientes enlaces: http://cvc.cs.yale.edu/cvc/projects/yalefacesB/yalefacesB.html y https://www.kaggle.com/datasets/hearfool/vggface2 

Dentro del directorio 'modelos' se encuentran las funciones usadas en los dos metodos basados en PCA junto con los 'main' donde se ejecutan, y también la red neuronal convolucional entrenada en su archivo respectivo. Asi los modelos usados son los siguientes:
1. Eigenfaces
2. Eigenfaces Bayesiano
3. Redes neuronales convolucionales.
Para cada modelo primero se ha realizado un ajuste de hiperparámetros, seguida de una validación cruzada.

El la carpets 'resultados' se pueden ver las figuras consegidas tanto como en las validación cruzadas y ajustes de hiperparametros que se han hecho, como los resultados de exactitud y perdida conseguidos época a época en el caso del modelo de CNN.

Por último, en el directorio raíz también se encentra el archivo con el que se lanza el programa de detección via web cam.

## Paquetes necesarios

En este trabajo se ha utilizado `Python 3.11.3`. Para el trabajo se han usado los siguientes paquetes:
```
Paquete                       Versión
----------------------------- --------------
matplotlib                    3.6.3
numpy                         1.23.5
pandas                        1.5.3
scikit-learn                  1.2.0
tensorflow                    2.12.0
keras                         2.11.0
ipywidgets                    8.0.4
ipycanvas                     0.13.2
voila                         0.5.7
```
