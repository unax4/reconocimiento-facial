# Modelos de aprendizaje automático para tareas de reconocimiento facial

Este repositorio complementa mi Trabajo de Fin de Grado, que ha consistido en entrenar varios **modelos de aprendizaje automático** para que sean capaces de clasificar imágenes de caras. Se han utilizado dos bases de datos, la de YaleB, con solo 37 clases  y 2480 fotos, siendo la más pequeña, y una versión adaptada de VGGFaces2, con un total de 534 clases y 94600 muestras en total, siendo además esta última base de datos desequilibrada.
Para dar una aplicación práctica a los modelos ademas se ha desarrollado un programa con una interfaz gráfica capaz de hacer reconocimiento facial en vivo mediante el uso de una camara. La interfaz además permite editar la base de datos, añadir nuevas clases y elegir entre los dos modelos de PCA para ver cual funciona mejor. Como ejemplo se ha añadido una base de datos para uso exclusivo de la cámara donde ya hay añadidas unas 40 clases, por lo que el usuario puede añadir a esta fotos suyas para ver si los modelos son capaz de reconocerle.

<p align="center">
<img width="590" alt="Celebrity" src="https://github.com/user-attachments/assets/7048461f-d1c8-4f0d-9886-70818616bc42" />
</p>

## Uso

En el directorio raiz se encuentra el archivo `mediapipe_facedetector.py`, que es con el cual se han preprocesado las caras de la base de datos de VGGFaces2.

El directorio `BD_camara` solo está la base de datos usada en la cámara, debido a que las que se usan en el trabajo son de tamaño demasiado grande. Estas en cambio se pueden descargar en los siguientes enlaces: http://cvc.cs.yale.edu/cvc/projects/yalefacesB/yalefacesB.html y https://www.kaggle.com/datasets/hearfool/vggface2 

Dentro del directorio `modelos` se encuentran las funciones usadas en los dos metodos basados en PCA junto con los `main` donde se ejecutan, y también el programa a partir del cual se entrenó la red neuronal convolucional. Asi los modelos usados son los siguientes:
1. Eigenfaces
2. Eigenfaces Bayesiano
3. Redes neuronales convolucionales.
   
Para cada modelo de PCA primero se ha realizado una validación cruzada. Las funciones usadas en los dos metodos basados en PCA se encuentran conjuntos en el archivo `eigenfaces_utils.py`, junto con la descripción detallada de lo que hace cada función. La dinamica descrita para llevar a cabo el reconocimiento se encuentra en los archivos `eigenfaces_main.py` y `bayes_main.py`.
Para la CNN se hace todo en un mismo archivo. Si el usuario quiere probar a entrenar la red se recomienda hacerlo en un servidor con GPUs dedicadas ya que sino el tiempo de ejecución sería muy largo.

El la carpeta `resultados` se pueden ver las figuras consegidas tanto como en las validación cruzadas y ajustes de hiperparametros que se han hecho, como los resultados de exactitud y perdida conseguidos época a época en el caso del modelo de CNN.

## Detección en vivo

También en el directorio raiz, se encuentran los archivos `camara_haar.py` y `camara_mp.py`, que pertenecen a la aplicación grafica de identificación facial creada para el trabajo, con la única diferencia del modelo de detección de caras usado en ellos; en el primero se usa el detector Haar Cascade, y en el segundo el más avanzado Mediapipe. Para un correcto funcionamiento del programa tambien hay que descargar el archivo de `eigenfaces_utils.py`, ya que el programa hace uso de él para llevar a cabo la identificación.

Si se quiere usar la misma base de datos que la del anexo, también se puede descargar desde el directorio raiz, para a partir de ella añadir los sujetos que se quieran y probar a identificarlos usando el archivo de python.

En definitiva, para poder lanzar la aplicación el directorio debedería de tener la siguiente forma:

camara_mp.py

camara_haar.py

models

   &ensp; &emsp; --eigenfaces_utils.py

BD_Camara

   &emsp; &emsp; --sujeto1

   &emsp; &emsp; --sujeto2

   &emsp; &emsp; ...


## Cuaderno
En el directorio raiz se ha preparado el cuaderno de jupyter `modelos-reconocimiento-facial.ipynb`, donde se ve paso a paso como funcionan cada uno de los metodos.
Aunque en realidad el cuaderno conste del codigo de los `main`, para el caso de la CNN se ha incluido todo el proceso de entrenamiento de la red, y se puede observar época a época como evoluciona el aprendizaje, además de como se incluyen las capas, aumentación de datos, etc.

## Paquetes necesarios

En este trabajo se ha utilizado `Python 3.11.3`. Para el trabajo se han usado los siguientes paquetes:
```
Paquete                       Versión 
---------------------------- -------------- 
matplotlib                    3.8.0 
numpy                         1.23.5
scikit-learn                  1.2.0
tensorflow                    2.12.0 
keras                         2.11.0 
opencv-python                 4.8.0 
scipy                         1.11.2 
mediapipe                     0.9.1 
```

Para elaborar la interfaz gráfica del programa de reconocimiento en vivo, se han usado los siguientes paquetes, por lo que si se quiere probar a ejecutarlo sería suficiente con descargar solo estos. Dependiendo del metodo de detección de caras a usar, habrá que descargar la librería open cv o la de mediapipe:
```
Paquete                       Versión
----------------------------- --------------
numpy                         1.23.5
scikit-learn                  1.2.0
PyQt5                         5.15.9

opencv-python                 4.8.0

mediapipe                     0.9.1
```
