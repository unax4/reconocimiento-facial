import cv2
import numpy as np
import os
import mediapipe as mp
import random
from PIL import Image
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
from threading import Thread, Lock
import sys
import mediapipe as mp  
from models.eigenfaces_utils import predict_single_image,compute_eigenfaces, project_images, calculate_class_statistics, predict_single_image_bayes

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def load_images(database_path, training_ratio, image_size):
    train_images, train_labels, test_images, test_labels = [], [], [], []
    
    for folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, folder)
        if os.path.isdir(folder_path):
            person_id = folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith((".pgm", ".jpg", ".png"))]
            random.shuffle(image_files)
            num_train = round(len(image_files) * training_ratio)

            for i, img_file in enumerate(image_files):
                try:
                    img = np.array(Image.open(os.path.join(folder_path, img_file)).convert("L").resize((image_size[1], image_size[0])))
                    if i < num_train:
                        train_images.append(img)
                        train_labels.append(person_id)
                    else:
                        test_images.append(img)
                        test_labels.append(person_id)
                except Exception as e:
                    print(f"Error al cargar la imagen {img_file}: {e}")
                    continue

    print(f"Cargadas {len(train_images)} imágenes de entrenamiento, {len(train_labels)} etiquetas de entrenamiento")
    print(f"Cargadas {len(test_images)} imágenes de prueba, {len(test_labels)} etiquetas de prueba")

    return (np.array(train_images), np.array(test_images), 
            np.array(train_labels), np.array(test_labels))

class FaceRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FAplicación de Reconocimiento Facial")
        self.setGeometry(100, 100, 800, 600)

        self.database_path = "BD_camara" 
        self.image_size = (112, 92)
        self.num_components = 4
        self.prob_threshold = 0.5
        self.running = False
        self.model_type = "Bayesiano"
        self.model_lock = Lock()
        self.prediction_buffer = deque(maxlen=5)

        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        self.load_model()

        # Main widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Webcam display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label)

        # Control panel
        self.control_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)

        # Model selection
        self.model_group = QtWidgets.QGroupBox("Selección de Modelo")
        self.model_layout = QtWidgets.QHBoxLayout()
        self.model_group.setLayout(self.model_layout)
        self.bayesian_radio = QtWidgets.QRadioButton("Bayesiano")
        self.pca_radio = QtWidgets.QRadioButton("PCA")
        self.bayesian_radio.setChecked(True)
        self.model_layout.addWidget(self.bayesian_radio)
        self.model_layout.addWidget(self.pca_radio)
        self.control_layout.addWidget(self.model_group)
        self.bayesian_radio.toggled.connect(self.update_model)
        self.pca_radio.toggled.connect(self.update_model)

        # Components selection
        self.components_group = QtWidgets.QGroupBox("Número de Componentes")
        self.components_layout = QtWidgets.QHBoxLayout()
        self.components_group.setLayout(self.components_layout)
        self.components_label = QtWidgets.QLabel("Componentes:")
        self.components_entry = QtWidgets.QSpinBox()
        self.components_entry.setRange(1, 100)
        self.components_entry.setValue(self.num_components)
        self.apply_components_button = QtWidgets.QPushButton("Aplicar")
        self.components_layout.addWidget(self.components_label)
        self.components_layout.addWidget(self.components_entry)
        self.components_layout.addWidget(self.apply_components_button)
        self.control_layout.addWidget(self.components_group)
        self.apply_components_button.clicked.connect(self.update_components)

        # Buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        self.launch_button = QtWidgets.QPushButton("Iniziar Cámara")
        self.manage_db_button = QtWidgets.QPushButton("Gestionar Base de Datos")
        self.button_layout.addWidget(self.launch_button)
        self.button_layout.addWidget(self.manage_db_button)
        self.launch_button.clicked.connect(self.launch_camera)
        self.manage_db_button.clicked.connect(self.open_db_manager)

    def load_model(self):
        try:
            self.train_images, self.test_images, self.train_labels, self.test_labels = load_images(
                self.database_path, 1.0, self.image_size)
            self.train_images = np.array(self.train_images)

            if len(self.train_images) == 0:
                raise ValueError("No se cargaron imágenes desde la base de datos")
            if len(self.train_images) != len(self.train_labels):
                raise ValueError("Desajuste entre el número de imágenes de entrenamiento y etiquetas")
            if not all(isinstance(label, str) for label in self.train_labels):
                raise ValueError("Las etiquetas de entrenamiento contienen valores no válidos")

            self.retrain_model()

        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            QtWidgets.QMessageBox.warning(self, "Advertencia", f"No se pudo cargar el modelo: {e}")
            self.train_images = np.array([])
            self.train_labels = np.array([])
            self.test_images = np.array([])
            self.test_labels = np.array([])
            self.eigenfaces = None
            self.mean = None
            self.trainE = None
            self.class_means = {}
            self.class_covariances = {}

    def retrain_model(self):
        # Reentrenar el modelo con los datos cargados
        with self.model_lock:
            if self.train_images.size == 0:
                QtWidgets.QMessageBox.warning(self, "Advertencia", "No hay datos de entrenamiento disponibles")
                self.eigenfaces = None
                self.mean = None
                self.trainE = None
                self.class_means = {}
                self.class_covariances = {}
                return

            if self.num_components > len(self.train_images):
                raise ValueError("El número de componentes excede el número de imágenes de entrenamiento")

            self.eigenfaces, self.mean = compute_eigenfaces(self.train_images, self.image_size)
            if self.num_components > self.eigenfaces.shape[0]:
                raise ValueError("El número de componentes excede el número de eigenfaces")

            self.trainE = project_images(self.train_images, self.mean, 
                                        self.eigenfaces, self.num_components, self.image_size)
            self.class_means = {}
            self.class_covariances = {}
            if self.model_type == "Bayesiano":
                self.class_means, self.class_covariances = calculate_class_statistics(
                    self.trainE, self.train_labels)

    def update_model(self):
        # Actualizar el tipo de modelo (Bayesiano o PCA)
        self.model_type = "Bayesiano" if self.bayesian_radio.isChecked() else "PCA"
        self.retrain_model()
        QtWidgets.QMessageBox.information(self, "Información", f"Cambiado al modelo {self.model_type}")

    def update_components(self):
        # Actualizar el número de componentes para el modelo
        try:
            new_components = self.components_entry.value()
            if new_components < 1:
                raise ValueError("El número de componentes debe ser positivo")
            if self.train_images.size > 0 and new_components > len(self.train_images):
                raise ValueError("El número de componentes no puede exceder el número de imágenes de entrenamiento")
            if self.eigenfaces is not None and new_components > self.eigenfaces.shape[0]:
                raise ValueError("El número de componentes no puede exceder el número de eigenfaces")
            self.num_components = new_components
            self.retrain_model()
            QtWidgets.QMessageBox.information(self, "Información", f"Actualizado a {self.num_components} componentes")
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Entrada no válida: {e}")

    def process_face(self, face_resized):
        with self.model_lock:
            if self.eigenfaces is None or self.mean is None:
                return "Desconocido", None

            if self.model_type == "Bayesiano":
                label, probs = predict_single_image_bayes(
                    face_resized, self.mean, self.eigenfaces,
                    self.class_means, self.class_covariances,
                    self.image_size, self.num_components
                )
                probs_exp = {k: np.exp(v) for k, v in probs.items()}
                total = sum(probs_exp.values())
                probs_percent = {k: (v / total) * 100 for k, v in probs_exp.items()}
                label = max(probs_percent, key=probs_percent.get)
                confidence = probs_percent[label] / 100
                predicted_label = label if confidence >= self.prob_threshold else "Desconocido"
            else:
                label = predict_single_image(
                    face_resized, self.mean, self.eigenfaces,
                    self.trainE, self.train_labels,
                    self.image_size, self.num_components
                )
                predicted_label = label if label else "Desconocido"
                confidence = None

            return predicted_label, confidence

    def smooth_prediction(self, current_label, current_confidence):
        if current_label != "Desconocido":
            self.prediction_buffer.append((current_label, current_confidence))
        
        if len(self.prediction_buffer) == 0:
            return current_label, current_confidence

        labels = [label for label, _ in self.prediction_buffer]
        confidences = [conf for _, conf in self.prediction_buffer if conf is not None]

        most_common_label = max(set(labels), key=labels.count)
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return most_common_label, avg_confidence

    def detect_faces_mediapipe(self, frame):
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)
                faces.append((x, y, w_box, h_box))
        return faces

    def camera_loop(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")

            self.running = True
            frame_count = 0
            detection_interval = 5
            last_result = []

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame de la cámara")
                    break

                if frame_count % detection_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detect_faces_mediapipe(frame)
                    last_result = []

                    for (x, y, w, h) in faces:
                        x, y = max(0, x), max(0, y)
                        face = gray[y:y+h, x:x+w]
                        if face.size == 0:
                            continue
                        face_resized = cv2.resize(face, (self.image_size[1], self.image_size[0]))
                        predicted_label, confidence = self.process_face(face_resized)
                        predicted_label, confidence = self.smooth_prediction(predicted_label, confidence)
                        last_result.append(((x, y, w, h), predicted_label, confidence))
                    if not faces:
                        self.prediction_buffer.clear()

                for (x, y, w_box, h_box), name, conf in last_result:
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    if conf is not None:
                        cv2.putText(frame, f"prob={round(conf, 2)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio))

                QtWidgets.QApplication.processEvents()
                frame_count += 1

        except Exception as e:
            print(f"Error de la cámara: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error de la cámara: {e}")
        finally:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.launch_button.setText("Iniciar Cámara")

    def launch_camera(self):
        # Iniciar o detener la cámara
        if not self.running:
            self.launch_button.setText("Detener Cámara")
            self.camera_thread = Thread(target=self.camera_loop)
            self.camera_thread.start()
        else:
            self.running = False
            self.launch_button.setText("Iniciar Cámara")

    def open_db_manager(self):
        # Abrir la ventana de gestión de la base de datos
        self.db_window = QtWidgets.QDialog(self)
        self.db_window.setWindowTitle("Gestionar Base de Datos")
        self.db_window.setGeometry(200, 200, 500, 400)
        layout = QtWidgets.QVBoxLayout()

        # Añadir nueva persona
        add_group = QtWidgets.QGroupBox("Añadir Nueva Persona")
        add_layout = QtWidgets.QFormLayout()
        self.name_entry = QtWidgets.QLineEdit()
        self.image_list = QtWidgets.QListWidget()
        add_images_button = QtWidgets.QPushButton("Añadir Imágenes")
        submit_button = QtWidgets.QPushButton("Enviar")
        add_layout.addRow("Nombre:", self.name_entry)
        add_layout.addRow(self.image_list)
        add_layout.addRow(add_images_button)
        add_layout.addRow(submit_button)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        add_images_button.clicked.connect(self.add_images)
        submit_button.clicked.connect(self.submit_new_subject)

        # Lista de etiquetas y número de fotos
        labels_group = QtWidgets.QGroupBox("Lista de Etiquetas")
        labels_layout = QtWidgets.QVBoxLayout()
        self.labels_list = QtWidgets.QListWidget()

        # Obtener las etiquetas y contar las imágenes
        for label in self.class_means.keys():
            folder_path = os.path.join(self.database_path, label)
            if os.path.isdir(folder_path):
                num_images = len([f for f in os.listdir(folder_path) if f.endswith((".pgm", ".jpg", ".png"))])
                self.labels_list.addItem(f"{label}: {num_images} fotos")
        
        labels_layout.addWidget(self.labels_list)
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)

        self.db_window.setLayout(layout)
        self.db_window.exec_()

    def add_images(self):
        # Añadir imágenes al listado para un nuevo sujeto
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Seleccionar Imágenes", "", "Imágenes (*.jpg *.png)")
        for file in files:
            self.image_list.addItem(file)

    def submit_new_subject(self):
        label = self.name_entry.text()
        images = [self.image_list.item(i).text() for i in range(self.image_list.count())]
        if not label:
            QtWidgets.QMessageBox.critical(self, "Error", "Por favor, introduce un nombre")
            return
        if len(images) < 5:
            QtWidgets.QMessageBox.critical(self, "Error", "Por favor, selecciona al menos 5 imágenes")
            return

        os.makedirs(os.path.join(self.database_path, label), exist_ok=True)
        new_images = []
        new_labels = []

        for i, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces_mediapipe(img)
            if len(faces) == 0:
                print(f"No se detectó rostro en {img_path}")
                continue

            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            face_resized = cv2.resize(face_roi, (self.image_size[1], self.image_size[0]))
            new_images.append(face_resized)
            new_labels.append(label)
            cv2.imwrite(os.path.join(self.database_path, label, f"img_{i}.jpg"), face_resized)

        if len(new_images) >= 5:
            new_images_array = np.array(new_images)
            with self.model_lock:
                if self.train_images.size == 0:
                    self.train_images = new_images_array
                    self.train_labels = np.array(new_labels)
                else:
                    self.train_images = np.concatenate((self.train_images, new_images_array), axis=0)
                    self.train_labels = np.concatenate((self.train_labels, np.array(new_labels)), axis=0)
                self.retrain_model()
            QtWidgets.QMessageBox.information(self, "Éxito", f"Sujeto {label} añadido correctamente")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "No se detectaron suficientes imágenes de rostros válidas (se necesitan al menos 5)")


    def closeEvent(self, event):
        self.running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.face_detector.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
