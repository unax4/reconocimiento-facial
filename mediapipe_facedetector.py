import os
import cv2
import shutil
import mediapipe as mp
from PIL import Image


output_path = "C:/Users/Unax/Desktop/LegacyTFG/Database/VGGFace"  # Base de datos original
database_path = "D:/Universidad/TFG IE/Databases/VGG/train"  # Nueva base de datos con caras
min_faces_required = 40  # Mínimo de imágenes necesarias para mantener a la persona

mp_face_detection = mp.solutions.face_detection

def detectar_y_recortar_cara(image_path):
    """Detecta la cara en una imagen y devuelve solo la región de la cara."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # MediaPipe requiere imágenes en formato RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            results = face_detection.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape

                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    box_width = int(bboxC.width * w)
                    box_height = int(bboxC.height * h)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_min + box_width)
                    y_max = min(h, y_min + box_height)

                    # Recortar la cara
                    face = img[y_min:y_max, x_min:x_max]

                    if face.size == 0:
                        return None
                    
                    return face

    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def procesar_database(database_path, output_path):
    """Recorre la base de datos y extrae las caras de todas las imágenes."""
    os.makedirs(output_path, exist_ok=True)

    for person in os.listdir(database_path):
        person_dir = os.path.join(database_path, person)
        output_person_dir = os.path.join(output_path, person)

        if not os.path.isdir(person_dir):
            continue

        os.makedirs(output_person_dir, exist_ok=True)
        images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        face_count = 0 

        for img in images:
            img_path = os.path.join(person_dir, img)
            face = detectar_y_recortar_cara(img_path)
            
            if face is not None:
                h, w = face.shape[:2]  # Obtener dimensiones de la cara detectada
                if h >= 100 and w >= 100:
                    save_path = os.path.join(output_person_dir, img)
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_pil.save(save_path)
                    face_count += 1

        if face_count < min_faces_required:
            shutil.rmtree(output_person_dir, ignore_errors=True)
            print(f"Eliminado {person} (solo {face_count} caras detectadas)")
        else:
            print(f"Guardado {person} ({face_count} caras detectadas)")

    print("Base de datos reconstruida con solo caras válidas en:", output_path)

if __name__ == "__main__":
    procesar_database(database_path, output_path)
