import cv2
import face_recognition
import os
import numpy as np

# Ruta donde están guardadas las imágenes de los rostros
save_path = '../faces'
known_face_encodings = []
known_face_names = []

# Cargar imágenes y extraer los encodings de los rostros
for filename in os.listdir(save_path):
    if filename.endswith(".jpg"):
        # Cargar la imagen y extraer el encoding
        image_path = os.path.join(save_path, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Intentar obtener el encoding
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            encoding = encodings[0]
            # Agregar el encoding y el nombre a las listas
            known_face_encodings.append(encoding)
            # El nombre es el nombre del archivo sin la extensión y el número
            name = filename.split('_')[0]
            known_face_names.append(name)
        else:
            print(f"No se encontró un rostro en la imagen: {filename}")

# Iniciar la webcam
webcam = cv2.VideoCapture(0)

while True:
    # Leer imagen de la webcam
    (_, frame) = webcam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar ubicaciones de rostros y calcular los encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Comparar cada rostro detectado con los encodings conocidos
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        # Si se encuentra una coincidencia
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        # Escribir el nombre de la persona
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Mostrar la imagen en tiempo real
    cv2.imshow('OpenCV', frame)

    # Esperar 10ms y salir si se presiona la tecla 'ESC'
    key = cv2.waitKey(10)
    if key == 27:
        break

# Liberar la cámara y cerrar todas las ventanas
webcam.release()
cv2.destroyAllWindows()
