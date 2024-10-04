import cv2
import face_recognition
import os
import numpy as np

save_path = '../faces'
known_face_encodings = []
known_face_names = []

for filename in os.listdir(save_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(save_path, filename)
        image = face_recognition.load_image_file(image_path)
        
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            name = filename.split('_')[0]
            known_face_names.append(name)
        else:
            print(f"No se encontró un rostro en la imagen: {filename}")

webcam = cv2.VideoCapture(0)

while True:
    (_, frame) = webcam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('OpenCV', frame)

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
