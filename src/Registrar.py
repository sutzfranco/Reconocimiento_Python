import cv2
import os

# Cargar el archivo de clasificación de rostros
file = "../frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)

# Iniciar la webcam
webcam = cv2.VideoCapture(0)

# Ruta donde se guardarán las imágenes de los rostros
save_path = "../faces/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Nombre del usuario que estás registrando
person_name = input("Ingresa el nombre de la persona: ")

# Contador para el número de fotos guardadas
count = 0

while True:
    # Leer imagen de la webcam
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor de la cara
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Guardar la imagen recortada de la cara
        face_img = im[y:y+h, x:x+w]
        count += 1
        img_name = f"{save_path}{person_name}_{count}.jpg"
        cv2.imwrite(img_name, face_img)
        print(f"Guardando imagen {img_name}")

    # Mostrar la imagen en tiempo real
    cv2.imshow('OpenCV', im)

    # Esperar 10ms y salir si se presiona la tecla 'ESC'
    key = cv2.waitKey(10)
    if key == 27 or count >= 10:  # Captura 10 fotos
        break

# Liberar la cámara y cerrar todas las ventanas
webcam.release()
cv2.destroyAllWindows()
