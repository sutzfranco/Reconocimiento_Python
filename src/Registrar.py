import cv2
import os

file = "../frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)
#cargo el clasificador de rostro en la variable

webcam = cv2.VideoCapture(0)


save_path = "../faces/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
#defino la ruta donde voy a guardar las imagenes, sino existe lo creo

person_name = input("Ingresa el nombre de la persona: ")
#le pido el nombre de la persona que toma la imagen

count = 0

while True:
#empiezo un bucle hasta que se tomen 10 fotos    
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #Capturo la imagen, convierto la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray)
    #aca devuelvo la lista de coordenadas que que te dice donde esta el rostro
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #hace un rectangulo por cada rostro que reconoce

        # Guardar la imagen recortada de la cara
        face_img = im[y:y+h, x:x+w]
        count += 1
        img_name = f"{save_path}{person_name}_{count}.jpg"
        cv2.imwrite(img_name, face_img)
        print(f"Guardando imagen {img_name}")

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27 or count >= 10:  
        break
    #espera 10 milisegundos para cancelar osi llega a 10 fotos el bucle se detiene

webcam.release()
cv2.destroyAllWindows()
#libero la camara y cierro las ventanas