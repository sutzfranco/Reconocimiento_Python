import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import threading
#importo bibliotecas

save_path = '../faces'
known_face_encodings = []
known_face_names = []
#almacenara los encodings de los rostros y los nombres


for filename in os.listdir(save_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(save_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        #recorro la carpeta faces para encontrar archivos jpg y extraer los encodings 
        if len(encodings) > 0:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            name = filename.split('_')[0]  
            known_face_names.append(name)
        else:
            print(f"No se encontró un rostro en la imagen: {filename}")
        #este if pregunta que si se encontro una cara guarde el encoding y el nombre


def registrar_evento(nombre, evento):
    now = datetime.now()
    fecha_hora = now.strftime("%Y-%m-%d %H:%M:%S") 
    if evento == "Entrada":
        archivo = "registro_entradas.txt"
    elif evento == "Salida":
        archivo = "registro_salidas.txt"
        
    with open(archivo, "a") as file:
        file.write(f"{fecha_hora} - {nombre} - {evento}\n")
#funcion que le inserto la fecha el nombre y el evento
def mostrar_botones(name, stop_camera_event):
    root = tk.Tk()
    root.title(f"Registrar asistencia para {name}")
    #creo la ventana
    def marcar_entrada():
        registrar_evento(name, "Entrada")
        messagebox.showinfo("Registro", f"Ingreso registrado para {name}")
        stop_camera_event.set() 
        root.destroy()
    def marcar_salida():
        registrar_evento(name, "Salida")
        messagebox.showinfo("Registro", f"Salida registrada para {name}")
        stop_camera_event.set()
        root.destroy()
#funcion de entrada y salida
   
    btn_ingreso = tk.Button(root, text="Marcar Ingreso", command=marcar_entrada, width=25, height=2)
    btn_ingreso.pack(pady=10)

    btn_salida = tk.Button(root, text="Marcar Salida", command=marcar_salida, width=25, height=2)
    btn_salida.pack(pady=10)
    #botones
    root.mainloop()
#funcion que muestra los botones de entrada y salida

#funcion para el reconocimiento facial
def reconocimiento_facial(stop_camera_event):
    webcam = cv2.VideoCapture(0)
    recognized_names = set()
    while not stop_camera_event.is_set():
        #lee la imagen de la camara web
        _, frame = webcam.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #detecta los rostros y calcula los encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        #compara los rostros con los encodigns
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            print(face_encoding) 
            name = "Desconocido"

            #calcula las distancia de los encodings de la camara con los conocidos
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                #muestra los botones
                if name not in recognized_names:
                    recognized_names.add(name)
                    threading.Thread(target=mostrar_botones, args=(name, stop_camera_event)).start()

        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        #dibuja un rectangulo en cada rostro que detecta
       
        cv2.imshow('OpenCV - Reconocimiento Facial', frame)
         #muestra la imagen en tiempo real

        key = cv2.waitKey(10)
        if key == 27:
            stop_camera_event.set()  
            break

    
    webcam.release()
    cv2.destroyAllWindows()


stop_camera_event = threading.Event()

# Ejecutar el reconocimiento facial en un hilo
threading.Thread(target=reconocimiento_facial, args=(stop_camera_event,)).start()
