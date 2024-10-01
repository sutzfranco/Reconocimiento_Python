import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import threading

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
            name = filename.split('_')[0]  # El nombre es el archivo sin extensión
            known_face_names.append(name)
        else:
            print(f"No se encontró un rostro en la imagen: {filename}")

# Función para registrar la entrada o salida en un archivo de texto
def registrar_evento(nombre, evento):
    now = datetime.now()  # Obtener la fecha y hora actual
    fecha_hora = now.strftime("%Y-%m-%d %H:%M:%S")  # Formatear la fecha y hora
    if evento == "Entrada":
        archivo = "registro_entradas.txt"
    elif evento == "Salida":
        archivo = "registro_salidas.txt"
        
    with open(archivo, "a") as file:
        file.write(f"{fecha_hora} - {nombre} - {evento}\n")

# Función que muestra los botones para registrar ingreso o salida
def mostrar_botones(name, stop_camera_event):
    # Crear la ventana
    root = tk.Tk()
    root.title(f"Registrar asistencia para {name}")

    # Función para registrar entrada y cerrar la ventana
    def marcar_entrada():
        registrar_evento(name, "Entrada")
        messagebox.showinfo("Registro", f"Ingreso registrado para {name}")
        stop_camera_event.set()  # Detener el thread de la cámara
        root.destroy()

    # Función para registrar salida y cerrar la ventana
    def marcar_salida():
        registrar_evento(name, "Salida")
        messagebox.showinfo("Registro", f"Salida registrada para {name}")
        stop_camera_event.set()  # Detener el thread de la cámara
        root.destroy()

    # Botones para marcar ingreso o salida
    btn_ingreso = tk.Button(root, text="Marcar Ingreso", command=marcar_entrada, width=25, height=2)
    btn_ingreso.pack(pady=10)

    btn_salida = tk.Button(root, text="Marcar Salida", command=marcar_salida, width=25, height=2)
    btn_salida.pack(pady=10)

    root.mainloop()

# Función para ejecutar el reconocimiento facial
def reconocimiento_facial(stop_camera_event):
    # Iniciar la webcam
    webcam = cv2.VideoCapture(0)
    recognized_names = set()  # Set para almacenar nombres ya reconocidos

    while not stop_camera_event.is_set():
        # Leer imagen de la webcam
        _, frame = webcam.read()
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

                # Mostrar los botones solo si no se ha mostrado antes para esta persona
                if name not in recognized_names:
                    recognized_names.add(name)
                    threading.Thread(target=mostrar_botones, args=(name, stop_camera_event)).start()

        # Dibujar un rectángulo alrededor del rostro
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Mostrar la imagen en tiempo real
        cv2.imshow('OpenCV - Reconocimiento Facial', frame)

        # Salir si se presiona la tecla 'ESC'
        key = cv2.waitKey(10)
        if key == 27:
            stop_camera_event.set()  # Detener el hilo
            break

    # Liberar la cámara y cerrar todas las ventanas
    webcam.release()
    cv2.destroyAllWindows()

# Crear un evento para detener la cámara
stop_camera_event = threading.Event()

# Ejecutar el reconocimiento facial en un hilo
threading.Thread(target=reconocimiento_facial, args=(stop_camera_event,)).start()
