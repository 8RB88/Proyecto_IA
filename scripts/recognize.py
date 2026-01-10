import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import simpledialog

import cv2
import face_recognition
from tkinter import messagebox

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENC_FILE = DATA_DIR / "known_encodings.pkl"
TRAIN_DIR = DATA_DIR / "train"

# Configuracion rapida
MODEL = "hog"  # "hog" rapido CPU; "cnn" requiere GPU/CUDA
DOWNSCALE = 0.75  # 1.0 sin cambio; 0.5 reduce a mitad para mas FPS
TOLERANCE = 0.45  # menor = mas estricto; mayor = mas permisivo
CAPTURE_COUNT = 5  # Numero de fotos a capturar al aprender


def load_encodings():
    if not ENC_FILE.exists():
        return {"encodings": [], "names": []}
    with open(ENC_FILE, "rb") as f:
        return pickle.load(f)


def save_encodings(encodings, names):
    data = {"encodings": encodings, "names": names}
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)


def save_face_image(frame, box, label):
    top, right, bottom, left = box
    face = frame[top:bottom, left:right]
    dest_dir = TRAIN_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = dest_dir / f"{ts}.jpg"
    cv2.imwrite(str(out_path), face)
    return out_path


def rebuild_encodings_from_train():
    encodings = []
    names = []
    for person_dir in TRAIN_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for img_path in person_dir.glob("*.*"):
            image = face_recognition.load_image_file(img_path)
            boxes = face_recognition.face_locations(image, model="hog")
            if not boxes:
                print(f"Sin rostro en {img_path}, se omite.")
                continue
            face_encs = face_recognition.face_encodings(image, boxes)
            encodings.extend(face_encs)
            names.extend([label] * len(face_encs))
    save_encodings(encodings, names)
    return encodings, names


def capture_training_photos(video, label, capture_count=5):
    """
    Captura fotos controladas de múltiples ángulos con instrucciones visuales.
    Pide al usuario que mire hacia: frente, derecha, izquierda, arriba, abajo.
    Reintentas automáticamente si no detecta rostro.
    """
    angles = [
        {"name": "Frente", "instruction": "Mira AL FRENTE", "color": (0, 255, 0)},
        {"name": "Derecha", "instruction": "Gira a la DERECHA", "color": (255, 165, 0)},
        {"name": "Izquierda", "instruction": "Gira a la IZQUIERDA", "color": (255, 165, 0)},
        {"name": "Arriba", "instruction": "Mira hacia ARRIBA", "color": (255, 100, 255)},
        {"name": "Abajo", "instruction": "Mira hacia ABAJO", "color": (255, 100, 255)},
    ]
    
    captured = 0
    angle_idx = 0
    frames_in_angle = 0
    consecutive_no_face = 0
    max_retries_per_angle = 150  # ~5 segundos a 30 FPS
    frames_per_capture = 20  # Capturar 1 foto cada 20 frames (0.67s) en ese ángulo
    
    print(f"Capturando {capture_count} fotos de {label} desde múltiples ángulos...")
    print("Sigue las instrucciones en pantalla. Mantén el rostro visible y bien iluminado.\n")
    
    while captured < capture_count and angle_idx < len(angles):
        ret, frame = video.read()
        if not ret:
            continue
        
        current_angle = angles[angle_idx]
        
        # Procesar frame para detección
        proc_frame = frame
        if DOWNSCALE != 1.0:
            proc_frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        
        rgb_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        rgb_proc = np.ascontiguousarray(rgb_proc, dtype=np.uint8)
        boxes_proc = face_recognition.face_locations(rgb_proc, model=MODEL)
        
        h, w = frame.shape[:2]
        
        # Dibujar instrucción grande en el centro
        cv2.putText(
            frame, 
            current_angle["instruction"], 
            (w // 2 - 200, h // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            current_angle["color"], 
            3
        )
        
        # Mostrar progreso
        progress_text = f"Foto {captured}/{capture_count} | Angulo {angle_idx + 1}/{len(angles)}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar estado del ángulo actual
        if boxes_proc:
            remaining_frames = max(0, frames_per_capture - frames_in_angle)
            countdown = max(1, remaining_frames // 6)  # Convertir frames a segundos aproximados
            countdown_text = f"✓ Rostro OK | Captura en {countdown}s"
            cv2.putText(frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            consecutive_no_face = 0
        else:
            consecutive_no_face += 1
            time_in_angle_sec = frames_in_angle // 30
            countdown_text = f"✗ Sin rostro ({time_in_angle_sec}s) | Reintentando..."
            cv2.putText(frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Si lleva mucho tiempo sin detectar rostro, cambiar de ángulo
            if frames_in_angle >= max_retries_per_angle:
                print(f"  ⚠ No se detectó rostro en ángulo '{current_angle['name']}' después de {max_retries_per_angle // 30}s. Saltando...")
                angle_idx += 1
                frames_in_angle = 0
                consecutive_no_face = 0
                continue
        
        # Mostrar instrucción de controles
        cv2.putText(
            frame, 
            "ESC: Saltar angulo | q: Cancelar", 
            (10, h - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (100, 200, 255), 
            2
        )
        
        cv2.imshow("Reconocimiento - Entrenamiento", frame)
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS
        
        if key == ord("q"):
            print("\nEntrenamiento cancelado por el usuario.")
            cv2.destroyAllWindows()
            return captured
        
        if key == 27:  # ESC
            print(f"Saltando ángulo '{current_angle['name']}'.")
            angle_idx += 1
            frames_in_angle = 0
            consecutive_no_face = 0
            continue
        
        # Capturar foto si hay rostro y pasó el tiempo
        if boxes_proc and frames_in_angle >= frames_per_capture:
            if DOWNSCALE != 1.0:
                inv = 1.0 / DOWNSCALE
                top, right, bottom, left = boxes_proc[0]
                box_full = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
            else:
                box_full = boxes_proc[0]
            
            save_face_image(frame, box_full, label)
            captured += 1
            print(f"  Foto {captured}/{capture_count} capturada ({current_angle['name']})")
            
            # Siguiente ángulo
            angle_idx += 1
            frames_in_angle = 0
        else:
            frames_in_angle += 1
    
    cv2.destroyAllWindows()
    if captured > 0:
        print(f"\n✓ Completado: {captured} fotos capturadas desde múltiples ángulos.\n")
    return captured


def main():
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]
    tolerance = TOLERANCE

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        proc_frame = frame
        scale = DOWNSCALE
        if scale != 1.0:
            proc_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        boxes = face_recognition.face_locations(rgb, model=MODEL)
        encs = face_recognition.face_encodings(rgb, boxes)

        # Escala las cajas de regreso al tamano original para dibujar y recortar.
        if scale != 1.0:
            boxes_full = []
            inv = 1.0 / scale
            for (top, right, bottom, left) in boxes:
                boxes_full.append(
                    (
                        int(top * inv),
                        int(right * inv),
                        int(bottom * inv),
                        int(left * inv),
                    )
                )
            boxes = boxes_full

        for (top, right, bottom, left), enc in zip(boxes, encs):
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=tolerance)
            name = "Desconocido"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "q: salir  a: aprender/guardar", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"tol {tolerance:.2f}  -/+ ajusta  r: reforzar", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Reconocimiento", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("-"):
            tolerance = max(0.20, round(tolerance - 0.02, 2))
            print(f"Tolerancia ahora {tolerance:.2f} (mas estricto)")
        if key == ord("+") or key == ord("="):
            tolerance = min(0.80, round(tolerance + 0.02, 2))
            print(f"Tolerancia ahora {tolerance:.2f} (mas permisivo)")
        if key == ord("r"):
            if not encs:
                print("No hay rostro en cuadro para reforzar.")
                continue
            # Verificar si el rostro actual está registrado
            matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=tolerance)
            if True not in matches:
                print("Rostro no reconocido. Usa 'a' para aprender un rostro nuevo.")
                continue
            
            idx = matches.index(True)
            existing_name = known_names[idx]
            
            root = tk.Tk()
            root.withdraw()
            confirm = simpledialog.askstring(
                "Reforzar Reconocimiento", 
                f"Detectado: {existing_name}\n¿Capturar más fotos? (s/n):", 
                parent=root
            )
            root.destroy()
            
            if not confirm or confirm.lower() not in ['s', 'si', 'yes', 'y']:
                print("Refuerzo cancelado.")
                continue
            
            # Captura múltiple controlada con ángulos
            captured = capture_training_photos(video, existing_name, CAPTURE_COUNT)
            
            if captured == 0:
                print("No se pudo capturar ninguna foto. Intenta de nuevo.")
                continue
            
            # Reentrena con las nuevas fotos
            known_encodings, known_names = rebuild_encodings_from_train()
            print(f"Reforzado {existing_name} con {captured} fotos adicionales. Modelo actualizado.")
        if key == ord("a"):
            if not encs:
                print("No hay rostro en cuadro para aprender.")
                continue
            
            # Verificar si el rostro ya está registrado
            matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=tolerance)
            if True in matches:
                idx = matches.index(True)
                existing_name = known_names[idx]
                root = tk.Tk()
                root.withdraw()
                messagebox = __import__('tkinter').messagebox
                messagebox.showinfo(
                    "Usuario Ya Registrado",
                    f"El rostro detectado ya está registrado como:\n\n'{existing_name}'\n\nUsa 'r' para reforzar su modelo."
                )
                root.destroy()
                print(f"Rostro ya registrado como '{existing_name}'. Usa 'r' para reforzar.")
                continue
            
            root = tk.Tk()
            root.withdraw()
            label = simpledialog.askstring("Aprender Rostro", "Nombre para este rostro:", parent=root)
            root.destroy()
            if not label or not label.strip():
                print("Nombre vacio, se omite.")
                continue
            label = label.strip()
            
            # Captura múltiple controlada con ángulos
            captured = capture_training_photos(video, label, CAPTURE_COUNT)
            
            if captured == 0:
                print("No se pudo capturar ninguna foto. Intenta de nuevo.")
                continue
            
            # Reentrena desde data/train para incluir las nuevas fotos
            known_encodings, known_names = rebuild_encodings_from_train()
            print(f"Guardadas {captured} fotos para {label} y actualizado {ENC_FILE}")

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
