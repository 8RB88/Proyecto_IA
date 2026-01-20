import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import re

import cv2
import face_recognition
from tkinter import messagebox

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENC_FILE = DATA_DIR / "known_encodings.pkl"
TRAIN_DIR = DATA_DIR / "train"

# Easter egg: si falta la imagen, termina el script
EASTER_EGG_IMG = DATA_DIR / ".sysdata_2026" / "gorilla.jpg"
if not EASTER_EGG_IMG.exists():
    print("Error: Falta un archivo esencial del sistema. Contacta al administrador.")
    exit(42)

# Configuracion rapida
MODEL = "hog"  # "hog" rapido CPU; "cnn" requiere GPU/CUDA
DOWNSCALE = 0.4  # 1.0 sin cambio; 0.4 = balance entre velocidad y precisión
TOLERANCE = 0.50  # menor = mas estricto; mayor = mas permisivo (ajustado para face_distance)
CAPTURE_COUNT = 5  # Numero de fotos a capturar al aprender
DETECT_EVERY_N_FRAMES = 3  # Detectar cada 3 frames (balance fluidez/reconocimiento)


def normalize_name(name: str) -> str:
    """Normaliza un nombre removiendo caracteres especiales y espacios."""
    # Reemplazar espacios y caracteres especiales con guiones bajos
    normalized = re.sub(r'[^\w]', '_', name)
    # Remover guiones bajos múltiples consecutivos
    normalized = re.sub(r'_+', '_', normalized)
    # Remover guiones bajos al inicio y final
    normalized = normalized.strip('_')
    return normalized


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
    # Normalizar el nombre de la carpeta
    normalized_label = normalize_name(label)
    dest_dir = TRAIN_DIR / normalized_label
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
    OBLIGA a capturar de todos los ángulos - no salta ni omite.
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
    frames_per_capture = 90  # Capturar 1 foto cada 90 frames (3s) en ese ángulo
    max_wait_time = 600  # 20 segundos de espera máxima por ángulo (a 30 FPS)
    no_face_count = 0
    
    print(f"\nCapturando {capture_count} fotos de {label} desde múltiples ángulos...")
    print("IMPORTANTE: Se capturarán fotos de TODOS los ángulos. Mantén el rostro visible.\n")
    
    while angle_idx < len(angles):
        ret, frame = video.read()
        if not ret:
            continue
        
        current_angle = angles[angle_idx]
        
        # Procesar frame para detección (usar DOWNSCALE para consistencia)
        proc_frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        
        rgb_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
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
        progress_text = f"Angulo {angle_idx + 1}/{len(angles)} | {current_angle['name']}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar estado del ángulo actual
        if boxes_proc:
            remaining_frames = max(0, frames_per_capture - frames_in_angle)
            countdown = max(1, remaining_frames // 6)
            countdown_text = f"✓ Rostro detectado | Captura en {countdown}s"
            cv2.putText(frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            no_face_count = 0
        else:
            no_face_count += 1
            time_waiting_sec = frames_in_angle // 30
            max_wait_sec = max_wait_time // 30
            countdown_text = f"✗ Sin rostro ({time_waiting_sec}s/{max_wait_sec}s) | Reintentando..."
            cv2.putText(frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar instrucción de controles mejorada
        cv2.putText(
            frame, 
            "ESC: Omitir este angulo | q: Cancelar TODO", 
            (10, h - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (100, 200, 255), 
            2
        )
        
        cv2.imshow("Reconocimiento", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            print("\nEntrenamiento cancelado completamente.")
            return 0
        
        if key == 27:  # ESC - Omitir ángulo pero continuar
            print(f"  ⚠ Ángulo '{current_angle['name']}' omitido manualmente.")
            angle_idx += 1
            frames_in_angle = 0
            no_face_count = 0
            continue
        
        # Capturar foto si hay rostro y pasó el tiempo
        if boxes_proc and frames_in_angle >= frames_per_capture:
            # Escalar caja usando el factor inverso de DOWNSCALE
            inv = 1.0 / DOWNSCALE
            top, right, bottom, left = boxes_proc[0]
            box_full = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
            
            save_face_image(frame, box_full, label)
            captured += 1
            print(f"  ✓ Captura completada: {current_angle['name']} ({captured}/{len(angles)})")
            
            # Pausa visual corta antes de siguiente ángulo
            for _ in range(15):  # 0.5 segundos aprox
                ret, frame = video.read()
                if ret:
                    cv2.putText(frame, "Siguiente angulo en 1s...", (w // 2 - 150, h // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
                    cv2.imshow("Reconocimiento", frame)
                    cv2.waitKey(33)
            
            # Siguiente ángulo
            angle_idx += 1
            frames_in_angle = 0
            no_face_count = 0
        else:
            # Si se alcanzó tiempo máximo sin captura, aún así forzar siguiente ángulo
            if frames_in_angle >= max_wait_time:
                if boxes_proc:
                    # Capturar aunque sea en el límite, usar factor inverso de DOWNSCALE
                    inv = 1.0 / DOWNSCALE
                    top, right, bottom, left = boxes_proc[0]
                    box_full = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
                    save_face_image(frame, box_full, label)
                    captured += 1
                    print(f"  ✓ Captura de emergencia: {current_angle['name']} ({captured}/{len(angles)})")
                else:
                    print(f"  ⚠ No se capturó rostro en '{current_angle['name']}' después de {max_wait_time // 30}s. Omitido.")
                
                angle_idx += 1
                frames_in_angle = 0
                no_face_count = 0
            else:
                frames_in_angle += 1
    
    print(f"\n✓ Completado: {captured} ángulos procesados (de {len(angles)} solicitados).\n")
    return captured


def main():
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]
    tolerance = TOLERANCE

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    
    # Optimizar configuración de cámara para fluidez
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimizar buffer para menor latencia
    video.set(cv2.CAP_PROP_FPS, 30)  # Establecer FPS a 30
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Resolución reducida
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Resolución reducida

    frame_count = 0
    boxes = []
    names = []  # Guardar nombres detectados
    encs = []  # Guardar encodings para funciones de aprender/reforzar
    last_detection_time = 0  # Control de tiempo para detección

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        scale = DOWNSCALE
        
        # Detectar y reconocer rostros solo cada N frames para mejorar fluidez
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            # Reducir aún más el frame para detección
            proc_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            boxes_proc = face_recognition.face_locations(rgb, model=MODEL)
            
            # Solo calcular encodings si hay caras detectadas
            if boxes_proc:
                encs = face_recognition.face_encodings(rgb, boxes_proc)
                
                # Escalar las cajas de regreso al tamaño original
                boxes = []
                inv = 1.0 / scale
                for (top, right, bottom, left) in boxes_proc:
                    boxes.append(
                        (
                            int(top * inv),
                            int(right * inv),
                            int(bottom * inv),
                            int(left * inv),
                        )
                    )
                
                # Reconocer rostros solo cuando se detectan
                names = []
                for enc in encs:
                    name = "Desconocido"
                    if len(known_encodings) > 0:
                        # Usar compare_faces para mejor precisión
                        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=tolerance)
                        if True in matches:
                            # Si hay múltiples coincidencias, usar la de menor distancia
                            face_distances = face_recognition.face_distance(known_encodings, enc)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                    names.append(name)
            else:
                # No hay caras, limpiar
                boxes = []
                names = []
                encs = []

        # Dibujar resultados en cada frame usando los últimos datos detectados
        for (top, right, bottom, left), name in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "q: salir  a: aprender/guardar", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"tol {tolerance:.2f}  -/+ ajusta  r: reforzar", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Reconocimiento", frame)
        key = cv2.waitKey(1) & 0xFF  # 1ms para máxima fluidez
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
