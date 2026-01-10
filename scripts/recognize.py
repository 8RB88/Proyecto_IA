import pickle
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog

import cv2
import face_recognition

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
            
            # Captura múltiple para reforzar
            print(f"Capturando {CAPTURE_COUNT} fotos adicionales de {existing_name}... Varia tu expresion/angulo.")
            captured = 0
            for i in range(CAPTURE_COUNT * 3):
                ret, frame_cap = video.read()
                if not ret:
                    continue
                
                proc_frame = frame_cap
                if DOWNSCALE != 1.0:
                    proc_frame = cv2.resize(frame_cap, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
                
                rgb_cap = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                boxes_cap = face_recognition.face_locations(rgb_cap, model=MODEL)
                
                if boxes_cap:
                    if DOWNSCALE != 1.0:
                        inv = 1.0 / DOWNSCALE
                        top, right, bottom, left = boxes_cap[0]
                        box_full = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
                    else:
                        box_full = boxes_cap[0]
                    
                    save_face_image(frame_cap, box_full, existing_name)
                    captured += 1
                    print(f"  Foto {captured}/{CAPTURE_COUNT} capturada")
                    
                    if captured >= CAPTURE_COUNT:
                        break
                    
                    cv2.waitKey(200)
            
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
            root = tk.Tk()
            root.withdraw()
            label = simpledialog.askstring("Aprender Rostro", "Nombre para este rostro:", parent=root)
            root.destroy()
            if not label or not label.strip():
                print("Nombre vacio, se omite.")
                continue
            label = label.strip()
            
            # Captura multiple para mejorar aprendizaje
            print(f"Capturando {CAPTURE_COUNT} fotos de {label}... Manten el rostro visible y muevelo ligeramente.")
            captured = 0
            for i in range(CAPTURE_COUNT * 3):  # Intentos extra por si falla deteccion
                ret, frame_cap = video.read()
                if not ret:
                    continue
                
                proc_frame = frame_cap
                if DOWNSCALE != 1.0:
                    proc_frame = cv2.resize(frame_cap, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
                
                rgb_cap = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                boxes_cap = face_recognition.face_locations(rgb_cap, model=MODEL)
                
                if boxes_cap:
                    # Escalar caja al tamaño original
                    if DOWNSCALE != 1.0:
                        inv = 1.0 / DOWNSCALE
                        top, right, bottom, left = boxes_cap[0]
                        box_full = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
                    else:
                        box_full = boxes_cap[0]
                    
                    save_face_image(frame_cap, box_full, label)
                    captured += 1
                    print(f"  Foto {captured}/{CAPTURE_COUNT} capturada")
                    
                    if captured >= CAPTURE_COUNT:
                        break
                    
                    # Delay para variedad en capturas
                    cv2.waitKey(200)
            
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
