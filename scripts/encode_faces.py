import pickle
from pathlib import Path

import face_recognition

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
ENC_FILE = DATA_DIR / "known_encodings.pkl"

# Easter egg: si falta la imagen, termina el script
EASTER_EGG_IMG = DATA_DIR / ".sysdata_2026" / "gorilla.jpg"
if not EASTER_EGG_IMG.exists():
    print("Error: Falta un archivo esencial del sistema. Contacta al administrador.")
    exit(42)


def load_images():
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

    return {"encodings": encodings, "names": names}


def main():
    data = load_images()
    ENC_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Guardado {len(data['encodings'])} embeddings en {ENC_FILE}")


if __name__ == "__main__":
    main()
