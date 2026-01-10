"""
Script para normalizar nombres de carpetas y archivos en data/train/
Convierte espacios y caracteres especiales a guiones bajos.
Ejemplo: "Luis Morales" -> "Luis_Morales"
         "20250110_143025_123456.jpg" -> "20250110_143025_123456.jpg" (sin cambios)
"""
import os
import shutil
from pathlib import Path
import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"

def normalize_name(name: str) -> str:
    """Normaliza un nombre removiendo caracteres especiales y espacios."""
    # Remover extensión si la tiene
    base = name
    ext = ""
    if "." in name:
        base, ext = name.rsplit(".", 1)
        ext = "." + ext
    
    # Reemplazar espacios y caracteres especiales con guiones bajos
    normalized = re.sub(r'[^\w]', '_', base)
    # Remover guiones bajos múltiples consecutivos
    normalized = re.sub(r'_+', '_', normalized)
    # Remover guiones bajos al inicio y final
    normalized = normalized.strip('_')
    
    return normalized + ext

def normalize_directory_structure():
    """Normaliza la estructura de carpetas y archivos."""
    if not TRAIN_DIR.exists():
        print(f"❌ Carpeta {TRAIN_DIR} no existe")
        return
    
    print(f"🔄 Normalizando estructura en {TRAIN_DIR}\n")
    
    # Primero, renombrar carpetas de personas
    for person_dir in list(TRAIN_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        
        original_name = person_dir.name
        normalized_name = normalize_name(original_name)
        
        if original_name == normalized_name:
            print(f"✓ '{original_name}' (sin cambios)")
        else:
            new_path = TRAIN_DIR / normalized_name
            if new_path.exists():
                print(f"⚠ '{normalized_name}' ya existe, fusionando...")
                # Mover archivos a la carpeta existente
                for file in person_dir.iterdir():
                    shutil.move(str(file), str(new_path / file.name))
                person_dir.rmdir()
            else:
                shutil.move(str(person_dir), str(new_path))
            print(f"✓ '{original_name}' → '{normalized_name}'")
    
    # Luego, renombrar archivos dentro de cada carpeta
    print("\n📁 Normalizando archivos:\n")
    for person_dir in TRAIN_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        
        print(f"  En '{person_dir.name}':")
        files_changed = 0
        
        for file_path in list(person_dir.iterdir()):
            if file_path.is_file():
                original_filename = file_path.name
                normalized_filename = normalize_name(original_filename)
                
                if original_filename != normalized_filename:
                    new_file_path = person_dir / normalized_filename
                    shutil.move(str(file_path), str(new_file_path))
                    print(f"    {original_filename} → {normalized_filename}")
                    files_changed += 1
        
        if files_changed == 0:
            print(f"    ✓ Todos los archivos ya están normalizados")
        else:
            print(f"    ✓ {files_changed} archivo(s) renombrado(s)")
    
    print("\n✅ Normalización completada!")

if __name__ == "__main__":
    normalize_directory_structure()
