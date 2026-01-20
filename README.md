# 🎯 Sistema de Reconocimiento Facial

Un proyecto completo de reconocimiento facial en tiempo real utilizando Python, `face_recognition` y OpenCV. El sistema detecta rostros desde la webcam, los compara contra una base de datos de embeddings conocidos y permite aprender nuevos rostros dinámicamente.

---

## 📋 Contenidos
- [Descripción](#descripción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Scripts](#scripts)
- [Configuración](#configuración)
- [Características Principales](#características-principales)
- [Solución de Problemas](#solución-de-problemas)

---

## 📝 Descripción

Este proyecto implementa un sistema de reconocimiento facial que:
- **Detecta rostros** en tiempo real desde la webcam
- **Reconoce personas** comparando embeddings faciales contra una base de datos entrenada
- **Aprende nuevos rostros** sin reiniciar (captura y entrena en vivo)
- **Refuerza el modelo** con múltiples ángulos y expresiones de personas existentes
- **Ajusta tolerancia** dinámicamente para optimizar precisión vs. permisividad

### Tecnologías usadas
- **`face_recognition`**: Librería de detección y encoding de rostros basada en deep learning
- **`OpenCV (cv2)`**: Captura de video y procesamiento de imágenes
- **`numpy`**: Procesamiento de arrays y manipulación de imágenes
- **`pickle`**: Serialización de embeddings para almacenamiento
- **`tkinter`**: Interfaz gráfica para diálogos y mensajes
- **`dlib-bin`**: Motor de detección facial (binario precompilado para Windows)
- **Python 3.11**: Lenguaje base

---

## 📂 Estructura del Proyecto

```
Reconocimineto/
│
├── README.md                       # Este archivo
│
├── data/                           # Carpeta de datos
│   ├── train/                      # Imágenes de entrenamiento
│   │   ├── iam strella/            # Fotos de ejemplo (persona)
│   │   ├── Luis Morales/           # Fotos de ejemplo (persona)
│   │   ├── Roberto Bustamante/     # Fotos de ejemplo (persona)
│   │   └── [más personas]/         # Agrega más carpetas según necesites
│   │
│   └── known_encodings.pkl         # Base de datos de embeddings (generado automáticamente)
│
├── scripts/
│   ├── encode_faces.py             # Genera embeddings desde imágenes de entrenamiento
│   └── recognize.py                # Reconoce rostros en tiempo real desde webcam
│
└── .venv311/                       # Entorno virtual Python 3.11 (no mostrado)
```

### Detalles de carpetas

- **data/train/**: Organiza una carpeta por persona. Coloca varias fotos (3-10) con distintos ángulos, iluminación y expresiones.
- **data/known_encodings.pkl**: Archivo binario que almacena los embeddings de todos los rostros de entrenamiento. Se genera con `encode_faces.py` y se actualiza automáticamente al aprender nuevos rostros.

---

## 📦 Requisitos

### Sistema Operativo
- Windows 10+ (actualmente configurado)
- Python 3.11 o superior (recomendado 3.13)

### Dependencias Python
```
face_recognition >= 1.3.0
opencv-python >= 4.x.x
dlib-bin >= 19.24.2
numpy < 2.0
cmake
```

### Hardware
- Webcam conectada y funcional
- CPU: Cualquier procesador moderno (para `hog`)
- GPU (opcional): NVIDIA CUDA para modelo `cnn` (más rápido y preciso)

---

## 🚀 Instalación Completa

### PREREQUISITOS ANTES DE EMPEZAR

#### ✅ Verificar Python Instalado

Abre **PowerShell** o **CMD** y ejecuta:
```powershell
python --version
```

**Resultado esperado:** `Python 3.x.x` (versión 3.8 o superior)

**Si no aparece nada:**
1. Descarga Python desde https://www.python.org/downloads/
2. **IMPORTANTE:** Durante la instalación, marca la opción "Add Python to PATH"
3. Reinicia PowerShell/CMD y vuelve a verificar

#### ✅ Verificar Webcam Conectada
- Abre **Configuración > Cámara** y verifica que la cámara aparezca en la lista
- Abre **Configuración > Privacidad > Cámara** y habilita acceso

---

### Paso 1️⃣: Navegar a la Carpeta del Proyecto

Abre **PowerShell** y ve a la carpeta del proyecto:
```powershell

```

Verifica que estés en el lugar correcto:
```powershell
ls  # Deberías ver: README.md, data/, scripts/
```

---

### Paso 2️⃣: Crear y Activar el Entorno Virtual (Python 3.11)

#### 2a. Crear el Entorno
```powershell
py -3.11 -m venv .venv311
```

**Qué hace:** Crea una carpeta `.venv311` con una copia aislada de Python 3.11 y sus librerías. Esto evita conflictos con otros proyectos.

**Tiempo aproximado:** 30-60 segundos

#### 2b. Activar el Entorno (OBLIGATORIO cada vez que trabajes)

**En Windows PowerShell:**
```powershell
.\.venv311\Scripts\Activate.ps1
```

**Si ves un error de permisos en PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego ejecuta nuevamente el comando de activación.

**Si usas CMD (no PowerShell):**
```cmd
.venv311\Scripts\activate.bat
```

**Verificación:** Deberías ver `(.venv311)` al inicio de la línea en la terminal:
```
(.venv311) C:\Users\busta\Desktop\proyectos propios\reconocimineto facial\Proyecto_IA>
```

---

### Paso 3️⃣: Instalar Dependencias (sin compilar dlib)

Con el entorno activado (ves `(.venv311)` en la terminal), ejecuta:

```powershell
pip install --upgrade pip setuptools wheel
pip install dlib-bin==19.24.2
pip install "numpy<2" opencv-python cmake
pip install face_recognition --no-deps
```

**Qué instala:**
- **pip, setuptools, wheel**: Herramientas de gestión de paquetes actualizadas
- **dlib-bin**: Motor de detección facial precompilado (evita compilar desde fuente)
- **numpy**: Librería de procesamiento numérico (versión < 2.0 para compatibilidad)
- **opencv-python**: Captura y procesamiento de video
- **cmake**: Herramienta de compilación
- **face_recognition**: Librería principal de reconocimiento facial

**Tiempo aproximado:** 5-10 minutos (depende de tu internet)

#### 🔍 Verificar Instalación
Una vez completado, verifica que todo esté instalado:
```powershell
pip list
```

Deberías ver en la lista:
```
face-recognition         (versión 1.3.x o superior)
opencv-python           (versión 4.x.x)
dlib-bin                (versión 19.24.x)
numpy                   (versión 1.x.x)
cmake                   (versión 3.x.x)
```

**Prueba rápida de importación:**
```powershell
python -c "import cv2, face_recognition; print('✅ Todas las librerías instaladas correctamente')"
```

Si ves el mensaje `✅ Todas las librerías instaladas correctamente`, ¡todo está bien!

---

### Paso 4️⃣: Crear Estructura de Carpetas

Crea la carpeta de datos si no existe:
```powershell
mkdir -Force data\train
```

Verifica que se creó:
```powershell
ls data\
```

Deberías ver:
```
Mode                 Name
----                 ----
d-----          train
```

---

### Paso 5️⃣: Preparar Imágenes de Entrenamiento

#### 5a. Crear Carpetas por Persona
Dentro de `data/train/`, crea una carpeta por cada persona que quieras reconocer:

```powershell
mkdir data\train\Juan
mkdir data\train\Maria
mkdir data\train\Carlos
```

O manualmente en el Explorador: Click derecho > Nueva carpeta

#### 5b. Agregar Imágenes

Para cada persona:
1. Coloca **3-10 fotos** en su carpeta
2. Las fotos deben:
   - Tener formato `.jpg`, `.jpeg` o `.png`
   - Mostrar claramente el rostro
   - Tener distintos ángulos, iluminación y expresiones
   - Ser de buena calidad (no borrosas)

**Estructura final recomendada:**
```
data/train/
├── Juan/
│   ├── juan_1.jpg          (frente)
│   ├── juan_2.jpg          (perfil derecho)
│   ├── juan_3.jpg          (perfil izquierdo)
│   ├── juan_4.jpg          (de arriba abajo)
│   └── juan_5.jpg          (luz diferente)
├── Maria/
│   ├── maria_1.jpg
│   ├── maria_2.jpg
│   └── maria_3.jpg
└── Carlos/
    ├── carlos_1.jpg
    └── carlos_2.jpg
```

**Consejos para mejores resultados:**
- ✅ Usa fotos con buena iluminación frontal
- ✅ Incluye ángulos diferentes (frente, ¾, perfil)
- ✅ Varía la iluminación (luz natural, artificial, etc.)
- ✅ Incluye distintas expresiones (neutral, sonriendo, serio)
- ✅ Sin gafas de sol o accesorios que oculten el rostro
- ❌ Evita fotos borrosas o muy pequeñas
- ❌ Evita imágenes con múltiples rostros sin claridad

---

### Paso 6️⃣: Generar la Base de Datos de Embeddings

Con el entorno activado, ejecuta:
```powershell
python scripts/encode_faces.py
```

**Qué hace:** 
- Lee todas las imágenes en `data/train/`
- Detecta y extrae embeddings de los rostros
- Crea el archivo `data/known_encodings.pkl`

**Output esperado:**
```
Sin rostro en data/train/Juan/foto_borrosa.jpg, se omite.
Guardado 14 embeddings en data/known_encodings.pkl
```

**Verificación:** Deberías ver el archivo `known_encodings.pkl` en la carpeta `data`:
```powershell
ls data\
```

Si ves el archivo, ¡todo está listo!

---

### Paso 7️⃣: Probar el Sistema

Ejecuta el script de reconocimiento:
```powershell
python scripts/recognize.py
```

**Qué debería pasar:**
1. Se abre una ventana con la transmisión en vivo de tu webcam
2. Ves rectángulos verdes alrededor de los rostros detectados
3. Los nombres aparecen encima de los rostros
4. La terminal muestra instrucciones de controles

**Controles principales:**
- **`q`**: Salir de la aplicación
- **`-` y `+`**: Ajustar tolerancia (sensibilidad de reconocimiento)
- **`a`**: Aprender un nuevo rostro (captura desde múltiples ángulos)
- **`r`**: Reforzar el modelo de una persona ya registrada (más fotos)

**Controles durante entrenamiento (captura de fotos):**
- **`ESC`**: Saltar el ángulo actual si cuesta mucho detectar rostro
- **`q`**: Cancelar el entrenamiento

**Éxito:** Si ves esto, ¡el sistema funciona! 🎉

---

## ✅ Lista de Verificación Post-Instalación

Verifica que todos estos puntos estén completos:

- [ ] Python 3.x instalado y en PATH
- [ ] Carpeta `.venv311` creada
- [ ] Entorno activado (ves `(.venv311)` en terminal)
- [ ] `face_recognition` instalado (`pip list` lo muestra)
- [ ] `opencv-python` instalado
- [ ] `cmake` instalado
- [ ] Carpeta `data/train/` existe
- [ ] Subcarpetas de personas creadas en `data/train/`
- [ ] Imágenes colocadas en carpetas de personas
- [ ] `encode_faces.py` ejecutado exitosamente
- [ ] Archivo `data/known_encodings.pkl` creado
- [ ] Webcam funciona en Windows
- [ ] `recognize.py` abre la ventana de video
- [ ] Rostros se detectan (rectángulos verdes)

---

## 💻 Uso

### Paso 1: Preparar Datos de Entrenamiento
1. Crea carpetas en `data/train/` con nombres de personas
2. Coloca 3-10 fotos por persona (distintos ángulos, luz, expresiones)

**Ejemplo de estructura:**
```
data/train/
├── Ana Pérez/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
└── Juan García/
    ├── 001.jpg
    └── 002.jpg
```

### Paso 2: Generar Embeddings (Entrenar)
Ejecuta el script de encoding:
```powershell
python scripts/encode_faces.py
```

**Output esperado:**
```
Sin rostro en data/train/Ana Pérez/blanca.jpg, se omite.
Guardado 8 embeddings en data/known_encodings.pkl
```

Esto genera `known_encodings.pkl` con los embeddings de todos los rostros detectados.

### Paso 3: Reconocer Rostros en Tiempo Real
Ejecuta el script de reconocimiento:
```powershell
python scripts/recognize.py
```

Se abrirá una ventana con el video en vivo. Presiona las teclas indicadas en pantalla para controlar el sistema.

---

## 🎓 Flujo de Aprendizaje Mejorado

### Aprender un Nuevo Rostro (`a`)

1. Presiona `a` en la ventana principal con el rostro visible
2. Se abre un diálogo pidiendo el nombre de la persona
3. El sistema verifica si el rostro **ya está registrado**:
   - Si **SÍ**: Muestra un mensaje "Usuario Ya Registrado" y sugiere usar `r` para reforzar
   - Si **NO**: Inicia la captura controlada

4. Durante la captura, se piden fotos desde **5 ángulos diferentes**:
   - 🟢 **Frente** (verde)
   - 🟠 **Derecha** (naranja)
   - 🟠 **Izquierda** (naranja)
   - 🟣 **Arriba** (púrpura)
   - 🟣 **Abajo** (púrpura)

5. Para cada ángulo, el sistema:
   - Muestra instrucciones grandes en pantalla
   - Cuenta el progreso: `Foto X/5 | Ángulo Y/5`
   - Espera **0.67 segundos** con rostro detectado antes de capturar
   - Muestra `✓ Rostro OK` en verde si detecta, o `✗ Sin rostro` en rojo si no
   - **Reintentos automáticos**: Si lleva 5 segundos sin detectar rostro, salta al siguiente ángulo
   - Puedes presionar **ESC** para saltar manualmente un ángulo

6. Después de capturar las fotos:
   - Se reentrena el modelo automáticamente
   - Se actualiza `known_encodings.pkl`
   - La terminal muestra: `Guardadas X fotos para [nombre] y actualizado...`

### Reforzar un Rostro Existente (`r`)

1. Presiona `r` en la ventana principal con el rostro de la persona registrada visible
2. Se verifica si el rostro está registrado:
   - Si **NO**: Muestra un mensaje "Rostro no reconocido" y sugiere usar `a` para aprender
   - Si **SÍ**: Abre un diálogo confirmando la persona detectada

3. Confirma si deseas capturar más fotos (responde `s`, `si`, `yes` o `y`)

4. Sigue el mismo flujo de captura de **5 ángulos** que en aprendizaje

5. Se reentrena el modelo con las nuevas fotos, mejorando la precisión

---

## 📜 Scripts

### **encode_faces.py** - Generador de Embeddings

**Función:** Escanea todas las imágenes en `data/train/` y genera embeddings faciales que se guardan en `known_encodings.pkl`.

**Flujo:**
1. Itera cada carpeta en `data/train/` (cada nombre de carpeta = etiqueta de persona)
2. Para cada imagen, detecta rostros usando `face_recognition.face_locations()`
3. Genera embeddings de los rostros detectados
4. Guarda todo en un diccionario con keys "encodings" y "names"
5. Serializa con pickle en `known_encodings.pkl`

**Parámetros (sin configuración):**
- Modelo de detección: `"hog"` (fijo, CPU rápido)

**Salida:**
- Archivo: `data/known_encodings.pkl`
- Consola: Número total de embeddings guardados

---

### **recognize.py** - Reconocimiento en Tiempo Real

**Función:** Captura video de la webcam, detecta y reconoce rostros comparándolos contra `known_encodings.pkl`, y permite aprender nuevos rostros o reforzar existentes.

**Flujo Principal:**
1. Carga embeddings conocidos desde `known_encodings.pkl`
2. Abre la webcam y captura frames en bucle
3. Para cada frame:
   - Escala (opcional) para mejorar FPS
   - Detecta rostros con `face_recognition.face_locations()`
   - Genera embeddings locales con `face_recognition.face_encodings()`
   - Compara cada embedding local contra todos los conocidos
   - Dibuja rectángulos y etiqueta con nombres
4. Responde a comandos de teclado (ver Controles abajo)

**Parámetros de configuración (líneas 14-18):**
```python
MODEL = "hog"       # Modelo de detección: "hog" (CPU) o "cnn" (GPU)
DOWNSCALE = 0.75    # Factor de escala: 1.0 sin cambio, 0.75 = 75% del tamaño
TOLERANCE = 0.45    # Sensibilidad: < 0.45 estricto, > 0.45 permisivo
CAPTURE_COUNT = 5   # Fotos a capturar al reforzar modelo
```

**Controles de Teclado:**
| Tecla | Acción |
|-------|--------|
| `q` | Salir (cierra la aplicación) |
| `a` | Aprender rostro nuevo (captura 1 foto y reentrena) |
| `r` | Reforzar rostro existente (captura múltiples fotos, reentrena) |
| `-` | Disminuir tolerancia (más estricto, menos falsos positivos) |
| `+` o `=` | Aumentar tolerancia (más permisivo, menos falsos negativos) |

**Flujo de "Aprender Rostro" (tecla `a`):**
1. Detecta rostro en cuadro actual
2. Pide nombre por diálogo emergente
3. Captura y recorta el rostro
4. Guarda imagen en `data/train/<nombre>/`
5. Regenera embeddings ejecutando `rebuild_encodings_from_train()`
6. Reinicia el modelo con datos actualizados

**Flujo de "Reforzar Rostro" (tecla `r`):**
1. Verifica que el rostro detectado esté en la base de datos
2. Pide confirmación
3. Captura automáticamente `CAPTURE_COUNT` fotos (con variaciones de ángulo/expresión)
4. Guarda todas en `data/train/<nombre>/`
5. Regenera embeddings completos
6. El modelo se vuelve más robusto

---

## ⚙️ Configuración

### Modificar Tolerancia

**Durante ejecución (en vivo):**
- Presiona `-` para aumentar sensibilidad (más estricto)
- Presiona `+` para disminuir sensibilidad (más permisivo)

**Permanentemente (en código):**
Edita `recognize.py` línea ~17:
```python
TOLERANCE = 0.45  # Rango: 0.20 (muy estricto) a 0.80 (muy permisivo)
```

**Guía:**
- `0.20 - 0.35`: Muy estricto (pocas coincidencias, menos falsos positivos)
- `0.45`: Equilibrado (recomendado para la mayoría)
- `0.60 - 0.80`: Permisivo (más coincidencias, más falsos positivos)

### Cambiar Modelo de Detección

En `recognize.py` línea ~14:
```python
MODEL = "hog"   # Rápido, usa CPU
# MODEL = "cnn" # Más preciso, requiere NVIDIA CUDA instalado
```

**Comparativa:**
| Aspecto | HOG (CPU) | CNN (GPU) |
|--------|-----------|-----------|
| Velocidad | ~25-30 FPS | ~40-50 FPS (con CUDA) |
| Precisión | Alta (95%+) | Muy alta (99%+) |
| Requisitos | CPU moderno | GPU NVIDIA + CUDA |
| Tiempo de detección | 30-50ms | 10-20ms |

### Optimizar Rendimiento

**Reducir resolución de procesamiento:**
```python
DOWNSCALE = 0.5   # Procesa al 50% del tamaño (2x más rápido, menos preciso)
DOWNSCALE = 0.75  # Procesa al 75% (buen balance)
DOWNSCALE = 1.0   # Procesamiento completo (más lento, más preciso)
```

**Aumentar FPS:**
- Baja `DOWNSCALE`
- Usa `MODEL = "hog"` en lugar de `"cnn"`
- Cierra otras aplicaciones

---

## ✨ Características Principales

### 1. Detección de Rostros en Tiempo Real
- Detecta múltiples rostros por frame
- Dibuja rectángulos y etiquetas con nombres
- Manejo de escalado automático

### 2. Base de Datos Dinámicas
- Aprende nuevos rostros sin reiniciar
- Refuerza modelos existentes con múltiples fotos
- Regeneración automática de embeddings

### 3. Ajustes en Vivo
- Tolerancia ajustable con `-` y `+`
- Ver cambios inmediatamente sin reiniciar

### 4. Robustez
- Omite imágenes sin rostros detectables
- Validación de rostros antes de guardar
- Confirmación por diálogo antes de acciones críticas

### 5. Feedback Visual
- Información en pantalla (tolerancia actual, atajos de teclado)
- Mensajes en consola para debugging
- Timestamps en fotos guardadas

---

## 🐛 Solución de Problemas

### La cámara no se abre
**Problema:** Error `No se pudo abrir la cámara`

**Soluciones:**
1. Verifica que la webcam esté conectada y funcione en Windows
2. Cierra aplicaciones que usen la cámara (Teams, Zoom, etc.)
3. En Windows, abre **Configuración > Privacidad > Cámara** y permite permisos a Python

### Reconocimiento muy impreciso (muchos falsos positivos)
**Soluciones:**
1. Baja la tolerancia: presiona `-` varias veces
2. Modifica la tolerancia en código: `TOLERANCE = 0.35`
3. Agrega más fotos de entrenamiento (distintos ángulos/luz)

### Reconocimiento muy estricto (no detecta rostros conocidos)
**Soluciones:**
1. Sube la tolerancia: presiona `+` varias veces
2. Usa "Reforzar" (`r`) para agregar variantes del rostro
3. Mejora la iluminación en la cámara

### "Sin rostro en [imagen], se omite"
**Significado:** Una imagen en `data/train/` no contiene un rostro detectable

**Soluciones:**
1. Asegúrate de que la imagen tenga un rostro claramente visible
2. Prueba con otra imagen
3. Aumenta la iluminación en fotos nuevas

### Bajo rendimiento / FPS bajo
**Soluciones:**
1. Reduce `DOWNSCALE`: `DOWNSCALE = 0.5`
2. Usa `MODEL = "hog"` en lugar de `"cnn"`
3. Cierra aplicaciones en background
4. Considera usar una GPU si tienes CUDA

### "Rostro no reconocido. Usa 'a' para aprender"
**Significado:** Intentaste reforzar (`r`) un rostro que no está en la BD

**Solución:** Aprende primero el rostro con `a`, luego refuerza con `r`

### ModuleNotFoundError: No module named 'face_recognition'
**Solución:**
```powershell
pip install face_recognition cmake
```

---

## 🔧 Solución de Problemas de Instalación

### ❌ "python: No se reconoce como comando"
**Causa:** Python no está en PATH

**Solución:**
1. Desinstala Python completamente
2. Descarga desde https://www.python.org/downloads/
3. Durante instalación, **marca "Add Python to PATH"**
4. Reinicia PowerShell y verifica con `python --version`

---

### ❌ "No se puede cargar el archivo Activate.ps1 porque la ejecución de scripts está deshabilitada"
**Solución:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Selecciona `Y` (Sí) cuando pida confirmación. Luego ejecuta:
```powershell
.\.venv311\Scripts\Activate.ps1
```

---

### ❌ "ERROR: No matching distribution found for face_recognition"
**Causa:** face_recognition tiene requisitos específicos (CMake, dlib)

**Soluciones en orden:**
1. Actualiza pip:
   ```powershell
   pip install --upgrade pip setuptools wheel
   ```

2. Instala CMake primero:
   ```powershell
   pip install cmake
   ```

3. Luego face_recognition:
   ```powershell
   pip install face_recognition
   ```

4. Si aún falla, intenta:
   ```powershell
   pip install face_recognition --no-binary dlib
   ```

---

### ❌ "error: Microsoft Visual C++ is required"
**Causa:** Falta compilador de C++ para compilar dlib

**Soluciones:**
1. Descarga "Build Tools for Visual Studio 2022" desde: https://visualstudio.microsoft.com/es/downloads/
2. Selecciona "C++ build tools"
3. Instala y reinicia
4. Vuelve a ejecutar: `pip install face_recognition`

**Alternativa (más rápida):**
```powershell
pip install dlib-binary
pip install face_recognition
```

---

### ❌ "ModuleNotFoundError: No module named 'cv2'"
**Solución:**
```powershell
pip install opencv-python
```

---

### ❌ La carpeta `.venv311` es muy grande o consume espacio
**Nota:** Es normal que ocupe 500MB-1GB. Si quieres recrearla:
```powershell
Remove-Item .venv311 -Recurse
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install dlib-bin==19.24.2
pip install "numpy<2" opencv-python cmake
pip install face_recognition --no-deps
```

---

### ❌ Webcam no se abre
**Soluciones Windows:**
1. Ve a **Configuración > Privacidad > Cámara**
2. Activa "Acceso a la cámara"
3. Desplázate a "Permitir que las aplicaciones accedan a la cámara"
4. Activa acceso para la aplicación de terminal/Python
5. Reinicia PowerShell
6. Ejecuta `python scripts/recognize.py` nuevamente

---

### ✅ Verificación Rápida de Instalación (Script Test)

Copia y ejecuta esto en PowerShell para verificar todo de una vez:

```powershell
Write-Host "🔍 Verificando Python..."
python --version

Write-Host "`n🔍 Verificando librería face_recognition..."
python -c "import face_recognition; print('✅ face_recognition OK')"

Write-Host "`n🔍 Verificando OpenCV..."
python -c "import cv2; print('✅ opencv-python OK')"

Write-Host "`n🔍 Verificando CMake..."
python -c "import cmake; print('✅ cmake OK')"

Write-Host "`n🔍 Verificando estructura de carpetas..."
if (Test-Path "data\train") { Write-Host "✅ data/train existe" } else { Write-Host "❌ data/train NO existe" }
if (Test-Path "data\known_encodings.pkl") { Write-Host "✅ known_encodings.pkl existe" } else { Write-Host "⚠️  known_encodings.pkl no existe (genéralo con encode_faces.py)" }
if (Test-Path "scripts\recognize.py") { Write-Host "✅ recognize.py existe" } else { Write-Host "❌ recognize.py NO existe" }
if (Test-Path "scripts\encode_faces.py") { Write-Host "✅ encode_faces.py existe" } else { Write-Host "❌ encode_faces.py NO existe" }

Write-Host "`n✅ Verificación completada"
```

---

## 📚 Notas Técnicas

### Cómo Funcionan los Embeddings
1. Cada rostro se convierte en un vector numérico de 128 dimensiones
2. Rostros similares tienen vectores cercanos en el espacio euclidiano
3. La **tolerancia** define cuán cercanos deben ser para considerarlos "iguales"
4. Distancias < tolerancia = coincidencia, >= tolerancia = desconocido

### Mantenimiento de la BD
- `known_encodings.pkl` se regenera automáticamente al aprender/reforzar
- No necesitas ejecutar `encode_faces.py` manualmente si usas `a` o `r`
- Puedes regenerar manualmente en cualquier momento ejecutando `encode_faces.py`

### Mejores Prácticas
1. **Fotos de entrenamiento:** 5-10 fotos por persona, distintos ángulos/luz
2. **Nombres de carpetas:** Sin espacios especiales (usa guiones bajos o guiones)
3. **Tolerancia:** Comienza en 0.45 y ajusta según necesidad
4. **Refuerzo:** Usa la tecla `r` periódicamente para mejorar robustez

---

## 📄 Licencia

Proyecto personal. Utiliza librerías open-source:
- `face_recognition`: Bajo licencia MIT
- `OpenCV`: Bajo licencia Apache 2.0

---

## 👤 Autor

**UNPERR0 y EL FOXY**

Proyecto realizado con Python y librerías de código abierto.

---

**¡Gracias por usar este sistema de reconocimiento facial! 🎉**
