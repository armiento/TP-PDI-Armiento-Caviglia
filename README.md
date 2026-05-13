# Editor Interactivo de Imágenes de Frutas
## Trabajo Práctico Final – Técnicas de Procesamiento Digital de Imágenes

**Carrera:** Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial  
**Instituto:** IFTS N°18 – CABA  
**Profesor:** Bonini Juan Ignacio  
**Ciclo:** 2026  
**Alumnos:** Armiento, Fernando – Caviglia, Paula  

---

## 1. Descripción del proyecto

Se desarrolló una aplicación interactiva en Python con OpenCV que permite editar y procesar imágenes de frutas en tiempo real. El usuario puede ajustar los parámetros de cada técnica mediante sliders y ver el efecto sobre la imagen al instante, sin necesidad de reiniciar el programa.

El foco del proyecto es la **edición y manipulación de imágenes**, no el procesamiento automático. Cada técnica tiene sus propios controles interactivos que el usuario maneja libremente.

---

## 2. Problema que resuelve

Las técnicas de procesamiento digital de imágenes son difíciles de entender en abstracto. Esta herramienta permite visualizar en tiempo real cómo afecta cada parámetro a una imagen concreta, facilitando tanto el aprendizaje como la exploración práctica de los algoritmos.

Casos de uso reales del mismo concepto:
- Clasificación automática de frutas en supermercados
- Detección de madurez en cosechas por drones
- Segmentación de tumores en imágenes médicas
- Guía visual para brazos robóticos

---

## 3. Técnicas implementadas

| Módulo | Técnica |
|--------|---------|
| `bordes.py` | Detección de bordes con Canny y Sobel |
| `color.py` | Conversión de espacios de color (BGR, HSV, HLS) |
| `histograma.py` | Histogramas y ecualización por canal |
| `morfologia.py` | Erosión, dilatación, apertura y cierre |
| `segmentacion_kmeans.py` | Segmentación por color con K-means |
| `segmentacion_otsu.py` | Umbralización automática con Otsu |
| `watershed.py` | Separación de objetos con Watershed |
| `main.py` | Interfaz visual con sliders y modos |

---

## 4. Modos de visualización interactivos

La aplicación tiene 8 modos que se activan con las teclas 1 al 8. Cada modo tiene sus propios sliders:

- Modo 1 – Blur gaussiano + brillo y contraste ajustables
- Modo 2 – Bordes con Canny (sliders de umbral bajo y alto)
- Modo 3 – Segmentación K-means (slider para elegir k de 2 a 6)
- Modo 4 – Umbralización Otsu (automática o manual)
- Modo 5 – Watershed (slider de umbral de transformada de distancia)
- Modo 6 – Histograma por canal R, G, B o grises con ecualización
- Modo 7 – Transformada de Fourier con filtro pasa bajos ajustable
- Modo 8 – Ajuste de tono, saturación y brillo en espacio HSV

Navegación: A / D para cambiar de imagen, S para guardar, Q para salir.

---

## 5. Dataset

| Campo | Detalle |
|-------|---------|
| Nombre | Fruits Dataset (Images) |
| Fuente | Kaggle – shreyapmaher/fruits-dataset-images |
| Motivo | Dataset liviano, imágenes limpias organizadas por carpeta de fruta |

---

## 6. Librerías utilizadas

| Librería | Uso |
|----------|-----|
| `opencv-python` | Procesamiento de imágenes, interfaz visual, sliders |
| `numpy` | Operaciones matriciales sobre píxeles |
| `scikit-learn` | Algoritmo K-means |
| `matplotlib` | Referencia para cálculo de histogramas |

---

## 7. Estado actual del proyecto

### Lo que está hecho
- 7 módulos de procesamiento implementados y funcionales
- Interfaz interactiva con sliders en tiempo real
- 8 modos de visualización cubriendo todas las unidades del programa
- Manejo de errores para dataset no encontrado
- Captura de pantalla del resultado disponible

### Lo que falta (próximas etapas)
- Aplicar Programación Orientada a Objetos (POO) — refactorizar módulos en clases
- Separar formalmente el Core de la capa de presentación
- Subir al repositorio Git con historia de commits representativa
- Versión web con Streamlit (sugerencia del profe)
- Mejorar interfaz: reemplazar teclas 1-8 por botones o menú desplegable
- Informe técnico final en PDF
- Pruebas automatizadas del Core (opcional)
- Anexo de uso de IA (opcional si la usamos)

---

## 8. Plan de trabajo – Próximas etapas

| Etapa | Tarea | Estado |
|-------|-------|--------|
| 1 | Subir proyecto a Git con commits | Pendiente |
| 2 | Refactorizar módulos aplicando POO | Pendiente |
| 3 | Crear versión web con Streamlit | Pendiente |
| 4 | Mejorar interfaz OpenCV (botones/menú) | Pendiente |
| 5 | Redactar informe técnico final en PDF | Pendiente |
| 6 | Agregar pruebas automatizadas del Core | Pendiente |
| 7 | Preparar anexo de uso de IA | Pendiente |

---

## 9. Cómo ejecutar el proyecto

**1. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**2. Descargar el dataset desde Kaggle y colocarlo en la carpeta `images/`**

**3. Ejecutar:**
```bash
python main.py
```
