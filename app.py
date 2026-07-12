"""
Versión web del Editor Interactivo de Imágenes de Frutas.

Trabajo Práctico Final – Técnicas de Procesamiento Digital de Imágenes
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
IFTS N°18 – CABA – Ciclo 2026
Profesor: Bonini Juan Ignacio

Interfaz web construida con Streamlit. Usa Motor como único punto de
acceso al core de procesamiento, exactamente igual que main.py.
El usuario sube su propia imagen y ajusta los parámetros con sliders;
cada cambio redibuja la imagen procesada automáticamente.

Ejecutar con:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from motor import Motor

# ===========================================================================
# PASO 1: CONFIGURACIÓN DE LA PÁGINA WEB (Debe ser el primer comando de Streamlit)
# ===========================================================================
st.set_page_config(
    page_title="Editor PDI – Frutas",
    layout="wide",
)

# Diccionario estático con los nombres de las técnicas de cada unidad
MODOS = {
    1: "1 – Blur / Brillo / Contraste",
    2: "2 – Bordes Canny",
    3: "3 – Segmentacion K-means HSV",
    4: "4 – Umbralizacion Otsu",
    5: "5 – Watershed",
    6: "6 – Histograma",
    7: "7 – Transformada de Fourier",
    8: "8 – Ajuste HSV",
}

ANCHO_MAX = 640  # Límite para redimensionar imágenes muy pesadas y optimizar el procesamiento

# ===========================================================================
# PASO 2: UTILIDADES DE CONVERSION (Puente entre matrices OpenCV y visualización Web)
# ===========================================================================

def pil_a_bgr(imagen_pil: Image.Image) -> np.ndarray:
    """Convierte PIL Image (RGB) a numpy array BGR para pasarle al Motor analítico."""
    rgb = np.array(imagen_pil.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def bgr_a_rgb(imagen_bgr: np.ndarray) -> np.ndarray:
    """Convierte numpy array BGR a RGB para que Streamlit pueda renderizarla con st.image()."""
    return imagen_bgr[:, :, ::-1].copy()


def redimensionar_bgr(imagen_bgr: np.ndarray, ancho_max: int) -> np.ndarray:
    """Reduce el tamaño de la imagen si supera el ancho_max, manteniendo el aspecto original."""
    h, w = imagen_bgr.shape[:2]
    if w <= ancho_max:
        return imagen_bgr
    nueva_h = int(h * ancho_max / w)
    pil = Image.fromarray(imagen_bgr[:, :, ::-1].astype(np.uint8))
    pil = pil.resize((ancho_max, nueva_h), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)[:, :, ::-1].copy()


# ===========================================================================
# PASO 3: CONSTRUCCIÓN DEL SIDEBAR (Panel izquierdo interactivo)
# ===========================================================================

with st.sidebar:
    st.header("Controles")

    # Componente nativo para que el usuario arrastre o suba su propio archivo de imagen
    archivo = st.file_uploader(
        "Subir imagen",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formatos soportados: JPG, PNG, BMP.",
    )

    st.markdown("---")

    # Selector de técnica desplegable (Dropdown web)
    nombre_modo = st.selectbox(
        "Tecnica de procesamiento",
        options=list(MODOS.values()),
    )
    # Mapeo invertido para extraer el número entero de modo actual en base al texto seleccionado
    modo = next(k for k, v in MODOS.items() if v == nombre_modo)

    st.markdown("---")
    st.subheader("Parametros")

    # Diccionario local dinámico para almacenar los valores de los sliders seleccionados
    params: dict = {}

    # Inyección condicional de sliders en la interfaz web según el modo activo
    if modo == 1: # MODO 1: BLUR / BRILLO / CONTRASTE
        params["blur_val"] = st.slider("Blur  (0 = sin blur, 10 = maximo)", 0, 10, 2)
        params["brillo"] = st.slider("Brillo (offset  −100 a +100)", -100, 100, 0)
        params["contraste"] = st.slider("Contraste (factor)", 0.1, 3.0, 1.0, step=0.1)

    elif modo == 2: # MODO 2: BORDES CANNY
        params["umbral_bajo"] = st.slider("Umbral bajo Canny", 0, 300, 40)
        params["umbral_alto"] = st.slider("Umbral alto Canny", 0, 300, 120)

    elif modo == 3: # MODO 3: SEGMENTACIÓN K-MEANS HSV
        params["k"] = st.slider("Numero de clusters (k)", 2, 6, 4)

    elif modo == 4: # MODO 4: UMBRALIZACIÓN OTSU
        params["umbral_manual"] = st.slider(
            "Umbral  (0 = automatico Otsu, 1–255 = manual)", 0, 255, 0
        )

    elif modo == 5: # MODO 5: WATERSHED
        params["umbral_pct"] = st.slider(
            "Umbral transformada de distancia (%)",
            1, 90, 50,
            help="Valores bajos: semillas grandes, menos separacion. Valores altos: semillas pequeñas, mas separacion.",
        )

    elif modo == 6: # MODO 6: HISTOGRAMA ANALÍTICO
        canal_nombre = st.radio(
            "Canal a visualizar",
            ["Grises", "R (rojo)", "G (verde)", "B (azul)"],
        )
        params["canal"] = ["Grises", "R (rojo)", "G (verde)", "B (azul)"].index(canal_nombre)

    elif modo == 7: # MODO 7: FILTRADO DE FOURIER (FFT)
        params["radio"] = st.slider(
            "Radio filtro pasa bajos (px)  –  0 = mostrar espectro",
            0, 100, 0,
            help="Con radio=0 se muestra el espectro de frecuencias. Con radio>0 se reconstruye la imagen.",
        )

    elif modo == 8: # MODO 8: AJUSTE CROMÁTICO HSV
        params["delta_h"] = st.slider("Tono H (offset)", -90, 90, 0)
        params["escala_s"] = st.slider("Saturacion (%)", 0, 200, 100)
        params["escala_v"] = st.slider("Brillo V (%)", 0, 200, 100)


# ===========================================================================
# PASO 4: ÁREA PRINCIPAL DE LA INTERFAZ WEB (Estructura de la vista)
# ===========================================================================

st.title("Editor Interactivo de Imagenes de Frutas")
st.caption("TP Final – Tecnicas de Procesamiento Digital de Imagenes · IFTS N°18 · Ciclo 2026")
st.markdown(f"**Tecnica activa:** {nombre_modo}")
st.markdown("---")

# Interrupción segura del flujo web si el usuario no ha arrastrado ningún archivo todavía
if archivo is None:
    st.info("Subi una imagen desde el panel izquierdo para comenzar.")
    st.stop()

# Carga de la imagen en memoria y normalización de su tamaño de forma relativa
imagen_bgr = redimensionar_bgr(pil_a_bgr(Image.open(archivo)), ANCHO_MAX)
h, w = imagen_bgr.shape[:2]

# Inicializamos el Motor analítico con la imagen bgr de la fruta
motor = Motor(imagen_bgr)


# ===========================================================================
# PASO 5: NÚCLEO DE PROCESAMIENTO (Invocación segura de los algoritmos del Motor)
# ===========================================================================
try:
    if modo == 1: # MODO 1: BLUR / BRILLO / CONTRASTE
        kernel = params["blur_val"] * 2 + 1  
        procesada_bgr = motor.blur_y_brillo(kernel, params["brillo"], params["contraste"])
        info = f"kernel={kernel}px   brillo={params['brillo']:+d}   contraste={params['contraste']:.1f}x"

    elif modo == 2: # MODO 2: BORDES CANNY
        procesada_bgr = motor.bordes_canny(params["umbral_bajo"], params["umbral_alto"])
        info = f"umbral bajo={params['umbral_bajo']}   umbral alto={params['umbral_alto']}"

    elif modo == 3: # MODO 3: SEGMENTACIÓN K-MEANS HSV
        procesada_bgr = motor.segmentar_kmeans(params["k"])
        info = f"k = {params['k']} clusters de color"

    elif modo == 4: # MODO 4: UMBRALIZACIÓN OTSU
        procesada_bgr = motor.umbralizar_otsu(params["umbral_manual"])
        info = "umbral = auto (Otsu)" if params["umbral_manual"] == 0 else f"umbral = {params['umbral_manual']}"

    elif modo == 5: # MODO 5: WATERSHED MORFOLÓGICO
        umbral_dist = max(0.01, params["umbral_pct"] / 100.0)
        procesada_bgr = motor.aplicar_watershed(umbral_dist)
        info = f"umbral transformada de distancia = {params['umbral_pct']}% del maximo"

    elif modo == 6: # MODO 6: VISUALIZACIÓN DE HISTOGRAMAS
        procesada_bgr = motor.histograma_canal(params["canal"], alto=h, ancho=w)
        nombres = {0: "Grises", 1: "R (rojo)", 2: "G (verde)", 3: "B (azul)"}
        info = f"canal = {nombres[params['canal']]}   (original + ecualizado superpuestos)"

    elif modo == 7: # MODO 7: TRANSFORMADA DE FOURIER (FFT)
        procesada_bgr = motor.fourier(params["radio"])
        info = f"radio = {params['radio']}px   " + ("espectro sin filtro" if params["radio"] == 0 else "imagen reconstruida con pasa bajos")

    elif modo == 8: # MODO 8: AJUSTE CROMÁTICO HSV
        escala_s = params["escala_s"] / 100.0
        escala_v = params["escala_v"] / 100.0
        procesada_bgr = motor.ajustar_hsv(params["delta_h"], escala_s, escala_v)
        info = f"H {params['delta_h']:+d}   Saturacion={params['escala_s']}%   Brillo={params['escala_v']}%"

    else:
        procesada_bgr = imagen_bgr.copy()
        info = ""

except Exception as e:
    st.error(f"Error al procesar la imagen: {e}")
    st.stop()


# ===========================================================================
# PASO 6: PERSISTENCIA WEB (Codificación y Botón de Descarga)
# ===========================================================================
# Una vez calculada con éxito la matriz 'procesada_bgr' en el paso 5, procedemos a exportarla
exito, buffer = cv2.imencode('.png', procesada_bgr)

if exito:
    # Transformamos el búfer binario en bytes puros para la transmisión web HTTP
    img_bytes = buffer.tobytes()
    
    # Renderizado del botón nativo de descargas del cliente web
    st.download_button(
        label=" Descargar Imagen Procesada",
        data=img_bytes,
        file_name=f"fruta_procesada_modo_{modo}.png",
        mime="image/png"
    )

st.markdown("---")


# ===========================================================================
# PASO 7: RENDERIZACIÓN GRÁFICA LADO A LADO (Visualización final en dos columnas)
# ===========================================================================
col_orig, col_proc = st.columns(2)

with col_orig:
    st.subheader("Original")
    # Pasamos la matriz BGR a RGB justo antes de renderizarla en la interfaz web
    st.image(bgr_a_rgb(imagen_bgr), use_container_width=True) # <-- CAMBIADO ACÁ

with col_proc:
    st.subheader(nombre_modo)
    # Pasamos el resultado BGR procesado a RGB justo antes de renderizarlo en la interfaz web
    st.image(bgr_a_rgb(procesada_bgr), use_container_width=True) # 