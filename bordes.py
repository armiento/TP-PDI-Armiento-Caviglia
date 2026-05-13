"""
Módulo de detección de bordes.

Unidad 5 – Filtros y detección de bordes:
  - Filtro gaussiano para reducción de ruido (preprocesamiento)
  - Detección de bordes con algoritmo de Canny
  - Detección de bordes con operador de Sobel

Los bordes de una imagen corresponden a cambios bruscos de intensidad entre
píxeles vecinos. Detectarlos permite identificar contornos, formas y límites
entre objetos, como el contorno de una fruta sobre el fondo.
"""

import cv2
import numpy as np


def aplicar_blur_gaussiano(imagen, kernel_size=(5, 5), sigma=0):
    """
    Aplica filtro gaussiano para reducir ruido antes de detectar bordes.

    El filtro gaussiano promedia cada píxel con sus vecinos usando pesos
    que decaen según una distribución gaussiana. Elimina variaciones de
    alta frecuencia (ruido) que, de no eliminarse, generarían falsos bordes
    en los pasos siguientes de detección.

    El parámetro sigma controla el grado de suavizado: mayor sigma implica
    más suavizado y menos sensibilidad al ruido pero también a detalles finos.
    Con sigma=0, OpenCV calcula el valor óptimo a partir de kernel_size.

    Args:
        imagen: Imagen de entrada (BGR o escala de grises)
        kernel_size: Tamaño del kernel gaussiano (ambas dimensiones deben
                     ser impares, e.g., (3,3), (5,5), (7,7))
        sigma: Desviación estándar de la gaussiana (0 = automático)

    Returns:
        Imagen suavizada con el mismo tipo y forma que la entrada
    """
    return cv2.GaussianBlur(imagen, kernel_size, sigma)


def detectar_bordes_canny(imagen_bgr, umbral_bajo=50, umbral_alto=150):
    """
    Detecta bordes usando el algoritmo de Canny (1986).

    Canny es considerado el detector de bordes óptimo. Su pipeline es:
      1. Suavizado gaussiano para eliminar ruido
      2. Gradiente de Sobel en X e Y para calcular magnitud y dirección
      3. Supresión de no-máximos: adelgaza los bordes a 1 píxel de grosor
      4. Histéresis de doble umbral:
         - Píxeles con gradiente > umbral_alto: borde seguro
         - Píxeles con gradiente < umbral_bajo: descartados
         - Intermedios: borde solo si conecta con un borde seguro

    La relación recomendada entre umbrales es 1:2 o 1:3.

    Args:
        imagen_bgr: Imagen en formato BGR
        umbral_bajo: Umbral inferior de histéresis (típico: 30–100)
        umbral_alto: Umbral superior de histéresis (típico: 100–300)

    Returns:
        Imagen binaria (uint8): 255 en bordes, 0 en el resto
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = aplicar_blur_gaussiano(gris, kernel_size=(5, 5))
    bordes = cv2.Canny(suavizada, umbral_bajo, umbral_alto)
    return bordes


def detectar_bordes_sobel(imagen_bgr):
    """
    Detecta bordes usando el operador de Sobel.

    Sobel aplica dos kernels de convolución 3×3 que aproximan las derivadas
    parciales de la imagen en dirección horizontal (Gx) y vertical (Gy):

        Gx = [[-1, 0, +1],      Gy = [[-1, -2, -1],
               [-2, 0, +2],            [ 0,  0,  0],
               [-1, 0, +1]]            [+1, +2, +1]]

    La magnitud del gradiente es: G = √(Gx² + Gy²)

    A diferencia de Canny, Sobel produce bordes con grosor variable y es
    más sensible al ruido, pero permite obtener información de dirección
    de los bordes, útil para análisis de orientación.

    Args:
        imagen_bgr: Imagen en formato BGR

    Returns:
        Imagen uint8 con la magnitud del gradiente Sobel (normalizada 0–255)
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = aplicar_blur_gaussiano(gris, kernel_size=(3, 3))

    # Derivadas parciales en X e Y con precisión float64
    sobel_x = cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitud del gradiente
    magnitud = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalizar a rango 0–255 para visualización
    if magnitud.max() > 0:
        magnitud = (magnitud / magnitud.max() * 255).astype(np.uint8)
    else:
        magnitud = magnitud.astype(np.uint8)

    return magnitud


def bordes_a_bgr(imagen_bordes):
    """
    Convierte una imagen de bordes (1 canal) a formato BGR para el panel.

    Las funciones de detección de bordes devuelven imágenes de un solo canal.
    Esta función las convierte a 3 canales para poder concatenarlas con
    otras imágenes BGR en el panel de visualización.

    Args:
        imagen_bordes: Imagen de bordes en escala de grises (H×W, uint8)

    Returns:
        Imagen BGR equivalente (H×W×3, uint8)
    """
    return cv2.cvtColor(imagen_bordes, cv2.COLOR_GRAY2BGR)


def canny_coloreado(imagen_bgr, umbral_bajo=50, umbral_alto=150):
    """
    Devuelve bordes Canny coloreados en verde sobre fondo negro.

    Versión visualmente mejorada de los bordes para el panel: en lugar de
    blanco/negro, los bordes se muestran en color verde brillante para
    contrastar mejor con el panel oscuro.

    Args:
        imagen_bgr: Imagen en formato BGR
        umbral_bajo: Umbral inferior de Canny
        umbral_alto: Umbral superior de Canny

    Returns:
        Imagen BGR con bordes en verde (H×W×3, uint8)
    """
    bordes = detectar_bordes_canny(imagen_bgr, umbral_bajo, umbral_alto)
    resultado = np.zeros((*bordes.shape, 3), dtype=np.uint8)
    resultado[bordes > 0] = [0, 255, 0]  # Verde
    return resultado
