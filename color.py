"""
Módulo de conversión de espacios de color.

Unidad 7 – Técnicas avanzadas: conversión a HSV para segmentación por color.
Unidad 2 – Fundamentos: representación de imágenes en distintos espacios.

El espacio BGR es el formato nativo de OpenCV (orden invertido de RGB).
Convertir a otros espacios de color permite separar la información de tono,
saturación y luminosidad, lo que facilita tareas de segmentación y análisis.
"""

import cv2
import numpy as np


def bgr_a_hsv(imagen):
    """
    Convierte una imagen de espacio BGR a HSV (Hue-Saturation-Value).

    HSV separa el tono del color (Hue) de la saturación (Saturation) y el
    brillo (Value). Esto es muy útil para segmentar por color sin que las
    variaciones de iluminación afecten el resultado, ya que el canal H
    codifica el color puro independientemente del brillo.

    En OpenCV:
      - H: 0–180 (rojo=0/180, verde=60, azul=120)
      - S: 0–255 (0 = gris, 255 = color puro)
      - V: 0–255 (0 = negro, 255 = máximo brillo)

    Args:
        imagen: Imagen en formato BGR (numpy array H×W×3)

    Returns:
        Imagen convertida a espacio HSV (numpy array H×W×3)
    """
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)


def bgr_a_hls(imagen):
    """
    Convierte una imagen de espacio BGR a HLS (Hue-Lightness-Saturation).

    Similar a HSV pero usa Lightness (luminosidad perceptual) en lugar de
    Value. Es útil cuando se necesita normalizar la luminosidad de la imagen
    para análisis de color más robustos. A diferencia de HSV, en HLS el
    blanco puro se representa con L=255 y S=0.

    En OpenCV:
      - H: 0–180
      - L: 0–255 (0 = negro, 128 = color puro, 255 = blanco)
      - S: 0–255

    Args:
        imagen: Imagen en formato BGR (numpy array H×W×3)

    Returns:
        Imagen convertida a espacio HLS (numpy array H×W×3)
    """
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)


def bgr_a_gris(imagen):
    """
    Convierte una imagen BGR a escala de grises.

    La conversión a grises reduce la dimensionalidad de 3 canales a 1,
    aplicando la fórmula ponderada por la sensibilidad del ojo humano:
        Gris = 0.114·B + 0.587·G + 0.299·R

    Es el preprocesamiento más común antes de aplicar umbralización,
    detección de bordes u operaciones morfológicas.

    Args:
        imagen: Imagen en formato BGR (numpy array H×W×3)

    Returns:
        Imagen en escala de grises (numpy array H×W, un solo canal)
    """
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def obtener_canal_hue(imagen_bgr):
    """
    Extrae el canal Hue de una imagen BGR.

    El canal H (tono) es el más discriminativo para identificar el tipo de
    fruta por su color característico: manzanas (rojas/verdes), bananas
    (amarillas), naranjas (naranja), etc. Al ser independiente del brillo,
    es robusto ante cambios de iluminación.

    Args:
        imagen_bgr: Imagen en formato BGR (numpy array H×W×3)

    Returns:
        Canal Hue como imagen de un solo canal (numpy array H×W, uint8)
    """
    hsv = bgr_a_hsv(imagen_bgr)
    return hsv[:, :, 0]


def ajustar_brillo_contraste(imagen_bgr, brillo=0, contraste=1.0):
    """
    Ajusta brillo y contraste con transformación lineal: resultado = clip(img * contraste + brillo).

    brillo: offset aditivo (-255 a +255), 0 = sin cambio
    contraste: factor multiplicativo (0.0 a 3.0), 1.0 = sin cambio

    Args:
        imagen_bgr: Imagen en formato BGR
        brillo: Desplazamiento de intensidad
        contraste: Factor de escala de intensidad

    Returns:
        Imagen BGR ajustada (numpy array H×W×3, uint8)
    """
    resultado = imagen_bgr.astype(np.float32) * contraste + brillo
    return np.clip(resultado, 0, 255).astype(np.uint8)


def ajustar_hsv(imagen_bgr, delta_h=0, escala_s=1.0, escala_v=1.0):
    """
    Modifica el tono (H), saturación (S) y brillo (V) de una imagen BGR.

    Convierte a HSV, aplica los ajustes y convierte de vuelta a BGR.
    El canal H usa módulo 180 para manejar el wrapping del tono (rojo ↔ rojo).

    delta_h: desplazamiento en H (−90 a +90). H en OpenCV es 0–180.
    escala_s: factor multiplicativo en S (0.0 a 2.0), 1.0 = sin cambio
    escala_v: factor multiplicativo en V (0.0 a 2.0), 1.0 = sin cambio

    Args:
        imagen_bgr: Imagen en formato BGR
        delta_h: Desplazamiento del tono
        escala_s: Factor de saturación
        escala_v: Factor de brillo (Value)

    Returns:
        Imagen BGR con ajustes HSV aplicados (numpy array H×W×3, uint8)
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta_h) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * escala_s, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * escala_v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def hsv_para_display(imagen_bgr):
    """
    Convierte BGR a HSV y devuelve una versión visual BGR del espacio HSV.

    Para mostrar en pantalla, los valores HSV se interpretan directamente
    como valores BGR, generando una representación con colores falsos que
    permite ver visualmente cómo el espacio HSV separa los tonos de color.

    Args:
        imagen_bgr: Imagen en formato BGR

    Returns:
        Imagen BGR donde los canales contienen los valores H, S, V
        (visualización de falso color del espacio HSV)
    """
    hsv = bgr_a_hsv(imagen_bgr)
    # Escalar H de 0–180 a 0–255 para visualización uniforme
    hsv_display = hsv.copy()
    hsv_display[:, :, 0] = (hsv[:, :, 0].astype(np.float32) / 180.0 * 255).astype(np.uint8)
    return hsv_display
