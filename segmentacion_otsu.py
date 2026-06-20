"""
Módulo de umbralización con método de Otsu.

Unidad 7 – Técnicas avanzadas: umbralización automática con método de Otsu.

La umbralización convierte una imagen en escala de grises en una imagen binaria
(blanco/negro) eligiendo un valor de umbral T: píxeles con intensidad > T se
vuelven blancos (objeto), el resto negro (fondo).

El método de Otsu (1979) calcula automáticamente el umbral óptimo que maximiza
la varianza inter-clase entre los dos grupos (fondo y objeto), sin necesidad de
parámetros manuales. Es ideal para imágenes con histograma bimodal.
"""

import cv2
import numpy as np
from morfologia import limpiar_mascara


def umbralizar_otsu(imagen_bgr):
    """
    Aplica umbralización automática con el método de Otsu.

    Proceso completo:
      1. Convertir BGR a escala de grises
      2. Aplicar filtro gaussiano para reducir ruido (mejora Otsu)
      3. Calcular umbral óptimo con el método de Otsu
      4. Binarizar la imagen con ese umbral
      5. Limpiar la máscara con operaciones morfológicas
      6. Aplicar la máscara a la imagen original

    El método de Otsu minimiza la varianza intra-clase, equivalente a
    maximizar la varianza entre las dos clases (fondo vs. objeto):

        σ²_b(T) = w₀(T)·w₁(T)·[μ₀(T) - μ₁(T)]²

    donde w₀, w₁ son las probabilidades de cada clase y μ₀, μ₁ sus medias.

    Args:
        imagen_bgr: Imagen en formato BGR (numpy array H×W×3)

    Returns:
        Tuple (mascara_limpia, imagen_recortada):
          - mascara_limpia: Máscara binaria donde la fruta es blanca (255)
          - imagen_recortada: Imagen original con el fondo enmascarado (negro)
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    # Preprocesamiento: blur gaussiano reduce ruido que distorsiona Otsu
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu calcula automáticamente el umbral óptimo (valor_umbral devuelto)
    valor_umbral, mascara = cv2.threshold(
        suavizada,
        0,       # Se ignora cuando se usa THRESH_OTSU
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Limpiar con morfología: elimina ruido y rellena huecos
    mascara_limpia = limpiar_mascara(mascara)

    # Aplicar máscara: solo conservar píxeles donde la máscara es blanca
    imagen_recortada = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_limpia)

    return mascara_limpia, imagen_recortada


def umbralizar_otsu_hsv(imagen_bgr):
    """
    Umbralización con Otsu aplicada al canal V (Value/Brillo) del espacio HSV.

    El canal V de HSV representa el brillo de cada píxel independientemente
    de su color. En imágenes de frutas con fondo claro o fondo oscuro,
    el canal V puede separar mejor el objeto del fondo que la escala de grises
    estándar (que promedia los tres canales BGR).

    Args:
        imagen_bgr: Imagen en formato BGR

    Returns:
        Tuple (mascara_limpia, imagen_recortada)
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    canal_v = hsv[:, :, 2]

    suavizado = cv2.GaussianBlur(canal_v, (5, 5), 0)
    _, mascara = cv2.threshold(
        suavizado, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mascara_limpia = limpiar_mascara(mascara)
    imagen_recortada = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_limpia)

    return mascara_limpia, imagen_recortada


def otsu_con_overlay(imagen_bgr):
    """
    Muestra la máscara de Otsu superpuesta sobre la imagen original.

    Genera una visualización donde:
      - Las zonas detectadas como objeto (blanco) se muestran con la imagen real
      - Las zonas de fondo se muestran oscurecidas en azul semi-transparente

    Esto facilita evaluar visualmente la calidad de la segmentación de Otsu.

    Args:
        imagen_bgr: Imagen en formato BGR

    Returns:
        Imagen BGR con el overlay de la máscara (numpy array H×W×3, uint8)
    """
    mascara, _ = umbralizar_otsu(imagen_bgr)

    # Crear versión oscurecida para el fondo
    fondo = (imagen_bgr * 0.3).astype(np.uint8)
    fondo[fondo > 0] = fondo[fondo > 0] // 2
    fondo[:, :, 0] = np.clip(fondo[:, :, 0].astype(int) + 60, 0, 255)  # Tinte azul

    resultado = fondo.copy()
    resultado[mascara == 255] = imagen_bgr[mascara == 255]

    # Contorno de la máscara en amarillo
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultado, contornos, -1, (0, 255, 255), 1)

    return resultado


def mascara_a_bgr(mascara):
    """
    Convierte una máscara binaria de 1 canal a formato BGR de 3 canales.

    Necesario para poder concatenar imágenes de 1 canal con imágenes BGR
    en el panel de visualización de OpenCV.

    Args:
        mascara: Imagen binaria de un canal (numpy array H×W, uint8)

    Returns:
        Imagen BGR equivalente (numpy array H×W×3, uint8)
    """
    return cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
