"""
Módulo de cálculo y visualización de histogramas.

Unidad 4 – Histogramas:
  - Cálculo del histograma de la imagen original
  - Ecualización de histograma para mejorar contraste
  - Comparación histograma antes/después del procesamiento

El histograma de una imagen es la distribución de frecuencias de los niveles
de intensidad (0–255). Permite analizar el contraste, brillo y rango dinámico
de la imagen de forma cuantitativa.
"""

import cv2
import numpy as np


def calcular_histograma_gris(imagen_gris):
    """
    Calcula el histograma de una imagen en escala de grises.

    Cuenta cuántos píxeles tienen cada nivel de intensidad (0 a 255).
    Un histograma concentrado en valores bajos indica imagen oscura;
    concentrado en valores altos indica imagen clara; un histograma
    extendido indica buen contraste.

    Args:
        imagen_gris: Imagen en escala de grises (numpy array H×W, uint8)

    Returns:
        Array 1D de 256 elementos con el conteo de píxeles por nivel
    """
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])
    return hist.flatten()


def ecualizar_histograma(imagen_gris):
    """
    Aplica ecualización de histograma para mejorar el contraste global.

    La ecualización transforma los niveles de intensidad de forma que el
    histograma resultante sea aproximadamente uniforme (plano). Esto estira
    el rango dinámico de la imagen y mejora el contraste, revelando detalles
    en zonas muy oscuras o muy claras que estaban comprimidas.

    Matemáticamente aplica la transformación:
        s = (L-1) · CDF(r)
    donde CDF es la función de distribución acumulada del histograma original.

    Args:
        imagen_gris: Imagen en escala de grises (numpy array H×W, uint8)

    Returns:
        Imagen con histograma ecualizado (mismo tamaño, uint8)
    """
    return cv2.equalizeHist(imagen_gris)


def dibujar_histograma_como_imagen(imagen_bgr, alto=256, ancho=256):
    """
    Dibuja el histograma de los tres canales BGR sobre un lienzo OpenCV.

    Renderiza el histograma directamente como imagen numpy para poder
    mostrarlo en el panel principal de OpenCV sin necesidad de matplotlib.
    Cada canal se dibuja en su color correspondiente:
      - Azul (B), Verde (G), Rojo (R)

    Args:
        imagen_bgr: Imagen original en formato BGR
        alto: Altura del lienzo en píxeles (determina la escala vertical)
        ancho: Ancho del lienzo en píxeles

    Returns:
        Imagen BGR (numpy array alto×ancho×3) con el histograma dibujado
    """
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    colores_bgr = [
        (255, 80, 80),   # Azul
        (80, 255, 80),   # Verde
        (80, 80, 255),   # Rojo
    ]

    for canal_idx, color in enumerate(colores_bgr):
        hist = cv2.calcHist([imagen_bgr], [canal_idx], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, alto - 15, cv2.NORM_MINMAX)
        hist = hist.flatten().astype(int)

        for x in range(1, 256):
            x1 = int((x - 1) * ancho / 256)
            x2 = int(x * ancho / 256)
            y1 = alto - hist[x - 1]
            y2 = alto - hist[x]
            cv2.line(lienzo, (x1, y1), (x2, y2), color, 1)

    # Etiquetas de canales
    cv2.putText(lienzo, "B", (5, alto - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 80, 80), 1)
    cv2.putText(lienzo, "G", (20, alto - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (80, 255, 80), 1)
    cv2.putText(lienzo, "R", (35, alto - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (80, 80, 255), 1)

    return lienzo


def dibujar_histograma_canal(imagen_bgr, canal=0, alto=256, ancho=256):
    """
    Dibuja el histograma de un canal individual con comparación antes/después de ecualizar.

    Permite ver cómo la distribución de intensidades varía por canal de color,
    lo que es útil para analizar el balance de color de la imagen de fruta.

    Mapeo de canal:
      0 → Grises (conversión a escala de grises)
      1 → R (canal rojo,  índice BGR 2)
      2 → G (canal verde, índice BGR 1)
      3 → B (canal azul,  índice BGR 0)

    Args:
        imagen_bgr: Imagen en formato BGR
        canal: Canal a visualizar (0=Gris, 1=R, 2=G, 3=B)
        alto: Altura del lienzo en píxeles
        ancho: Ancho del lienzo en píxeles

    Returns:
        Imagen BGR con el histograma original (claro) y ecualizado (color del canal)
    """
    # Extraer el canal seleccionado
    _nombres = {0: "Gris", 1: "R", 2: "G", 3: "B"}
    _colores_orig = {
        0: (160, 160, 160),  # gris
        1: (80,  80,  220),  # rojo (BGR: R alto)
        2: (80,  200, 80),   # verde
        3: (220, 80,  80),   # azul
    }
    _colores_ecual = {
        0: (0,   220, 220),  # cian
        1: (50,  50,  255),  # rojo brillante
        2: (50,  255, 50),   # verde brillante
        3: (255, 50,  50),   # azul brillante
    }
    # Índice en el array BGR (canal 0=Gris es especial)
    _bgr_idx = {1: 2, 2: 1, 3: 0}

    if canal == 0:
        datos = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    else:
        datos = imagen_bgr[:, :, _bgr_idx[canal]]

    datos_ecual = cv2.equalizeHist(datos)

    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for img_c, color in [(datos, _colores_orig[canal]),
                         (datos_ecual, _colores_ecual[canal])]:
        hist = cv2.calcHist([img_c], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, alto - 20, cv2.NORM_MINMAX)
        hist = hist.flatten().astype(int)
        for x in range(1, 256):
            x1 = int((x - 1) * ancho / 256)
            x2 = int(x * ancho / 256)
            cv2.line(lienzo, (x1, alto - hist[x - 1]), (x2, alto - hist[x]), color, 1)

    nombre = _nombres[canal]
    cv2.putText(lienzo, f"Canal: {nombre}", (5, alto - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _colores_orig[canal], 1)
    cv2.putText(lienzo, "Ecualizado", (5, alto - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _colores_ecual[canal], 1)

    return lienzo


def dibujar_histograma_con_ecualizacion(imagen_bgr, alto=256, ancho=256):
    """
    Dibuja histograma original (gris) y ecualizado (amarillo) superpuestos.

    Permite comparar visualmente el efecto de la ecualización: el histograma
    original muestra la distribución real, mientras que el ecualizado muestra
    cómo la transformación distribuye los píxeles de forma más uniforme.

    Args:
        imagen_bgr: Imagen en formato BGR
        alto: Altura del lienzo
        ancho: Ancho del lienzo

    Returns:
        Imagen BGR con ambos histogramas superpuestos
    """
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gris_ecualizado = cv2.equalizeHist(gris)

    for img, color in [(gris, (160, 160, 160)), (gris_ecualizado, (0, 220, 220))]:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, alto - 20, cv2.NORM_MINMAX)
        hist = hist.flatten().astype(int)

        for x in range(1, 256):
            x1 = int((x - 1) * ancho / 256)
            x2 = int(x * ancho / 256)
            y1 = alto - hist[x - 1]
            y2 = alto - hist[x]
            cv2.line(lienzo, (x1, y1), (x2, y2), color, 1)

    cv2.putText(lienzo, "Orig", (5, alto - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (160, 160, 160), 1)
    cv2.putText(lienzo, "Ecual.", (40, alto - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 220, 220), 1)

    return lienzo
