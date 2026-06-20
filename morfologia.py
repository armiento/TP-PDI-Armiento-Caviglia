"""
Módulo de operaciones morfológicas.

Unidad 2 – Fundamentos: erosión, dilatación, apertura y cierre morfológico.
Unidad 7 – Técnicas avanzadas: limpieza de máscaras de segmentación.

La morfología matemática opera sobre imágenes binarias usando un elemento
estructurante (kernel) que define la forma del "vecindario" analizado.
Las cuatro operaciones básicas son: erosión, dilatación, apertura y cierre.
Se aplican típicamente después de la segmentación para limpiar el resultado.
"""

import cv2
import numpy as np


def obtener_elemento_estructurante(forma=cv2.MORPH_ELLIPSE, tamanio=(5, 5)):
    """
    Crea un elemento estructurante para operaciones morfológicas.

    El elemento estructurante define la forma y tamaño del vecindario que se
    analiza en cada operación morfológica. La elipse es ideal para objetos
    naturales como frutas, que tienen contornos redondeados. El rectángulo
    es más agresivo en los bordes, y la cruz afecta menos las esquinas.

    Formas disponibles:
      - cv2.MORPH_ELLIPSE: Elipse (recomendado para frutas)
      - cv2.MORPH_RECT:    Rectángulo
      - cv2.MORPH_CROSS:   Cruz (+)

    Args:
        forma: Constante OpenCV para la forma del kernel
        tamanio: Tupla (ancho, alto) en píxeles

    Returns:
        Elemento estructurante como numpy array uint8
    """
    return cv2.getStructuringElement(forma, tamanio)


def erosion(imagen_binaria, kernel=None, iteraciones=1):
    """
    Aplica erosión morfológica a una imagen binaria.

    Un píxel de salida es blanco (255) solo si TODOS los píxeles del vecindario
    definido por el kernel también son blancos. En caso contrario, el píxel
    se vuelve negro (0). Efecto: las regiones blancas se encogen desde sus bordes.

    Usos:
      - Eliminar pequeños objetos blancos (ruido de sal)
      - Separar objetos que se tocan levemente
      - Reducir el tamaño de las regiones segmentadas

    Args:
        imagen_binaria: Imagen binaria o máscara (numpy array H×W, uint8)
        kernel: Elemento estructurante (None = elipse 5×5 por defecto)
        iteraciones: Número de veces que se aplica la erosión

    Returns:
        Imagen erosionada (mismo tamaño, uint8)
    """
    if kernel is None:
        kernel = obtener_elemento_estructurante()
    return cv2.erode(imagen_binaria, kernel, iterations=iteraciones)


def dilatacion(imagen_binaria, kernel=None, iteraciones=1):
    """
    Aplica dilatación morfológica a una imagen binaria.

    Un píxel de salida es blanco (255) si AL MENOS UN píxel del vecindario
    definido por el kernel es blanco. Efecto: las regiones blancas se expanden
    hacia sus bordes.

    Usos:
      - Rellenar pequeños huecos negros en objetos blancos (ruido de pimienta)
      - Conectar regiones blancas cercanas
      - Ampliar las regiones segmentadas

    Args:
        imagen_binaria: Imagen binaria o máscara (numpy array H×W, uint8)
        kernel: Elemento estructurante (None = elipse 5×5 por defecto)
        iteraciones: Número de veces que se aplica la dilatación

    Returns:
        Imagen dilatada (mismo tamaño, uint8)
    """
    if kernel is None:
        kernel = obtener_elemento_estructurante()
    return cv2.dilate(imagen_binaria, kernel, iterations=iteraciones)


def apertura(imagen_binaria, kernel=None):
    """
    Aplica apertura morfológica (erosión seguida de dilatación).

    La apertura es erosión + dilatación con el mismo elemento estructurante.
    Elimina objetos blancos pequeños (más pequeños que el kernel) sin cambiar
    significativamente el tamaño de los objetos grandes. También suaviza los
    contornos de los objetos.

    Fórmula: Apertura(A, B) = Dilatación(Erosión(A, B), B)

    Usos:
      - Eliminar ruido puntual en máscaras de segmentación
      - Remover conexiones delgadas entre objetos
      - Suavizar contornos irregulares

    Args:
        imagen_binaria: Imagen binaria o máscara (numpy array H×W, uint8)
        kernel: Elemento estructurante (None = elipse 5×5 por defecto)

    Returns:
        Imagen con apertura morfológica aplicada (mismo tamaño, uint8)
    """
    if kernel is None:
        kernel = obtener_elemento_estructurante()
    return cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)


def cierre(imagen_binaria, kernel=None):
    """
    Aplica cierre morfológico (dilatación seguida de erosión).

    El cierre es dilatación + erosión con el mismo elemento estructurante.
    Rellena pequeños huecos negros dentro de regiones blancas y conecta
    objetos blancos cercanos, sin cambiar significativamente el tamaño.

    Fórmula: Cierre(A, B) = Erosión(Dilatación(A, B), B)

    Usos:
      - Rellenar huecos en la segmentación causados por reflejos de luz
      - Cerrar pequeñas brechas en contornos
      - Conectar regiones fragmentadas de la misma fruta

    Args:
        imagen_binaria: Imagen binaria o máscara (numpy array H×W, uint8)
        kernel: Elemento estructurante (None = elipse 5×5 por defecto)

    Returns:
        Imagen con cierre morfológico aplicado (mismo tamaño, uint8)
    """
    if kernel is None:
        kernel = obtener_elemento_estructurante()
    return cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)


def limpiar_mascara(mascara):
    """
    Pipeline completo de limpieza morfológica de una máscara de segmentación.

    Aplica apertura (elimina ruido puntual) seguida de cierre (rellena huecos).
    Esta combinación es el post-procesamiento estándar en segmentación de
    imágenes y produce máscaras más limpias y compactas.

    Usa un kernel elíptico de tamaño mayor (7×7) para ser más efectivo en
    imágenes redimensionadas a resolución moderada.

    Args:
        mascara: Máscara binaria a limpiar (numpy array H×W, uint8)

    Returns:
        Máscara limpiada con operaciones morfológicas (mismo tamaño, uint8)
    """
    kernel = obtener_elemento_estructurante(tamanio=(7, 7))
    limpia = apertura(mascara, kernel)
    limpia = cierre(limpia, kernel)
    return limpia


def visualizar_operaciones(mascara, alto=256, ancho=512):
    """
    Genera una imagen comparativa con las 4 operaciones morfológicas.

    Muestra lado a lado: máscara original, erosionada, dilatada y abierta+cerrada.
    Útil para demostrar el efecto de cada operación en la presentación del TP.

    Args:
        mascara: Máscara binaria de entrada
        alto: Alto de cada celda de visualización
        ancho: Ancho total de la imagen compuesta

    Returns:
        Imagen BGR con las 4 operaciones morfológicas visualizadas
    """
    ancho_celda = ancho // 4
    kernel = obtener_elemento_estructurante(tamanio=(5, 5))

    ops = [
        ("Original", mascara),
        ("Erosion",  erosion(mascara, kernel)),
        ("Dilatac.", dilatacion(mascara, kernel)),
        ("Ap+Cierre", limpiar_mascara(mascara)),
    ]

    celdas = []
    for nombre, img in ops:
        celda = cv2.resize(img, (ancho_celda, alto))
        celda_bgr = cv2.cvtColor(celda, cv2.COLOR_GRAY2BGR)
        cv2.putText(celda_bgr, nombre, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        celdas.append(celda_bgr)

    return np.hstack(celdas)
