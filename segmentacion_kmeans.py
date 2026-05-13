"""
Módulo de segmentación por color con K-means.

Unidad 7 – Técnicas avanzadas: segmentación con K-means (k=3 o k=4).

K-means es un algoritmo de clustering no supervisado que agrupa los píxeles
en k grupos (clusters) según su similitud de color. Cada píxel del resultado
recibe el color promedio del cluster al que pertenece.

El algoritmo minimiza iterativamente la suma de distancias cuadráticas entre
cada punto y el centroide de su cluster (criterio de varianza intra-cluster).
"""

import cv2
import numpy as np


def segmentar_kmeans(imagen_bgr, k=4, intentos=10, max_iter=100):
    """
    Segmenta una imagen por color usando K-means en espacio BGR.

    Proceso:
      1. Aplana la imagen de H×W×3 a (H*W)×3 (lista de píxeles)
      2. Ejecuta K-means: asigna cada píxel a uno de k clusters
      3. Reemplaza cada píxel por el color promedio de su cluster
      4. Reconstruye la imagen con la forma original

    Con k=3 se obtienen típicamente: fondo / fruta / zona de sombra o reflejo.
    Con k=4 se puede distinguir además zonas de transición de color.

    Args:
        imagen_bgr: Imagen en formato BGR (numpy array H×W×3)
        k: Número de clusters (grupos de color). 3 o 4 para frutas.
        intentos: Cuántas veces se reinicia con centros aleatorios distintos
                  (se elige el resultado con menor distorsión total)
        max_iter: Máximo de iteraciones por intento antes de converger

    Returns:
        Imagen BGR segmentada donde cada región tiene el color uniforme
        del centroide de su cluster (numpy array H×W×3, uint8)
    """
    # Aplanar la imagen a una lista de píxeles: (N, 3) con N = H*W
    datos = imagen_bgr.reshape((-1, 3)).astype(np.float32)

    # Criterio de parada: cuando el error < 0.2 O se alcanzan max_iter iteraciones
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)

    _, etiquetas, centros = cv2.kmeans(
        datos,
        k,
        None,
        criterio,
        intentos,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Reconstruir imagen: cada píxel toma el color del centroide de su cluster
    centros = np.uint8(centros)
    imagen_segmentada = centros[etiquetas.flatten()]
    imagen_segmentada = imagen_segmentada.reshape(imagen_bgr.shape)

    return imagen_segmentada


def segmentar_kmeans_hsv(imagen_bgr, k=4, intentos=10, max_iter=100):
    """
    Segmentación K-means en espacio HSV para mejor separación de colores.

    Operar en espacio HSV mejora la segmentación por color porque:
      - El canal H (tono) codifica el color de forma perceptualmente uniforme
      - Los clusters en H son más coherentes con la percepción humana del color
      - La segmentación es menos afectada por variaciones de iluminación

    La imagen se convierte a HSV antes del clustering y el resultado se
    devuelve en BGR para visualización.

    Args:
        imagen_bgr: Imagen en formato BGR (numpy array H×W×3)
        k: Número de clusters (3 o 4 para imágenes de frutas)
        intentos: Número de reinicios del algoritmo
        max_iter: Máximo de iteraciones por intento

    Returns:
        Imagen segmentada en formato BGR (numpy array H×W×3, uint8)
    """
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    datos = imagen_hsv.reshape((-1, 3)).astype(np.float32)

    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)

    _, etiquetas, centros = cv2.kmeans(
        datos,
        k,
        None,
        criterio,
        intentos,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centros = np.uint8(centros)
    imagen_segmentada_hsv = centros[etiquetas.flatten()]
    imagen_segmentada_hsv = imagen_segmentada_hsv.reshape(imagen_hsv.shape)

    # Convertir de vuelta a BGR para visualización en el panel
    return cv2.cvtColor(imagen_segmentada_hsv, cv2.COLOR_HSV2BGR)


def obtener_mascara_cluster_dominante(imagen_bgr, k=4):
    """
    Obtiene la máscara binaria del cluster más grande (objeto principal).

    Identifica cuál de los k clusters tiene más píxeles (presumiblemente
    la fruta o el objeto principal) y devuelve una máscara binaria de ese cluster.

    Útil para aislar el objeto principal de la imagen después del clustering.

    Args:
        imagen_bgr: Imagen en formato BGR
        k: Número de clusters

    Returns:
        Tuple (mascara, cluster_idx):
          - mascara: Imagen binaria con 255 donde está el cluster dominante
          - cluster_idx: Índice del cluster seleccionado
    """
    datos = imagen_bgr.reshape((-1, 3)).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, etiquetas, _ = cv2.kmeans(
        datos, k, None, criterio, 5, cv2.KMEANS_RANDOM_CENTERS
    )

    etiquetas_flat = etiquetas.flatten()

    # Encontrar el cluster con más píxeles
    conteos = [np.sum(etiquetas_flat == i) for i in range(k)]
    cluster_dominante = int(np.argmax(conteos))

    # Crear máscara binaria para ese cluster
    mascara_flat = (etiquetas_flat == cluster_dominante).astype(np.uint8) * 255
    mascara = mascara_flat.reshape(imagen_bgr.shape[:2])

    return mascara, cluster_dominante
