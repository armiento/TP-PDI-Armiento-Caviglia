"""
Módulo de segmentación con algoritmo Watershed.

Unidad 7 – Técnicas avanzadas: separación de objetos con Watershed.

El algoritmo Watershed (línea divisoria de aguas) trata la imagen como un
mapa topográfico donde los valores de intensidad representan alturas. El
algoritmo "inunda" el terreno desde los mínimos locales (semillas) y
construye "presas" (bordes de segmentación) donde distintas inundaciones
se encuentran. Es especialmente útil para separar frutas que se tocan.

Implementación basada en la variante con marcadores (marker-based watershed)
que usa la transformada de distancia para encontrar semillas robustas.
"""

import cv2
import numpy as np
from morfologia import obtener_elemento_estructurante


def aplicar_watershed(imagen_bgr, umbral_dist=0.5):
    """
    Separa objetos que se tocan usando el algoritmo Watershed con marcadores.

    Pipeline completo:
      1. Binarizar con Otsu para separar objetos del fondo
      2. Apertura morfológica para eliminar ruido en la máscara
      3. Dilatación para obtener el "fondo seguro" (fuera de los objetos)
      4. Transformada de distancia: cada píxel toma el valor de su distancia
         al borde más cercano (centro = alto, borde = bajo)
      5. Umbralizar la transformada de distancia para obtener el "frente
         seguro" (el núcleo de cada objeto, lejos de los bordes).
         umbral_dist controla qué fracción del máximo se usa como umbral:
           - Valor bajo (0.1-0.3): semillas grandes, menos objetos separados
           - Valor alto (0.6-0.9): semillas pequeñas, más objetos detectados
      6. Zona desconocida = fondo_seguro - frente_seguro (la frontera)
      7. Etiquetar componentes conectadas como marcadores (semillas)
      8. Asignar zona desconocida = 0 (Watershed la resolverá)
      9. Aplicar cv2.watershed() que expande las semillas
      10. Colorear cada segmento con un color único

    Los bordes calculados por Watershed se pintan en rojo (etiqueta = -1).

    Args:
        imagen_bgr: Imagen en formato BGR (numpy array H×W×3)
        umbral_dist: Fracción del máximo de la transformada de distancia
                     usada como umbral para las semillas (0.1–0.9, default 0.5)

    Returns:
        Imagen BGR con los objetos segmentados coloreados (numpy array H×W×3)
        Los bordes entre objetos se muestran en rojo.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

    # Paso 1: Binarizar con Otsu
    _, binaria = cv2.threshold(
        suavizada, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Paso 2: Apertura morfológica para limpiar ruido
    kernel = obtener_elemento_estructurante(tamanio=(3, 3))
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)

    # Paso 3: Fondo seguro (dilatando la máscara)
    fondo_seguro = cv2.dilate(limpia, kernel, iterations=3)

    # Paso 4: Transformada de distancia euclidiana
    # Cada píxel blanco toma el valor de su distancia al borde negro más cercano
    transformada_dist = cv2.distanceTransform(limpia, cv2.DIST_L2, 5)

    # Paso 5: Frente seguro = centro de los objetos (distancia alta)
    # umbral_dist % del máximo para quedarnos con los núcleos de cada objeto
    umbral_val = max(0.01, umbral_dist) * transformada_dist.max()
    _, frente_seguro = cv2.threshold(
        transformada_dist,
        umbral_val,
        255,
        0
    )
    frente_seguro = np.uint8(frente_seguro)

    # Paso 6: Zona desconocida (frontera entre objetos y fondo)
    desconocido = cv2.subtract(fondo_seguro, frente_seguro)

    # Paso 7: Etiquetar cada componente conectada del frente seguro como semilla
    num_etiquetas, marcadores = cv2.connectedComponents(frente_seguro)

    # Paso 8: Desplazar etiquetas para que el fondo sea 1 (no 0)
    # Watershed usa 0 como zona desconocida, fondo debe ser >= 1
    marcadores = marcadores + 1

    # Marcar zona desconocida como 0 para que Watershed la resuelva
    marcadores[desconocido == 255] = 0

    # Paso 9: Aplicar Watershed (modifica marcadores in-place, -1 = borde)
    marcadores_resultado = cv2.watershed(imagen_bgr.copy(), marcadores.copy())

    # Paso 10: Colorear segmentos
    return colorear_segmentos(marcadores_resultado, imagen_bgr.shape)


def colorear_segmentos(marcadores, forma):
    """
    Colorea cada segmento de Watershed con un color distinto y reproducible.

    Asigna a cada etiqueta un color pseudo-aleatorio pero fijo (semilla 42)
    para que la visualización sea consistente entre ejecuciones.

    - Etiqueta 1: fondo → negro
    - Etiqueta -1: bordes Watershed → rojo brillante
    - Resto: objetos → colores variados

    Args:
        marcadores: Array 2D con etiquetas de Watershed (numpy array H×W, int32)
        forma: Forma de la imagen de salida como tupla (alto, ancho, 3)

    Returns:
        Imagen BGR con cada segmento coloreado (numpy array H×W×3, uint8)
    """
    np.random.seed(42)
    resultado = np.zeros(forma, dtype=np.uint8)

    etiquetas_unicas = np.unique(marcadores)

    # Paleta de colores predefinidos para los primeros segmentos
    paleta = [
        (255, 100, 100),  # Azul claro
        (100, 255, 100),  # Verde
        (100, 100, 255),  # Rojo claro
        (255, 255, 100),  # Cian
        (255, 100, 255),  # Magenta claro
        (100, 255, 255),  # Amarillo claro
        (200, 150, 50),   # Azul medio
        (50, 200, 150),   # Verde-amarillo
    ]

    idx_color = 0
    for etiqueta in etiquetas_unicas:
        if etiqueta <= 1:
            continue  # Saltar fondo (1) y borde (-1)

        if idx_color < len(paleta):
            color = paleta[idx_color]
        else:
            color = tuple(np.random.randint(80, 255, 3).tolist())

        resultado[marcadores == etiqueta] = color
        idx_color += 1

    # Pintar bordes de Watershed en rojo brillante
    resultado[marcadores == -1] = [0, 0, 255]

    return resultado


def watershed_con_contornos(imagen_bgr):
    """
    Aplica Watershed y dibuja los contornos sobre la imagen original.

    En lugar de colorear con regiones planas, superpone los bordes de
    Watershed (en rojo) sobre la imagen original. Visualiza dónde el
    algoritmo encontró los límites entre objetos.

    Args:
        imagen_bgr: Imagen en formato BGR

    Returns:
        Imagen original con bordes Watershed superpuestos en rojo
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

    _, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = obtener_elemento_estructurante(tamanio=(3, 3))
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
    fondo_seguro = cv2.dilate(limpia, kernel, iterations=3)

    transformada_dist = cv2.distanceTransform(limpia, cv2.DIST_L2, 5)
    _, frente_seguro = cv2.threshold(
        transformada_dist, 0.5 * transformada_dist.max(), 255, 0
    )
    frente_seguro = np.uint8(frente_seguro)
    desconocido = cv2.subtract(fondo_seguro, frente_seguro)

    _, marcadores = cv2.connectedComponents(frente_seguro)
    marcadores = marcadores + 1
    marcadores[desconocido == 255] = 0

    resultado = imagen_bgr.copy()
    marcadores_ws = cv2.watershed(resultado, marcadores)
    resultado[marcadores_ws == -1] = [0, 0, 255]

    return resultado
