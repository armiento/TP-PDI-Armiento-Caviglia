"""
Módulo Motor: fachada principal del core de procesamiento de imágenes.

Es la única clase que main.py necesita importar. Expone un método por
cada transformación disponible y delega internamente en:
  - PillowProcessor: operaciones básicas (blur, brillo, contraste, grises)
  - OpenCVProcessor: operaciones avanzadas (bordes, kmeans, otsu, watershed,
                     histograma, fourier, hsv)

Esta separación permite reutilizar el core desde cualquier capa de
presentación (OpenCV, Streamlit, etc.) sin modificar los procesadores.
"""

import numpy as np

from pillow_processor import PillowProcessor
from opencv_processor import OpenCVProcessor


class Motor:
    """
    Fachada del core de procesamiento de imágenes del TP Final.

    Recibe una imagen BGR y expone todos los procesamientos disponibles como
    métodos independientes. Cada método construye el procesador apropiado,
    aplica la transformación y devuelve el resultado listo para mostrar.
    """

    def __init__(self, imagen_bgr: np.ndarray):
        """
        Inicializa el Motor con la imagen a procesar.

        Args:
            imagen_bgr: Imagen de entrada (H×W×3, uint8, BGR).
        """
        self._imagen = imagen_bgr.copy()

    # ------------------------------------------------------------------
    # Operaciones básicas — delegadas a PillowProcessor
    # ------------------------------------------------------------------

    def blur_y_brillo(self, kernel_size: int, brillo: int, contraste: float) -> np.ndarray:
        """
        Aplica desenfoque gaussiano seguido de ajuste de brillo y contraste.

        Usa PillowProcessor para todas las operaciones. El parámetro brillo
        (offset −100..+100) se convierte a factor Pillow antes de delegar.

        Args:
            kernel_size: Tamaño del kernel gaussiano (impar, ≥ 1). 1 = sin blur.
            brillo: Offset aditivo de brillo (−100 a +100). 0 = sin cambio.
            contraste: Factor multiplicativo de contraste (0.1–3.0). 1.0 = sin cambio.

        Returns:
            Imagen BGR con blur, brillo y contraste aplicados.
        """
        radio = max(0.0, (kernel_size - 1) / 2.0)
        # Convierte el offset lineal de brillo al factor multiplicativo de Pillow:
        # offset=0 → factor=1.0 (sin cambio), offset=+100 → ~1.78, offset=−100 → ~0.22
        factor_brillo = max(0.01, (128.0 + brillo) / 128.0)
        proc = PillowProcessor(self._imagen)
        proc.aplicar_blur(radio).ajustar_brillo(factor_brillo).ajustar_contraste(contraste)
        return proc.obtener_imagen()

    def convertir_grises(self) -> np.ndarray:
        """
        Convierte la imagen a escala de grises usando la ponderación luminosa de Pillow.

        Returns:
            Imagen BGR en escala de grises (3 canales con valores iguales en BGR).
        """
        return PillowProcessor(self._imagen).convertir_grises().obtener_imagen()

    # ------------------------------------------------------------------
    # Operaciones avanzadas — delegadas a OpenCVProcessor
    # ------------------------------------------------------------------

    def bordes_canny(self, umbral_bajo: int = 40, umbral_alto: int = 120) -> np.ndarray:
        """
        Detecta bordes con el algoritmo de Canny (Unidad 5).

        Args:
            umbral_bajo: Umbral inferior de histéresis (0–300).
            umbral_alto: Umbral superior de histéresis (0–300).

        Returns:
            Imagen BGR con bordes en verde sobre fondo negro.
        """
        return OpenCVProcessor(self._imagen).bordes_canny(umbral_bajo, umbral_alto)

    def segmentar_kmeans(self, k: int = 4) -> np.ndarray:
        """
        Segmenta la imagen por color usando K-means en espacio HSV (Unidad 7).

        Args:
            k: Número de clusters de color (2–6).

        Returns:
            Imagen BGR con regiones de color uniforme por cluster.
        """
        return OpenCVProcessor(self._imagen).segmentar_kmeans(k)

    def umbralizar_otsu(self, umbral_manual: int = 0) -> np.ndarray:
        """
        Aplica umbralización con el método de Otsu o con umbral manual (Unidad 7).

        Args:
            umbral_manual: 0 = Otsu automático; 1–255 = umbral fijo manual.

        Returns:
            Imagen BGR con overlay de máscara binaria.
        """
        proc = OpenCVProcessor(self._imagen)
        if umbral_manual == 0:
            return proc.umbralizar_otsu()
        return proc.umbralizar_manual(umbral_manual)

    def aplicar_watershed(self, umbral_dist: float = 0.5) -> np.ndarray:
        """
        Separa objetos que se tocan usando el algoritmo Watershed (Unidad 7).

        Args:
            umbral_dist: Fracción del máximo de la transformada de distancia (0.01–0.9).

        Returns:
            Imagen BGR con objetos coloreados y bordes en rojo.
        """
        return OpenCVProcessor(self._imagen).aplicar_watershed(umbral_dist)

    def histograma_canal(self, canal: int = 0, alto: int = 340, ancho: int = 400) -> np.ndarray:
        """
        Visualiza el histograma de un canal con comparación antes/después de ecualizar (Unidad 4).

        Args:
            canal: Canal a visualizar (0=Gris, 1=R, 2=G, 3=B).
            alto: Alto del lienzo en píxeles.
            ancho: Ancho del lienzo en píxeles.

        Returns:
            Imagen BGR con el histograma dibujado.
        """
        return OpenCVProcessor(self._imagen).histograma_canal(canal, alto, ancho)

    def fourier(self, radio: int = 0) -> np.ndarray:
        """
        Calcula el espectro de Fourier o aplica un filtro pasa bajos (Unidad 6).

        Args:
            radio: Radio del filtro en píxeles. 0 = mostrar espectro sin filtrar.

        Returns:
            Espectro BGR (radio=0) o imagen reconstruida con el filtro pasa bajos.
        """
        proc = OpenCVProcessor(self._imagen)
        if radio == 0:
            return proc.espectro_fourier()
        return proc.filtrar_fourier(radio)

    def ajustar_hsv(self, delta_h: int = 0, escala_s: float = 1.0, escala_v: float = 1.0) -> np.ndarray:
        """
        Modifica el tono, saturación y brillo en espacio HSV (Unidad 7).

        Args:
            delta_h: Desplazamiento del tono (−90 a +90).
            escala_s: Factor de saturación (0.0–2.0). 1.0 = sin cambio.
                escala_v: Factor de brillo/value (0.0–2.0). 1.0 = sin cambio.

        Returns:
            Imagen BGR con los ajustes HSV aplicados.
        """
        return OpenCVProcessor(self._imagen).ajustar_hsv(delta_h, escala_s, escala_v)
