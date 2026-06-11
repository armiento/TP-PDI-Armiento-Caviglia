"""
Módulo de procesamiento básico de imágenes con Pillow.

Unidades 3 y 5 – Transformaciones básicas:
  - Desenfoque gaussiano para reducción de ruido
  - Ajuste de brillo y contraste
  - Conversión a escala de grises

Pillow (PIL) proporciona operaciones de imagen de alto nivel con una API
distinta a OpenCV. Mientras que OpenCV trabaja con numpy arrays BGR,
Pillow trabaja con objetos Image en formato RGB. Esta clase encapsula la
conversión entre ambos formatos y expone las operaciones como métodos
encadenables.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


class PillowProcessor:
    """
    Aplica operaciones básicas de procesamiento de imágenes usando Pillow.

    Recibe un numpy array BGR en el constructor, trabaja internamente con
    objetos PIL.Image y devuelve el resultado como numpy array BGR mediante
    el método obtener_imagen().

    Los métodos devuelven self para permitir encadenamiento:
        PillowProcessor(img).aplicar_blur(2).ajustar_brillo(1.2).obtener_imagen()
    """

    def __init__(self, imagen_bgr: np.ndarray):
        """
        Inicializa el procesador con una imagen en formato numpy BGR.

        Convierte internamente de BGR (OpenCV) a RGB (Pillow) para operar
        con la API de Pillow correctamente.

        Args:
            imagen_bgr: Imagen de entrada (H×W×3, uint8, BGR).
        """
        rgb = imagen_bgr[:, :, ::-1]  # BGR → RGB
        self._imagen = Image.fromarray(rgb.astype(np.uint8))

    def aplicar_blur(self, radio: float = 2.0) -> "PillowProcessor":
        """
        Aplica desenfoque gaussiano con Pillow.

        Usa ImageFilter.GaussianBlur para suavizar la imagen. A mayor radio,
        mayor suavizado y pérdida de detalle fino. Con radio=0 no se aplica
        ningún filtro.

        Args:
            radio: Radio del kernel gaussiano en píxeles (≥ 0). 0 = sin blur.

        Returns:
            self, para encadenar métodos.
        """
        if radio > 0:
            self._imagen = self._imagen.filter(ImageFilter.GaussianBlur(radius=radio))
        return self

    def ajustar_brillo(self, factor: float = 1.0) -> "PillowProcessor":
        """
        Ajusta el brillo de la imagen con ImageEnhance.Brightness.

        Un factor de 1.0 devuelve la imagen sin cambios.
        Un factor de 0.0 produce negro absoluto.
        Un factor de 2.0 duplica el brillo percibido.

        Args:
            factor: Factor multiplicativo de brillo (≥ 0.0).

        Returns:
            self, para encadenar métodos.
        """
        self._imagen = ImageEnhance.Brightness(self._imagen).enhance(max(0.0, factor))
        return self

    def ajustar_contraste(self, factor: float = 1.0) -> "PillowProcessor":
        """
        Ajusta el contraste de la imagen con ImageEnhance.Contrast.

        Un factor de 1.0 devuelve la imagen sin cambios.
        Un factor de 0.0 produce una imagen gris uniforme (contraste nulo).
        Un factor > 1.0 aumenta la diferencia entre zonas claras y oscuras.

        Args:
            factor: Factor multiplicativo de contraste (≥ 0.0).

        Returns:
            self, para encadenar métodos.
        """
        self._imagen = ImageEnhance.Contrast(self._imagen).enhance(max(0.0, factor))
        return self

    def convertir_grises(self) -> "PillowProcessor":
        """
        Convierte la imagen a escala de grises usando la ponderación luminosa de Pillow.

        Pillow aplica la fórmula ITU-R 601: L = 0.299·R + 0.587·G + 0.114·B.
        El resultado se reconvierte a RGB para mantener 3 canales en la salida.

        Returns:
            self, para encadenar métodos.
        """
        self._imagen = ImageOps.grayscale(self._imagen).convert("RGB")
        return self

    def obtener_imagen(self) -> np.ndarray:
        """
        Devuelve la imagen procesada como numpy array en formato BGR.

        Convierte internamente de RGB (Pillow) a BGR (OpenCV) para que el
        resultado sea compatible con el resto del pipeline.

        Returns:
            Imagen procesada (H×W×3, uint8, BGR).
        """
        rgb = np.array(self._imagen.convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()  # RGB → BGR
