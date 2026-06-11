"""
Módulo de procesamiento avanzado de imágenes con OpenCV.

Agrupa todas las técnicas avanzadas del programa de la materia en una
única clase. Cada técnica es un método que delega en las funciones de
los módulos existentes (bordes, segmentacion_kmeans, segmentacion_otsu,
watershed, histograma, color), sin reescribir su lógica.

Unidades cubiertas:
  - Unidad 4: histogramas y ecualización
  - Unidad 5: detección de bordes (Canny)
  - Unidad 6: Transformada de Fourier
  - Unidad 7: K-means HSV, Otsu, Watershed, ajuste HSV
"""

import cv2
import numpy as np

from bordes import canny_coloreado
from segmentacion_kmeans import segmentar_kmeans_hsv
from segmentacion_otsu import otsu_con_overlay
from watershed import aplicar_watershed as _aplicar_watershed
from histograma import dibujar_histograma_canal
from color import ajustar_hsv as _ajustar_hsv


class OpenCVProcessor:
    """
    Agrupa todos los procesamientos avanzados de imágenes usando OpenCV.

    Recibe un numpy array BGR en el constructor. Cada método aplica una
    técnica distinta y devuelve el resultado como numpy array BGR.
    Los módulos existentes son reutilizados internamente sin duplicar código.
    """

    def __init__(self, imagen_bgr: np.ndarray):
        """
        Inicializa el procesador con una imagen en formato numpy BGR.

        Args:
            imagen_bgr: Imagen de entrada (H×W×3, uint8, BGR).
        """
        self._imagen = imagen_bgr.copy()

    def bordes_canny(self, umbral_bajo: int = 40, umbral_alto: int = 120) -> np.ndarray:
        """
        Detecta bordes usando el algoritmo de Canny con visualización en verde.

        Unidad 5 – Filtros y detección de bordes. Delega en bordes.canny_coloreado,
        que aplica suavizado gaussiano previo y muestra los bordes en verde
        sobre fondo negro para mayor contraste en el panel.

        Args:
            umbral_bajo: Umbral inferior de histéresis (0–300).
            umbral_alto: Umbral superior de histéresis (0–300).

        Returns:
            Imagen BGR con bordes en verde sobre fondo negro.
        """
        return canny_coloreado(self._imagen, umbral_bajo=umbral_bajo, umbral_alto=umbral_alto)

    def segmentar_kmeans(self, k: int = 4) -> np.ndarray:
        """
        Segmenta la imagen por color usando K-means en espacio HSV.

        Unidad 7 – Técnicas avanzadas. Delega en segmentacion_kmeans.segmentar_kmeans_hsv.
        Opera en HSV para que el clustering sea menos sensible a variaciones
        de iluminación que en BGR.

        Args:
            k: Número de clusters de color (2–6).

        Returns:
            Imagen BGR con cada región coloreada con el centroide de su cluster.
        """
        return segmentar_kmeans_hsv(self._imagen, k=k, intentos=3, max_iter=50)

    def umbralizar_otsu(self) -> np.ndarray:
        """
        Aplica umbralización automática con el método de Otsu y overlay visual.

        Unidad 7 – Técnicas avanzadas. Delega en segmentacion_otsu.otsu_con_overlay.
        El umbral óptimo se calcula automáticamente maximizando la varianza
        inter-clase entre fondo y objeto.

        Returns:
            Imagen BGR con la máscara de Otsu superpuesta sobre el original.
        """
        return otsu_con_overlay(self._imagen)

    def umbralizar_manual(self, umbral: int) -> np.ndarray:
        """
        Aplica umbralización binaria con un valor de umbral definido por el usuario.

        Genera el mismo overlay visual que umbralizar_otsu() pero usando el
        umbral fijo indicado en lugar del calculado por Otsu. Permite al
        usuario comparar el resultado automático contra uno manual.

        Args:
            umbral: Valor de umbral fijo (1–255).

        Returns:
            Imagen BGR con overlay: objeto con colores reales, fondo oscurecido en azul.
        """
        gris = cv2.cvtColor(self._imagen, cv2.COLOR_BGR2GRAY)
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
        _, mascara = cv2.threshold(suavizada, umbral, 255, cv2.THRESH_BINARY)

        fondo = (self._imagen * 0.3).astype(np.uint8)
        fondo[:, :, 0] = np.clip(fondo[:, :, 0].astype(int) + 60, 0, 255)
        resultado = fondo.copy()
        resultado[mascara == 255] = self._imagen[mascara == 255]

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(resultado, contornos, -1, (0, 255, 255), 1)
        return resultado

    def aplicar_watershed(self, umbral_dist: float = 0.5) -> np.ndarray:
        """
        Separa objetos que se tocan usando el algoritmo Watershed con marcadores.

        Unidad 7 – Técnicas avanzadas. Delega en watershed.aplicar_watershed.
        Usa la transformada de distancia para encontrar semillas robustas y
        el algoritmo Watershed para definir los bordes entre objetos.

        Args:
            umbral_dist: Fracción del máximo de la transformada de distancia
                         usada como umbral de semillas (0.01–0.9).

        Returns:
            Imagen BGR con cada objeto coloreado distinto y bordes en rojo.
        """
        return _aplicar_watershed(self._imagen, umbral_dist=umbral_dist)

    def histograma_canal(self, canal: int = 0, alto: int = 340, ancho: int = 400) -> np.ndarray:
        """
        Dibuja el histograma del canal indicado con comparación antes/después de ecualizar.

        Unidad 4 – Histogramas. Delega en histograma.dibujar_histograma_canal.
        Muestra superpuestos el histograma original (color tenue) y el
        ecualizado (color brillante) para comparar el efecto de la ecualización.

        Args:
            canal: Canal a visualizar (0=Gris, 1=R, 2=G, 3=B).
            alto: Altura del lienzo en píxeles.
            ancho: Ancho del lienzo en píxeles.

        Returns:
            Imagen BGR con el histograma original y ecualizado superpuestos.
        """
        return dibujar_histograma_canal(self._imagen, canal=canal, alto=alto, ancho=ancho)

    def espectro_fourier(self) -> np.ndarray:
        """
        Calcula y visualiza el espectro de frecuencias con la Transformada Discreta de Fourier.

        Unidad 6 – Transformada de Fourier. La componente DC (baja frecuencia)
        se centra con fftshift. La escala logarítmica (log1p) comprime el rango
        dinámico para hacer visibles tanto las frecuencias bajas (centro) como
        las altas (periferia).

        Returns:
            Imagen BGR del espectro con colormap INFERNO para facilitar
            la visualización del rango dinámico.
        """
        gris = cv2.cvtColor(self._imagen, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gris), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_centrado = np.fft.fftshift(dft)
        magnitud = cv2.magnitude(dft_centrado[:, :, 0], dft_centrado[:, :, 1])
        espectro = cv2.normalize(np.log1p(magnitud), None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(espectro.astype(np.uint8), cv2.COLORMAP_INFERNO)

    def filtrar_fourier(self, radio: int) -> np.ndarray:
        """
        Aplica un filtro pasa bajos circular en el dominio de la frecuencia.

        Unidad 6 – Transformada de Fourier. Elimina las frecuencias superiores
        al radio dado (detalles finos y ruido) y reconstruye la imagen con la
        Transformada Inversa. A mayor radio, más detalle conservado.

        Args:
            radio: Radio del filtro en píxeles. Conserva solo las frecuencias
                   dentro del círculo de ese radio centrado en la componente DC.

        Returns:
            Imagen BGR reconstruida únicamente con las frecuencias bajas.
        """
        gris = cv2.cvtColor(self._imagen, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gris), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_centrado = np.fft.fftshift(dft)

        h, w = gris.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        mascara = ((X - cx) ** 2 + (Y - cy) ** 2 <= radio ** 2).astype(np.float32)

        dft_filtrado = dft_centrado * mascara[:, :, np.newaxis]
        dft_ishift = np.fft.ifftshift(dft_filtrado)
        img_rec = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        img_rec = cv2.normalize(img_rec, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(img_rec.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def ajustar_hsv(self, delta_h: int = 0, escala_s: float = 1.0, escala_v: float = 1.0) -> np.ndarray:
        """
        Modifica el tono (H), saturación (S) y brillo (V) en espacio HSV.

        Unidad 7 – Técnicas avanzadas. Delega en color.ajustar_hsv.
        Permite editar los tres componentes del espacio HSV de forma
        independiente para manipular el color de las frutas.

        Args:
            delta_h: Desplazamiento del tono (−90 a +90). H en OpenCV es 0–180.
            escala_s: Factor multiplicativo de saturación (0.0–2.0). 1.0 = sin cambio.
            escala_v: Factor multiplicativo de brillo/value (0.0–2.0). 1.0 = sin cambio.

        Returns:
            Imagen BGR con los ajustes HSV aplicados.
        """
        return _ajustar_hsv(self._imagen, delta_h=delta_h, escala_s=escala_s, escala_v=escala_v)
