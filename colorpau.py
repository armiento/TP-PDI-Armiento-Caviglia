import cv2
import numpy as np

class ProcesadorColor:
    def __init__(self):
        """
        Constructor de la clase. Aquí se guardan las variables de estado
        (los parámetros que el usuario mueve en los sliders).
        """
        self.brillo = 0          # Rango: -100 a 100
        self.contraste = 1.0     # Rango: 0.1 a 3.0
        self.delta_h = 0         # Rango: -90 a 90 (Tono)
        self.escala_s = 1.0      # Rango: 0.0 a 2.0 (Saturación)
        self.escala_v = 1.0      # Rango: 0.0 a 2.0 (Brillo HSV)

    def bgr_a_gris(self, imagen):
        """Convierte una imagen BGR a escala de grises (1 canal)."""
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    def aplicar_brillo_contraste(self, imagen_bgr):
        """
        Aplica el brillo y contraste guardados en la instancia usando
        transformación lineal.
        """
        resultado = imagen_bgr.astype(np.float32) * self.contraste + self.brillo
        return np.clip(resultado, 0, 255).astype(np.uint8)

    def aplicar_ajuste_hsv(self, imagen_bgr):
        """
        Modifica el tono, saturación y brillo usando los valores guardados.
        """
        hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.delta_h) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.escala_s, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.escala_v, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def obtener_hsv_display(self, imagen_bgr):
        """Devuelve una versión visual BGR del espacio HSV (falso color)."""
        hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        hsv_display = hsv.copy()
        hsv_display[:, :, 0] = (hsv[:, :, 0].astype(np.float32) / 180.0 * 255).astype(np.uint8)
        return hsv_display