"""
Editor Interactivo de Imagenes de Frutas - INTERFAZ DROPDOWN (Orientada a Objetos)
==================================================================================
Trabajo Practico Final - Tecnicas de Procesamiento Digital de Imagenes
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
IFTS N18 - CABA - Ciclo 2026
Profesor: Bonini Juan Ignacio
Alumnos: Armiento, Fernando - Caviglia, Paula
"""

import cv2
import numpy as np
import os
import glob
import sys

from motor import Motor

# ===========================================================================
# CONSTANTES DE CONFIGURACIÓN ESTÁTICAS (PEP 8 - Solo lectura, no son globales dañinas)
# ===========================================================================
ANCHO_IMG = 400        # Ancho de cada imagen en el panel lateral
ALTO_IMG  = 340        # Alto de cada imagen en el panel lateral
CARPETA_DATOS = "images"
EXTENSIONES = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

VENTANA_DISPLAY = "Editor de Frutas"
VENTANA_CTRL    = "Controles"

# Diccionario con los nombres de los modos correspondientes a las unidades
MODOS = {
    1: "1-Blur / Brillo",
    2: "2-Bordes Canny",
    3: "3-K-means HSV",
    4: "4-Otsu",
    5: "5-Watershed",
    6: "6-Histograma",
    7: "7-Fourier",
    8: "8-Ajuste HSV",
}


# ===========================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN (Encapsula el estado eliminando las globales)
# ===========================================================================
class EditorApp:
    def __init__(self):
        """Inicializa los atributos de instancia y configura el entorno de forma portable."""
        # Detección automática del directorio del script 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.carpeta_datos = os.path.join(self.script_dir, "images")
        
        # Carga el listado de imágenes de la carpeta dinámica calculada arriba
        self.rutas = self.cargar_rutas_imagenes(self.carpeta_datos)
        
        # atributos de instancia encapsulados
        self.necesita_actualizar = True
        self.estado = {
            "indice": 0,                # Índice de la imagen actual en el arreglo
            "total": len(self.rutas),   # Total de imágenes encontradas en el dataset
            "modo": 1,                  # Modo/Unidad seleccionada (1 al 8)
            "guardar": False,           # Flag interactivo para guardar la imagen
            "menu_abierto": False       # Flag para controlar la apertura del menú desplegable
        }

    def _cb(self, _val):
        """Callback para los sliders estandart de OpenCV. Activa el redibujado interactivo."""
        self.necesita_actualizar = True

    def cargar_rutas_imagenes(self, carpeta_datos):
        """Escanea la carpeta de datos relativa y construye el set indexado de imágenes."""
        if not os.path.exists(carpeta_datos):
            print(f"[ERROR] No se encontro la carpeta: '{carpeta_datos}'")
            sys.exit(1)

        rutas = []
        for nombre_carpeta in sorted(os.listdir(carpeta_datos)):
            ruta_carpeta = os.path.join(carpeta_datos, nombre_carpeta)
            if not os.path.isdir(ruta_carpeta):
                continue
            vistas = set()
            lista = []
            for ext in EXTENSIONES:
                for ruta in glob.glob(os.path.join(ruta_carpeta, ext)):
                    clave = os.path.normcase(ruta)
                    if clave not in vistas:
                        vistas.add(clave)
                        lista.append(ruta)
            lista.sort()
            for ruta in lista:
                rutas.append((ruta, nombre_carpeta))

        if not rutas:
            print(f"[ERROR] No se encontraron imagenes en '{carpeta_datos}'")
            sys.exit(1)

        print(f"[OK] {len(rutas)} imagenes listas de forma portable.")
        return rutas

    def mouse_handler(self, event, x, y, flags, param):
        """Manejador de eventos del mouse para botones y menú flotante."""
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # 1. SI EL MENU ESTÁ DESPLEGADO: Evalúa qué opción de la lista vertical tocó el usuario
            if self.estado["menu_abierto"]:
                if 230 <= x <= 410 and 42 <= y <= 42 + (8 * 25):
                    opcion_seleccionada = ((y - 42) // 25) + 1
                    self.estado["modo"] = opcion_seleccionada
                    self.estado["menu_abierto"] = False
                    self.necesita_actualizar = True
                    return
                else:
                    self.estado["menu_abierto"] = False
                    self.necesita_actualizar = True
                    return

            # 2. BOTONES EN EL ENCABEZADO: Evalúa clicks horizontales en la barra superior
            if 12 <= y <= 38:
                if 10 <= x <= 110:     # Click en Botón << ANTERIOR
                    self.estado["indice"] = (self.estado["indice"] - 1) % self.estado["total"]
                    self.necesita_actualizar = True
                elif 120 <= x <= 220:  # Click en Botón SIGUIENTE >>
                    self.estado["indice"] = (self.estado["indice"] + 1) % self.estado["total"]
                    self.necesita_actualizar = True
                elif 230 <= x <= 410:  # Click en la caja del MENU DESPLEGABLE
                    self.estado["menu_abierto"] = True
                    self.necesita_actualizar = True
                elif 420 <= x <= 510:  # Click en Botón GUARDAR
                    self.estado["guardar"] = True
                    self.necesita_actualizar = True

    def crear_controles(self, modo):
        """Destruye y recrea dinámicamente los sliders específicos de la ventana secundaria."""
        try:
            cv2.destroyWindow(VENTANA_CTRL)
        except Exception:
            pass
        cv2.waitKey(1)

        cv2.namedWindow(VENTANA_CTRL, cv2.WINDOW_NORMAL)

        # Inyección de barras deslizantes según el modo seleccionado
        if modo == 1: # MODO 1: BLUR Y BRILLO
            cv2.createTrackbar("Blur 0=sin 10=max", VENTANA_CTRL, 2, 10, self._cb)
            cv2.createTrackbar("Brillo 0=-100 200=+100", VENTANA_CTRL, 100, 200, self._cb)
            cv2.createTrackbar("Contraste x0.1 (10=1.0x)", VENTANA_CTRL, 10, 30, self._cb)
        elif modo == 2: # MODO 2: BORDES CANNY
            cv2.createTrackbar("Canny Umbral Bajo (0-300)", VENTANA_CTRL, 40, 300, self._cb)
            cv2.createTrackbar("Canny Umbral Alto (0-300)", VENTANA_CTRL, 120, 300, self._cb)
        elif modo == 3: # MODO 3: K-MEANS HSV
            cv2.createTrackbar("K-means k-2 (0=k2 4=k6)", VENTANA_CTRL, 2, 4, self._cb)
        elif modo == 4: # MODO 4: OTSU
            cv2.createTrackbar("Umbral 0=auto Otsu 1-255=manual", VENTANA_CTRL, 0, 255, self._cb)
        elif modo == 5: # MODO 5: WATERSHED
            cv2.createTrackbar("Umbral dist % (1-90 def=50)", VENTANA_CTRL, 50, 90, self._cb)
        elif modo == 6: # MODO 6: HISTOGRAMA
            cv2.createTrackbar("Canal 0=Gris 1=R 2=G 3=B", VENTANA_CTRL, 0, 3, self._cb)
        elif modo == 7: # MODO 7: FOURIER
            cv2.createTrackbar("Filtro pasa bajo radio px (0=sin)", VENTANA_CTRL, 0, 100, self._cb)
        elif modo == 8: # MODO 8: AJUSTE HSV
            cv2.createTrackbar("Tono H offset (90=sin cambio)", VENTANA_CTRL, 90, 180, self._cb)
            cv2.createTrackbar("Saturacion x% (100=sin cambio)", VENTANA_CTRL, 100, 200, self._cb)
            cv2.createTrackbar("Brillo V x% (100=sin cambio)", VENTANA_CTRL, 100, 200, self._cb)

        cv2.resizeWindow(VENTANA_CTRL, 460, 150)
        self.necesita_actualizar = True

    def procesar_modo(self, img, modo):
        """Enlaza las posiciones de las barras con los métodos matemáticos del Motor."""
        motor = Motor(img)
        if modo == 1: # MODO 1: BLUR Y BRILLO
            blur_val      = cv2.getTrackbarPos("Blur 0=sin 10=max", VENTANA_CTRL)
            brillo_raw    = cv2.getTrackbarPos("Brillo 0=-100 200=+100", VENTANA_CTRL)
            contraste_raw = cv2.getTrackbarPos("Contraste x0.1 (10=1.0x)", VENTANA_CTRL)
            return motor.blur_y_brillo(blur_val * 2 + 1, brillo_raw - 100, max(0.1, contraste_raw / 10.0))
        elif modo == 2: # MODO 2: BORDES CANNY
            bajo = cv2.getTrackbarPos("Canny Umbral Bajo (0-300)", VENTANA_CTRL)
            alto = cv2.getTrackbarPos("Canny Umbral Alto (0-300)", VENTANA_CTRL)
            return motor.bordes_canny(bajo, alto)
        elif modo == 3: # MODO 3: K-MEANS HSV
            k_offset = cv2.getTrackbarPos("K-means k-2 (0=k2 4=k6)", VENTANA_CTRL)
            return motor.segmentar_kmeans(k_offset + 2)
        elif modo == 4: # MODO 4: OTSU
            umbral_manual = cv2.getTrackbarPos("Umbral 0=auto Otsu 1-255=manual", VENTANA_CTRL)
            return motor.umbralizar_otsu(umbral_manual)
        elif modo == 5: # MODO 5: WATERSHED
            umbral_pct = cv2.getTrackbarPos("Umbral dist % (1-90 def=50)", VENTANA_CTRL)
            return motor.aplicar_watershed(max(0.01, umbral_pct / 100.0))
        elif modo == 6: # MODO 6: HISTOGRAMA
            canal = cv2.getTrackbarPos("Canal 0=Gris 1=R 2=G 3=B", VENTANA_CTRL)
            return motor.histograma_canal(canal, ALTO_IMG, ANCHO_IMG)
        elif modo == 7: # MODO 7: FOURIER
            radio = cv2.getTrackbarPos("Filtro pasa bajo radio px (0=sin)", VENTANA_CTRL)
            return motor.fourier(radio)
        elif modo == 8: # MODO 8: AJUSTE HSV
            delta_h  = cv2.getTrackbarPos("Tono H offset (90=sin cambio)", VENTANA_CTRL) - 90
            escala_s = cv2.getTrackbarPos("Saturacion x% (100=sin cambio)", VENTANA_CTRL) / 100.0
            escala_v = cv2.getTrackbarPos("Brillo V x% (100=sin cambio)", VENTANA_CTRL) / 100.0
            return motor.ajustar_hsv(delta_h, escala_s, escala_v)
        return img.copy()

    def _info_slider(self, modo):
        """Genera dinámicamente las cadenas métricas de los sliders e inyecta la teoría de cada modo."""
        try:
            # --- MODO 1: BLUR Y BRILLO ---
            # Suaviza la piel de la fruta con un filtro Gaussiano y ajusta brillo/contraste píxel a píxel.
            if modo == 1:
                b  = cv2.getTrackbarPos("Blur 0=sin 10=max", VENTANA_CTRL)
                br = cv2.getTrackbarPos("Brillo 0=-100 200=+100", VENTANA_CTRL) - 100
                c  = cv2.getTrackbarPos("Contraste x0.1 (10=1.0x)", VENTANA_CTRL) / 10.0
                return f"kernel={b*2+1}px   brillo={br:+d}   contraste={c:.1f}x"
            
            # --- MODO 2: BORDES CANNY ---
            # Detecta los contornos de la fruta usando gradientes espaciales y umbralización por histéresis.
            elif modo == 2:
                bajo = cv2.getTrackbarPos("Canny Umbral Bajo (0-300)", VENTANA_CTRL)
                alto = cv2.getTrackbarPos("Canny Umbral Alto (0-300)", VENTANA_CTRL)
                return f"Canny Bajo={bajo}  Alto={alto}"
            
            # --- MODO 3: K-MEANS HSV ---
            # Segmenta por color usando clustering no supervisado para agrupar los píxeles más similares.
            elif modo == 3:
                k = cv2.getTrackbarPos("K-means k-2 (0=k2 4=k6)", VENTANA_CTRL) + 2
                return f"k={k} clusters de color"
            
            # --- MODO 4: OTSU ---
            # Binariza la imagen calculando estadísticamente el umbral óptimo que minimiza la varianza interna.
            elif modo == 4:
                u = cv2.getTrackbarPos("Umbral 0=auto Otsu 1-255=manual", VENTANA_CTRL)
                return f"umbral={'Otsu Automatico' if u == 0 else str(u)}"
            
            # --- MODO 5: WATERSHED ---
            # Segmentación morfológica avanzada basada en transformada de distancia para separar objetos tocándose.
            elif modo == 5:
                pct = cv2.getTrackbarPos("Umbral dist % (1-90 def=50)", VENTANA_CTRL)
                return f"Distancia = {pct}% del maximo"
            
            # --- MODO 6: HISTOGRAMA ---
            # Muestra el análisis estadístico de la distribución de tonos y la curva acumulada (CDF) del canal.
            elif modo == 6:
                c = cv2.getTrackbarPos("Canal 0=Gris 1=R 2=G 3=B", VENTANA_CTRL)
                return f"canal = {['Grises', 'R (rojo)', 'G (verde)', 'B (azul)'][c]}"
            
            # --- MODO 7: FOURIER ---
            # Filtra en el dominio de la frecuencia con un filtro pasa-bajos ideal para lograr un suavizado homogéneo.
            elif modo == 7:
                r = cv2.getTrackbarPos("Filtro pasa bajo radio px (0=sin)", VENTANA_CTRL)
                return f"Filtro Fourier Radio = {r}px"
            
            # --- MODO 8: AJUSTE HSV ---
            # Permite alterar el color (Tono), pureza (Saturación) y brillo (Valor) de forma totalmente independiente.
            elif modo == 8:
                dh = cv2.getTrackbarPos("Tono H offset (90=sin cambio)", VENTANA_CTRL) - 90
                s  = cv2.getTrackbarPos("Saturacion x% (100=sin cambio)", VENTANA_CTRL)
                v  = cv2.getTrackbarPos("Brillo V x% (100=sin cambio)", VENTANA_CTRL)
                return f"H {dh:+d}  Saturacion={s}%   Brillo={v}%"
        except Exception:
            pass
        return ""

    def construir_display(self, img_orig, img_proc, modo, nombre_fruta, indice, total):
        """Ensambla los lienzos parciales (original y procesado) montando la botonería gráfica."""
        orig = cv2.resize(img_orig, (ANCHO_IMG, ALTO_IMG))
        proc = cv2.resize(img_proc, (ANCHO_IMG, ALTO_IMG))

        cv2.rectangle(orig, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
        cv2.putText(orig, "Original", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.rectangle(proc, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
        cv2.putText(proc, MODOS[modo], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

        sep_v = np.full((ALTO_IMG, 4, 3), (55, 55, 55), dtype=np.uint8)
        cuerpo = np.hstack([orig, sep_v, proc])
        ancho_panel = cuerpo.shape[1]

        header = np.full((55, ancho_panel, 3), (25, 25, 25), dtype=np.uint8)
        
        # Render del Botón Anterior
        cv2.rectangle(header, (10, 12), (110, 38), (50, 50, 50), -1)
        cv2.rectangle(header, (10, 12), (110, 38), (100, 100, 100), 1)
        cv2.putText(header, "<< ANTERIOR", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Render del Botón Siguiente
        cv2.rectangle(header, (120, 12), (220, 38), (50, 50, 50), -1)
        cv2.rectangle(header, (120, 12), (220, 38), (100, 100, 100), 1)
        cv2.putText(header, "SIGUIENTE >>", (126, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Render de la caja del Menú Desplegable
        color_btn_menu = (75, 45, 15) if self.estado["menu_abierto"] else (40, 40, 40)
        cv2.rectangle(header, (230, 12), (410, 38), color_btn_menu, -1)
        cv2.rectangle(header, (230, 12), (410, 38), (120, 120, 120), 1)
        texto_dropdown = f" {MODOS[modo]}  v"
        cv2.putText(header, texto_dropdown[:22], (235, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1, cv2.LINE_AA)

        # Render del Botón Guardar
        cv2.rectangle(header, (420, 12), (510, 38), (20, 80, 20), -1)
        cv2.rectangle(header, (420, 12), (510, 38), (50, 150, 50), 1)
        cv2.putText(header, "GUARDAR", (435, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1, cv2.LINE_AA)

        # Render del rótulo informativo de Estado de la Fruta actual
        cv2.putText(header, f"Fruta: {nombre_fruta.title()}  [{indice + 1}/{total}]", (530, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(header, (0, 54), (ancho_panel, 54), (70, 70, 70), 1)

        panel_completo = np.vstack([header, cuerpo])

        # Renderizado de la lista flotante vertical si menu_abierto es True
        if self.estado["menu_abierto"]:
            for i, nombre_modo in MODOS.items():
                y_opcion = 42 + (i - 1) * 25
                bg_color = (100, 65, 25) if i == modo else (30, 30, 30)
                cv2.rectangle(panel_completo, (230, y_opcion), (410, y_opcion + 24), bg_color, -1)
                cv2.rectangle(panel_completo, (230, y_opcion), (410, y_opcion + 24), (60, 60, 60), 1)
                cv2.putText(panel_completo, nombre_modo, (240, y_opcion + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        info_texto = self._info_slider(modo)
        info_bar = np.full((26, ancho_panel, 3), (12, 12, 12), dtype=np.uint8)
        if info_texto:
            cv2.putText(info_bar, info_texto, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 0), 1, cv2.LINE_AA)

        return np.vstack([panel_completo, info_bar])

    def ejecutar(self):
        """Loop infinito principal de renderizado y escucha de eventos físicos de OpenCV."""
        modo_anterior = -1
        indice_cache = -1
        img_actual = None
        nombre_fruta = ""
        panel = None

        cv2.namedWindow(VENTANA_DISPLAY, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA_DISPLAY, ANCHO_IMG * 2 + 4, ALTO_IMG + 81)
        
        # Pasa de forma segura el método reactivo de ratón vinculado a esta instancia pura
        cv2.setMouseCallback(VENTANA_DISPLAY, self.mouse_handler)

        while True:
            indice = self.estado["indice"]
            modo = self.estado["modo"]
            total = self.estado["total"]

            # Evaluación de avance o retroceso de fruta en el dataset
            if indice != indice_cache:
                ruta, nombre_fruta = self.rutas[indice]
                imagen = cv2.imread(ruta)
                if imagen is None:
                    self.estado["indice"] = (indice + 1) % total
                    continue
                img_actual = cv2.resize(imagen, (ANCHO_IMG, ALTO_IMG))
                indice_cache = indice
                self.necesita_actualizar = True

            # Recreación de sliders al saltar entre técnicas
            if modo != modo_anterior:
                self.crear_controles(modo)
                modo_anterior = modo

            # Renderizado reactivo por variaciones en sliders o clicks
            if self.necesita_actualizar:
                self.necesita_actualizar = False
                try:
                    img_proc = self.procesar_modo(img_actual, modo)
                except Exception as e:
                    img_proc = np.zeros((ALTO_IMG, ANCHO_IMG, 3), dtype=np.uint8)
                    cv2.putText(img_proc, f"ERROR: {str(e)[:30]}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

                panel = self.construir_display(img_actual, img_proc, modo, nombre_fruta, indice, total)
                cv2.imshow(VENTANA_DISPLAY, panel)

            # Persistencia de salida local utilizando el Path dinámico calculado en el constructor
            if self.estado["guardar"] and panel is not None:
                self.estado["guardar"] = False
                nombre_archivo = f"procesado_{indice:03d}_{nombre_fruta.replace(' ', '_')}_m{modo}.png"
                cv2.imwrite(os.path.join(self.script_dir, nombre_archivo), panel)
                print(f"  [GUARDADO] {nombre_archivo}")

            # Captura de teclado para romper el loop con la tecla 'ESC' o 'q'
            tecla = cv2.waitKey(30) & 0xFF
            if tecla in (ord('q'), 27):
                break

            # Cierre seguro si el usuario destruye físicamente la ventana principal con la cruz
            if cv2.getWindowProperty(VENTANA_DISPLAY, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


# ===========================================================================
# PUNTO DE ARRANQUE GENERAL DEL SCRIPT PYTHON
# ===========================================================================
def main():
    # Instanciación limpia del objeto controlador de la interfaz
    app = EditorApp()
    # Disparo de la ejecución encapsulada sin intervención de memoria global
    app.ejecutar()


if __name__ == "__main__":
    main()