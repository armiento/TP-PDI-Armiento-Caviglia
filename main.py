"""
Editor Interactivo de Imágenes de Frutas – INTERFAZ DROPDOWN
===========================================================
Trabajo Práctico Final – Técnicas de Procesamiento Digital de Imágenes
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
IFTS N°18 – CABA – Ciclo 2026
Profesor: Bonini Juan Ignacio
Alumnos: Armiento, Fernando – Caviglia, Paula
"""

import cv2
import numpy as np
import os
import glob
import sys

from motor import Motor

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ANCHO_IMG = 400        # Ancho de cada imagen en el panel lateral
ALTO_IMG  = 340        # Alto de cada imagen en el panel lateral
CARPETA_DATOS = "images"
EXTENSIONES = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

VENTANA_DISPLAY = "Editor de Frutas"
VENTANA_CTRL    = "Controles"

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

# ---------------------------------------------------------------------------
# Estado global de navegación (Sincronizado con Mouse y Menú Desplegable)
# ---------------------------------------------------------------------------
_necesita_actualizar = [True]
_estado_navegacion = {"indice": 0, "total": 0, "modo": 1, "guardar": False, "menu_abierto": False}


def _cb(_val):
    """Callback para los sliders estándar."""
    _necesita_actualizar[0] = True


def mouse_handler(event, x, y, flags, param):
    """Manejador del mouse para botones y Menú Desplegable flotante."""
    global _estado_navegacion
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. SI EL MENÚ ESTÁ ABIERTO: Detectar qué opción tocó de la lista vertical
        if _estado_navegacion["menu_abierto"]:
            # El menú se despliega verticalmente debajo del botón central (X: 230-410)
            if 230 <= x <= 410 and 42 <= y <= 42 + (8 * 25):
                opcion_seleccionada = ((y - 42) // 25) + 1
                _estado_navegacion["modo"] = opcion_seleccionada
                _estado_navegacion["menu_abierto"] = False
                _necesita_actualizar[0] = True
                return
            else:
                # Si hace click afuera, se cierra el menú automáticamente
                _estado_navegacion["menu_abierto"] = False
                _necesita_actualizar[0] = True
                return

        # 2. BOTONES DEL ENCABEZADO (Cuando el menú está cerrado)
        if 12 <= y <= 38:
            if 10 <= x <= 110:     # Botón << ANTERIOR
                _estado_navegacion["indice"] = (_estado_navegacion["indice"] - 1) % _estado_navegacion["total"]
                _necesita_actualizar[0] = True
            elif 120 <= x <= 220:  # Botón SIGUIENTE >>
                _estado_navegacion["indice"] = (_estado_navegacion["indice"] + 1) % _estado_navegacion["total"]
                _necesita_actualizar[0] = True
            elif 230 <= x <= 410:  # Botón central del MENÚ DESPLEGABLE
                _estado_navegacion["menu_abierto"] = True
                _necesita_actualizar[0] = True
            elif 420 <= x <= 510:  # Botón GUARDAR
                _estado_navegacion["guardar"] = True
                _necesita_actualizar[0] = True


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------
def cargar_rutas_imagenes(carpeta_datos):
    if not os.path.exists(carpeta_datos):
        print(f"[ERROR] No se encontró la carpeta: '{carpeta_datos}'")
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
        print(f"[ERROR] No se encontraron imágenes en '{carpeta_datos}'")
        sys.exit(1)

    print(f"[OK] {len(rutas)} imágenes listas.")
    return rutas


# ---------------------------------------------------------------------------
# Interfaz Gráfica: Ventana de Parámetros Específicos
# ---------------------------------------------------------------------------
def crear_controles(modo):
    """Recrea los sliders específicos sin duplicar el control de modo."""
    try:
        cv2.destroyWindow(VENTANA_CTRL)
    except Exception:
        pass
    cv2.waitKey(1)

    cv2.namedWindow(VENTANA_CTRL, cv2.WINDOW_NORMAL)

    if modo == 1:
        cv2.createTrackbar("Blur  0=sin  10=max", VENTANA_CTRL, 2, 10, _cb)
        cv2.createTrackbar("Brillo  0=-100  200=+100", VENTANA_CTRL, 100, 200, _cb)
        cv2.createTrackbar("Contraste x0.1  (10=1.0x)", VENTANA_CTRL, 10, 30, _cb)
    elif modo == 2:
        cv2.createTrackbar("Canny  Umbral Bajo  (0-300)", VENTANA_CTRL, 40, 300, _cb)
        cv2.createTrackbar("Canny  Umbral Alto  (0-300)", VENTANA_CTRL, 120, 300, _cb)
    elif modo == 3:
        cv2.createTrackbar("K-means  k-2  (0=k2  4=k6)", VENTANA_CTRL, 2, 4, _cb)
    elif modo == 4:
        cv2.createTrackbar("Umbral  0=auto Otsu  1-255=manual", VENTANA_CTRL, 0, 255, _cb)
    elif modo == 5:
        cv2.createTrackbar("Umbral dist %  (1-90  def=50)", VENTANA_CTRL, 50, 90, _cb)
    elif modo == 6:
        cv2.createTrackbar("Canal  0=Gris 1=R 2=G 3=B", VENTANA_CTRL, 0, 3, _cb)
    elif modo == 7:
        cv2.createTrackbar("Filtro pasa bajo  radio px  (0=sin)", VENTANA_CTRL, 0, 100, _cb)
    elif modo == 8:
        cv2.createTrackbar("Tono H  offset  (90=sin cambio)", VENTANA_CTRL, 90, 180, _cb)
        cv2.createTrackbar("Saturacion x%   (100=sin cambio)", VENTANA_CTRL, 100, 200, _cb)
        cv2.createTrackbar("Brillo V  x%    (100=sin cambio)", VENTANA_CTRL, 100, 200, _cb)

    cv2.resizeWindow(VENTANA_CTRL, 460, 150)
    _necesita_actualizar[0] = True


def procesar_modo(img, modo):
    motor = Motor(img)
    if modo == 1:
        blur_val      = cv2.getTrackbarPos("Blur  0=sin  10=max", VENTANA_CTRL)
        brillo_raw    = cv2.getTrackbarPos("Brillo  0=-100  200=+100", VENTANA_CTRL)
        contraste_raw = cv2.getTrackbarPos("Contraste x0.1  (10=1.0x)", VENTANA_CTRL)
        return motor.blur_y_brillo(blur_val * 2 + 1, brillo_raw - 100, max(0.1, contraste_raw / 10.0))
    elif modo == 2:
        bajo = cv2.getTrackbarPos("Canny  Umbral Bajo  (0-300)", VENTANA_CTRL)
        alto = cv2.getTrackbarPos("Canny  Umbral Alto  (0-300)", VENTANA_CTRL)
        return motor.bordes_canny(bajo, alto)
    elif modo == 3:
        k_offset = cv2.getTrackbarPos("K-means  k-2  (0=k2  4=k6)", VENTANA_CTRL)
        return motor.segmentar_kmeans(k_offset + 2)
    elif modo == 4:
        umbral_manual = cv2.getTrackbarPos("Umbral  0=auto Otsu  1-255=manual", VENTANA_CTRL)
        return motor.umbralizar_otsu(umbral_manual)
    elif modo == 5:
        umbral_pct = cv2.getTrackbarPos("Umbral dist %  (1-90  def=50)", VENTANA_CTRL)
        return motor.aplicar_watershed(max(0.01, umbral_pct / 100.0))
    elif modo == 6:
        canal = cv2.getTrackbarPos("Canal  0=Gris 1=R 2=G 3=B", VENTANA_CTRL)
        return motor.histograma_canal(canal, ALTO_IMG, ANCHO_IMG)
    elif modo == 7:
        radio = cv2.getTrackbarPos("Filtro pasa bajo  radio px  (0=sin)", VENTANA_CTRL)
        return motor.fourier(radio)
    elif modo == 8:
        delta_h  = cv2.getTrackbarPos("Tono H  offset  (90=sin cambio)", VENTANA_CTRL) - 90
        escala_s = cv2.getTrackbarPos("Saturacion x%   (100=sin cambio)", VENTANA_CTRL) / 100.0
        escala_v = cv2.getTrackbarPos("Brillo V  x%    (100=sin cambio)", VENTANA_CTRL) / 100.0
        return motor.ajustar_hsv(delta_h, escala_s, escala_v)
    return img.copy()


def _info_slider(modo):
    try:
        if modo == 1:
            b  = cv2.getTrackbarPos("Blur  0=sin  10=max", VENTANA_CTRL)
            br = cv2.getTrackbarPos("Brillo  0=-100  200=+100", VENTANA_CTRL) - 100
            c  = cv2.getTrackbarPos("Contraste x0.1  (10=1.0x)", VENTANA_CTRL) / 10.0
            return f"kernel={b*2+1}px   brillo={br:+d}   contraste={c:.1f}x"
        elif modo == 2:
            bajo = cv2.getTrackbarPos("Canny  Umbral Bajo  (0-300)", VENTANA_CTRL)
            alto = cv2.getTrackbarPos("Canny  Umbral Alto  (0-300)", VENTANA_CTRL)
            return f"Canny Bajo={bajo}  Alto={alto}"
        elif modo == 3:
            k = cv2.getTrackbarPos("K-means  k-2  (0=k2  4=k6)", VENTANA_CTRL) + 2
            return f"k={k} clusters de color"
        elif modo == 4:
            u = cv2.getTrackbarPos("Umbral  0=auto Otsu  1-255=manual", VENTANA_CTRL)
            return f"umbral={'Otsu Automatico' if u == 0 else str(u)}"
        elif modo == 5:
            pct = cv2.getTrackbarPos("Umbral dist %  (1-90  def=50)", VENTANA_CTRL)
            return f"Distancia = {pct}% del maximo"
        elif modo == 6:
            c = cv2.getTrackbarPos("Canal  0=Gris 1=R 2=G 3=B", VENTANA_CTRL)
            return f"canal = {['Grises', 'R (rojo)', 'G (verde)', 'B (azul)'][c]}"
        elif modo == 7:
            r = cv2.getTrackbarPos("Filtro pasa bajo  radio px  (0=sin)", VENTANA_CTRL)
            return f"Filtro Fourier Radio = {r}px"
        elif modo == 8:
            dh = cv2.getTrackbarPos("Tono H  offset  (90=sin cambio)", VENTANA_CTRL) - 90
            s  = cv2.getTrackbarPos("Saturacion x%   (100=sin cambio)", VENTANA_CTRL)
            v  = cv2.getTrackbarPos("Brillo V  x%    (100=sin cambio)", VENTANA_CTRL)
            return f"H {dh:+d}  Saturacion={s}%   Brillo={v}%"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Ensamblado del Display (Con Botonera Virtual Integrada y Menú Flotante)
# ---------------------------------------------------------------------------
def construir_display(img_orig, img_proc, modo, nombre_fruta, indice, total):
    global _estado_navegacion
    orig = cv2.resize(img_orig, (ANCHO_IMG, ALTO_IMG))
    proc = cv2.resize(img_proc, (ANCHO_IMG, ALTO_IMG))

    cv2.rectangle(orig, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
    cv2.putText(orig, "Original", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(proc, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
    cv2.putText(proc, MODOS[modo], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    sep_v = np.full((ALTO_IMG, 4, 3), (55, 55, 55), dtype=np.uint8)
    cuerpo = np.hstack([orig, sep_v, proc])
    ancho_panel = cuerpo.shape[1]

    # CABECERA PRINCIPAL (Gris oscuro)
    header = np.full((55, ancho_panel, 3), (25, 25, 25), dtype=np.uint8)
    
    # Botón 1: Anterior
    cv2.rectangle(header, (10, 12), (110, 38), (50, 50, 50), -1)
    cv2.rectangle(header, (10, 12), (110, 38), (100, 100, 100), 1)
    cv2.putText(header, "<< ANTERIOR", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Botón 2: Siguiente
    cv2.rectangle(header, (120, 12), (220, 38), (50, 50, 50), -1)
    cv2.rectangle(header, (120, 12), (220, 38), (100, 100, 100), 1)
    cv2.putText(header, "SIGUIENTE >>", (126, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Botón 3: DISEÑO DEL MENÚ DESPLEGABLE
    color_btn_menu = (75, 45, 15) if _estado_navegacion["menu_abierto"] else (40, 40, 40)
    cv2.rectangle(header, (230, 12), (410, 38), color_btn_menu, -1)
    cv2.rectangle(header, (230, 12), (410, 38), (120, 120, 120), 1)
    texto_dropdown = f" {MODOS[modo]}  ▾"
    cv2.putText(header, texto_dropdown[:22], (235, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1, cv2.LINE_AA)

    # Botón 4: Guardar
    cv2.rectangle(header, (420, 12), (510, 38), (20, 80, 20), -1)
    cv2.rectangle(header, (420, 12), (510, 38), (50, 150, 50), 1)
    cv2.putText(header, "GUARDAR", (435, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1, cv2.LINE_AA)

    # Texto de Estado de Frutas
    cv2.putText(header, f"Fruta: {nombre_fruta.title()}  [{indice + 1}/{total}]", (530, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(header, (0, 54), (ancho_panel, 54), (70, 70, 70), 1)

    panel_completo = np.vstack([header, cuerpo])

    # ── RENDERIZAR LA LISTA DESPLEGADA (Si el menú está abierto) ──
    if _estado_navegacion["menu_abierto"]:
        for i, nombre_modo in MODOS.items():
            y_opcion = 42 + (i - 1) * 25
            bg_color = (100, 65, 25) if i == modo else (30, 30, 30)
            cv2.rectangle(panel_completo, (230, y_opcion), (410, y_opcion + 24), bg_color, -1)
            cv2.rectangle(panel_completo, (230, y_opcion), (410, y_opcion + 24), (60, 60, 60), 1)
            cv2.putText(panel_completo, nombre_modo, (240, y_opcion + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Barra inferior estática
    info_texto = _info_slider(modo)
    info_bar = np.full((26, ancho_panel, 3), (12, 12, 12), dtype=np.uint8)
    if info_texto:
        cv2.putText(info_bar, info_texto, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 0), 1, cv2.LINE_AA)

    return np.vstack([panel_completo, info_bar])


# ---------------------------------------------------------------------------
# Función Ejecutora Principal
# ---------------------------------------------------------------------------
def main():
    global _estado_navegacion
    
    # Forzamos las rutas absolutas de tu computadora para evitar cierres instantáneos
    carpeta_datos = r"C:\Users\Paula\Documents\TP Integrador - Tec Proc de Imagenes\images"
    script_dir = r"C:\Users\Paula\Documents\TP Integrador - Tec Proc de Imagenes"

    rutas = cargar_rutas_imagenes(carpeta_datos)
    
    _estado_navegacion["total"] = len(rutas)
    _estado_navegacion["indice"] = 0
    _estado_navegacion["modo"] = 1
    
    modo_anterior = -1
    indice_cache = -1
    img_actual = None
    nombre_fruta = ""
    panel = None

    cv2.namedWindow(VENTANA_DISPLAY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_DISPLAY, ANCHO_IMG * 2 + 4, ALTO_IMG + 81)
    
    cv2.setMouseCallback(VENTANA_DISPLAY, mouse_handler)

    while True:
        indice = _estado_navegacion["indice"]
        modo = _estado_navegacion["modo"]
        total = _estado_navegacion["total"]

        if indice != indice_cache:
            ruta, nombre_fruta = rutas[indice]
            imagen = cv2.imread(ruta)
            if imagen is None:
                _estado_navegacion["indice"] = (indice + 1) % total
                continue
            img_actual = cv2.resize(imagen, (ANCHO_IMG, ALTO_IMG))
            indice_cache = indice
            _necesita_actualizar[0] = True

        if modo != modo_anterior:
            crear_controles(modo)
            modo_anterior = modo

        if _necesita_actualizar[0]:
            _necesita_actualizar[0] = False
            try:
                img_proc = procesar_modo(img_actual, modo)
            except Exception as e:
                img_proc = np.zeros((ALTO_IMG, ANCHO_IMG, 3), dtype=np.uint8)
                cv2.putText(img_proc, f"ERROR: {str(e)[:30]}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            panel = construir_display(img_actual, img_proc, modo, nombre_fruta, indice, total)
            cv2.imshow(VENTANA_DISPLAY, panel)

        if _estado_navegacion["guardar"] and panel is not None:
            _estado_navegacion["guardar"] = False
            nombre_archivo = f"procesado_{indice:03d}_{nombre_fruta.replace(' ', '_')}_m{modo}.png"
            cv2.imwrite(os.path.join(script_dir, nombre_archivo), panel)
            print(f"  [GUARDADO] {nombre_archivo}")

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord('q'), 27):
            break

        if cv2.getWindowProperty(VENTANA_DISPLAY, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()