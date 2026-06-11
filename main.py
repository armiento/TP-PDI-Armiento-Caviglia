"""
Editor Interactivo de Imágenes de Frutas
=========================================
Trabajo Práctico Final – Técnicas de Procesamiento Digital de Imágenes
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
IFTS N°18 – CABA – Ciclo 2025
Profesor: Bonini Juan Ignacio

Herramienta interactiva con sliders (cv2.createTrackbar) que permite ajustar
los parámetros de cada técnica y ver el efecto en tiempo real.

Teclas de modo:
  1 — Blur gaussiano + Brillo / Contraste    (Unidad 5)
  2 — Detección de bordes Canny              (Unidad 5)
  3 — Segmentación K-means HSV               (Unidad 7)
  4 — Umbralización Otsu (auto o manual)     (Unidad 7)
  5 — Separación con Watershed               (Unidad 7)
  6 — Histograma antes/después ecualización  (Unidad 4)
  7 — Espectro de Fourier                    (Unidad 6)
  8 — Ajuste de Tono / Saturación / Brillo   (Unidad 7)

Navegación:
  A / ← : imagen anterior
  D / → : imagen siguiente
  s     : guardar la imagen procesada actual
  q / Esc : salir
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

VENTANA_DISPLAY = "Editor de Frutas – PDI TP Final"
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
# Estado global (accesible desde los callbacks de trackbar)
# ---------------------------------------------------------------------------
_necesita_actualizar = [True]


def _cb(_val):
    """Callback para todos los trackbars: señala que hay que reprocesar."""
    _necesita_actualizar[0] = True


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------

def cargar_rutas_imagenes(carpeta_datos):
    """
    Carga todas las rutas de imágenes del dataset organizado por fruta.

    Estructura esperada:
        carpeta_datos/nombre_fruta/imagen.jpg

    Args:
        carpeta_datos: Ruta a la carpeta raíz del dataset

    Returns:
        Lista de tuplas (ruta_imagen, nombre_fruta) ordenada por fruta
    """
    if not os.path.exists(carpeta_datos):
        print(f"[ERROR] No se encontró la carpeta: '{carpeta_datos}'")
        print(f"        Ruta absoluta: {os.path.abspath(carpeta_datos)}")
        print(f"        Estructura esperada: {carpeta_datos}/nombre_fruta/imagen.jpg")
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
        print(f"        Formatos soportados: JPG, JPEG, PNG, BMP")
        sys.exit(1)

    print(f"[OK] {len(rutas)} imágenes en {len(set(n for _, n in rutas))} frutas")
    return rutas


# ---------------------------------------------------------------------------
# Ventana de controles (trackbars)
# ---------------------------------------------------------------------------

def crear_controles(modo):
    """
    Recrea la ventana de controles con los sliders específicos del modo activo.

    Cada modo de visualización tiene sus propios parámetros ajustables.
    La ventana anterior se destruye y se crea una nueva con los trackbars
    correspondientes al modo seleccionado.

    Todos los modos tienen al menos un slider interactivo.

    Args:
        modo: Modo de visualización activo (1-8)
    """
    try:
        cv2.destroyWindow(VENTANA_CTRL)
    except Exception:
        pass
    cv2.waitKey(1)  # vaciar la cola de eventos de OpenCV

    cv2.namedWindow(VENTANA_CTRL, cv2.WINDOW_NORMAL)

    if modo == 1:
        # Unidad 5: Blur gaussiano y ajuste de brillo/contraste
        # Blur: trackbar 0-10 → kernel 1,3,5,...,21 (solo impares)
        # Brillo: 0-200, centro en 100 (100 = sin cambio, equivale a offset 0)
        # Contraste: 1-30, centro en 10 (10 = 1.0x)
        cv2.createTrackbar("Blur  0=sin  10=max", VENTANA_CTRL, 2, 10, _cb)
        cv2.createTrackbar("Brillo  0=-100  200=+100", VENTANA_CTRL, 100, 200, _cb)
        cv2.createTrackbar("Contraste x0.1  (10=1.0x)", VENTANA_CTRL, 10, 30, _cb)

    elif modo == 2:
        # Unidad 5: Detección de bordes Canny (doble umbral)
        cv2.createTrackbar("Canny  Umbral Bajo  (0-300)", VENTANA_CTRL, 40, 300, _cb)
        cv2.createTrackbar("Canny  Umbral Alto  (0-300)", VENTANA_CTRL, 120, 300, _cb)

    elif modo == 3:
        # Unidad 7: K-means — trackbar 0-4 mapea a k=2,3,4,5,6
        cv2.createTrackbar("K-means  k-2  (0=k2  4=k6)", VENTANA_CTRL, 2, 4, _cb)

    elif modo == 4:
        # Unidad 7: Otsu — 0 = automático, 1-255 = umbral manual
        cv2.createTrackbar("Umbral  0=auto Otsu  1-255=manual", VENTANA_CTRL, 0, 255, _cb)

    elif modo == 5:
        # Unidad 7: Watershed — fracción del máximo de la transformada de distancia
        # 10 = 10 % del máximo (semillas grandes) … 90 = 90 % (semillas pequeñas)
        cv2.createTrackbar("Umbral dist %  (1-90  def=50)", VENTANA_CTRL, 50, 90, _cb)

    elif modo == 6:
        # Unidad 4: Histograma — canal a visualizar (0=Gris 1=R 2=G 3=B)
        cv2.createTrackbar("Canal  0=Gris 1=R 2=G 3=B", VENTANA_CTRL, 0, 3, _cb)

    elif modo == 7:
        # Unidad 6: Fourier — radio del filtro pasa bajos en píxeles
        # 0 = sin filtro (muestra espectro); 1-100 = mantiene frecuencias dentro del radio
        cv2.createTrackbar("Filtro pasa bajo  radio px  (0=sin)", VENTANA_CTRL, 0, 100, _cb)

    elif modo == 8:
        # Unidad 7: Ajuste de espacio HSV
        # H: 0-180, centro 90 (90 = sin cambio, delta_h = val - 90)
        # S y V: 0-200, centro 100 (100 = 1.0x, sin cambio)
        cv2.createTrackbar("Tono H  offset  (90=sin cambio)", VENTANA_CTRL, 90, 180, _cb)
        cv2.createTrackbar("Saturacion x%   (100=sin cambio)", VENTANA_CTRL, 100, 200, _cb)
        cv2.createTrackbar("Brillo V  x%    (100=sin cambio)", VENTANA_CTRL, 100, 200, _cb)

    cv2.resizeWindow(VENTANA_CTRL, 440, 160)
    _necesita_actualizar[0] = True


# ---------------------------------------------------------------------------
# Procesamiento según modo activo
# ---------------------------------------------------------------------------

def procesar_modo(img, modo):
    """
    Aplica la técnica de procesamiento correspondiente al modo activo.

    Lee los valores actuales de los trackbars y delega en Motor, que
    internamente usa PillowProcessor (modos básicos) u OpenCVProcessor
    (modos avanzados).

    Unidades cubiertas:
      Modo 1 → Unidades 3 y 5 (blur, brillo, contraste — Pillow)
      Modo 2 → Unidad 5 (Canny)
      Modo 3 → Unidad 7 (K-means HSV)
      Modo 4 → Unidad 7 (Otsu automático o umbral manual)
      Modo 5 → Unidad 7 (Watershed — umbral de transformada de distancia)
      Modo 6 → Unidad 4 (histograma por canal + ecualización)
      Modo 7 → Unidad 6 (Fourier — espectro o reconstrucción con filtro pasa bajos)
      Modo 8 → Unidad 7 (ajuste HSV)

    Args:
        img: Imagen BGR redimensionada (ANCHO_IMG × ALTO_IMG)
        modo: Modo de visualización activo (1-8)

    Returns:
        Imagen BGR procesada lista para mostrar
    """
    motor = Motor(img)

    if modo == 1:
        blur_val      = cv2.getTrackbarPos("Blur  0=sin  10=max", VENTANA_CTRL)
        brillo_raw    = cv2.getTrackbarPos("Brillo  0=-100  200=+100", VENTANA_CTRL)
        contraste_raw = cv2.getTrackbarPos("Contraste x0.1  (10=1.0x)", VENTANA_CTRL)
        kernel    = blur_val * 2 + 1                    # 1, 3, 5, ..., 21
        brillo    = brillo_raw - 100                    # −100 a +100
        contraste = max(0.1, contraste_raw / 10.0)      # 0.1 a 3.0
        return motor.blur_y_brillo(kernel, brillo, contraste)

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
    """
    Devuelve un string con los valores actuales de los sliders del modo activo.
    Retorna cadena vacía para modos sin sliders o si ocurre algún error.
    """
    try:
        if modo == 1:
            b  = cv2.getTrackbarPos("Blur  0=sin  10=max", VENTANA_CTRL)
            br = cv2.getTrackbarPos("Brillo  0=-100  200=+100", VENTANA_CTRL) - 100
            c  = cv2.getTrackbarPos("Contraste x0.1  (10=1.0x)", VENTANA_CTRL) / 10.0
            return f"kernel={b*2+1}px   brillo={br:+d}   contraste={c:.1f}x"
        elif modo == 2:
            bajo = cv2.getTrackbarPos("Canny  Umbral Bajo  (0-300)", VENTANA_CTRL)
            alto = cv2.getTrackbarPos("Canny  Umbral Alto  (0-300)", VENTANA_CTRL)
            return f"umbral bajo={bajo}   umbral alto={alto}"
        elif modo == 3:
            k = cv2.getTrackbarPos("K-means  k-2  (0=k2  4=k6)", VENTANA_CTRL) + 2
            return f"k={k} clusters de color"
        elif modo == 4:
            u = cv2.getTrackbarPos("Umbral  0=auto Otsu  1-255=manual", VENTANA_CTRL)
            return f"umbral={'auto (Otsu)' if u == 0 else str(u)}"
        elif modo == 5:
            pct = cv2.getTrackbarPos("Umbral dist %  (1-90  def=50)", VENTANA_CTRL)
            return f"umbral transformada de distancia = {pct}% del maximo"
        elif modo == 6:
            c = cv2.getTrackbarPos("Canal  0=Gris 1=R 2=G 3=B", VENTANA_CTRL)
            nombre = {0: "Grises", 1: "R (rojo)", 2: "G (verde)", 3: "B (azul)"}[c]
            return f"canal = {nombre}   (original gris + ecualizado color)"
        elif modo == 7:
            r = cv2.getTrackbarPos("Filtro pasa bajo  radio px  (0=sin)", VENTANA_CTRL)
            return f"radio = {r}px   ({'espectro sin filtro' if r == 0 else 'imagen reconstruida con pasa bajos'})"
        elif modo == 8:
            dh = cv2.getTrackbarPos("Tono H  offset  (90=sin cambio)", VENTANA_CTRL) - 90
            s  = cv2.getTrackbarPos("Saturacion x%   (100=sin cambio)", VENTANA_CTRL)
            v  = cv2.getTrackbarPos("Brillo V  x%    (100=sin cambio)", VENTANA_CTRL)
            return f"H {dh:+d}°   Saturacion={s}%   Brillo={v}%"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Construcción del panel de visualización
# ---------------------------------------------------------------------------

def construir_display(img_orig, img_proc, modo, nombre_fruta, indice, total):
    """
    Construye el panel con imagen original (izquierda) y procesada (derecha).

    Estructura del panel:
      ┌────────────────────────────────────────────┐
      │  Encabezado: fruta, índice, modo, atajos   │ 50 px
      ├──────────────────────┬─────────────────────┤
      │  Original            │  <Modo activo>      │ ALTO_IMG px
      ├──────────────────────┴─────────────────────┤
      │  Valores actuales de sliders               │ 26 px
      └────────────────────────────────────────────┘

    Args:
        img_orig: Imagen original BGR (se redimensionará a ANCHO_IMG × ALTO_IMG)
        img_proc: Imagen procesada BGR
        modo: Modo activo (1-8)
        nombre_fruta: Nombre de la carpeta/fruta
        indice: Índice de la imagen actual (base 0)
        total: Total de imágenes en el dataset

    Returns:
        Panel BGR completo como numpy array
    """
    orig = cv2.resize(img_orig, (ANCHO_IMG, ALTO_IMG))
    proc = cv2.resize(img_proc, (ANCHO_IMG, ALTO_IMG))

    # Etiqueta sobre imagen original
    cv2.rectangle(orig, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
    cv2.putText(orig, "Original", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Etiqueta sobre imagen procesada (modo activo)
    cv2.rectangle(proc, (0, 0), (ANCHO_IMG, 22), (20, 20, 20), -1)
    cv2.putText(proc, MODOS[modo], (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    sep_v = np.full((ALTO_IMG, 4, 3), (55, 55, 55), dtype=np.uint8)
    cuerpo = np.hstack([orig, sep_v, proc])
    ancho_panel = cuerpo.shape[1]

    # Encabezado
    header = np.full((50, ancho_panel, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(
        header,
        f"Fruta: {nombre_fruta.title()}   [{indice + 1}/{total}]   |   Modo: {MODOS[modo]}",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA
    )
    cv2.putText(
        header,
        "1-8: cambiar modo   |   A / D: navegar   |   s: guardar   |   q: salir",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 140, 140), 1, cv2.LINE_AA
    )
    cv2.line(header, (0, 49), (ancho_panel, 49), (65, 65, 65), 1)

    # Barra de información con valores actuales de los sliders
    info_texto = _info_slider(modo)
    info_bar = np.full((26, ancho_panel, 3), (12, 12, 12), dtype=np.uint8)
    if info_texto:
        cv2.putText(info_bar, info_texto, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 0), 1, cv2.LINE_AA)

    return np.vstack([header, cuerpo, info_bar])


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main():
    """
    Función principal: carga el dataset y ejecuta el bucle interactivo.

    Lógica del bucle:
      1. Si cambió el índice de imagen → recargar y redimensionar
      2. Si cambió el modo → recrear ventana de controles con nuevos sliders
      3. Si _necesita_actualizar es True (slider movido o imagen/modo cambiado)
         → procesar imagen y actualizar el display
      4. Leer tecla y actualizar estado según corresponda
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_datos = os.path.join(script_dir, CARPETA_DATOS)

    print("=" * 60)
    print("  Editor Interactivo de Frutas – TP Final PDI")
    print("  TSCD – IFTS N°18 – Ciclo 2025")
    print("=" * 60)

    rutas = cargar_rutas_imagenes(carpeta_datos)
    total = len(rutas)
    indice = 0
    modo = 1
    modo_anterior = -1
    indice_cache = -1
    img_actual = None
    nombre_fruta = ""
    panel = None

    cv2.namedWindow(VENTANA_DISPLAY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_DISPLAY, ANCHO_IMG * 2 + 4, ALTO_IMG + 76)

    print(f"\n  {total} imágenes listas. Teclas 1-8 para modo, A/D para navegar, q para salir.\n")

    while True:
        # ── Cargar nueva imagen si el índice cambió ──────────────────────────
        if indice != indice_cache:
            ruta, nombre_fruta = rutas[indice]
            imagen = cv2.imread(ruta)
            if imagen is None:
                print(f"  [WARN] No se pudo leer: {ruta}")
                indice = (indice + 1) % total
                continue
            img_actual = cv2.resize(imagen, (ANCHO_IMG, ALTO_IMG))
            indice_cache = indice
            _necesita_actualizar[0] = True
            print(f"  [{indice + 1}/{total}] {nombre_fruta} – {os.path.basename(ruta)}")

        # ── Recrear controles si el modo cambió ─────────────────────────────
        if modo != modo_anterior:
            crear_controles(modo)
            modo_anterior = modo

        # ── Reprocesar si hay cambios ────────────────────────────────────────
        if _necesita_actualizar[0]:
            _necesita_actualizar[0] = False
            try:
                img_proc = procesar_modo(img_actual, modo)
            except Exception as e:
                print(f"  [ERROR modo {modo}]: {e}")
                img_proc = np.zeros((ALTO_IMG, ANCHO_IMG, 3), dtype=np.uint8)
                cv2.putText(img_proc, "ERROR: " + str(e)[:35], (8, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

            panel = construir_display(
                img_actual, img_proc, modo, nombre_fruta, indice, total
            )
            cv2.imshow(VENTANA_DISPLAY, panel)

        # ── Leer tecla (30 ms) ───────────────────────────────────────────────
        tecla = cv2.waitKey(30) & 0xFF

        if tecla in (ord('q'), 27):
            print("\n  Cerrando la aplicación.")
            break

        elif tecla in (ord('d'), 83):   # d o flecha derecha
            indice = (indice + 1) % total

        elif tecla in (ord('a'), 81):   # a o flecha izquierda
            indice = (indice - 1) % total

        elif ord('1') <= tecla <= ord('8'):
            nuevo_modo = tecla - ord('0')
            if nuevo_modo != modo:
                modo = nuevo_modo
                print(f"  Modo: {MODOS[modo]}")

        elif tecla == ord('s') and panel is not None:
            nombre_archivo = (
                f"procesado_{indice:03d}_{nombre_fruta.replace(' ', '_')}_m{modo}.png"
            )
            ruta_guardado = os.path.join(script_dir, nombre_archivo)
            cv2.imwrite(ruta_guardado, panel)
            print(f"  [GUARDADO] {ruta_guardado}")

        # Detectar cierre de ventana por el usuario
        if cv2.getWindowProperty(VENTANA_DISPLAY, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
