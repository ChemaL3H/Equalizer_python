#!/usr/bin/env python3
"""
Ecualizador Visual + Letras en Tiempo Real
Little Jesus - La Magia
Requiere: pip install pygame numpy
"""

import pygame
import numpy as np
import sys
import os
import re
import time

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
AUDIO_FILE = "02. Little Jesus - La Magia.mp3"
LRC_FILE   = "02. Little Jesus - La Magia.lrc"          # guarda el archivo .lrc con este nombre
                                    # o cambia aquí el nombre de tu archivo

ANCHO      = 1000
ALTO       = 600
FPS        = 60
NUM_BARRAS = 60
SR         = 44100
CHUNK      = 2048

# Colores
COLOR_FONDO      = (5,   5,  12)
COLOR_LETRA      = (255, 80, 160)   # rosa
COLOR_LETRA_DIM  = (100, 30,  60)   # rosa apagado (línea anterior)
COLOR_BARRA_LOW  = (255, 60,  60)   # rojo  — graves
COLOR_BARRA_MID  = (255, 200,  0)   # amarillo — medios
COLOR_BARRA_HIGH = (0,  210, 255)   # cian  — agudos
COLOR_TITULO     = (180, 180, 180)

# ─── PARSEAR .LRC ─────────────────────────────────────────────────────────────
def parsear_lrc(ruta):
    """Devuelve lista de (segundos_float, texto)."""
    patron = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2})\](.*)')
    letras = []
    if not os.path.exists(ruta):
        return letras
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            m = patron.match(linea.strip())
            if m:
                mins, segs, cents, texto = m.groups()
                t = int(mins)*60 + int(segs) + int(cents)/100
                letras.append((t, texto.strip()))
    letras.sort(key=lambda x: x[0])
    return letras

def linea_actual(letras, pos_seg):
    """Devuelve índice de la línea que corresponde a pos_seg."""
    idx = -1
    for i, (t, _) in enumerate(letras):
        if pos_seg >= t:
            idx = i
        else:
            break
    return idx

# ─── ANÁLISIS FFT SIMPLE (sobre array de samples) ─────────────────────────────
class AudioCaptura:
    """
    pygame.mixer no expone samples en tiempo real en todas las plataformas.
    Usamos numpy para generar una onda sintética sincronizada al tiempo
    de reproducción cuando no hay acceso al buffer directo.
    Como alternativa robusta, cargamos el audio con numpy via soundfile
    si está disponible; si no, generamos barras animadas sincronizadas.
    """
    def __init__(self, archivo):
        self.samples = None
        self.sr      = SR
        self._cargar(archivo)

    def _cargar(self, archivo):
        try:
            import soundfile as sf
            data, sr = sf.read(archivo, dtype='float32', always_2d=True)
            self.samples = data[:, 0]   # canal izquierdo
            self.sr      = sr
            print("[OK] Audio cargado con soundfile para análisis FFT real.")
        except Exception:
            try:
                import scipy.io.wavfile as wf
                sr, data = wf.read(archivo)
                if data.ndim > 1:
                    data = data[:, 0]
                self.samples = data.astype(np.float32) / 32768.0
                self.sr      = sr
                print("[OK] Audio cargado con scipy para análisis FFT real.")
            except Exception:
                print("[INFO] Sin soundfile/scipy: usando barras animadas sintéticas.")

    def obtener_fft(self, pos_seg):
        if self.samples is not None:
            inicio = int(pos_seg * self.sr)
            fin    = inicio + CHUNK
            if fin > len(self.samples):
                frag = self.samples[inicio:]
                frag = np.pad(frag, (0, CHUNK - len(frag)))
            else:
                frag = self.samples[inicio:fin]
            ventana  = frag * np.hanning(len(frag))
            espectro = np.abs(np.fft.rfft(ventana))[:512]
            return espectro
        else:
            # Barras sintéticas animadas (fallback)
            t = pos_seg
            espectro = np.array([
                abs(np.sin(t * (i + 1) * 0.3 + i)) * (512 - i)
                for i in range(512)
            ], dtype=np.float32)
            return espectro

# ─── DIBUJAR ──────────────────────────────────────────────────────────────────
def dibujar_barras(superficie, espectro, ancho, alto_disp, picos):
    bandas = np.array_split(espectro, NUM_BARRAS)
    ancho_barra  = ancho // NUM_BARRAS
    margen       = 2
    max_altura   = alto_disp - 10

    for i, banda in enumerate(bandas):
        intensidad = float(np.mean(banda)) * 5
        picos[i]   = max(intensidad, picos[i] * 0.92)
        altura     = min(int(picos[i]), max_altura)
        if altura < 2:
            continue

        # Color según frecuencia
        t = i / NUM_BARRAS
        if t < 0.33:
            r = int(COLOR_BARRA_LOW[0]  + t/0.33 * (COLOR_BARRA_MID[0]  - COLOR_BARRA_LOW[0]))
            g = int(COLOR_BARRA_LOW[1]  + t/0.33 * (COLOR_BARRA_MID[1]  - COLOR_BARRA_LOW[1]))
            b = int(COLOR_BARRA_LOW[2]  + t/0.33 * (COLOR_BARRA_MID[2]  - COLOR_BARRA_LOW[2]))
        else:
            tt = (t - 0.33) / 0.67
            r  = int(COLOR_BARRA_MID[0] + tt * (COLOR_BARRA_HIGH[0] - COLOR_BARRA_MID[0]))
            g  = int(COLOR_BARRA_MID[1] + tt * (COLOR_BARRA_HIGH[1] - COLOR_BARRA_MID[1]))
            b  = int(COLOR_BARRA_MID[2] + tt * (COLOR_BARRA_HIGH[2] - COLOR_BARRA_MID[2]))

        color = (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))
        x     = i * ancho_barra + margen
        y     = alto_disp - altura
        pygame.draw.rect(superficie, color,
                         (x, y, ancho_barra - margen*2, altura),
                         border_radius=3)

        # Pico
        pygame.draw.rect(superficie, (255,255,255),
                         (x, y - 3, ancho_barra - margen*2, 3),
                         border_radius=1)

def dibujar_letras(superficie, letras, idx, fuente_grande, fuente_peq, ancho, y_base):
    if idx < 0 or not letras:
        return

    # Línea anterior (dim)
    if idx > 0 and letras[idx-1][1]:
        surf = fuente_peq.render(letras[idx-1][1], True, COLOR_LETRA_DIM)
        r    = surf.get_rect(centerx=ancho//2, bottom=y_base - 10)
        superficie.blit(surf, r)

    # Línea actual (rosa brillante)
    texto = letras[idx][1]
    if texto:
        surf = fuente_grande.render(texto, True, COLOR_LETRA)
        # Sombra
        sombra = fuente_grande.render(texto, True, (80, 0, 40))
        rs = sombra.get_rect(centerx=ancho//2 + 2, centery=y_base + 2)
        r  = surf.get_rect(centerx=ancho//2,       centery=y_base)
        superficie.blit(sombra, rs)
        superficie.blit(surf,   r)

    # Siguiente línea (dim)
    if idx + 1 < len(letras) and letras[idx+1][1]:
        surf = fuente_peq.render(letras[idx+1][1], True, COLOR_LETRA_DIM)
        r    = surf.get_rect(centerx=ancho//2, top=y_base + 30)
        superficie.blit(surf, r)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=CHUNK)

    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Little Jesus — La Magia  |  Ecualizador")
    reloj = pygame.time.Clock()

    # Fuentes
    try:
        fuente_titulo  = pygame.font.SysFont("DejaVu Sans", 18, bold=False)
        fuente_grande  = pygame.font.SysFont("DejaVu Sans", 32, bold=True)
        fuente_peq     = pygame.font.SysFont("DejaVu Sans", 22, bold=False)
    except Exception:
        fuente_titulo  = pygame.font.Font(None, 22)
        fuente_grande  = pygame.font.Font(None, 38)
        fuente_peq     = pygame.font.Font(None, 26)

    # Cargar audio
    if not os.path.exists(AUDIO_FILE):
        print(f"[ERROR] No encontré '{AUDIO_FILE}'")
        print("        Pon el mp3 en la misma carpeta que este script.")
        sys.exit(1)

    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()

    captura = AudioCaptura(AUDIO_FILE)
    letras  = parsear_lrc(LRC_FILE)
    if not letras:
        print(f"[AVISO] No encontré '{LRC_FILE}'. Se mostrarán solo las barras.")

    picos = np.zeros(NUM_BARRAS)

    ALTO_BARRAS  = int(ALTO * 0.55)
    Y_LETRAS     = ALTO_BARRAS + int((ALTO - ALTO_BARRAS) * 0.40)

    print("¡Ecualizador iniciado!  Ctrl+C / cierra ventana para salir.")

    corriendo = True
    while corriendo:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                corriendo = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    corriendo = False
                if ev.key == pygame.K_SPACE:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                    else:
                        pygame.mixer.music.unpause()

        if not pygame.mixer.music.get_busy():
            # Canción terminó
            pantalla.fill(COLOR_FONDO)
            msg = fuente_grande.render("♪  Fin  ♪", True, COLOR_LETRA)
            pantalla.blit(msg, msg.get_rect(center=(ANCHO//2, ALTO//2)))
            pygame.display.flip()
            reloj.tick(FPS)
            continue

        pos_seg = pygame.mixer.music.get_pos() / 1000.0

        # FFT
        espectro = captura.obtener_fft(pos_seg)

        # Fondo
        pantalla.fill(COLOR_FONDO)

        # Línea divisoria sutil
        pygame.draw.line(pantalla, (30, 30, 50), (0, ALTO_BARRAS), (ANCHO, ALTO_BARRAS), 1)

        # Barras
        dibujar_barras(pantalla, espectro, ANCHO, ALTO_BARRAS, picos)

        # Letras
        idx = linea_actual(letras, pos_seg)
        dibujar_letras(pantalla, letras, idx, fuente_grande, fuente_peq, ANCHO, Y_LETRAS)

        # Título + tiempo
        t_total = int(pos_seg)
        m, s    = divmod(t_total, 60)
        info    = fuente_titulo.render(
            f"Little Jesus — La Magia    {m:02d}:{s:02d}    [SPACE] pausa  [ESC] salir",
            True, COLOR_TITULO)
        pantalla.blit(info, (10, ALTO - 28))

        pygame.display.flip()
        reloj.tick(FPS)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
