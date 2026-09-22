#!/usr/bin/env python3
"""Puente MediaPipe (Face Landmarker) -> GestureController de openface.py.

Python 3.9+; requiere: opencv-python, mediapipe, numpy.

Ejemplo:
    python mediapipe_gestures.py --preview --debug --invert-x

stdout: mismos eventos JSONL que openface.py (STOP/MOVE/INTERACT/MODE/...)

Documentación:
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
"""

import argparse
import math
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


if __package__:
    from .gesture_controller import GestureController, log
    from .config import add_config_arguments, parse_settings
else:
    from gesture_controller import GestureController, log
    from config import add_config_arguments, parse_settings

FACE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
)

# Índices del mesh de 468/478 puntos de MediaPipe.
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 144, 153]
LEFT_BROW, LEFT_EYE_TOP = 105, 159
RIGHT_BROW, RIGHT_EYE_TOP = 334, 386
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
NOSE_TIP, NASION, LEFT_TEMPLE, RIGHT_TEMPLE, MOUTH_L, MOUTH_R = 1, 168, 127, 356, 61, 291
# MOUTH_L/MOUTH_R se conservan solo para mouth_metric(); ya no entran en head_pose().

POSE_LANDMARKS = [NOSE_TIP, NASION, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_TEMPLE, RIGHT_TEMPLE]
# Puntos rígidos frente a la articulación mandibular: nariz, nasión, esquinas
# oculares externas y sienes. Se retiraron mentón y comisuras de boca del set
# de pose porque solvePnP los trataba como geometría rígida y atribuía la
# apertura de mandíbula a un falso cambio de pitch.

# Modelo 3D canónico (mm) para solvePnP — 6 puntos, ninguno afectado por la mandíbula.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # punta de nariz
    (0.0, 45.0, -30.0),       # nasión (entre las cejas)
    (-225.0, 170.0, -135.0),  # esquina externa ojo (visualmente izquierda)
    (225.0, 170.0, -135.0),   # esquina externa ojo (visualmente derecha)
    (-320.0, 90.0, -270.0),   # sien/pómulo izquierdo
    (320.0, 90.0, -270.0),    # sien/pómulo derecho
], dtype=np.float64)

EAR_OPEN_BASELINE = 0.30  # EAR típico ojo abierto; ajusta si tu cámara/ángulo difiere
BLINK_SCALE = 12.0  # heurístico: mapea (baseline - EAR) a una escala tipo AU
BROW_SCALE = 20.0  # heurístico: mapea distancia ceja/ojo normalizada a escala tipo AU

_camera_matrix_cache = {}


def euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def camera_matrix(w, h):
    key = (w, h)
    if key not in _camera_matrix_cache:
        focal = float(w)
        _camera_matrix_cache[key] = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
    return _camera_matrix_cache[key]


def head_pose(landmarks, w, h):
    """Devuelve (yaw_rad, pitch_rad) vía solvePnP, o None si falla.

    Usa seis puntos rígidos (nariz, nasión, esquinas oculares externas y
    sienes); ninguno se desplaza al abrir la mandíbula, así que abrir la
    boca ya no se filtra como una falsa inclinación de cabeza.
    """
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LANDMARKS],
        dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix(w, h),
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch, yaw, _roll = (float(v[0]) for v in euler)
    # decomposeProjectionMatrix es ambiguo en signo cerca de +-90°; corrige.
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    return math.radians(yaw), math.radians(pitch)


def ear(landmarks, idxs, w, h):
    c = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
    d_v1 = euclid(c[1], c[5])
    d_v2 = euclid(c[2], c[4])
    d_h = euclid(c[0], c[3])
    return (d_v1 + d_v2) / (2.0 * d_h) if d_h > 0 else 0.0


def blink_metric(landmarks, w, h):
    avg_ear = (ear(landmarks, LEFT_EYE_EAR, w, h) + ear(landmarks, RIGHT_EYE_EAR, w, h)) / 2
    return max(0.0, EAR_OPEN_BASELINE - avg_ear) * BLINK_SCALE


def mouth_metric(landmarks, w, h):
    """Apertura interior (13–14) / ancho de boca (61–291), sin unidades."""
    def point(i):
        return (landmarks[i].x * w, landmarks[i].y * h)
    width = euclid(point(61), point(291))
    return euclid(point(13), point(14)) / width if width > 0 else float('nan')


def brow_metric(landmarks, w, h):
    interocular = euclid(
        (landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h),
        (landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h))
    if interocular <= 0:
        return 0.0

    def side(brow_idx, eye_idx):
        return max(0.0, (landmarks[eye_idx].y * h - landmarks[brow_idx].y * h) / interocular)
    raised = (side(LEFT_BROW, LEFT_EYE_TOP) + side(RIGHT_BROW, RIGHT_EYE_TOP)) / 2
    return raised * BROW_SCALE


class EMASmoother:
    """Filtro paso-bajo de un polo (media móvil exponencial) por señal.

    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

    Por qué funciona: cuadro a cuadro, el ruido de detección de landmarks
    (temblor de +-1 px, iluminación, compresión de la cámara) cambia mucho
    más rápido que el movimiento real de cabeza o cejas, que está limitado
    por la inercia del cuello y los músculos faciales. Ponderar cada nueva
    lectura contra el valor suavizado anterior es, matemáticamente, un
    filtro IIR de un polo: atenúa las componentes de alta frecuencia (el
    temblor) y deja pasar las de baja frecuencia (el gesto real), al precio
    de un pequeño retardo de fase. `alpha` controla el compromiso: más bajo
    = más suavizado pero más retardo; `alpha=1.0` equivale a no suavizar.
    """

    def __init__(self, alpha):
        if not (0.0 < alpha <= 1.0):
            raise ValueError('smooth-alpha debe estar en (0, 1]')
        self.alpha = alpha
        self._state = {}

    def reset(self, key=None):
        """Olvida el estado (todo, o solo `key`). Llamar tras perder la cara
        para no arrastrar un valor suavizado obsoleto al recuperar seguimiento."""
        if key is None:
            self._state.clear()
        else:
            self._state.pop(key, None)

    def apply(self, key, value):
        if value is None or not math.isfinite(value):
            return value
        prev = self._state.get(key)
        if prev is None:
            self._state[key] = value
            return value
        smoothed = self.alpha * value + (1.0 - self.alpha) * prev
        self._state[key] = smoothed
        return smoothed


def ensure_model(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        log(f'[System] Descargando modelo: {path.name} ...')
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
        log(f'[System] Guardado en {path}')


def open_camera(index):
    backend_map = {'Linux': cv2.CAP_V4L2, 'Windows': cv2.CAP_DSHOW, 'Darwin': cv2.CAP_AVFOUNDATION}
    cap = cv2.VideoCapture(index, backend_map.get(platform.system(), cv2.CAP_ANY))
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_landmarks(image, landmarks, w, h):
    for i in POSE_LANDMARKS + LEFT_EYE_EAR + RIGHT_EYE_EAR + [LEFT_BROW, RIGHT_BROW, LEFT_EYE_TOP, RIGHT_EYE_TOP, 13, 14]:
        cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(image, (cx, cy), 2, (255, 255, 0), -1)


def _dot_color(controller):
    if controller.active is not None:
        return (0, 140, 255)  # naranja: gesto MOVIMIENTO activo/sostenido
    if controller.candidate == 'MODO':
        return (60, 60, 255)  # rojo: cejas levantadas, candidato a cambio de modo
    if controller.candidate == 'CENTRO':
        return (0, 255, 0) if controller.armed else (0, 255, 255)  # verde=armado, amarillo=sosteniendo
    if controller.candidate in ('IZQUIERDA', 'DERECHA', 'ATRAS', 'ADELANTE'):
        return (255, 200, 0) if controller.armed else (130, 130, 130)  # gris = bloqueado, falta CENTRO
    return (170, 170, 170)  # AMBIGUO / sin candidato


def _draw_gauge(image, x, y, w_px, label, value, threshold, max_value, unit=''):
    h_px = 14
    cv2.rectangle(image, (x, y), (x + w_px, y + h_px), (70, 70, 70), 1)
    if value is not None and max_value > 0:
        frac = max(0.0, min(1.0, value / max_value))
        fill_w = int(w_px * frac)
        over = value >= threshold
        color = (0, 0, 255) if over else (0, 200, 0)
        cv2.rectangle(image, (x, y), (x + fill_w, y + h_px), color, -1)
        if max_value > 0:
            tick_x = x + int(w_px * max(0.0, min(1.0, threshold / max_value)))
            cv2.line(image, (tick_x, y - 2), (tick_x, y + h_px + 2), (255, 255, 255), 1)
    cv2.putText(image, label, (x, y - 4), FONT, 0.4, (255, 255, 255), 1)


def _draw_hold_bar(image, controller, a, now, w, h):
    candidate = controller.candidate
    if candidate in (None, 'AMBIGUO'):
        return
    if candidate == 'CENTRO':
        required, label = a.center_hold, 'CENTRO (armando)'
        locked = False
    else:
        required = a.mouth_hold if candidate == 'BOCA' else a.mode_hold if candidate == 'MODO' else a.hold
        label = candidate
        locked = not controller.mouth_ready if candidate == 'BOCA' else not controller.armed
    elapsed = max(0.0, now - controller.since)
    frac = max(0.0, min(1.0, elapsed / required))
    bar_w, bar_h = 220, 16
    x0, y0 = w // 2 - bar_w // 2, h - 30
    cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (70, 70, 70), 1)
    color = (120, 120, 120) if locked else (0, 200, 255)
    cv2.rectangle(image, (x0, y0), (x0 + int(bar_w * frac), y0 + bar_h), color, -1)
    text = (f'{label}: cierra la boca' if candidate == 'BOCA' else f'{label}: vuelve al CENTRO') if locked else label
    cv2.putText(image, text, (x0, y0 - 6), FONT, 0.45, (255, 255, 255), 1)


def draw_hud(image, controller, a, disp_x, disp_y, disp_brow, blink_raw, now, w, h):
    """Overlay directo sobre el frame: marcador central, punto de gesto,
    zonas de umbral, medidores de cejas/parpadeo y barra de sostenimiento."""
    cx, cy = w // 2, h // 2
    scale = min(w, h) * 0.45 / (.22 * 2.5)  # escala visual fija
    r_release = int(a.release * scale)
    rx = int(a.threshold_x * scale)
    ry = int(a.threshold_y * scale)
    r_threshold = max(rx, ry)
    cv2.rectangle(image, (cx - r_release, cy - r_release), (cx + r_release, cy + r_release), (90, 90, 90), 1)
    cv2.line(image, (cx + rx, cy - ry), (cx + rx, cy + ry), (100, 100, 100), 1)
    cv2.line(image, (cx - rx, cy - ry), (cx - rx, cy + ry), (100, 100, 100), 1)
    cv2.line(image, (cx - rx, cy - ry), (cx + rx, cy - ry), (100, 100, 100), 1)
    cv2.line(image, (cx - rx, cy + ry), (cx + rx, cy + ry), (100, 100, 100), 1)
    cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    labels = {
        'DERECHA': (cx + r_threshold + 8, cy + 5),
        'IZQUIERDA': (cx - r_threshold - 95, cy + 5),
        'ATRAS': (cx - 25, cy + r_threshold + 20),
        'ADELANTE': (cx - 35, cy - r_threshold - 10),
    }
    for text, pos in labels.items():
        cv2.putText(image, text, pos, FONT, 0.45, (150, 150, 150), 1)
    if disp_x is None:
        cv2.putText(image, 'CALIBRANDO...', (cx - 70, cy - r_threshold - 30),
                    FONT, 0.6, (0, 255, 255), 2)
    else:
        px = max(0, min(w - 1, int(cx + disp_x * scale)))
        py = max(0, min(h - 1, int(cy + disp_y * scale)))
        color = _dot_color(controller)
        cv2.line(image, (cx, cy), (px, py), color, 1)
        cv2.circle(image, (px, py), 9, color, -1)
    _draw_gauge(image, 10, h - 55, 150, 'CEJAS', disp_brow, a.brow, a.brow * 2)
    _draw_gauge(image, 10, h - 25, 150, 'OJOS', blink_raw, a.blink, a.blink * 2)
    _draw_gauge(image, 10, h - 85, 150, 'BOCA', controller.mouth_raw, a.mouth_open, a.mouth_open * 2)
    _draw_hold_bar(image, controller, a, now, w, h)
    status = f'modo={controller.mode} activo={controller.active} candidato={controller.candidate}'
    cv2.putText(image, status, (10, 20), FONT, 0.5, (0, 255, 0), 1)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--models-dir', type=Path, default=Path('./models'))
    p.add_argument('--min-face-confidence', type=float, default=.7)
    p.add_argument('--min-tracking-confidence', type=float, default=.5)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--invert-x', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--invert-y', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--threshold', type=float, default=.22, help='umbral direccional, radianes')
    p.add_argument('--release', type=float, default=.12, help='zona central, radianes')
    p.add_argument('--brow', type=float, default=.9, help='umbral cejas, escala heurística propia')
    p.add_argument('--blink', type=float, default=1.8, help='umbral parpadeo, escala heurística propia')
    p.add_argument('--hold', type=float, default=.45)
    p.add_argument('--mode-hold', type=float, default=1.2)
    p.add_argument('--center-hold', type=float, default=.3)
    p.add_argument('--calibration', type=float, default=2.0)
    p.add_argument('--stale', type=float, default=.5, help='timeout de datos, segundos')
    p.add_argument('--startup-timeout', type=float, default=15)
    p.add_argument('--smooth-alpha', type=float, default=.35,
                    help='suavizado EMA de yaw/pitch/cejas/parpadeo, en (0,1]; 1.0 = sin suavizar')
    p.add_argument('--gpu', action='store_true',
                    help='usa el delegado GPU de MediaPipe (OpenGL ES vía EGL); si falla, cae a CPU automáticamente')
    add_config_arguments(p, 'mediapipe')
    return p


def _create_face_landmarker(model_path, a, delegate):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=a.min_face_confidence,
        min_tracking_confidence=a.min_tracking_confidence,
    ))


def main():
    p = parser()
    a = parse_settings(p, 'mediapipe')
    positives = ('threshold', 'release', 'brow', 'blink', 'hold', 'mode_hold',
                 'center_hold', 'calibration', 'stale', 'startup_timeout')
    if any(not math.isfinite(getattr(a, k)) or getattr(a, k) <= 0 for k in positives):
        p.error('Los umbrales y tiempos deben ser positivos y finitos.')
    if not (0.0 < a.smooth_alpha <= 1.0):
        p.error('--smooth-alpha debe estar en (0, 1].')

    model_path = a.models_dir.expanduser().resolve() / 'face_landmarker.task'
    ensure_model(model_path)

    BaseOptions = mp.tasks.BaseOptions
    delegate = BaseOptions.Delegate.GPU if a.gpu else BaseOptions.Delegate.CPU
    try:
        face_landmarker = _create_face_landmarker(model_path, a, delegate)
    except Exception as exc:
        if a.gpu:
            log(f'[System] Delegado GPU no disponible ({exc}); usando CPU.')
            face_landmarker = _create_face_landmarker(model_path, a, BaseOptions.Delegate.CPU)
        else:
            raise

    controller = GestureController(a)
    smoother = EMASmoother(a.smooth_alpha)
    cap = open_camera(a.device)
    if not cap.isOpened():
        log('[Critical Error] No se pudo abrir la cámara.')
        cap.release()
        face_landmarker.close()
        return 1

    controller.event('STOP', reason='inicio')
    log('Mira al centro con cejas relajadas durante la calibración. Ctrl+C o ESC para salir.')

    start_mono = time.monotonic()
    last_ts_ms = -1
    try:
        while True:
            success, frame = cap.read()
            if not success:
                controller.fault('lectura_camara')
                smoother.reset()
                time.sleep(.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int((time.monotonic() - start_mono) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            result = face_landmarker.detect_for_video(mp_image, ts_ms)
            now = time.monotonic()

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                pose = head_pose(landmarks, w, h)
                if pose is None:
                    controller.fault('rostro_no_confiable')
                    smoother.reset()
                    yaw = pitch = brow = blink = None
                else:
                    yaw_raw, pitch_raw = pose
                    yaw = smoother.apply('yaw', yaw_raw)
                    pitch = smoother.apply('pitch', pitch_raw)
                    brow = smoother.apply('brow', brow_metric(landmarks, w, h))
                    blink = smoother.apply('blink', blink_metric(landmarks, w, h))
                controller.feed(yaw, pitch, brow, blink, True, now, mouth=mouth_metric(landmarks, w, h))

                if a.preview:
                    draw_landmarks(frame, landmarks, w, h)
                    if controller.baseline is not None and yaw is not None:
                        bx, by, bbrow = controller.baseline
                        disp_x = (yaw - bx) * (-1 if a.invert_x else 1)
                        disp_y = (pitch - by) * (-1 if a.invert_y else 1)
                        disp_brow = brow - bbrow
                    else:
                        disp_x = disp_y = disp_brow = None
                    draw_hud(frame, controller, a, disp_x, disp_y, disp_brow, blink, now, w, h)
            else:
                controller.fault('rostro_no_detectado')
                smoother.reset()

            if a.preview:
                cv2.imshow('mediapipe_gestures', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        log('Cámara liberada.')
    finally:
        controller.stop('cierre', force=True)
        cap.release()
        face_landmarker.close()
        if a.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())