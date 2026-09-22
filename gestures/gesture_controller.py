"""Controlador de gestos independiente de cámara y hardware."""
import json
import math
import statistics
import sys
import time


def log(message):
    print(message, file=sys.stderr, flush=True)


class GestureController:
    def __init__(self, args, emit=None):
        self.a = args
        self.emit = emit or self.print_event
        self.mode = 'INTERACCION'
        self.active = None
        self.armed = False
        self.candidate = None
        self.since = 0.0
        self.last = None
        self.last_heartbeat = 0.0
        self.last_fault = None
        self.samples = []
        self.baseline = None
        self.debug_at = 0.0
        self.threshold_x = getattr(args, 'threshold_x', args.threshold)
        self.threshold_y = getattr(args, 'threshold_y', args.threshold)
        self.diagonal_ratio = getattr(args, 'diagonal_ratio', 1.0)
        self.sustain_ratio = getattr(args, 'sustain_ratio', .65)
        # Primero debe observarse un cierre; perder seguimiento no rearma boca.
        self.mouth_ready = False
        self.mouth_since = None
        self.mouth_closed_since = None
        self.mouth_raw = None

    @staticmethod
    def print_event(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def event(self, kind, **fields):
        self.emit(dict(type=kind, mode=self.mode, monotonic=time.monotonic(), **fields))

    def stop(self, reason, force=False):
        if self.active is not None or force:
            self.event('STOP', reason=reason)
        self.active = None

    def fault(self, reason):
        self.stop(reason, force=self.last_fault != reason)
        self.armed = False
        self.candidate = None
        self.mouth_ready = False
        self.mouth_since = self.mouth_closed_since = None
        if self.baseline is None:
            self.samples.clear()
        self.last_fault = reason

    def tick(self, now):
        if self.last is not None and now - self.last > self.a.stale:
            self.fault('sin_datos_recientes')

    def mouth_gesture(self, mouth, now):
        """Cierre confirmado rearma; apertura sostenida emite una sola acción."""
        if mouth is None:
            return False
        if mouth <= self.a.mouth_close:
            self.mouth_since = None
            if self.mouth_closed_since is None:
                self.mouth_closed_since = now
            if now - self.mouth_closed_since >= self.a.mouth_close_hold:
                self.mouth_ready = True
            return False
        self.mouth_closed_since = None
        if mouth < self.a.mouth_open:
            self.mouth_since = None
            return False
        self.stop('boca_abierta')
        self.armed = False
        self.candidate = 'BOCA'
        if self.mouth_since is None:
            self.mouth_since = self.since = now
        if self.mouth_ready and now - self.mouth_since >= self.a.mouth_hold:
            self.mouth_ready = False
            self.event('MOUTH_OPEN')
        return True

    def direction(self, x, y):
        # Mantener una dirección usa el límite de retorno y un cono más ancho.
        if self.active in ('IZQUIERDA', 'DERECHA'):
            signed = x if self.active == 'DERECHA' else -x
            if signed > self.a.release and signed > self.sustain_ratio * abs(y):
                return self.active
        elif self.active in ('ADELANTE', 'ATRAS'):
            signed = y if self.active == 'ATRAS' else -y
            if signed > self.a.release and signed > self.sustain_ratio * abs(x):
                return self.active
        if abs(x) < self.a.release and abs(y) < self.a.release:
            return 'CENTRO'
        if abs(x) >= self.threshold_x and abs(x) > self.diagonal_ratio * abs(y):
            return 'DERECHA' if x > 0 else 'IZQUIERDA'
        if abs(y) >= self.threshold_y and abs(y) > self.diagonal_ratio * abs(x):
            return 'ATRAS' if y > 0 else 'ADELANTE'
        return 'AMBIGUO'

    def feed(self, x, y, brow, blink, valid, now, mouth=None):
        self.tick(now)
        self.last = now
        if not valid or not all(math.isfinite(v) for v in (x, y, brow, blink)):
            self.fault('rostro_no_confiable')
            return
        if mouth is not None and not math.isfinite(mouth):
            self.fault('boca_no_confiable')
            return
        self.mouth_raw = mouth
        self.last_fault = None
        if blink > self.a.blink:
            self.fault('ojos_cerrados')
            return
        if self.baseline is None:
            self.samples.append((now, x, y, brow))
            if now - self.samples[0][0] < self.a.calibration or len(self.samples) < 15:
                return
            # Reject unstable calibration instead of learning a moving face.
            xs, ys = [s[1] for s in self.samples], [s[2] for s in self.samples]
            if max(statistics.pstdev(xs), statistics.pstdev(ys)) > min(self.threshold_x, self.threshold_y) / 3:
                self.samples.clear()
                log('Calibración inestable: mira al centro y relaja la cara.')
                return
            self.baseline = (statistics.median(xs), statistics.median(ys),
                             statistics.median(s[3] for s in self.samples))
            self.samples.clear()
            self.event('CALIBRATED', baseline=self.baseline)
            log('Calibrado. Mantén el centro para habilitar los gestos.')
            return
        x = (x - self.baseline[0]) * (-1 if self.a.invert_x else 1)
        y = (y - self.baseline[1]) * (-1 if self.a.invert_y else 1)
        brow -= self.baseline[2]
        if self.a.debug and now - self.debug_at >= .25:
            log(f'modo={self.mode} x={x:+.3f} y={y:+.3f} cejas={brow:.2f} boca={mouth} armado={self.armed}')
            self.debug_at = now
        if self.mouth_gesture(mouth, now):
            return
        if brow >= self.a.brow:
            gesture = 'MODO'
        else:
            gesture = self.direction(x, y)
        if self.active is not None and gesture != self.active:
            self.stop('gesto_liberado_o_cambiado')
        if gesture != self.candidate:
            self.candidate, self.since = gesture, now
        elapsed = now - self.since
        if gesture == 'CENTRO':
            if elapsed >= self.a.center_hold:
                self.armed = True
            return
        if gesture == 'AMBIGUO':
            return
        if self.active == gesture:
            if now - self.last_heartbeat >= .1:
                self.event('HEARTBEAT', direction=self.active)
                self.last_heartbeat = now
            return
        required = self.a.mode_hold if gesture == 'MODO' else self.a.hold
        if not self.armed or elapsed < required:
            return
        self.armed = False
        if gesture == 'MODO':
            self.stop('cambio_de_modo', force=True)
            self.mode = 'MOVIMIENTO' if self.mode == 'INTERACCION' else 'INTERACCION'
            self.event('MODE', gesture=gesture)
        elif self.mode == 'MOVIMIENTO':
            self.active = gesture
            self.last_heartbeat = now
            self.event('MOVE', direction=gesture)
        else:
            self.event('INTERACT', direction=gesture)
