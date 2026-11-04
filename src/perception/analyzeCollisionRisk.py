"""Velocidades e conflitos previstos exclusivamente entre veiculo e pedestre.

Camera fixa; sem calibracao, as medidas sao apenas projecoes em pixels.
Nao calcula probabilidade estatistica de acidente.
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path


class MotionHistory:
    """Regressao em uma janela temporal, sem reutilizar movimento apos lacunas."""

    def __init__(self, window=0.8, minimum=0.4, max_gap=0.25):
        self.samples = {}
        self.window, self.minimum, self.max_gap = window, minimum, max_gap

    def update(self, key, time, point):
        samples = self.samples.setdefault(key, deque())
        if samples and time - samples[-1][0] > self.max_gap:
            samples.clear()
        samples.append((time, point))
        while samples and time - samples[0][0] > self.window:
            samples.popleft()
        if len(samples) < 3 or time - samples[0][0] < self.minimum:
            return None
        mean_t = sum(t for t, _ in samples) / len(samples)
        denominator = sum((t - mean_t) ** 2 for t, _ in samples)
        return tuple(sum((t - mean_t) * p[axis] for t, p in samples) / denominator
                     for axis in (0, 1))

    def expire(self, time):
        for key in list(self.samples):
            if time - self.samples[key][-1][0] > self.max_gap:
                del self.samples[key]


def predict_conflict(pedestrian, vehicle, radius, horizon):
    """Primeira entrada na distancia de seguranca, sob velocidade constante.

    Retorna (tempo de entrada, distancia minima prevista) ou None.
    Cada estado e (posicao, velocidade); velocidades usam unidades/segundo.
    """
    p, pv = pedestrian
    v, vv = vehicle
    if pv is None or vv is None:
        return None
    relative = tuple(v[i] - p[i] for i in (0, 1))
    velocity = tuple(vv[i] - pv[i] for i in (0, 1))
    distance2 = sum(x*x for x in relative)
    speed2 = sum(x*x for x in velocity)
    dot = sum(relative[i] * velocity[i] for i in (0, 1))
    closest_t = max(0.0, min(horizon, -dot / speed2)) if speed2 > 1e-12 else 0.0
    closest = math.hypot(*(relative[i] + velocity[i] * closest_t for i in (0, 1)))
    if distance2 <= radius * radius:
        return 0.0, closest
    if speed2 <= 1e-12 or dot >= 0:
        return None
    discriminant = dot*dot - speed2 * (distance2 - radius*radius)
    if discriminant < 0:
        return None
    entry = (-dot - math.sqrt(discriminant)) / speed2
    return (entry, closest) if 0 <= entry <= horizon else None


def cross_conflicts(objects, radius, horizon):
    # Categorias fazem parte da chave: IDs iguais de modelos diferentes nao colidem.
    pedestrians = [obj for obj in objects if obj['kind'] == 'pedestre']
    vehicles = [obj for obj in objects if obj['kind'] == 'veiculo']
    for pedestrian in pedestrians:
        for vehicle in vehicles:
            prediction = predict_conflict(
                (pedestrian['point'], pedestrian['velocity']),
                (vehicle['point'], vehicle['velocity']), radius, horizon)
            if prediction is not None:
                yield pedestrian, vehicle, prediction


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Use um numero positivo e finito.')
    return value


def main():
    import cv2
    import numpy as np
    # Funciona como script e como modulo (python -m src.perception.analyzeCollisionRisk).
    if __package__:
        from . import findPedestrian as pedestrians, findVehicle as vehicles
    else:
        import findPedestrian as pedestrians
        import findVehicle as vehicles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=pedestrians.PROJECT_ROOT / 'data/raw/TwoWayTraffic720p.mp4')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--homography', type=Path, help='JSON com matriz 3x3 de pixels para metros no chao.')
    calibration.add_argument('--meters-per-pixel', type=positive, help='Escala uniforme; apenas aproximacao sem perspectiva.')
    parser.add_argument('--fps', type=positive, help='Sobrescreve FPS do video quando os metadados estiverem incorretos.')
    parser.add_argument('--horizon', type=positive, default=3.0, help='Horizonte de previsao em segundos.')
    parser.add_argument('--safety-distance', type=positive, help='Raio combinado de seguranca: metros se calibrado, pixels caso contrario.')
    parser.add_argument('--conf', type=pedestrians.probability, default=0.10)
    parser.add_argument('--imgsz', type=pedestrians.image_size, default=640)
    parser.add_argument('--pedestrian-model', default=pedestrians.MODEL_WEIGHTS)
    parser.add_argument('--vehicle-model', default=vehicles.MODEL_WEIGHTS)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', type=Path, help='Video anotado .mp4; opcional.')
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f'Video nao encontrado: {args.source}')
    if args.output and args.output.resolve() == args.source.resolve():
        parser.error('O video de saida deve ser diferente do video de entrada.')
    matrix = None
    if args.homography:
        try:
            matrix = np.asarray(json.loads(args.homography.read_text(encoding='utf-8')), dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
                raise ValueError('matriz deve ser 3x3, finita e invertivel')
        except (OSError, ValueError) as error:
            parser.error(f'Calibracao invalida: {error}')
    metric = matrix is not None or args.meters_per_pixel is not None
    radius = args.safety_distance if args.safety_distance is not None else (2.0 if metric else 50.0)
    unit = 'm' if metric else 'px'
    print(f'Previsao geometrica, sem probabilidade estatistica. Distancia: {radius} {unit}; horizonte: {args.horizon} s.')
    if not metric:
        print('SEM CALIBRACAO: velocidades em px/s e alertas somente no plano da imagem.')
    capture = cv2.VideoCapture(str(args.source.resolve()))
    writer = None
    try:
        if not capture.isOpened():
            parser.error('Nao foi possivel abrir o video.')
        fps = args.fps or capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            parser.error('FPS invalido; informe --fps com o valor correto.')
        models = [('pedestre', pedestrians, pedestrians.YOLO(args.pedestrian_model)),
                  ('veiculo', vehicles, vehicles.YOLO(args.vehicle_model))]
        history = MotionHistory()
        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            # Tempo do video, nunca o tempo de inferencia dos dois modelos.
            time = frame_number / fps
            history.expire(time)
            objects = []
            for kind, detector, model in models:
                result = detector.track_frame(model, frame, conf=args.conf, imgsz=args.imgsz)[0]
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue
                for box, identifier in zip(boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist()):
                    x1, y1, x2, y2 = box
                    anchor = ((x1 + x2) / 2, y2)  # Aproximacao do contato com o chao.
                    point = anchor
                    if matrix is not None:
                        projected = matrix @ np.array([*anchor, 1.0])
                        if abs(projected[2]) < 1e-9:
                            continue
                        point = tuple(projected[:2] / projected[2])
                    elif metric:
                        point = tuple(x * args.meters_per_pixel for x in anchor)
                    if not all(math.isfinite(x) for x in point):
                        continue
                    velocity = history.update((kind, identifier), time, point)
                    objects.append(dict(kind=kind, id=identifier, point=point, velocity=velocity,
                                        anchor=tuple(map(int, anchor)), box=tuple(map(int, box))))
            conflicts = list(cross_conflicts(objects, radius, args.horizon))
            danger_ids = {(obj['kind'], obj['id']) for p, v, _ in conflicts for obj in (p, v)}
            for obj in objects:
                color = (0, 0, 255) if (obj['kind'], obj['id']) in danger_ids else (0, 200, 0)
                x1, y1, x2, y2 = obj['box']
                speed = 'calibrando'
                if obj['velocity'] is not None:
                    magnitude = math.hypot(*obj['velocity'])
                    speed = f'{magnitude * 3.6:.1f} km/h' if metric else f'{magnitude:.1f} px/s'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{obj['kind']} #{obj['id']} {speed}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for p, v, (entry, closest) in conflicts:
                cv2.line(frame, p['anchor'], v['anchor'], (0, 0, 255), 2)
                label = f"PERIGO P{p['id']}/V{v['id']}: {entry:.1f}s, dmin={closest:.1f}{unit}"
                cv2.putText(frame, label, p['anchor'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f'{time:.2f}s {label}')
            status = 'METRICO: estimativa' if metric else 'SEM CALIBRACAO: risco apenas na imagem'
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Nao foi possivel gravar: {args.output}')
                writer.write(frame)
            if not args.no_show:
                cv2.imshow('Velocidades e risco veiculo-pedestre', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
