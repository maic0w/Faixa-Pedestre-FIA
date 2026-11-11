"""Motor Cinemático e Análise Espaço-Temporal de Conflitos em Faixas de Pedestres.

Responsabilidades Metodológicas:
1. Ponto de contato com o solo: utiliza estritamente o centro inferior da caixa:
   u = x_centro = x1 + (x2 - x1) / 2,  v = y2
2. Homografia planar e calibração:
   Se calibration_valid for False, emite aviso e calcula cinemática estritamente em pixels/s.
   Apenas quando calibrado projeta (u, v) -> (X_m, Y_m) e computa velocidades e distâncias em metros.
3. Estruturação em Dois Níveis do Contrato C:
   - Nível 1: Eventos individuais de conflito (event_id, timestamps de entrada/saída de ROIs, PET, cessão).
   - Nível 2: Sumário agregado da sessão (total_crossings, p_crit, e_norm, d_faixa, ico_score, risk_level).
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    Polygon = None
    Point = None

from src.decision.ico import ConflictEvent, ICOCalculator, ICOWeights


def compute_ground_contact_point(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Calcula o ponto de contato com o solo (base inferior central da bounding box).

    u = x1 + (x2 - x1) / 2
    v = y2
    """
    u = (x1 + x2) / 2.0
    v = float(y2)
    return (u, v)


def project_point_homography(
    point_uv: Tuple[float, float],
    matrix_3x3: np.ndarray,
) -> Tuple[float, float]:
    """Aplica a matriz de homografia planar H para transformar (u, v) no solo em (X_m, Y_m) da via."""
    u, v = point_uv
    vec = np.array([u, v, 1.0], dtype=float)
    projected = matrix_3x3 @ vec
    if abs(projected[2]) < 1e-9:
        raise ValueError("Coordenada homogênea projetada com denominador nulo.")
    x_m = projected[0] / projected[2]
    y_m = projected[1] / projected[2]
    return (float(x_m), float(y_m))


@dataclass
class AgentTrajectoryPoint:
    """Ponto de trajetória de um agente em um determinado instante temporal."""
    frame_id: int
    timestamp_s: float
    track_id: int
    class_name: str
    u: float
    v: float
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    calibration_valid: bool = False
    confidence: float = 1.0


class KinematicEngine:
    """Motor cinemático responsável pela detecção de eventos e geração do Contrato C em dois níveis."""

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        meters_per_pixel: Optional[float] = None,
        roi_crosswalk_polygon: Optional[List[Tuple[float, float]]] = None,
        roi_sidewalk_polygon: Optional[List[Tuple[float, float]]] = None,
        pet_critical_threshold_s: float = 2.0,
        t_max_wait_s: float = 30.0,
    ):
        self.matrix = homography_matrix
        self.meters_per_pixel = meters_per_pixel
        self.is_calibrated = (self.matrix is not None) or (self.meters_per_pixel is not None)
        self.pet_critical_threshold = pet_critical_threshold_s
        self.t_max_wait_s = t_max_wait_s

        # Polígonos de Regiões de Interesse (ROIs) em coordenadas (u, v)
        self.roi_crosswalk = Polygon(roi_crosswalk_polygon) if roi_crosswalk_polygon and Polygon else None
        self.roi_sidewalk = Polygon(roi_sidewalk_polygon) if roi_sidewalk_polygon and Polygon else None

    def process_ground_point(self, u: float, v: float) -> Tuple[Optional[float], Optional[float], bool]:
        """Calcula coordenadas métricas (X_m, Y_m) a partir de (u, v) no solo se houver calibração válida."""
        if not self.is_calibrated:
            return None, None, False

        try:
            if self.matrix is not None:
                xm, ym = project_point_homography((u, v), self.matrix)
                return xm, ym, True
            elif self.meters_per_pixel is not None:
                return u * self.meters_per_pixel, v * self.meters_per_pixel, True
        except Exception:
            return None, None, False

        return None, None, False

    def calculate_velocity(
        self,
        p1: Tuple[float, float],
        t1: float,
        p2: Tuple[float, float],
        t2: float,
        is_metric: bool,
    ) -> Tuple[float, str]:
        """Calcula velocidade escalar média entre dois instantes.

        Garante que sem calibração a unidade seja estritamente 'px/s'.
        """
        dt = t2 - t1
        if dt <= 1e-6:
            return 0.0, "px/s" if not is_metric else "m/s"

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        speed = dist / dt
        unit = "m/s" if is_metric else "px/s"
        return round(speed, 3), unit

    def extract_roi_occupancy(
        self,
        trajectory: List[AgentTrajectoryPoint],
        roi_polygon: Optional[Polygon],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extrai instantes t_entry e t_exit da ocupação do polígono da ROI pelo ponto no solo (u, v)."""
        if roi_polygon is None or not trajectory:
            return None, None

        inside_times = []
        for pt in trajectory:
            point = Point(pt.u, pt.v)
            if roi_polygon.contains(point) or roi_polygon.touches(point):
                inside_times.append(pt.timestamp_s)

        if not inside_times:
            return None, None

        return min(inside_times), max(inside_times)

    def analyze_interactions_and_generate_contract_c(
        self,
        video_id: str,
        conflict_events: List[ConflictEvent],
        ico_calculator: Optional[ICOCalculator] = None,
    ) -> Dict[str, any]:
        """Gera a estrutura formal em Dois Níveis conforme o Contrato C (JSON Schema).

        Nível 1: events (lista de eventos com t_entry, t_exit, pet_s, is_pet_critical, cedência)
        Nível 2: summary (indicadores agregados da sessão + score e nível de risco do ICO)
        """
        calc = ico_calculator or ICOCalculator(
            t_max_wait_s=self.t_max_wait_s,
            pet_critical_threshold_s=self.pet_critical_threshold,
        )

        # 1. Estruturação do Nível 1 (Eventos Individuais Auditáveis)
        level_1_events = []
        for ev in conflict_events:
            level_1_events.append({
                "event_id": ev.event_id,
                "pedestrian_track_id": ev.pedestrian_track_id,
                "vehicle_track_id": ev.vehicle_track_id,
                "roi_id": ev.roi_id,
                "t_entry_pedestrian": ev.t_entry_pedestrian,
                "t_exit_pedestrian": ev.t_exit_pedestrian,
                "t_entry_vehicle": ev.t_entry_vehicle,
                "pet_s": ev.pet_s,
                "is_pet_critical": ev.is_pet_critical,
                "pedestrian_wait_time_s": ev.pedestrian_wait_time_s,
                "vehicle_yielded": ev.vehicle_yielded,
                "is_valid": ev.is_valid,
                "notes": "Calculado a partir do ponto de contato com o solo na faixa de travessia."
            })

        # 2. Estruturação do Nível 2 (Sumário Agregado da Sessão e Escore ICO)
        level_2_summary = calc.aggregate_session(video_id=video_id, events=conflict_events)
        # Remove video_id do summary interno para conformidade com schema
        if "video_id" in level_2_summary:
            del level_2_summary["video_id"]

        contract_c_output = {
            "video_id": video_id,
            "events": level_1_events,
            "summary": level_2_summary,
        }

        return contract_c_output
