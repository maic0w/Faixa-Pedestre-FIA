import json
from pathlib import Path
import numpy as np
import pytest
from src.kinematics.engine import (
    AgentTrajectoryPoint,
    KinematicEngine,
    compute_ground_contact_point,
    project_point_homography,
)
from src.decision.ico import ConflictEvent
from contracts.validator import validate_contract_c


def test_ground_contact_point_bottom_center():
    # Box [x1=100, y1=50, x2=200, y2=250]
    # Centro inferior: u = 100 + (200-100)/2 = 150, v = 250
    u, v = compute_ground_contact_point(100, 50, 200, 250)
    assert u == 150.0
    assert v == 250.0


def test_kinematic_velocity_without_calibration_is_pixels():
    engine = KinematicEngine()  # Sem calibracao
    p1 = (100.0, 100.0)
    p2 = (130.0, 140.0)  # dist = 50px
    speed, unit = engine.calculate_velocity(p1, 0.0, p2, 1.0, is_metric=False)
    assert speed == 50.0
    assert unit == "px/s"


def test_kinematic_velocity_with_calibration_is_meters():
    engine = KinematicEngine(meters_per_pixel=0.1)  # 0.1 m/px
    p1 = (10.0, 10.0)
    p2 = (13.0, 14.0)  # dist = 5.0m
    speed, unit = engine.calculate_velocity(p1, 0.0, p2, 1.0, is_metric=True)
    assert speed == 5.0
    assert unit == "m/s"


def test_contract_c_two_level_structure():
    engine = KinematicEngine(pet_critical_threshold_s=2.0)
    events = [
        ConflictEvent(
            event_id="EV_01",
            pedestrian_track_id=1,
            vehicle_track_id=2,
            roi_id="crosswalk_01",
            t_entry_pedestrian=4.0,
            t_exit_pedestrian=7.0,
            t_entry_vehicle=8.2,  # PET = 1.2s (< 2.0s -> critico)
            pedestrian_wait_time_s=6.0,
            vehicle_yielded=False,
            is_valid=True,
        )
    ]

    output = engine.analyze_interactions_and_generate_contract_c("session_test", events)

    # Validacao dos dois niveis
    assert "events" in output
    assert "summary" in output
    assert len(output["events"]) == 1
    assert output["events"][0]["event_id"] == "EV_01"
    assert output["events"][0]["is_pet_critical"] is True
    assert output["summary"]["critical_pet_count"] == 1
    assert output["summary"]["ico_score"] > 0.0

    # Validacao formal com o validador de schema do Contrato C
    assert validate_contract_c(output) is True
