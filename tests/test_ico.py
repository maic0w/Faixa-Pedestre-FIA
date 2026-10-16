import pytest
from src.decision.ico import (
    ConflictEvent,
    ICOCalculator,
    ICOWeights,
)


def test_ico_weights_validation():
    # Valido
    w = ICOWeights(w_pet=0.50, w_wait=0.25, w_yield=0.25)
    assert w.w_pet == 0.50

    # Invalido: soma != 1.0
    with pytest.raises(ValueError):
        ICOWeights(w_pet=0.50, w_wait=0.50, w_yield=0.50)

    # Invalido: negativo
    with pytest.raises(ValueError):
        ICOWeights(w_pet=1.2, w_wait=-0.1, w_yield=-0.1)


def test_ico_calculation_extremes():
    calc = ICOCalculator()

    # Cenário seguro ideal (tudo zero)
    score_zero = calc.calculate_from_rates(p_crit=0.0, e_norm=0.0, d_faixa=0.0)
    assert score_zero == 0.0
    assert calc.classify_risk(score_zero) == "Baixo"

    # Cenário crítico máximo (tudo 1.0)
    score_max = calc.calculate_from_rates(p_crit=1.0, e_norm=1.0, d_faixa=1.0)
    assert score_max == 100.0
    assert calc.classify_risk(score_max) == "Crítico"


def test_ico_session_aggregation():
    calc = ICOCalculator(t_max_wait_s=30.0, pet_critical_threshold_s=2.0)

    events = [
        # Evento 1: Conflito critico (PET = 1.2s < 2.0s), espera 15s, desrespeito
        ConflictEvent(
            event_id="E1",
            pedestrian_track_id=1,
            vehicle_track_id=2,
            roi_id="crosswalk",
            t_entry_pedestrian=10.0,
            t_exit_pedestrian=14.0,
            t_entry_vehicle=15.2,
            pedestrian_wait_time_s=15.0,
            vehicle_yielded=False,
            is_valid=True,
        ),
        # Evento 2: Seguro (PET = 4.0s >= 2.0s), espera 5s, motorista cedeu
        ConflictEvent(
            event_id="E2",
            pedestrian_track_id=3,
            vehicle_track_id=4,
            roi_id="crosswalk",
            t_entry_pedestrian=20.0,
            t_exit_pedestrian=25.0,
            t_entry_vehicle=29.0,
            pedestrian_wait_time_s=5.0,
            vehicle_yielded=True,
            is_valid=True,
        ),
    ]

    summary = calc.aggregate_session("video_test", events)

    assert summary["total_crossings"] == 2
    assert summary["eligible_events"] == 2
    assert summary["valid_pet_count"] == 2
    assert summary["critical_pet_count"] == 1
    # P_crit = 1 / 2 = 0.50
    assert summary["p_crit"] == 0.50
    # Espera media = (15 + 5)/2 = 10s -> E_norm = 10/30 = 0.3333
    assert summary["mean_wait_time_s"] == 10.0
    assert summary["e_norm"] == pytest.approx(0.3333, abs=1e-3)
    # D_faixa = 1 / 2 = 0.50
    assert summary["d_faixa"] == 0.50

    # ICO = 100 * (0.50*0.50 + 0.25*0.3333 + 0.25*0.50) = 100 * (0.25 + 0.0833 + 0.125) = 45.83
    assert summary["ico_score"] == pytest.approx(45.83, abs=0.1)
    assert summary["risk_level"] == "Moderado"


def test_ico_sensitivity_analysis():
    calc = ICOCalculator()
    results = calc.sensitivity_analysis(p_crit=0.6, e_norm=0.4, d_faixa=0.5)
    assert len(results) >= 5
    variations = [r["variation"] for r in results]
    assert "0% (Base)" in variations
