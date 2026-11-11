import json
from pathlib import Path
import pandas as pd
import pytest
from contracts.validator import (
    validate_contract_a,
    validate_contract_b,
    validate_contract_c,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_contract_a_mock_valid():
    with open(FIXTURES_DIR / "mock_contract_a.json", "r", encoding="utf-8") as f:
        frames = json.load(f)
    assert len(frames) > 0
    for frame in frames:
        assert validate_contract_a(frame) is True


def test_contract_a_invalid_missing_fields():
    invalid_data = {
        "video_id": "test_01",
        "frame_id": 1,
        # missing timestamp_s e detections
    }
    with pytest.raises(Exception):
        validate_contract_a(invalid_data)


def test_contract_b_mock_valid():
    df = pd.read_csv(FIXTURES_DIR / "mock_contract_b.csv")
    assert validate_contract_b(df) is True


def test_contract_b_invalid_missing_column():
    df = pd.read_csv(FIXTURES_DIR / "mock_contract_b.csv")
    df_invalid = df.drop(columns=["X_m"])
    with pytest.raises(ValueError):
        validate_contract_b(df_invalid)


def test_contract_c_mock_valid():
    with open(FIXTURES_DIR / "mock_contract_c.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert validate_contract_c(data) is True


def test_contract_c_invalid_score_range():
    with open(FIXTURES_DIR / "mock_contract_c.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    data["summary"]["ico_score"] = 150.0  # maior que 100
    with pytest.raises(Exception):
        validate_contract_c(data)
