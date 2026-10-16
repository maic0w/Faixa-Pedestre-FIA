import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import jsonschema

SCHEMAS_DIR = Path(__file__).parent / "schemas"

CONTRACT_B_COLUMNS = [
    "video_id",
    "frame_id",
    "timestamp_s",
    "track_id",
    "class_name",
    "u",
    "v",
    "X_m",
    "Y_m",
    "calibration_valid",
    "detection_confidence",
]


def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema nao encontrado: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_contract_a(data: Dict[str, Any]) -> bool:
    """Valida se um frame com deteccoes esta em conformidade com o Contrato A."""
    schema = load_schema("contract_a_detection.json")
    jsonschema.validate(instance=data, schema=schema)
    return True


def validate_contract_b(df: pd.DataFrame) -> bool:
    """Valida se uma tabela de trajetorias esta em conformidade com o Contrato B."""
    missing = [col for col in CONTRACT_B_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Contrato B invalido. Colunas ausentes: {missing}")

    # Checagens basicas de tipo
    if not pd.api.types.is_integer_dtype(df["frame_id"]):
        raise TypeError("Coluna 'frame_id' deve ser do tipo inteiro.")
    if not pd.api.types.is_integer_dtype(df["track_id"]):
        raise TypeError("Coluna 'track_id' deve ser do tipo inteiro.")
    if not pd.api.types.is_numeric_dtype(df["timestamp_s"]):
        raise TypeError("Coluna 'timestamp_s' deve ser numerica.")
    if not pd.api.types.is_bool_dtype(df["calibration_valid"]):
        raise TypeError("Coluna 'calibration_valid' deve ser booleana.")

    return True


def validate_contract_c(data: Dict[str, Any]) -> bool:
    """Valida se o evento e sumario estao em conformidade com o Contrato C."""
    schema = load_schema("contract_c_event.json")
    jsonschema.validate(instance=data, schema=schema)
    return True
