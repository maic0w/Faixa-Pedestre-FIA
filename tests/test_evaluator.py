import pytest
from src.evaluation.evaluator import (
    BoundingBox,
    DetectionEvaluator,
    calculate_iou,
    yolo_to_xyxy,
)


def test_calculate_iou_identical():
    b1 = BoundingBox(10.0, 10.0, 30.0, 30.0, class_id=0)
    b2 = BoundingBox(10.0, 10.0, 30.0, 30.0, class_id=0)
    assert calculate_iou(b1, b2) == pytest.approx(1.0)


def test_calculate_iou_disjoint():
    b1 = BoundingBox(0.0, 0.0, 10.0, 10.0, class_id=0)
    b2 = BoundingBox(20.0, 20.0, 30.0, 30.0, class_id=0)
    assert calculate_iou(b1, b2) == 0.0


def test_calculate_iou_partial():
    # Box 1: [0, 0, 10, 10] -> area 100
    # Box 2: [5, 0, 15, 10] -> area 100
    # Intersecao: [5, 0, 10, 10] -> area 50
    # Uniao: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    b1 = BoundingBox(0.0, 0.0, 10.0, 10.0, class_id=0)
    b2 = BoundingBox(5.0, 0.0, 15.0, 10.0, class_id=0)
    assert calculate_iou(b1, b2) == pytest.approx(1.0 / 3.0)


def test_evaluator_frame_matching():
    evaluator = DetectionEvaluator(iou_threshold=0.50)

    gt = [
        BoundingBox(10.0, 10.0, 50.0, 50.0, class_id=0, class_name="person"),
        BoundingBox(100.0, 100.0, 200.0, 200.0, class_id=2, class_name="car"),
    ]
    preds = [
        # Match com person (IoU alto) -> TP
        BoundingBox(12.0, 11.0, 49.0, 51.0, class_id=0, confidence=0.92, class_name="person"),
        # Falsa predicao de moto -> FP
        BoundingBox(300.0, 300.0, 350.0, 350.0, class_id=3, confidence=0.85, class_name="motorcycle"),
    ]

    matches = evaluator.evaluate_frame(gt, preds)

    types = [m[2] for m in matches]
    assert "TP" in types
    assert "FP" in types
    assert "FN" in types  # O carro foi omitido -> FN

    report = evaluator.evaluate_dataset([matches])
    assert "person" in report
    assert report["person"]["TP"] == 1
    assert report["person"]["FP"] == 0
    assert report["person"]["FN"] == 0
    assert report["person"]["precision"] == 1.0
    assert report["person"]["recall"] == 1.0
    assert report["person"]["f1_score"] == 1.0

    assert "car" in report
    assert report["car"]["FN"] == 1
    assert report["car"]["recall"] == 0.0

    assert "motorcycle" in report
    assert report["motorcycle"]["FP"] == 1
    assert report["motorcycle"]["precision"] == 0.0


def test_yolo_to_xyxy():
    # Linha YOLO normalizada: class_id x_center y_center width height
    line = "0 0.5 0.5 0.2 0.4"
    box = yolo_to_xyxy(line, img_width=1000, img_height=500)
    assert box is not None
    assert box.class_id == 0
    # xc = 500, yc = 250, w = 200, h = 200
    # x1 = 400, y1 = 150, x2 = 600, y2 = 350
    assert box.x1 == 400.0
    assert box.y1 == 150.0
    assert box.x2 == 600.0
    assert box.y2 == 350.0
