"""Modulo de Avaliacao Quantitativa de Deteccao de Objetos (IoU, Precisao, Revocacao e F1-Score).

Suporta:
- Correspondencia um-para-um gulosa com limiar de IoU configuravel (padrao: >= 0.50).
- Desacoplamento entre limiar de rastreamento (tracking) e limiar de deteccao pura (conf_threshold).
- Varredura de confianca (confidence sweep) entre 0.10 e 0.70 para analise de sensibilidade de falsos positivos.
- Leitura de anotacoes Ground Truth e Predicoes no formato canônico YOLO (.txt normalizado) e Contrato A.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class BoundingBox:
    """Caixa delimitadora padronizada em coordenadas [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0
    class_name: Optional[str] = None


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calcula a Intersection over Union (IoU) entre duas caixas."""
    xi1 = max(box1.x1, box2.x1)
    yi1 = max(box1.y1, box2.y1)
    xi2 = min(box1.x2, box2.x2)
    yi2 = min(box1.y2, box2.y2)

    inter_width = max(0.0, xi2 - xi1)
    inter_height = max(0.0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = max(0.0, box1.x2 - box1.x1) * max(0.0, box1.y2 - box1.y1)
    area2 = max(0.0, box2.x2 - box2.x1) * max(0.0, box2.y2 - box2.y1)

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def yolo_to_xyxy(
    line: str,
    img_width: float = 1.0,
    img_height: float = 1.0,
    class_mapping: Optional[Dict[int, str]] = None,
) -> Optional[BoundingBox]:
    """Converte linha YOLO normalizada 'class_id x_center y_center w h [conf]' para BoundingBox."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(parts[0])
    xc = float(parts[1]) * img_width
    yc = float(parts[2]) * img_height
    w = float(parts[3]) * img_width
    h = float(parts[4]) * img_height
    conf = float(parts[5]) if len(parts) >= 6 else 1.0

    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0

    class_name = class_mapping.get(class_id) if class_mapping else str(class_id)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id, confidence=conf, class_name=class_name)


class DetectionEvaluator:
    """Avaliador quantitativo para correspondencia um-para-um e metricas por classe.

    Desacopla explicitamente o limiar de deteccao pura (conf_threshold, padrao 0.25)
    do limiar permissivo usado pelo rastreador (ex: 0.10), evitando a inundacao artificial
    de falsos positivos na avaliacao do detector.
    """

    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        iou_threshold: float = 0.50,
        conf_threshold: float = 0.25,
        class_mapping: Optional[Dict[int, str]] = None,
    ):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.class_mapping = class_mapping or self.COCO_CLASSES

    def filter_predictions_by_confidence(
        self,
        predictions: List[BoundingBox],
        min_conf: Optional[float] = None,
    ) -> List[BoundingBox]:
        """Filtra predicoes aplicando o limiar de confianca de avaliacao."""
        threshold = self.conf_threshold if min_conf is None else min_conf
        return [b for b in predictions if b.confidence >= threshold]

    def evaluate_frame(
        self,
        ground_truth: List[BoundingBox],
        predictions: List[BoundingBox],
        conf_threshold: Optional[float] = None,
    ) -> List[Tuple[Optional[BoundingBox], Optional[BoundingBox], str]]:
        """Realiza correspondencia um-para-um para um unico quadro.

        Filtra predições abaixo de conf_threshold e ordena de forma gulosa por confiança.
        Retorna lista de tuplas (gt_box, pred_box, match_type) com 'TP', 'FP' ou 'FN'.
        """
        filtered_preds = self.filter_predictions_by_confidence(predictions, conf_threshold)
        sorted_preds = sorted(filtered_preds, key=lambda b: b.confidence, reverse=True)
        matched_gt_indices = set()
        matches = []

        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = None

            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt_indices:
                    continue
                if gt.class_id != pred.class_id:
                    continue

                iou = calculate_iou(pred, gt)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_gt_idx is not None:
                matched_gt_indices.add(best_gt_idx)
                matches.append((ground_truth[best_gt_idx], pred, "TP"))
            else:
                matches.append((None, pred, "FP"))

        # Ground truths nao associados contam como False Negatives (FN)
        for idx, gt in enumerate(ground_truth):
            if idx not in matched_gt_indices:
                matches.append((gt, None, "FN"))

        return matches

    def evaluate_dataset(
        self,
        dataset_matches: List[List[Tuple[Optional[BoundingBox], Optional[BoundingBox], str]]],
    ) -> Dict[str, Dict[str, float]]:
        """Calcula Precisao, Revocacao e F1-Score detalhados por classe a partir dos matches."""
        stats: Dict[int, Dict[str, int]] = {}

        for frame_matches in dataset_matches:
            for gt, pred, match_type in frame_matches:
                cid = gt.class_id if gt is not None else pred.class_id
                if cid not in stats:
                    stats[cid] = {"TP": 0, "FP": 0, "FN": 0}

                stats[cid][match_type] += 1

        results = {}
        total_tp, total_fp, total_fn = 0, 0, 0

        for cid, counts in stats.items():
            tp = counts["TP"]
            fp = counts["FP"]
            fn = counts["FN"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            class_name = self.class_mapping.get(cid, f"class_{cid}")
            results[class_name] = {
                "class_id": cid,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }

        # Micro-Average (global agregado)
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        results["MICRO_AVG"] = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1_score": round(micro_f1, 4),
        }

        # Macro-Average (media aritmetica simples das classes)
        classes_only = [v for k, v in results.items() if k != "MICRO_AVG"]
        if classes_only:
            macro_prec = float(np.mean([c["precision"] for c in classes_only]))
            macro_rec = float(np.mean([c["recall"] for c in classes_only]))
            macro_f1 = float(np.mean([c["f1_score"] for c in classes_only]))
        else:
            macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0

        results["MACRO_AVG"] = {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(macro_f1, 4),
        }

        return results

    def confidence_sweep(
        self,
        dataset_gt: List[List[BoundingBox]],
        dataset_preds: List[List[BoundingBox]],
        thresholds: Sequence[float] = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70),
    ) -> List[Dict[str, Union[float, Dict]]]:
        """Executa varredura de limiar de confianca para analisar trade-off P x R e falsos positivos."""
        sweep_results = []
        for conf in thresholds:
            dataset_matches = [
                self.evaluate_frame(gt, preds, conf_threshold=conf)
                for gt, preds in zip(dataset_gt, dataset_preds)
            ]
            summary = self.evaluate_dataset(dataset_matches)
            micro = summary.get("MICRO_AVG", {})
            macro = summary.get("MACRO_AVG", {})
            sweep_results.append({
                "conf_threshold": round(conf, 2),
                "micro_precision": micro.get("precision", 0.0),
                "micro_recall": micro.get("recall", 0.0),
                "micro_f1": micro.get("f1_score", 0.0),
                "macro_f1": macro.get("f1_score", 0.0),
                "total_fp": micro.get("FP", 0),
                "total_tp": micro.get("TP", 0),
                "total_fn": micro.get("FN", 0),
                "full_summary": summary,
            })
        return sweep_results

    def format_table(self, report: Dict[str, Dict[str, float]]) -> str:
        """Formata o relatorio quantitativo em tabela legivel estilo markdown."""
        header = f"| {'Classe':<14} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Precisão':>9} | {'Revocação':>9} | {'F1-Score':>9} |"
        sep = f"|{'-'*16}|{'-'*6}|{'-'*6}|{'-'*6}|{'-'*11}|{'-'*11}|{'-'*11}|"
        lines = [header, sep]

        for name, metrics in report.items():
            if name in ("MICRO_AVG", "MACRO_AVG"):
                continue
            lines.append(
                f"| {name:<14} | {metrics['TP']:>4} | {metrics['FP']:>4} | {metrics['FN']:>4} | "
                f"{metrics['precision']:>9.4f} | {metrics['recall']:>9.4f} | {metrics['f1_score']:>9.4f} |"
            )

        lines.append(sep)
        if "MICRO_AVG" in report:
            m = report["MICRO_AVG"]
            lines.append(
                f"| {'[Micro Avg]':<14} | {m['TP']:>4} | {m['FP']:>4} | {m['FN']:>4} | "
                f"{m['precision']:>9.4f} | {m['recall']:>9.4f} | {m['f1_score']:>9.4f} |"
            )
        if "MACRO_AVG" in report:
            m = report["MACRO_AVG"]
            lines.append(
                f"| {'[Macro Avg]':<14} | {'-':>4} | {'-':>4} | {'-':>4} | "
                f"{m['precision']:>9.4f} | {m['recall']:>9.4f} | {m['f1_score']:>9.4f} |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    evaluator = DetectionEvaluator(iou_threshold=0.50, conf_threshold=0.25)
    gt_sample = [BoundingBox(100, 100, 200, 200, class_id=0, class_name="person")]
    pred_sample = [
        # Detecção de alta confiança válida (TP)
        BoundingBox(105, 105, 205, 205, class_id=0, confidence=0.88, class_name="person"),
        # Detecção fraca que o rastreador aproveitaria mas o avaliador descarta (conf=0.15 < 0.25)
        BoundingBox(400, 400, 450, 450, class_id=2, confidence=0.15, class_name="car"),
    ]

    frame_result = evaluator.evaluate_frame(gt_sample, pred_sample)
    summary = evaluator.evaluate_dataset([frame_result])
    print(evaluator.format_table(summary))
