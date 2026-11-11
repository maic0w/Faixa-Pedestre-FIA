"""Script de Exportacao de Predicoes do Detector Baseline no Formato Canonico YOLO (.txt).

Gera arquivos .txt de predicao para cada frame da amostra controlada em:
data/predictions/yolo_baseline/<image_stem>.txt

Formato por linha:
<class_id> <x_center> <y_center> <width> <height> <confidence>

Mapeamento estrito das classes COCO:
0: person
1: bicycle
2: car
3: motorcycle
5: bus
7: truck
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from ultralytics import YOLO

COCO_TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def run_baseline_inference_and_export(
    images_dir: str = "data/raw/frames",
    output_dir: str = "data/predictions/yolo_baseline",
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    classes: Optional[List[int]] = None,
) -> int:
    """Executa o detector em cada quadro e salva as predicoes no formato canonico normalizado."""
    img_path = Path(images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_classes = classes if classes is not None else list(COCO_TARGET_CLASSES.keys())

    image_files = sorted([f for f in img_path.glob("frame_*.png")] + [f for f in img_path.glob("frame_*.jpg")])
    if not image_files:
        print(f"[AVISO] Nenhuma imagem 'frame_*.png/jpg' encontrada em {images_dir}")
        return 0

    print(f"--- Inicializando Detector Baseline ({model_name}) ---")
    print(f"Diretorio de entrada: {img_path} ({len(image_files)} quadros)")
    print(f"Diretorio de predicoes: {out_path}")
    print(f"Limiar de deteccao de avaliacao: conf >= {conf_threshold} | imgsz = {imgsz}")
    print(f"Classes COCO filtradas: {target_classes}")

    model = YOLO(model_name)
    processed_count = 0

    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue

        h, w, _ = frame.shape

        # Executa deteccao pura (sem associador de tracking para a avaliacao quantitativa)
        results = model.predict(
            source=frame,
            classes=target_classes,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        pred_txt_path = out_path / f"{img_file.stem}.txt"
        boxes = results[0].boxes

        with open(pred_txt_path, "w", encoding="utf-8") as f_out:
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()

                for box, conf, cid in zip(xyxy, confs, cls_ids):
                    x1, y1, x2, y2 = box
                    # Normalizacao para o formato YOLO canônico:
                    # x_center, y_center, width, height no intervalo [0, 1]
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # Formato: <class_id> <x_center> <y_center> <width> <height> <confidence>
                    line = f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n"
                    f_out.write(line)

        processed_count += 1

    print(f"--- Predicoes Concluidas ---")
    print(f"Total de {processed_count} arquivos de predicao salvos em '{out_path}'.")
    return processed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta predicoes do baseline no formato canonico YOLO.")
    parser.add_argument("--images", type=str, default="data/raw/frames", help="Pasta com os frames extraidos")
    parser.add_argument("--output", type=str, default="data/predictions/yolo_baseline", help="Pasta de saida")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Modelo YOLO da familia Ultralytics")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confianca para deteccao pura")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolucao de inferencia")

    args = parser.parse_args()
    run_baseline_inference_and_export(
        images_dir=args.images,
        output_dir=args.output,
        model_name=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
