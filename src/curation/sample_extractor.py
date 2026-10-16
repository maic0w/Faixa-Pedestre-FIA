"""Extrator de Amostragem de Quadros para Curadoria e Anotacao (Ground Truth).

Permite extrair de 100 a 300 quadros representativos de um video de trafego
com amostragem temporal uniforme, gerando manifesto CSV para auditoria.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional
import cv2


def extract_frames(
    video_path: str,
    output_dir: str = "data/annotations/raw_frames",
    target_frames: int = 200,
    prefix: str = "frame",
) -> int:
    """Extrai uma quantidade alvo de quadros distribuidos uniformemente pelo video."""
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        raise RuntimeError("O video reportou 0 quadros.")

    target_count = min(target_frames, total_frames)
    step = max(1, total_frames // target_count)

    manifest_path = out_path / "manifest_frames.csv"
    extracted_count = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow(["frame_id", "timestamp_s", "filename", "video_source"])

        curr_frame = 0
        while cap.isOpened() and extracted_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            if curr_frame % step == 0:
                timestamp_s = round(curr_frame / fps, 3)
                filename = f"{prefix}_{curr_frame:06d}.jpg"
                img_path = out_path / filename

                cv2.imwrite(str(img_path), frame)
                writer.writerow([curr_frame, timestamp_s, filename, video_file.name])
                extracted_count += 1

            curr_frame += 1

    cap.release()
    print(f"Extracao concluida: {extracted_count} quadros salvos em '{output_dir}'.")
    print(f"Manifesto gerado em '{manifest_path}'.")
    return extracted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai amostra de quadros de video para anotacao.")
    parser.add_argument("--video", type=str, required=True, help="Caminho do video de entrada")
    parser.add_argument("--output", type=str, default="data/annotations/raw_frames", help="Pasta de saida")
    parser.add_argument("--frames", type=int, default=200, help="Quantidade de quadros (100 a 300)")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.frames)
