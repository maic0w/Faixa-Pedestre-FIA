"""Script de Extracao e Amostragem Temporal Controlada de Quadros (OpenCV).

Responsavel por isolar uma amostra controlada de 100 a 200 frames a partir de gravacoes
urbanas abertas para rotulagem manual no CVAT/Label Studio e avaliacao de detectores.
Amostra a uma taxa de 1 a 2 quadros por segundo para evitar redundancia temporal entre frames vizinhos.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional
import cv2


def extract_controlled_frames(
    video_path: str,
    output_dir: str = "data/raw/frames",
    sample_rate_fps: float = 1.5,
    min_frames: int = 100,
    max_frames: int = 200,
    prefix: str = "frame",
    img_extension: str = "png",
) -> int:
    """Extrai frames de um video a uma taxa controlada de 1 a 2 fps ate atingir entre 100 e 200 imagens.

    Args:
        video_path: Caminho do arquivo de video.
        output_dir: Diretorio onde os frames serao salvos (ex: data/raw/frames).
        sample_rate_fps: Taxa de amostragem temporal em quadros por segundo (padrao: 1.5 fps).
        min_frames: Quantidade minima desejada de quadros (padrao: 100).
        max_frames: Quantidade maxima desejada de quadros (padrao: 200).
        prefix: Prefixo do nome do arquivo (padrao: 'frame').
        img_extension: Extensao da imagem (padrao: 'png').

    Returns:
        int: Total de quadros extraidos com sucesso.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Arquivo de video nao encontrado: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video com OpenCV: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_native_frames / native_fps if native_fps > 0 else 0.0

    print(f"--- Processando Video: {video_file.name} ---")
    print(f"Resolucao Nativa: {width}x{height} | FPS Nativo: {native_fps:.2f}")
    print(f"Total de Quadros Nativos: {total_native_frames} | Duracao: {duration_s:.2f} s")

    # Calcula o passo de quadros para atingir o sample_rate_fps desejado
    # step_frames = round(native_fps / sample_rate_fps)
    step_frames = max(1, round(native_fps / sample_rate_fps))
    effective_fps = native_fps / step_frames
    print(f"Taxa de Amostragem Alvo: {sample_rate_fps} fps (1 frame a cada {step_frames} frames nativos -> ~{effective_fps:.2f} fps)")

    index_log_path = out_path / "frames_index.csv"
    extracted_count = 0
    current_native_frame = 0

    with open(index_log_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame_seq_id", "filename", "native_frame_idx", "timestamp_s", "video_source"])

        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_native_frame % step_frames == 0:
                extracted_count += 1
                seq_name = f"{prefix}_{extracted_count:04d}.{img_extension}"
                target_file = out_path / seq_name
                timestamp_s = round(current_native_frame / native_fps, 3)

                cv2.imwrite(str(target_file), frame)
                writer.writerow([extracted_count, seq_name, current_native_frame, timestamp_s, video_file.name])

            current_native_frame += 1

    cap.release()

    print(f"--- Extracao Concluida ---")
    print(f"Total de quadros salvos em '{out_path}': {extracted_count}")
    print(f"Arquivo de indice gerado: '{index_log_path}'")

    if extracted_count < min_frames:
        print(f"[AVISO] Quantidade extraida ({extracted_count}) ficou abaixo do minimo esperado ({min_frames}).")
    elif extracted_count > max_frames:
        print(f"[AVISO] Quantidade extraida ultrapassou o maximo estipulado ({max_frames}).")
    else:
        print(f"[OK] Criterio de Aceitacao (DoD) satisfeito: {extracted_count} frames estao entre {min_frames} e {max_frames}.")

    return extracted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai amostra controlada de 100-200 frames a 1-2 fps.")
    parser.add_argument("--video", type=str, default="data/raw/urban_crosswalk_test.webm", help="Caminho do video")
    parser.add_argument("--output", type=str, default="data/raw/frames", help="Pasta de saida dos frames")
    parser.add_argument("--fps", type=float, default=1.35, help="Taxa de amostragem em frames por segundo (1.0 a 2.0)")
    parser.add_argument("--min", type=int, default=100, help="Minimo de frames esperado")
    parser.add_argument("--max", type=int, default=200, help="Maximo de frames esperado")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="Formato da imagem")

    args = parser.parse_args()
    extract_controlled_frames(
        video_path=args.video,
        output_dir=args.output,
        sample_rate_fps=args.fps,
        min_frames=args.min,
        max_frames=args.max,
        img_extension=args.format,
    )
