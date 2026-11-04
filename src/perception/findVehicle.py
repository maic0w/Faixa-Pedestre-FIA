"""Deteccao e rastreamento de veiculos com parametros de calibracao."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("Use um valor entre 0 e 1.")
    return value


def image_size(value):
    value = int(value)
    if value <= 0 or value % 32:
        raise argparse.ArgumentTypeError("Use um inteiro positivo multiplo de 32.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=PROJECT_ROOT / "data/raw/BackVehcilesTraffic720p.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    # 0.10 permite ao BoT-SORT recuperar tracks com deteccoes fracas;
    # o limiar proprio do tracker controla a criacao de novos IDs.
    parser.add_argument("--conf", type=probability, default=0.10)
    parser.add_argument("--iou", type=probability, default=0.70,
                        help="Limiar de IoU da supressao de caixas (NMS).")
    parser.add_argument("--imgsz", type=image_size, default=640,
                        help="Use 960 para avaliar objetos pequenos; aumenta o processamento.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Video nao encontrado: {source}")

    # Valida a leitura e fecha a captura antes de o YOLO abrir o video.
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened() or not capture.read()[0]:
            parser.error(f"Nao foi possivel ler o video: {source}")
    finally:
        capture.release()

    model = YOLO(args.model)
    results = model.track(
        source=str(source),
        tracker="botsort.yaml",  # BoT-SORT, nao DeepSORT.
        classes=[2, 3, 5, 7],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        vid_stride=1,  # Processa todos os quadros para preservar a continuidade.
        persist=True,
        line_width=1,
        save=args.save,
        stream=True,
        verbose=False,
    )
    try:
        for result in results:
            if not args.no_show:
                cv2.imshow("Rastreamento de Veiculos", result.plot(conf=True, line_width=1))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # Fecha o gerador inclusive quando o usuario interrompe com 'q'.
        results.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
