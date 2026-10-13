from ultralytics import YOLO
import cv2
import csv


model = YOLO("yolo26n.pt")

video = cv2.VideoCapture("./data/video.mp4")

frame_num = 0

arquivo_csv = open("deteccoes_pedestres.csv", "w", newline="", encoding="utf-8")
escritor = csv.writer(arquivo_csv)

escritor.writerow(["frame", "id", "x1", "y1", "x2", "y2", "confianca"])

while True:
    ret, frame = video.read()

    if not ret:
        break
    resultado = model.track(
        frame,
        persist = True,
        tracker = "configs/botsort_custom.yaml",
        classes = [0],
        conf=0.10,
        imgsz = 800
    )

    boxes = resultado[0].boxes

    if boxes is not None and boxes.id is not None:
        coordenadas = boxes.xyxy.cpu().tolist()
        ids = boxes.id.int().cpu().tolist()
        confiancas = boxes.conf.cpu().tolist()

        for coord, id_pessoa, confianca in zip(coordenadas, ids, confiancas):
            escritor.writerow([frame_num, id_pessoa, *coord, confianca])

    frame_completo = resultado[0].plot(conf=False)

    cv2.imshow("YOLO", frame_completo)

    frame_num += 1

    if cv2.waitKey(1) == ord("q"):
         break

video.release()
arquivo_csv.close()
cv2.destroyAllWindows()
