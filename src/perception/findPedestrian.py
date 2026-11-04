from ultralytics import YOLO
import cv2


model = YOLO("yolo26n.pt")

video = cv2.VideoCapture("./data/video.mp4")

frame_num = 0

while True:
    ret, frame = video.read()

    if not ret:
        break
    resultado = model.track(
        frame,
        persist = True,
        tracker = "botsort.yaml",
        classes = [0],
        conf=0.10,
        imgsz = 640
    )

    boxes = resultado[0].boxes

    frame_completo = resultado[0].plot(conf=False)

    cv2.imshow("YOLO", frame_completo)

    frame_num += 1

    if cv2.waitKey(1) == ord("q"):
         break

video.release()
cv2.destroyAllWindows()
