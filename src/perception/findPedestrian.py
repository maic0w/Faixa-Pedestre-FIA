from ultralytics import YOLO
import cv2


model = YOLO("yolo26n.pt")

video = cv2.VideoCapture("./data/video4.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    resultado = model.track(
        frame,
        persist = True,
        tracker = "botsort.yaml",
        classes = [0],
        conf=0.25
    )

    frame_completo = resultado[0].plot()

    cv2.imshow("YOLO", frame_completo)

    if cv2.waitKey(1) == ord("q"):
         break

video.release()
cv2.destroyAllWindows()
