from ultralytics import YOLO
import cv2
import socket
import json

# load model
model = YOLO("best.pt")

# UDP setup
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1)
            })

    # kirim ke unity
    data = json.dumps(detections)
    sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)
    if cv2.waitKey(1) == 27:
        break