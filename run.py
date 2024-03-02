import cv2
import numpy as np
import winsound
frequency = 500
duration = 20

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == "person" and confidence > 0.5:
                center_x, center_y, width, height = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x = center_x - width // 2
                y = center_y - height // 2
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                winsound.Beep(frequency, duration)
    cv2.imshow('Human Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
