import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

image = cv2.imread('E:\sih\image3.jpg')

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if classes[class_id] == "person" and confidence > 0.5:
            center_x, center_y, width, height = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
            x = center_x - width // 2
            y = center_y - height // 2
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.imshow('Human Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
