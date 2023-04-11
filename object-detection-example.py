import dlr
import numpy as np
import os
import cv2
from dlr.counter.phone_home import PhoneHome

device = 'cpu'
model = dlr.DLRModel('./dlr_model', device)
PhoneHome.disable_feature()

classLabels = []
filename = './labelmap.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')

inputImage = cv2.imread("./input_image.jpg")

# Format image so model can make predictions
#resized_image = image.resize((300, 300))
resized_image = cv2.resize(inputImage, (300, 300), interpolation = cv2.INTER_AREA)

# Model is quantized, so convert the image to uint8
x = np.array(resized_image).astype('uint8')
out = model.run(x)

detection_boxes = np.squeeze(out[0])
detection_classes = np.squeeze(out[1])
detection_scores = np.squeeze(out[2]) 
num_detections = np.squeeze(out[3])

# Loop over the detections
for i in range(num_detections):
    confidence = detection_scores[i]
    if confidence > 0.6:  # You can adjust the confidence threshold as needed
        class_id = int(detection_classes[i])
        class_name = classLabels[class_id + 1]
        box = detection_boxes[i] * np.array([inputImage.shape[1], inputImage.shape[0], inputImage.shape[1], inputImage.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(inputImage, (startX, startY), (endX, endY), (255, 0, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = "{}: {:.2f}%".format(class_name, (confidence * 100))
        cv2.putText(inputImage, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
