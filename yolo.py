# Importing the libraries
import numpy as np
import argparse
import cv2

# Reading command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--config", required=True, help="path to config file")
ap.add_argument("-w", "--weights", required=True, help="path to yolo weights")
ap.add_argument("-cl", "--classes", required=True, help="path to text file conraining class names")
args = vars(ap.parse_args())


def get_output_layers(net):
	# FUnction to return the name of all the output layers
	layers = net.getLayerNames()
	return [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw_bounding_box(image, cid, x, y, xw, yh):
	# Function to draw the box with color and label according to class of object detected
	label = str(classes[cid])
	color = COLORS[cid]
	cv2.rectangle(image, (x, y), (xw, yh), color, 2)
	cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Reading the input image and defining scale factor
image = cv2.imread(args['image'])
h, w = image.shape[:2]
scale = 0.00392

# Reading the different classes of objects from the file
classes = None
with open(args['classes'], 'r') as f:
	classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))  # Random distribution of colors for classes
net = cv2.dnn.readNet(args['weights'], args['config'])
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0), True, crop=False)
net.setInput(blob)
predictions = net.forward(get_output_layers(net))  # Building the net

# Getting all predictions and filtering out to get strong predictions
class_ids, boxes, confidences = [], [], []
conf_threshold, nms_threshold = 0.5, 0.4
for pred in predictions:
	for detection in pred:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence > conf_threshold:
			centerX, centerY = int(detection[0] * w), int(detection[1] * h)
			W, H = int(detection[2] * w), int(detection[3] * h)
			x, y = centerX - W / 2, centerY - H / 2
			class_ids.append(class_id)
			confidences.append(float(confidence))
			boxes.append([x, y, W, H])

# Preforming Non-Max Supression to avoid multiple boxes for the same object
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
	i = i[0]
	x, y, W, H = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
	draw_bounding_box(image, class_ids[i], round(x), round(y), round(x + W), round(y + H))

cv2.imshow('Detections', image)
cv2.waitKey()
cv2.destroyAllWindows()
