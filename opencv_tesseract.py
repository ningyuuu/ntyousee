from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

height, width = 640, 320

def decode_predictions(scores, geometry):
    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        # extract probabilities and BB coords
        scores_data = scores[0, 0, y]
        x_data_0 = geometry[0, 0, y]
        x_data_1 = geometry[0, 1, y]
        x_data_2 = geometry[0, 2, y]
        x_data_3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < 0.5:
                continue

            offset_x, offset_y = x * 4, y * 4
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data_0[x] + x_data_2[x]
            w = x_data_1[x] + x_data_3[x]

            end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
            end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
            start_y = int(end_y - h)
            start_x = int(end_x - w)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return (rects, confidences)

parser = argparse.ArgumentParser()
parser.add_argument('img_link')
args = vars(parser.parse_args())

image = cv2.imread(args["img_link"])
orig = image.copy()
orig_h, orig_w = image.shape[:2]

r_w = orig_w / width
r_h = orig_h / height

image = cv2.resize(image, (width, height))
h, w = image.shape[:2]

print("Using EAST text detector...")
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"
]
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []

for (start_x, start_y, end_x, end_y) in boxes:
    padding = 0.01

    start_x = int(start_x * r_w)
    start_y = int(start_y * r_h)
    end_x = int(end_x * r_w)
    end_y = int(end_y * r_h)

    d_x = int((end_x - start_x) * padding)
    d_y = int((end_y - start_y) * padding)

    start_x = max(0, start_x - d_x)
    start_y = max(0, start_y - d_y)

    end_x = min(w, end_x + d_x)
    end_y - min(h, end_y + d_y)

    roi = orig[start_y:end_y, start_x:end_x]

    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    results.append(((start_x, start_y, end_x, end_y), text))

results = sorted(results, key=lambda r:r[0][1])

# loop over the results
for ((start_x, start_y, end_x, end_y), text) in results:
	# display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))

    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
    cv2.putText(orig, text, (start_x, start_y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
