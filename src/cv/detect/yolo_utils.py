import sys
import numpy as np
import cv2
import os


from utils.constant import MODEL_DIR, DET_YOLO3, DET_YOLO2
from src.cv.detect.settings import COCO_LABELS, COCO_COLORS, C_VEHICLES
import utils.logger as logger

from src.cv.detect.detect_utils import non_max_suppression_cv

# derive the paths to the YOLO weights and model configuration
YOLO_V3_MODEL = [os.path.join(MODEL_DIR, 'yolo/yolo_v3', fn) for fn in ["yolov3.weights", "yolov3.cfg"]]
YOLO_V2_MODEL = [os.path.join(MODEL_DIR, 'yolo/yolo_v2', fn) for fn in ["yolov2.weights", "yolov2.cfg"]]

# load the COCO class labels our YOLO model was trained on
LABELS = COCO_LABELS
# initialize a list of colors to represent each possible class label
COLORS = COCO_COLORS
# target objects
TARGET_OBJS = C_VEHICLES

CONFIDENCE_THRESH = 0.7
OVERLAP_THRESH = 0.2


class YoloUtils:
    def __init__(self, model_type=DET_YOLO3, targets=C_VEHICLES, score_threshold=CONFIDENCE_THRESH,
                 overlap_threshold=OVERLAP_THRESH):
        logger.info("loading {} model[configure and weight files]".format(model_type))
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        if len(targets) == 0:
            self.targets = range(len(LABELS))
        else:
            self.targets = targets

        # [confidence]
        self.score_threshold = score_threshold

        # [threshol of overlap]
        self.overlap_threshold = overlap_threshold

        if model_type == DET_YOLO2:
            self.net = cv2.dnn.readNetFromDarknet(YOLO_V2_MODEL[1], YOLO_V2_MODEL[0])
            self.scale = 1 / 255.0
            self.sz = (608, 608)
        elif model_type == DET_YOLO3:
            self.net = cv2.dnn.readNetFromDarknet(YOLO_V3_MODEL[1], YOLO_V3_MODEL[0])
            self.scale = 1 / 255.0
            self.sz = (320, 320)
        else:
            logger.error("invalid yolo version")
            sys.exit(1)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, img):
        # construct a blob from the input frame and then perform a forward pass of the YOLO object detector, giving us
        # our bounding boxes and associated probabilities

        # img = cv2.resize(img, self.sz)
        blob = cv2.dnn.blobFromImage(image=img, scalefactor=self.scale, size=self.sz, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        label_ids = []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                label_id = np.argmax(scores)
                confidence = scores[label_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.score_threshold and label_id in self.targets:
                    # scale the bounding box coordinates back relative to the size of the image,
                    # keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box
                    # followed by the boxes' width and height
                    (center_x, center_y, width, height) = detection[0:4]

                    # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                    x = center_x - (width / 2)
                    y = center_y - (height / 2)

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    label_ids.append(label_id)

        return non_max_suppression_cv(img_sz=img.shape[:2], boxes=boxes, confidences=confidences, labels=LABELS,
                                      label_ids=label_ids, colors=COLORS, score_threshold=self.score_threshold,
                                      overlap_threshold=self.overlap_threshold)


if __name__ == '__main__':
    import os
    from src.cv.draw.draw_utils import DrawUtils
    from utils.constant import SAVE_DIR

    fn = "SAVE_2018-12-18_15-12-01.avi"
    path = os.path.join(SAVE_DIR, fn)

    cap = cv2.VideoCapture(path)

    ret, frame = cap.read()

    YoloUtils().detect(img=frame)
