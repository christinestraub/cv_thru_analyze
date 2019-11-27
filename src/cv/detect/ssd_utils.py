import os
import sys
import cv2
import numpy as np

import utils.logger as logger
from src.cv.detect.settings import PASCAL_LABELS, PASCAL_COLORS, P_VEHICLES
from utils.constant import MODEL_DIR, KEY_LABEL, KEY_CONFIDENCE, KEY_COLOR, KEY_FRECT, DET_SSD, DET_VGG

# [model path]
SSD_MODEL = [os.path.join(MODEL_DIR, 'ssd', fn) for fn in ['MobileNetSSD_deploy.caffemodel',
                                                           'MobileNetSSD_deploy.prototxt']]
VGG_MODEL = [os.path.join(MODEL_DIR, 'ssd', fn) for fn in ['VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel',
                                                           'VGG_VOC0712Plus_SSD_512x512_ft_iter_160000_deploy.prototxt']
             ]

# labels
LABELS = PASCAL_LABELS
# colors
COLORS = PASCAL_COLORS
# object indexes
TARGET_OBJS = P_VEHICLES
# confidence
CONFIDENCE_THRESH = 0.5


class SsdUtils:
    def __init__(self, model_type=DET_SSD, targets=P_VEHICLES, score_threshold=CONFIDENCE_THRESH):
        # [target object ids]
        if len(targets) == 0:
            self.targets = range(len(LABELS))
        else:
            self.targets = targets

        # [confidence]
        self.score_threshold = score_threshold

        # [model type ssd-mobile or vgg-voco712]
        if model_type == DET_SSD:  # "SSD"
            prototxt = os.path.join(MODEL_DIR, SSD_MODEL[1])
            model = os.path.join(MODEL_DIR, SSD_MODEL[0])
            scale = 0.00784  # 2/256
            mean_subtraction = (127.5, 127.5, 127.5)
            self.ssd_sz = (300, 300)
        elif model_type == DET_VGG:  # "VGG"
            prototxt = os.path.join(MODEL_DIR, VGG_MODEL[1])
            model = os.path.join(MODEL_DIR, VGG_MODEL[0])
            scale = 1.0
            mean_subtraction = (104, 117, 123)
            self.ssd_sz = (512, 512)
        else:
            logger.error("Invalid model type {}".format(model_type))
            sys.exit(1)

        if os.path.exists(prototxt) and os.path.exists(model):
            logger.info("loading SSD cnn_model...")
            self.scale = scale
            self.mean_subtraction = mean_subtraction
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        else:
            logger.error("no exist SSD camera_models")
            sys.exit(1)

    def detect(self, img, ssd_sz=None):
        h, w = img.shape[:2]
        if ssd_sz is None:
            ssd_sz = (int(w * self.ssd_sz[1] / h), self.ssd_sz[1])

        blob = cv2.dnn.blobFromImage(image=img, scalefactor=self.scale, size=ssd_sz, mean=self.mean_subtraction)

        self.net.setInput(blob)
        detections = self.net.forward()

        objs = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            label_id = int(detections[0, 0, i, 1])

            if confidence > self.score_threshold and label_id in self.targets:
                box = detections[0, 0, i, 3:7]  # [x, y, x2, y2]

                objs.append({KEY_LABEL: LABELS[label_id],
                             KEY_CONFIDENCE: confidence,
                             KEY_COLOR: [int(c) for c in COLORS[label_id]],
                             KEY_FRECT: box})
        return objs


if __name__ == '__main__':
    pass
