# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import os
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model


from src.cv.detect.detect_utils import non_max_suppression_np
from src.cv.detect.keras_yolov3.model import yolo_eval, yolo_body, tiny_yolo_body
# from src.cv.detect.keras_yolov3.utils import letterbox_image
from utils.constant import MODEL_DIR
from src.cv.detect.settings import COCO_COLORS, COCO_LABELS, C_VEHICLES, C_PERSON
import utils.logger as logger


# [model path] (the weight of yolo_v2_darknet)
YOLO_V3_MODEL = [os.path.join(MODEL_DIR, 'yolo/darknet', fn) for fn in ["yolov3_anchors.txt", "yolov3.h5"]]
# [labels]
LABELS = COCO_LABELS
# [colors]
COLORS = COCO_COLORS
# target objects
TARGET_OBJS = C_VEHICLES

CONFIDENCE_THRESH = 0.25
OVERLAP_THRESH = 0.3

LABEL_PATH = os.path.join(MODEL_DIR, 'label_names', 'coco.names')
ANCHOR_PATH = YOLO_V3_MODEL[0]


class YoloV3DarknetUtils(object):
    _defaults = {
        "model_path": YOLO_V3_MODEL[1],
        "anchors_path": ANCHOR_PATH,
        "classes_path": LABEL_PATH,
        "score": 0.3,
        "iou": 0.45,
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, targets=TARGET_OBJS, score_threshold=CONFIDENCE_THRESH):

        self.gpu_num = None
        self.model_path = None
        self.anchors_path = None
        self.classes_path = None
        self.input_image_shape = K.placeholder(shape=(2,))
        self.colors = COLORS
        self.yolo_model = None
        self.model_image_size = None
        self.score_threshold = None
        self.iou = None

        self.__dict__.update(self._defaults)  # set up default values

        if len(targets) == 0:
            self.targets = range(len(LABELS))
        else:
            self.targets = targets

        self.score_threshold = score_threshold
        self.iou = OVERLAP_THRESH

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        logger.info("loading YOLO v3 darknet {} model, anchors, and classes loaded.".format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score_threshold,
                                           iou_threshold=self.iou)
        return boxes, scores, classes

    def detect(self, img):

        img = cv2.resize(img, (320, 320))  # reduce the gpu memory
        img_h, img_w = img.shape[:2]

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            # boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            resized_img = cv2.resize(img, tuple(reversed(self.model_image_size)), cv2.INTER_CUBIC)
        else:
            new_image_size = (img_w - (img_w % 32),
                              img_h - (img_h % 32))
            # boxed_image = letterbox_image(image, new_image_size)
            resized_img = cv2.resize(img, new_image_size, cv2.INTER_CUBIC)

        re_img_h, re_img_w = resized_img.shape[:2]
        image_data = np.array(resized_img, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [re_img_h, re_img_w],
                K.learning_phase(): 0
            })

        boxes = []
        confidences = []
        label_ids = []

        for i, c in reversed(list(enumerate(out_classes))):
            if c not in self.targets:
                continue

            # predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            [x, y, x2, y2] = left, top, right, bottom

            if score > self.score_threshold:
                boxes.append(np.divide(
                    np.array([x, y, x2 - x, y2 - y]),
                    np.array([re_img_w, re_img_h, re_img_w, re_img_h])).tolist()
                )
                confidences.append(score)
                label_ids.append(c)

        return non_max_suppression_np(img_sz=img.shape[:2],
                                      boxes=boxes, confidences=confidences, labels=LABELS, label_ids=label_ids,
                                      colors=COLORS, overlap_threshold=self.iou)

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    from src.cv.draw.draw_utils import DrawUtils

    detector = YoloV3DarknetUtils(targets=C_PERSON, score_threshold=0.1)

    import os
    from src.cv.draw.draw_utils import DrawUtils
    from utils.constant import SAVE_DIR

    fn = "SAVE_2018-12-18_15-12-01.avi"
    path = os.path.join(SAVE_DIR, fn)

    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        # cv2.imwrite("frame.jpg", frame)
        dets = detector.detect(img=frame)

        show_img = DrawUtils().show_objects(objects=dets, img=frame)
        cv2.imshow("show", cv2.resize(show_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(1)

    # frame = cv2.imread("person.jpg")
    # dets = detector.detect(img=frame)
    #
    # show_img = DrawUtils().show_objects(objects=dets, img=frame)
    # cv2.imshow("show", show_img)
    # cv2.waitKey(0)
