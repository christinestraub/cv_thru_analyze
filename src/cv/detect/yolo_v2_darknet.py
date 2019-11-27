import os
import cv2
import numpy as np


from keras import backend as kera
from keras.models import load_model


from src.cv.detect.keras_yolov2.keras_yolov2 import yolo_eval, yolo_head
from src.cv.detect.detect_utils import non_max_suppression_np
from utils.constant import MODEL_DIR
from src.cv.detect.settings import COCO_COLORS, COCO_LABELS, C_VEHICLES
import utils.logger as logger


# [model path] (the weight of yolo_v2_darknet)
YOLO_V2_MODEL = [os.path.join(MODEL_DIR, 'yolo/darknet', fn) for fn in ["yolov2_anchors.txt", "yolov2.h5"]]
# [labels]
LABELS = COCO_LABELS
# [colors]
COLORS = COCO_COLORS
# target objects
TARGET_OBJS = C_VEHICLES

CONFIDENCE_THRESH = 0.25
OVERLAP_THRESH = 0.3


class YoloV2DarknetUtils:

    def __init__(self, targets=TARGET_OBJS, score_threshold=CONFIDENCE_THRESH):
        logger.info("loading YOLO v2 darknet model[model and anchors]")
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        if len(targets) == 0:
            self.targets = range(len(LABELS))
        else:
            self.targets = targets

        # [confidence]
        self.score_threshold = score_threshold

        # [threshold of overlap]
        self.overlap_threshold = OVERLAP_THRESH

        # [model] paths to h5 model file containing body and anchors file of a YOLO_v2 model
        self.anchors_path, self.model_path = YOLO_V2_MODEL[:2]

        self.sz = (608, 608)

        self.class_names = LABELS
        self.anchors, self.yolo_model = self.__config_model()

        # settings for self.session
        self.model_input_sz = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_input_sz != (None, None)
        yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.input_image_shape = kera.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs=yolo_outputs,
            image_shape=self.input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.overlap_threshold)

    def __config_model(self):

        self.sess = kera.get_session()

        # load anchors info
        anchors_path = os.path.expanduser(self.anchors_path)
        os.path.exists(anchors_path)
        with open(anchors_path) as anchor_f:
            anchor_f.seek(0, 0)
            anchors = anchor_f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        anchors = anchors

        # load yolo model
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(self.class_names)
        num_anchors = len(anchors)

        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and --classes_path flags.'

        return anchors, yolo_model

    def detect(self, img):
        img = cv2.resize(img, self.sz)
        img_h, img_w = img.shape[:2]  # reduce the gpu memory

        if self.is_fixed_size:
            resized = cv2.resize(img, tuple(reversed(self.model_input_sz)), cv2.INTER_CUBIC)
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have width and height as multiples of 32.
            new_img_sz = (img_w - (img_w % 32), img_h - (img_h % 32))
            resized = cv2.resize(img, new_img_sz, cv2.INTER_CUBIC)

        re_img_h, re_img_w = resized.shape[:2]
        data = np.array(resized, dtype='float32')

        data /= 255.
        data = np.expand_dims(data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: data,
                self.input_image_shape: [re_img_h, re_img_w],
                kera.learning_phase(): 0
            })

        boxes = []
        confidences = []
        label_ids = []

        for i, c in reversed(list(enumerate(out_classes))):
            if c not in self.targets:
                continue

            # predicted_class = self.class_names[c]
            box = out_boxes[i]  # top, left, bottom, right = box
            top, left, bottom, right = box
            [x, y, x2, y2] = left, top, right, bottom
            score = out_scores[i]

            if score > self.score_threshold:
                boxes.append(np.divide(
                    np.array([x, y, x2 - x, y2 - y]),
                    np.array([re_img_w, re_img_h, re_img_w, re_img_h])).tolist())

                confidences.append(score)
                label_ids.append(c)

        return non_max_suppression_np(img_sz=img.shape[:2], boxes=boxes, confidences=confidences, labels=LABELS,
                                      label_ids=label_ids, colors=COLORS, overlap_threshold=self.overlap_threshold)

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    from src.cv.draw.draw_utils import DrawUtils
    detector = YoloV2DarknetUtils(targets=C_VEHICLES)

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
