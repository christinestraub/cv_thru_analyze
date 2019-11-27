import os
import numpy as np
import tensorflow as tf
import cv2

import utils.logger as logger
from utils.constant import MODEL_DIR
from src.cv.detect.settings import PASCAL_LABELS, PASCAL_COLORS, P_VEHICLES
from utils.constant import KEY_LABEL, KEY_CONFIDENCE, KEY_COLOR, KEY_FRECT

# [model path] (the weight of yolo)
YOLO_SMALL = os.path.join(MODEL_DIR, 'yolo/yolo_small', 'yolo_small.ckpt')

# [labels]
LABELS = PASCAL_LABELS
# [colors]
COLORS = PASCAL_COLORS
# [pascal labels]
TARGET_OBJS = P_VEHICLES

CONFIDENCE_THRESH = 0.3
OVERLAP_THRESH = 0.3


class YoloSmallUtils:
    alpha = 0.1
    iou_threshold = OVERLAP_THRESH

    # init the class variables
    x = None
    conv_1 = None
    pool_2 = None
    conv_3 = None
    pool_4 = None
    conv_5 = None
    conv_6 = None
    conv_7 = None
    conv_8 = None
    pool_9 = None
    conv_10 = None
    conv_11 = None
    conv_12 = None
    conv_13 = None
    conv_14 = None
    conv_15 = None
    conv_16 = None
    conv_17 = None
    conv_18 = None
    conv_19 = None
    pool_20 = None
    conv_21 = None
    conv_22 = None
    conv_23 = None
    conv_24 = None
    conv_25 = None
    conv_26 = None
    conv_27 = None
    conv_28 = None
    fc_29 = None
    fc_30 = None
    # skip dropout_31
    fc_32 = None
    sess = None
    saver = None

    def __init__(self, targets=None):
        if targets is None:
            self.targets = P_VEHICLES
        elif len(targets) == 0:
            self.targets = range(len(LABELS))
        else:
            self.targets = targets

        # [confidence]
        self.score_threshold = CONFIDENCE_THRESH

        # [weight]
        self.weights_file = YOLO_SMALL

        self.config_network()

    def config_network(self):
        logger.info("configure YOLO_small graph...")
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
        self.fc_29 = self.fc_layer(29, self.conv_28, 512, flat=True, linear=False)
        self.fc_30 = self.fc_layer(30, self.fc_29, 4096, flat=False, linear=False)
        # skip dropout_31
        self.fc_32 = self.fc_layer(32, self.fc_30, 1470, flat=False, linear=True)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        # print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
        #       idx, size, size, stride, filters, int(channels)))
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    @staticmethod
    def pooling_layer(idx, inputs, size, stride):
        # print('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride))
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):

        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        # print('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
        #     idx, hiddens, int(dim), int(flat), 1 - int(linear)))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def detect(self, img):
        img_resized = cv2.resize(img, (448, 448))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_rgb)
        inputs = np.zeros((1, 448, 448, 3), dtype=np.float32)
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_32, feed_dict=in_dict)

        box_output = self.__interpret_output(img_sz=img.shape[:2], output=net_output[0])
        return box_output

    def __interpret_output(self, img_sz, output):
        img_h, img_w = img_sz
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.score_threshold, dtype=np.bool)
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype=np.bool)
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        objs = []
        for i in range(len(boxes_filtered)):

            [center_x, center_y, w, h] = [boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2],
                                          boxes_filtered[i][3]]

            x = center_x - (w / 2)
            y = center_y - (h / 2)

            offset_label_idx = classes_num_filtered[i] + 1  # from 1 instead of zero("background")
            if offset_label_idx in self.targets:
                objs.append({
                    KEY_LABEL: LABELS[offset_label_idx],
                    KEY_CONFIDENCE: probs_filtered[i],
                    KEY_COLOR: [int(c) for c in COLORS[classes_num_filtered[i]]],
                    KEY_FRECT: (
                        np.divide(np.array([x, y, x + w, y + h]), np.array([img_w, img_h, img_w, img_h]))).astype(
                        np.float)
                })

        return objs

    @staticmethod
    def iou(box1, box2):
        cen_x1, cen_y1, w1, h1 = box1[:4]
        cen_x2, cen_y2, w2, h2 = box2[:4]

        tb = min(cen_x1 + 0.5 * w1, cen_x2 + 0.5 * w2) - max(cen_x1 - 0.5 * w1, cen_x2 - 0.5 * w2)
        lr = min(cen_y1 + 0.5 * h1, cen_y2 + 0.5 * h2) - max(cen_y1 - 0.5 * h1, cen_y2 - 0.5 * h2)
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (w1 * h1 + w2 * h2 - intersection)


if __name__ == '__main__':
    yolo = YoloSmallUtils()
    from src.cv.draw.draw_utils import DrawUtils
    drawer = DrawUtils()
    cap = cv2.VideoCapture("sample.mp4")

    while True:
        ret, frame = cap.read()

        dets = yolo.detect(frame)
        show_img = drawer.show_objects(img=frame, objects=dets)

        cv2.imshow("frame", show_img)
        cv2.waitKey(1)
