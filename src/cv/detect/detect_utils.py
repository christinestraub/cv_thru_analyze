import numpy as np
import cv2
from operator import itemgetter


from utils.constant import DET_SSD, DET_YOLO3, DET_YOLO2, DET_DARKNET2, DET_DARKNET3, \
    KEY_CONFIDENCE, KEY_LABEL, KEY_COLOR, KEY_FRECT


def filter_confidence_dets(dets, car_thresh, truck_thresh):
    confidence_dets = []
    for det in dets:
        if det[KEY_LABEL] == "truck" and det[KEY_CONFIDENCE] > truck_thresh:
            confidence_dets.append(det)
        elif det[KEY_LABEL] != "truck" and det[KEY_CONFIDENCE] > car_thresh:
            confidence_dets.append(det)

    return confidence_dets


def __non_max_suppression(boxes, overlap_threshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = x2 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlap_threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return pick


def non_max_suppression_np(img_sz, boxes, confidences, labels, label_ids, colors, overlap_threshold):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats this is important since we'll be doing a bunch of divisions
    img_h, img_w = img_sz
    for i, box in enumerate(boxes):
        boxes[i] = (boxes[i] * np.array([img_w, img_h, img_w, img_h])).astype(np.float)
    boxes = np.array(boxes)

    pick = __non_max_suppression(boxes=boxes, overlap_threshold=overlap_threshold)
 
    # return only the bounding boxes that were picked using the
    # integer data type
    objs = []
    for i in pick:
        # extract the bounding box coordinates
        (x, y, w, h) = boxes[i]

        objs.append({
            KEY_LABEL: labels[label_ids[i]],
            KEY_CONFIDENCE: confidences[i],
            KEY_COLOR: [int(c) for c in colors[label_ids[i]]],
            # convert x, y, w, h to x, y, x2, y2
            KEY_FRECT: (
                np.divide(np.array([x, y, x + w, y + h]).astype(np.float), np.array([img_w, img_h, img_w, img_h]))),
            "x_pos": x + w / 2

        })
    return sorted(objs, key=itemgetter("x_pos"))


def non_max_suppression_cv(img_sz, boxes, confidences, labels, label_ids, colors, score_threshold, overlap_threshold):
    objs = []
    if len(boxes) == 0:
        return objs

    img_h, img_w = img_sz
    for i, box in enumerate(boxes):
        boxes[i] = (boxes[i] * np.array([img_w, img_h, img_w, img_h])).astype(np.float)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, overlap_threshold)
    idxs = idxs.flatten()

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs:
            # extract the bounding box coordinates
            (x, y, w, h) = boxes[i]

            objs.append({
                KEY_LABEL: labels[label_ids[i]],
                KEY_CONFIDENCE: confidences[i],
                KEY_COLOR: [int(c) for c in colors[label_ids[i]]],
                # convert x, y, w, h to x, y, x2, y2
                KEY_FRECT: (
                    np.divide(np.array([x, y, x + w, y + h]).astype(np.float), np.array([img_w, img_h, img_w, img_h]))),
                "x_pos": x + w / 2

            })

    return sorted(objs, key=itemgetter("x_pos"))


class DetectUtils:
    def __init__(self, det_mode, score_threshold=0.7):
        if det_mode == DET_YOLO3 or det_mode == DET_YOLO2:
            from src.cv.detect.yolo_utils import YoloUtils
            detector = YoloUtils(model_type=det_mode, score_threshold=score_threshold)
        elif det_mode == DET_SSD:
            from src.cv.detect.ssd_utils import SsdUtils
            detector = SsdUtils(model_type=DET_SSD, score_threshold=score_threshold)
        elif det_mode == DET_DARKNET2:
            from src.cv.detect.yolo_v2_darknet import YoloV2DarknetUtils
            detector = YoloV2DarknetUtils(score_threshold=score_threshold)
        elif det_mode == DET_DARKNET3:
            from src.cv.detect.yolo_v3_darknet import YoloV3DarknetUtils
            detector = YoloV3DarknetUtils(score_threshold=score_threshold)
        else:
            detector = None

        self.detector = detector


if __name__ == '__main__':
    import os
    from src.cv.draw.draw_utils import DrawUtils
    from utils.constant import SAVE_DIR

    _detector = DetectUtils(det_mode=DET_YOLO3, score_threshold=0.1)
    _drawer = DrawUtils()

    fn = "SAVE_2018-12-18_15-12-01.avi"
    path = os.path.join(SAVE_DIR, fn)

    cap = cv2.VideoCapture(path)
    cnt = 0
    while True:
        cnt += 1
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame = frame[91:625, 111:965]

        _dets = _detector.detector.detect(frame)
        show_img = _drawer.show_objects(img=frame, objects=_dets)

        cv2.imshow("frame", cv2.resize(show_img, None, fx=0.5, fy=0.5))
        # if len(dets) > 0:
        #     cv2.imwrite("frame_{}.jpg".format(cnt), show_img)
        key = cv2.waitKey(1)
        if key == ord('n'):
            _pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            _pos += 300
            cap.set(cv2.CAP_PROP_POS_FRAMES, _pos)
