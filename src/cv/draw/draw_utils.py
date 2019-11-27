import cv2
import numpy as np
from utils.constant import KEY_FRECT, KEY_LABEL, KEY_RECT, KEY_CONFIDENCE, KEY_DWELLING_FRAMES, \
    ROI_TRACK_AREA, ROI_CROSS_LINE, ROI_ORDER_AREA_1, ROI_ORDER_POINT_1, ROI_ORDER_AREA_2, ROI_ORDER_POINT_2


import utils.logger as logger

# draw utils
SHOW_IMG_WIDTH = 2000

# grid setting
GRID_LINE_COLOR = (200, 200, 200)
GRID_LINE_THICK = 3

COLR_RECT_FILL = (255, 125, 0)
COLR_RECT_BORDER = (0, 0, 255)
COLR_TEXT = (255, 255, 0)


class DrawUtils:
    def __init__(self, b_log=True):
        self.b_log = b_log

    @staticmethod
    def show_objects(img, objects, offset=(0, 0)):
        ofst_x, ofst_y = offset
        show_img = img.copy()
        img_h, img_w = img.shape[:2]

        for obj in objects:
            (x, y, x2, y2) = (obj[KEY_FRECT] * np.array([img_w, img_h, img_w, img_h])).astype(np.int)
            label = obj[KEY_LABEL]
            confidence = float(obj[KEY_CONFIDENCE])
            str_label = "{}: {:.1f}%".format(label, confidence * 100)
            # str_label = "{}".format(label)

            if 0 < x < x2 < img_w and 0 < y < y2 < img_h:
                cv2.rectangle(show_img, (x + ofst_x, y + ofst_y), (x2 + ofst_x, y2 + ofst_y),
                              COLR_RECT_BORDER, 2)
                cv2.putText(show_img, str_label, (x + ofst_x, y + ofst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COLR_TEXT, 1)

        return show_img

    def show_trackers(self, trk_img, trackers, mode="rect", fps=25.0, offset=(0, 0)):
        ofst_x, ofst_y = offset
        show_img = trk_img.copy()
        img_h, img_w = trk_img.shape[:2]

        for tid in trackers.keys():
            color = COLR_RECT_BORDER
            try:
                [t_x, t_y, t_w, t_h] = trackers[tid][KEY_RECT]
                t_x2 = t_x + t_w
                t_y2 = t_y + t_h

                if 0 < t_x < t_x + t_w < img_w and 0 < t_y < t_y + t_h < img_h:
                    if mode == "point":
                        cv2.circle(show_img, (int(t_x + t_w / 2 + ofst_x), int(t_y + t_h / 2 + ofst_y)), 5, color, -1)
                    else:  # "rect"
                        cv2.rectangle(show_img, (int(t_x + ofst_x), int(t_y + ofst_y)),
                                      (int(t_x2 + ofst_x), int(t_y2 + ofst_y)), (0, 0, 255), 2)

                        str_lbl = "{}_{}".format(tid,
                                                 round(trackers[tid][KEY_DWELLING_FRAMES]/fps, 1),
                                                 # round(trackers[tid][KEY_SPEED][0] + trackers[tid][KEY_SPEED][1], 1)
                                                 )
                        cv2.putText(show_img, str_lbl, (int(t_x + ofst_x), int(t_y + ofst_y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 2)
            except Exception as e:
                if self.b_log:
                    logger.warn("{}_tid:{}".format(e, tid))
        return show_img

    @staticmethod
    def show_roi(img, roi_objs):
        bck = np.zeros_like(img)
        for key in [ROI_TRACK_AREA, ROI_CROSS_LINE, ROI_ORDER_AREA_1, ROI_ORDER_POINT_1, ROI_ORDER_AREA_2,
                    ROI_ORDER_POINT_2]:
            roi = roi_objs[key]
            if key == ROI_TRACK_AREA:
                cv2.drawContours(bck, [roi], -1, (0, 255, 0), -1)
            elif key == ROI_CROSS_LINE:
                cv2.line(bck, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]), (255, 0, 0), 3)
            elif key == ROI_ORDER_AREA_1:
                cv2.drawContours(bck, [roi], -1, (0, 0, 255), -1)
            elif key == ROI_ORDER_POINT_1:
                cv2.circle(bck, (roi[0][0], roi[0][1]), 3, (255, 0, 0), -1)
            elif key == ROI_ORDER_AREA_2:
                cv2.drawContours(bck, [roi], -1, (0, 0, 255), -1)
            elif key == ROI_ORDER_POINT_2:
                cv2.circle(bck, (roi[0][0], roi[0][1]), 3, (255, 0, 0), -1)

        return cv2.addWeighted(img, 0.8, bck, 0.2, 0)
