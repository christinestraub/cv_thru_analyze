import cv2
import math
import numpy as np
import os


import utils.logger as logger
from utils.common import load_json
from utils.constant import KEY_FRECT, KEY_PREV_RECT, KEY_RECT, ROOT_DIR, ROI_TRACK_AREA, \
    ROI_CROSS_LINE, KEY_UPDATE_STATE, KEY_CROSSLINE_STATE, UPDATE_STATE_THRESH, \
    ORDER_PT_RANGE_THRESH, KEY_DWELLING_FRAMES, KEY_ENTER_FRAME_POS, \
    ROI_ORDER_POINT_1, ROI_ORDER_POINT_2, KEY_FRAMES_TO_ORDER_PT_1, KEY_FRAMES_AT_ORDER_PT_1, \
    KEY_FRAMES_TO_ORDER_PT_2, KEY_FRAMES_AT_ORDER_PT_2, \
    KEY_FRONT_FLAG, OVERLAP_THRESH


error_dis = -1000000.0


def merge_rects(det1, det2):
    x, y, x2, y2 = det1[KEY_FRECT]
    _x, _y, _x2, _y2 = det2[KEY_FRECT]

    xx = (x + _x) // 2
    yy = (y + _y) // 2
    xx2 = (x2 + _x2) // 2
    yy2 = (y2 + _y2) // 2

    det2[KEY_FRECT] = [xx, yy, xx2, yy2]
    return det2


def is_overlap_rects(rect1, rect2):
    x, y, x2, y2 = rect1
    _x, _y, _x2, _y2 = rect2

    xx = max(x, _x)
    yy = max(y, _y)
    xx2 = min(x2, _x2)
    yy2 = min(y2, _y2)

    if xx < xx2 and yy < yy2:
        # the size of overlap rect must be bigger than average size
        o_sz = (xx2 - xx) * (yy2 - yy)
        sz1 = (x2 - x) * (y2 - y)
        sz2 = (_x2 - _x) * (_y2 - _y)
        avg_sz = (sz1 + sz2) / 2
        if o_sz / avg_sz > (OVERLAP_THRESH / 100.0):
            return True
    return False


def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def find_nearest_border(self, rect, roi):
    t_x, t_y, t_w, t_h = rect
    pt = (t_x + t_w / 2, t_y + t_h / 2)

    dis_list = []
    for i in range(len(roi)):
        dis = self.distance(pt, roi[i]) + self.distance(pt, roi[(i + 1) % len(roi)])
        dis_list.append(dis)

    min_border_id = dis_list.index(min(dis_list))

    return min_border_id


class RoiUtils:
    def __init__(self, img_sz, roi_path=os.path.join(ROOT_DIR, "roi.json"), b_log=True):
        img_h, img_w = img_sz
        self.img_h, self.img_w = img_sz

        self.b_log = b_log

        roi_data = load_json(json_path=roi_path)

        # float coordinate to integer coordinate
        self.roi_objs = {}
        for key in roi_data.keys():
            roi_obj = [[pt[0] * img_w, pt[1] * img_h] for pt in roi_data[key]]
            self.roi_objs[key] = np.array(roi_obj, dtype=np.int)

        # calculate the optimized cropping rectangle
        margin_w, margin_h = 30, 200
        crop_x1, crop_y1 = np.min(self.roi_objs[ROI_TRACK_AREA], axis=0).tolist()
        crop_x1 = max(crop_x1 - margin_w, 0)
        crop_y1 = max(crop_y1 - margin_h, 0)
        crop_x2, crop_y2 = np.max(self.roi_objs[ROI_TRACK_AREA], axis=0).tolist()
        crop_x2 = min(crop_x2 + margin_w, img_w)
        crop_y2 = min(crop_y2 + margin_h, img_h)
        self.crop_to_detect = [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]

        # recover the cropping offset
        self.cropped_roi_objs = {}
        for key in self.roi_objs.keys():
            cropped_roi_obj = [[pt[0] * img_w - crop_x1, pt[1] * img_h - crop_y1] for pt in roi_data[key]]
            self.cropped_roi_objs[key] = np.array(cropped_roi_obj, dtype=np.int)

    @staticmethod
    def __is_crossed_line(tracker, cross_line):
        prev_rect = tracker[KEY_PREV_RECT]
        cur_rect = tracker[KEY_RECT]

        (x1, y1, w1, h1) = prev_rect
        (x2, y2, w2, h2) = cur_rect

        (p1x, p1y) = (x1 + w1 / 2, y1 + h1 * 2 / 3)
        (p2x, p2y) = (x2 + w2 / 2, y2 + h2 * 2 / 3)

        [[p3x, p3y], [p4x, p4y]] = cross_line

        sign1 = (p2x - p1x) * (p3y - p1y) - (p3x - p1x) * (p2y - p1y)
        sign2 = (p2x - p1x) * (p4y - p1y) - (p4x - p1x) * (p2y - p1y)
        sign3 = (p4x - p3x) * (p1y - p3y) - (p1x - p3x) * (p4y - p3y)
        sign4 = (p4x - p3x) * (p2y - p3y) - (p2x - p3x) * (p4y - p3y)
        if ((sign1 * sign2) < 0) and ((sign3 * sign4) <= 0):
            return True
        else:
            return False

    @staticmethod
    def __is_reached_order_pt(tracker, order_pt, thresh):
        t_x, t_y, t_w, t_h = tracker[KEY_RECT]
        t_pt = [t_x + t_w / 3, t_y + t_h / 2]
        if distance(pt1=t_pt, pt2=[order_pt[0][0], order_pt[0][1]]) < thresh:
            return True
        else:
            return False

    @staticmethod
    def __is_det_rect_in_roi_area(det_rect, roi_area):
        r_x, r_y, r_x2, r_y2 = det_rect
        r_w, r_h = r_x2 - r_x, r_y2 - r_y
        r_pt = (int(r_x + r_w / 2), int(r_y + r_h * 3 / 4))

        dis = int(cv2.pointPolygonTest(contour=roi_area, pt=r_pt, measureDist=True))
        return dis >= 0

    @staticmethod
    def __is_trk_rect_in_roi_area(trk_rect, roi_area):
        t_x, t_y, t_w, t_h = trk_rect
        t_pt = (int(t_x + t_w / 2), int(t_y + t_h * 2 / 3))

        dis = int(cv2.pointPolygonTest(contour=roi_area, pt=t_pt, measureDist=True))
        return dis >= 0

    def is_det_located_by_crossline(self, det, crop_sz):
        # new tracking object should be located on right side of crossline
        if crop_sz is None:
            roi_objs = self.roi_objs
        else:
            roi_objs = self.cropped_roi_objs

        crop_h, crop_w = crop_sz
        x, _, _, _ = (det[KEY_FRECT] * np.array([crop_w, crop_h, crop_w, crop_h])).astype(np.int)
        if x > max(roi_objs[ROI_CROSS_LINE][0][0], roi_objs[ROI_CROSS_LINE][1][0]):
            return True
        else:
            return False

    def crop_area(self, img):
        crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_to_detect
        return img[crop_y1:crop_y2, crop_x1:crop_x2]

    def filter_objects_in_roi(self, objects, crop_sz=None):
        if crop_sz is None:
            crop_w = self.img_h
            crop_h = self.img_h
            roi_objs = self.roi_objs
        else:
            crop_h, crop_w = crop_sz
            roi_objs = self.cropped_roi_objs

        dets_in_roi = []
        for obj_idx, obj in enumerate(objects):
            int_rect = (obj[KEY_FRECT] * np.array([crop_w, crop_h, crop_w, crop_h])).astype(dtype=np.int).tolist()

            if obj_idx > 0 and (obj_idx + 1) < len(objects):
                if is_overlap_rects(objects[obj_idx][KEY_FRECT], objects[obj_idx + 1][KEY_FRECT]) and \
                        self.is_det_located_by_crossline(det=objects[obj_idx], crop_sz=crop_sz):
                    continue

            if self.__is_det_rect_in_roi_area(det_rect=int_rect, roi_area=roi_objs[ROI_TRACK_AREA]):
                dets_in_roi.append(obj)

        return dets_in_roi

    def check_roi_events(self, trackers, fps=20.0, crop_sz=None):
        if crop_sz is None:
            roi_objs = self.roi_objs
        else:
            roi_objs = self.cropped_roi_objs

        tids_to_del = []
        tids = list(trackers.keys())

        # ----------------- check the roi events --------------------------------------------------------
        for index, tid in enumerate(tids):

            self.check_cross_line_state(tracker=trackers[tid], cross_line=roi_objs[ROI_CROSS_LINE])

            if not trackers[tid][KEY_CROSSLINE_STATE]:
                continue

            self.check_1st_order_point_state(trackers=trackers, cur_tid=tid, fps=fps,
                                             order_point=roi_objs[ROI_ORDER_POINT_1])

            self.check_2nd_order_point_state(trackers=trackers, cur_tid=tid, fps=fps,
                                             order_point=roi_objs[ROI_ORDER_POINT_2])

        # ----------------- trackers to remove ----------------------------------------------------------
        if len(tids) > 0 and trackers[tids[0]][KEY_RECT] is None:
            tids_to_del.append(tids[0])
        else:
            for idx in range(len(tids) - 1):
                x1, y1, w1, h1 = trackers[tids[idx]][KEY_RECT]
                x2, y2, w2, h2 = trackers[tids[idx + 1]][KEY_RECT]
                if is_overlap_rects(rect1=[x1, y1, x1 + w1, y1 + h1], rect2=[x2, y2, x2 + w2, y2 + h2]):
                    tids_to_del.append(tids[idx])

        for tid in tids:
            if tid in tids_to_del:
                continue
            _ratio = (tids.index(tid) + 1) * 0.5
            if trackers[tid][KEY_UPDATE_STATE] > UPDATE_STATE_THRESH * _ratio:
                tids_to_del.append(tid)
                continue
            if not self.__is_trk_rect_in_roi_area(trk_rect=trackers[tid][KEY_RECT], roi_area=roi_objs[ROI_TRACK_AREA]):
                tids_to_del.append(tid)
                continue

        # move disappeared trackers from tracking [state] to [lost]
        ret_hist = []
        for tid in tids_to_del:
            if trackers[tid][KEY_CROSSLINE_STATE] and \
                    trackers[tid][KEY_FRAMES_TO_ORDER_PT_1] != 0 and \
                    trackers[tid][KEY_FRAMES_TO_ORDER_PT_2] != 0:

                ret_hist.append(self.check_tracker_state_info(trackers=trackers, tid=tid, fps=fps))

            trackers.pop(tid)
        return ret_hist

    @staticmethod
    def __is_with_front_vehicle(trackers, tids, idx):
        if idx == 0:
            return False  # no exist in front of current vehicle
        else:
            front_tid = tids[idx - 1]
            if trackers[front_tid][KEY_UPDATE_STATE] == 0:
                return True

            # front_rect = trackers[front_tid][KEY_RECT]
            # front_pt = [front_rect[0] + front_rect[2] / 2, front_rect[1] + front_rect[3] / 2]
            #
            # cur_tid = tids[idx]
            # cur_rect = trackers[cur_tid][KEY_RECT]
            # cur_pt = [cur_rect[0] + cur_rect[2] / 2, cur_rect[1] + cur_rect[3] / 2]
            #
            # if front_pt[0] < cur_pt[0] and distance(pt1=front_pt, pt2=cur_pt) < ORDER_PT_RANGE_THRESH * 7:
            #     return True
            # else:
            #     return False

    def check_cross_line_state(self, tracker, cross_line):
        if not tracker[KEY_CROSSLINE_STATE] and self.__is_crossed_line(tracker=tracker, cross_line=cross_line):
            tracker[KEY_CROSSLINE_STATE] = True

    def check_1st_order_point_state(self, trackers, cur_tid, order_point, fps):
        tracker = trackers[cur_tid]
        tids = list(trackers.keys())

        cur_index = tids.index(cur_tid)

        if self.__is_reached_order_pt(tracker=tracker, order_pt=order_point, thresh=ORDER_PT_RANGE_THRESH):

            if tracker[KEY_FRAMES_TO_ORDER_PT_1] == 0:
                tracker[KEY_FRAMES_TO_ORDER_PT_1] = tracker[KEY_DWELLING_FRAMES]

                if self.__is_with_front_vehicle(trackers=trackers, tids=tids, idx=cur_index):
                    tracker[KEY_FRONT_FLAG] = True
                else:
                    tracker[KEY_FRONT_FLAG] = False

                if self.b_log:
                    logger.info("tid [{}] reached_1 [{}] front_vehicle [{}]".format(
                        cur_tid,
                        round(tracker[KEY_FRAMES_TO_ORDER_PT_1] / fps, 2),
                        tracker[KEY_FRONT_FLAG])
                    )

        else:
            if tracker[KEY_FRAMES_TO_ORDER_PT_1] != 0 and tracker[KEY_FRAMES_AT_ORDER_PT_1] == 0 and \
                    tracker[KEY_RECT][0] + tracker[KEY_RECT][2] / 2 < order_point[0][0]:

                tracker[KEY_FRAMES_AT_ORDER_PT_1] = \
                    tracker[KEY_DWELLING_FRAMES] - tracker[KEY_FRAMES_TO_ORDER_PT_1]

                if self.b_log:
                    logger.info("tid [{}] stopped_1 [{}]".format(
                        cur_tid,
                        round(tracker[KEY_FRAMES_AT_ORDER_PT_1] / fps, 2))
                    )

    def check_2nd_order_point_state(self, trackers, cur_tid, order_point, fps):
        tracker = trackers[cur_tid]

        if self.__is_reached_order_pt(tracker=tracker, order_pt=order_point, thresh=ORDER_PT_RANGE_THRESH * 1.5):

            if tracker[KEY_FRAMES_TO_ORDER_PT_2] == 0:
                tracker[KEY_FRAMES_TO_ORDER_PT_2] = tracker[KEY_DWELLING_FRAMES]

                if self.b_log:
                    logger.info("tid [{}] reached_2 [{}]".format(
                        cur_tid,
                        round(tracker[KEY_FRAMES_TO_ORDER_PT_2] / fps, 2)
                    ))

        else:
            if tracker[KEY_FRAMES_TO_ORDER_PT_2] != 0 and tracker[KEY_FRAMES_AT_ORDER_PT_2] == 0 and \
                    tracker[KEY_RECT][0] + tracker[KEY_RECT][2] / 2 < order_point[0][0]:

                tracker[KEY_FRAMES_AT_ORDER_PT_2] = \
                    tracker[KEY_DWELLING_FRAMES] - tracker[KEY_FRAMES_TO_ORDER_PT_2]

                if self.b_log:
                    logger.info("tid [{}] stopped_2 [{}]".format(
                        cur_tid,
                        round(tracker[KEY_FRAMES_AT_ORDER_PT_2] / fps, 2)
                    ))

    def check_tracker_state_info(self, trackers, tid, fps=20.0):
        tracker_state_info = [
            tid,
            round(trackers[tid][KEY_ENTER_FRAME_POS] / fps, 1),
            round(trackers[tid][KEY_FRAMES_TO_ORDER_PT_1] / fps, 1),
            round(trackers[tid][KEY_FRAMES_AT_ORDER_PT_1] / fps, 1),
            trackers[tid][KEY_FRONT_FLAG],
            round(trackers[tid][KEY_FRAMES_TO_ORDER_PT_2] / fps, 1),
            round(trackers[tid][KEY_FRAMES_AT_ORDER_PT_2] / fps, 1),
        ]

        if self.b_log:
            log_msg = "- tid [{}] ".format(tid) + \
                      "appearance [{}], ".format(round(trackers[tid][KEY_ENTER_FRAME_POS] / fps, 1)) + \
                      "time_to_order_1 [{}], ".format(round(trackers[tid][KEY_FRAMES_TO_ORDER_PT_1] / fps, 1)) + \
                      "time_at_order_1 [{}], ".format(round(trackers[tid][KEY_FRAMES_AT_ORDER_PT_1] / fps, 1)) + \
                      "front_flag [{}], ".format(trackers[tid][KEY_FRONT_FLAG]) + \
                      "time_to_order_2 [{}], ".format(round(trackers[tid][KEY_FRAMES_TO_ORDER_PT_2] / fps, 1)) + \
                      "time_at_order_2 [{}]".format(round(trackers[tid][KEY_FRAMES_AT_ORDER_PT_2] / fps, 1))
            logger.info(
                log_msg
            )
        return tracker_state_info
