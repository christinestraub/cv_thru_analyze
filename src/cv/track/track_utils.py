import dlib
import cv2
import sys
import numpy as np


from utils.constant import TRK_DLIB, TRK_CSRT, TRK_MOSSE, KEY_FRECT, KEY_RECT, KEY_PREV_RECT, KEY_TRACKER, \
    KEY_LABEL, KEY_CONFIDENCE, KEY_UPDATE_STATE, KEY_DWELLING_FRAMES, KEY_ENTER_FRAME_POS, \
    GOOD_TRACK_QUALITY, KEY_CROSSLINE_STATE, KEY_SPEED, \
    KEY_FRAMES_AT_ORDER_PT_1, KEY_FRAMES_TO_ORDER_PT_1, KEY_FRAMES_AT_ORDER_PT_2, KEY_FRAMES_TO_ORDER_PT_2, \
    KEY_FRONT_FLAG


import utils.logger as logger
from src.analyze.roi_utils import distance

# trackers = {
#     'tid': {
#         'rect': 'rect of current tracker',
#         'tracker': 'tracker'}
# }


class TrackUtils:
    def __init__(self, trk_type=TRK_DLIB):
        self.trk_type = trk_type

    def create_tracker(self, trk_img, det, frame_pos=0):
        img_h, img_w = trk_img.shape[:2]
        [x, y, x2, y2] = (det[KEY_FRECT] * np.array([img_w, img_h, img_w, img_h])).astype(np.int)
        w, h = x2 - x, y2 - y

        # dlib tracker
        if self.trk_type == TRK_DLIB:
            tracker = dlib.correlation_tracker()
            # tracker.start_track(trk_img, dlib.rectangle(x, y, x2, y2))
            tracker.start_track(trk_img, dlib.rectangle(int(x), int(y), int(x2), int(y2)))
        # cv trackers
        elif self.trk_type == TRK_MOSSE:
            tracker = cv2.TrackerMOSSE_create()
            tracker.init(trk_img, (x, y, w, h))
        elif self.trk_type == TRK_CSRT:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(trk_img, (x, y, w, h))
        else:
            logger.error("invalid tracker type")
            sys.exit(1)

        return {
            KEY_TRACKER: tracker,
            # KEY_COLOR: det[KEY_COLOR],
            KEY_LABEL: det[KEY_LABEL],
            KEY_CONFIDENCE: det[KEY_CONFIDENCE],
            KEY_RECT: [x, y, w, h],
            KEY_PREV_RECT: [x, y, w, h],
            KEY_UPDATE_STATE: 0,
            KEY_DWELLING_FRAMES: 0,
            KEY_ENTER_FRAME_POS: frame_pos,
            KEY_CROSSLINE_STATE: False,
            KEY_SPEED: [0, 0],
            KEY_FRAMES_AT_ORDER_PT_1: 0,
            KEY_FRAMES_TO_ORDER_PT_1: 0,
            KEY_FRAMES_AT_ORDER_PT_2: 0,
            KEY_FRAMES_TO_ORDER_PT_2: 0,
            KEY_FRONT_FLAG: "n/a"
        }

    def __update_tracker(self, tracker, trk_img):
        suc = False
        rect = None
        if self.trk_type in TRK_DLIB:
            quality = tracker.update(trk_img)
            if quality > GOOD_TRACK_QUALITY:
                tracked_position = tracker.get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                suc = True
                rect = [t_x, t_y, t_w, t_h]
        else:  # in CV_TRKs
            suc, (t_x, t_y, t_w, t_h) = tracker.update(trk_img)
            if suc:
                rect = [int(t_x), int(t_y), int(t_w), int(t_h)]
        return suc, rect

    def keep_trackers(self, trk_img, trackers):

        for tid in trackers.keys():
            suc, cur_rect = self.__update_tracker(tracker=trackers[tid][KEY_TRACKER], trk_img=trk_img)
            if not suc:
                continue

            trackers[tid][KEY_DWELLING_FRAMES] += 1
            trackers[tid][KEY_PREV_RECT] = trackers[tid][KEY_RECT]
            trackers[tid][KEY_RECT] = ((np.array(cur_rect) + np.array(trackers[tid][KEY_RECT])) / 2.0).tolist()

            trackers[tid][KEY_SPEED] = self.calc_speed(tracker=trackers[tid])

    def __update_tracker_with_det(self, tracker, det, trk_img):
        # backup prev states
        prev_rect = tracker[KEY_RECT]
        prev_dwelling_frames = tracker[KEY_DWELLING_FRAMES]
        prev_enter_frame_pos = tracker[KEY_ENTER_FRAME_POS]
        prev_crossline_state = tracker[KEY_CROSSLINE_STATE]

        prev_frames_to_order_1 = tracker[KEY_FRAMES_TO_ORDER_PT_1]
        prev_frames_at_order_1 = tracker[KEY_FRAMES_AT_ORDER_PT_1]
        prev_frames_to_order_2 = tracker[KEY_FRAMES_TO_ORDER_PT_2]
        prev_frames_at_order_2 = tracker[KEY_FRAMES_AT_ORDER_PT_2]

        prev_speed = tracker[KEY_SPEED]

        prev_front_flag = tracker[KEY_FRONT_FLAG]

        """ --------------- create tracker from det ------------------------------"""
        new_tracker = self.create_tracker(trk_img=trk_img, det=det)
        """ --------------------------------------------------------------------- """

        # recover the prev states
        new_tracker[KEY_PREV_RECT] = prev_rect
        new_tracker[KEY_UPDATE_STATE] = 0
        new_tracker[KEY_DWELLING_FRAMES] += (prev_dwelling_frames + 1)
        new_tracker[KEY_ENTER_FRAME_POS] = prev_enter_frame_pos

        new_tracker[KEY_CROSSLINE_STATE] = prev_crossline_state

        new_tracker[KEY_SPEED] = self.calc_speed(tracker=new_tracker, prev_speed=prev_speed)

        new_tracker[KEY_FRAMES_TO_ORDER_PT_1] = prev_frames_to_order_1
        new_tracker[KEY_FRAMES_AT_ORDER_PT_1] = prev_frames_at_order_1
        new_tracker[KEY_FRAMES_TO_ORDER_PT_2] = prev_frames_to_order_2
        new_tracker[KEY_FRAMES_AT_ORDER_PT_2] = prev_frames_at_order_2

        new_tracker[KEY_FRONT_FLAG] = prev_front_flag
        return new_tracker

    def upgrade_trackers(self, dets, trk_img, trackers):
        remain_dets = []

        img_h, img_w = trk_img.shape[:2]
        for tid in trackers.keys():
            trackers[tid][KEY_UPDATE_STATE] += 1

        # calculate the average distance between det and tracker
        for det in reversed(dets):
            [x, y, x2, y2] = (det[KEY_FRECT] * np.array([img_w, img_h, img_w, img_h])).astype(np.int)
            w, h = x2 - x, y2 - y
            det_pt = [x + w / 4, y + h / 2]

            min_dis = img_w
            min_tid = None
            for tid in trackers.keys():
                t_x, t_y, t_w, t_h = trackers[tid][KEY_RECT]
                t_x2, t_y2 = t_x + t_w, t_y + t_h
                trk_pt = [(t_x + t_x2) / 2, (t_y + t_y2) / 2]

                _dis = distance(pt1=det_pt, pt2=trk_pt)
                if _dis < min_dis and det_pt[0] < trk_pt[0]:  # det will be located left side by tracking object
                    min_dis = _dis
                    min_tid = tid

            if min_tid is not None and trackers[min_tid][KEY_UPDATE_STATE] != 0 and min_dis < 2 * w:
                trackers[min_tid] = self.__update_tracker_with_det(tracker=trackers[min_tid], det=det, trk_img=trk_img)
            else:
                remain_dets.append(det)

        return remain_dets

    @staticmethod
    def calc_speed(tracker, prev_speed=None):
        if prev_speed is None:
            prev_movements = tracker[KEY_SPEED]
        else:
            prev_movements = prev_speed

        prev_rect = tracker[KEY_PREV_RECT]
        cur_rect = tracker[KEY_RECT]
        prev_loc = [prev_rect[0] + prev_rect[2] / 2.0, prev_rect[1] + prev_rect[3] / 2.0]
        cur_loc = [cur_rect[0] + cur_rect[2] / 2.0, cur_rect[1] + cur_rect[3] / 2.0]

        cur_movements = np.array(cur_loc) - np.array(prev_loc)

        return ((np.array(prev_movements) * 0.7 + np.array(cur_movements) * 0.3) / 2.0).tolist()

    # def init_trackers(self, dets, trk_img):
    #     trackers = {}
    #     for i, det in enumerate(dets):
    #         trackers[i] = self.create_tracker(trk_img=trk_img, det=det)
    #     return trackers
