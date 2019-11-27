import cv2
import os
import datetime
import time

from src.cv.calib.camera_utils import Camera
from src.cv.detect.detect_utils import DetectUtils
# from src.cv.detect.detect_utils import filter_confidence_dets
from src.cv.draw.draw_utils import DrawUtils
from src.cv.track.track_utils import TrackUtils
from src.analyze.roi_utils import RoiUtils
from utils.common import logger

from utils.constant import TRACKER, DETECTOR, SKIP1, SKIP2, \
    B_SAVE_VIDEO, B_SHOW_VIDEO, PREFIX_SAVED_VIDEO, PREFIX_RESULT_VIDEO, SCALE_FACTOR


class DriveThru:
    def __init__(self):

        self.cap = None

        self.cam = Camera(overwrite=False)

        self.detector = DetectUtils(det_mode=DETECTOR, score_threshold=0.1).detector

        self.tracker = TrackUtils(trk_type=TRACKER)

        self.drawer = DrawUtils(b_log=False)

        self.roi = None

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.saver = None

        self.scale_factor = SCALE_FACTOR

    def run(self, video_path):
        # ---------------- initialize -------------------------------------------------
        self.cap = cv2.VideoCapture(video_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_factor
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_factor
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.info("video infos:\n" +
                    "\tfps: {}\n".format(fps) +
                    "\twidth: {}\n".format(width) +
                    "\theight: {}\n".format(height) +
                    "\tnum_frames: {}\n".format(num_frames))

        self.roi = RoiUtils(img_sz=[height, width], b_log=True)

        output_path = video_path.replace(PREFIX_SAVED_VIDEO, PREFIX_RESULT_VIDEO)
        self.saver = cv2.VideoWriter(output_path, self.fourcc, fps, (int(width), int(height)))

        # ---------------- validate --------------------------------------------------
        # model_h, model_w = self.cam.distort.get_map_sz()
        # if model_h != height or model_w != width:
        #     logger.info(
        #         "input frame size {}x{} should be matched with camera model {}x{}".format(width, height, model_w,
        #                                                                                   model_h))
        #     return
        # ----------------------------------------------------------------------------

        vehicles = {}
        current_id = 0
        log_history = []
        logger.info("start analyzing...")
        frm_cnt = -1
        is_first_frame = True

        __last_fps_t = 0

        while True:
            frm_cnt += 1
            ret, raw_frame = self.cap.read()

            if not ret:
                break

            # if frm_cnt == (fps * 300):
            #     break

            if frm_cnt % int(fps * 600) == 0:
                print("{} / {}".format(frm_cnt, num_frames))
                __dur = time.time() - __last_fps_t
                __fps = (fps * 600) / __dur
                print("fps: {}".format(round(__fps, 2)))
                __last_fps_t = time.time()

                # print("time pos: {}".format(round(frm_cnt / fps, 2)))

            # skip1
            if frm_cnt % SKIP1 != 0:
                continue

            # -------------------- pre-processing (undistorting) -------------------------------------------------------
            frame = cv2.resize(raw_frame, None, fx=self.scale_factor, fy=self.scale_factor)
            crop = self.roi.crop_area(img=frame)

            # -------------------- detect and tracking -----------------------------------------------------------------
            if frm_cnt % SKIP2 == 0:  # detect the object

                # detect the object ----------------------------------------------------
                dets = self.detector.detect(img=crop)

                # confidence_dets = filter_confidence_dets(dets, car_thresh=0.1, truck_thresh=0.35)  # no need in yolov3

                # only in roi(tracking area) -------------------------------------------
                dets_in_roi = self.roi.filter_objects_in_roi(objects=dets, crop_sz=crop.shape[:2])

                # update trackers ------------------------------------------------------
                if is_first_frame:
                    is_first_frame = False
                    cv2.imwrite("first_frame.jpg", frame)

                    # remain_dets = dets_in_roi.copy()
                    remain_dets = list(dets_in_roi)

                    for det in remain_dets:
                        current_id += 1
                        vehicles[current_id] = self.tracker.create_tracker(trk_img=crop, det=det, frame_pos=frm_cnt)
                else:
                    remain_dets = self.tracker.upgrade_trackers(dets=dets_in_roi, trk_img=crop, trackers=vehicles)

                    remain_dets_img = self.drawer.show_objects(img=crop, objects=dets_in_roi)
                    remain_dets_img = self.drawer.show_roi(img=remain_dets_img, roi_objs=self.roi.cropped_roi_objs)
                    cv2.imshow("remain_dets_img", remain_dets_img)
                    cv2.waitKey(1)

                    for det in remain_dets:
                        if self.roi.is_det_located_by_crossline(det=det, crop_sz=crop.shape[:2]):
                            current_id += 1
                            vehicles[current_id] = self.tracker.create_tracker(trk_img=crop, det=det, frame_pos=frm_cnt)

            else:
                # keep trackers --------------------------------------------------------
                self.tracker.keep_trackers(trk_img=crop, trackers=vehicles)

            # -------------------- show the frame ----------------------------------------------------------------------
            # recover the cropped trackers
            show_img = self.drawer.show_roi(img=frame, roi_objs=self.roi.roi_objs)
            show_img = self.drawer.show_trackers(trk_img=show_img, trackers=vehicles, mode="rect", fps=fps,
                                                 offset=self.roi.crop_to_detect[:2])
            if B_SHOW_VIDEO:
                # showing the result
                cv2.imshow("result", cv2.resize(show_img, None, fx=0.7, fy=0.7))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('n'):  # next
                    pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    pos += 100
                    frm_cnt += 100
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    print(frm_cnt)
                elif key == ord('p'):  # prev
                    pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    pos -= max(0, pos - 100)
                    frm_cnt -= 100
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    print(frm_cnt)

            # -------------------- check roi events --------------------------------------------------------------------
            event_hist = self.roi.check_roi_events(trackers=vehicles, fps=fps, crop_sz=crop.shape[:2])
            if len(event_hist) > 0:
                log_history.append(event_hist)
                print("new event at {}".format(round(frm_cnt / fps, 2)))

            if B_SAVE_VIDEO:  # rename
                self.saver.write(show_img)

        self.export_to_csv(video_path=video_path, history=log_history)
        self.release()

        if not B_SAVE_VIDEO:  # rename
            new_path = video_path.replace(PREFIX_SAVED_VIDEO, PREFIX_RESULT_VIDEO)
            try:
                os.rename(video_path, new_path)
            except Exception as e:
                print(e)
                os.remove(new_path)
                os.rename(video_path, new_path)
            time.sleep(1)

        return frm_cnt

    def release(self):
        if self.saver is not None:
            self.saver.release()
        if self.cap is not None:
            self.cap.release()

    def export_to_csv(self, video_path, history):
        s_time = self.get_time_from_fn(file_path=video_path)
        s_time = s_time.timestamp()

        csv_path = os.path.join(os.path.splitext(video_path)[0] + ".csv")

        lines = "ID, AppearanceTime (sec), TimeToOrderPoint1 (sec), TimeAtOrderPoint1 (sec), " \
                "Exist Vehicle in Front (T/F), TimeToOrderPoint2 (sec), TimeAtOrderPoint2 (sec) \n"
        for events in history:
            for event in events:
                tid, time_appearance, time_to_order_1, time_at_order_1, front_flag, \
                    time_to_order_2, time_at_order_2 = event

                # 2007-04-05T14:30:20 | ISO8601 standard
                epoch_t_appearance = s_time + time_appearance
                time_appearance = datetime.datetime.fromtimestamp(epoch_t_appearance).strftime('%Y-%m-%d %H:%M:%S')

                line = "{}, {}, {}, {}, {}, {}, {}\n".format(
                    tid,
                    time_appearance,
                    round(time_to_order_1, 2),
                    round(time_at_order_1, 2),
                    front_flag,
                    round(time_to_order_2, 2),
                    round(time_at_order_2, 2)
                )

                lines += line
        # save the csv file
        with open(csv_path, 'w') as fp:
            fp.write(lines)

    @staticmethod
    def get_time_from_fn(file_path):
        try:
            _, fn = os.path.split(file_path)
            base, ext = os.path.splitext(fn)
            base = base.replace(PREFIX_RESULT_VIDEO, '').replace(PREFIX_SAVED_VIDEO, '')
            t_obj = datetime.datetime.strptime(base, '%Y-%m-%d_%H-%M-%S')
        except Exception as e:
            logger.warn("{}".format(e))
            t_obj = datetime.datetime.utcnow()
        return t_obj


if __name__ == '__main__':
    dt = DriveThru()

    from utils.constant import SAVE_DIR, VIDEO_EXT

    paths = [os.path.join(SAVE_DIR, fn) for fn in os.listdir(SAVE_DIR) if
             os.path.splitext(fn)[1] == VIDEO_EXT and fn.find(PREFIX_SAVED_VIDEO) != -1]
    paths.sort()
    for path in paths:
        logger.info(os.path.split(path)[1])
        s_t = time.time()
        frames = dt.run(video_path=path)
        e_t = time.time()
        print(e_t - s_t)
