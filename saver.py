import datetime
import os
import time
import cv2
import queue
import threading

from utils.constant import VIDEO_SRC, SAVE_DIR, VIDEO_EXT, PREFIX_SAVED_VIDEO, VIDEO_ENCODE, ERASE_TIME
from utils.common import logger


erase_time = datetime.datetime.strptime(ERASE_TIME.split(' ')[1], '%H:%M')
erase_weekday = ERASE_TIME.split(' ')[0]

timeout = 0.5
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


class Saver(threading.Thread):

    _b_stop = threading.Event()
    _last_erase_time = 0

    def __init__(self):
        super().__init__()

        self.video_src = VIDEO_SRC

        logger.info('Start recording (video source - {})'.format(self.video_src))

        self.cap = None

        self.saver = None
        self.fourcc = cv2.VideoWriter_fourcc(*VIDEO_ENCODE)

        self.frame_queue = queue.Queue(maxsize=100)

        self._b_stop.clear()

    def capture_loop(self):
        # cnt = 0
        while True:
            ret, frame = self.cap.read()

            _cur_t = datetime.datetime.utcnow()
            if not ret or (_cur_t.minute == 0 and _cur_t.second == 0):  # restart the recording once start of hour
                self.frame_queue.put(None)
                logger.info('end the capture_loop')
                break

            # cnt += 1
            # if cur_t.minute % 1 == 0 and cur_t.second == 0 and cnt > 100:
            #     logger.log(cnt)
            #     cnt = 0

            try:
                self.frame_queue.put(frame, True, timeout)
                cv2.waitKey(1)

            except queue.Full:
                logger.warn("full")
                time.sleep(timeout)
                continue

            time.sleep(0.01)
        return

    def save_loop(self):
        cnt1 = 0
        cnt2 = 0

        while True:
            try:
                frame = self.frame_queue.get(True, timeout)
                cnt1 += 1

                if frame is None:
                    logger.info('end the save_loop')
                    break
                else:
                    self.saver.write(frame)
                    cnt2 += 1

                    _cur_t = datetime.datetime.utcnow()
                    if _cur_t.minute % 10 == 0 and _cur_t.second == 0 and cnt2 > 100:
                        logger.info("\tcaptured: {}, saved: {}".format(cnt1, cnt2))
                        cnt1 = 0
                        cnt2 = 0

            except queue.Empty:
                logger.info("empty")
                time.sleep(timeout)
                continue
            time.sleep(0.01)
        return

    def record_an_hour(self):
        logger.info(" ")
        self._b_stop.clear()

        cur_time = datetime.datetime.utcnow()
        save_fn = PREFIX_SAVED_VIDEO + cur_time.strftime("%Y-%m-%d_%H-%M-%S") + VIDEO_EXT
        # 'SAVE_2011-11-03_00-00-00.avi'

        self.cap = cv2.VideoCapture(self.video_src)
        if not self.cap.isOpened():
            logger.error("\ncannot open video - {}".format(self.video_src))
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fps = min(fps, 25.0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_path = os.path.join(SAVE_DIR, save_fn)
        self.saver = cv2.VideoWriter(save_path, self.fourcc, fps, (width, height))

        logger.info(">>> start recording : {}".format(save_fn))
        logger.info("save_path: {}".format(save_path))
        logger.info("\t{}, {}, {}".format(fps, width, height))

        threads = [threading.Thread(target=self.capture_loop, args=()),
                   threading.Thread(target=self.save_loop, args=())]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        logger.info(">>> finished recording : {}".format(save_fn))

    def run(self):
        while True:
            self.record_an_hour()

            self.stop()

            time.sleep(0.1)
            continue

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        if self.saver is not None:
            self.saver.release()

        cv2.destroyAllWindows()
        self._b_stop.set()

    def check_erase(self):
        now = datetime.datetime.utcnow()
        if erase_weekday == weekdays[now.weekday()] and now.hour == erase_time.hour:
            if time.time() - self._last_erase_time > 3600:
                paths_to_del = []
                for fn in os.listdir(SAVE_DIR):
                    if os.path.splitext(fn)[1] != VIDEO_EXT:
                        continue
                    if fn.find(PREFIX_SAVED_VIDEO) != -1:
                        continue
                    paths_to_del.append(os.path.join(SAVE_DIR, fn))

                for path in paths_to_del:
                    os.remove(path)
                    time.sleep(1)

                self._last_erase_time = time.time()
                return True

        return False

    def convert_cvi_to_mp4(self, path):
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = min(fps, 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # fourcc = cv2.VideoWriter_fourcc(*'x264')
        saver = cv2.VideoWriter("output.mp4", self.fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            cv2.imshow("frame", frame)
            saver.write(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        saver.release()
        cap.release()


if __name__ == '__main__':

    svr = Saver()
    svr.run()
    # Saver().convert_cvi_to_mp4(path="sample.avi")
