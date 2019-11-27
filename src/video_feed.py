import os
import cv2
import queue
import sys
from threading import Thread
from utils.logger import logger


cur_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.join(cur_path, os.path.pardir)


class ProcQueue:
    @staticmethod
    def put(q, data, timeout=1.0):
        try:
            q.put(data, True, timeout)
        except queue.Full:
            q.empty()
            q.put(data, True, timeout)

    @staticmethod
    def get(q, timeout=1.0):
        """
        :param q:
        :param timeout:
        :return:
        """
        try:
            data = q.get(True, timeout)
        except queue.Empty:
            data = None
        return data


class VideoFeed:

    def __init__(self, stream_src):
        self.stream = cv2.VideoCapture(stream_src)

        if not self.stream.isOpened():
            sys.stdout.err("Cannot open stream.")

        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.stopped = False
        # print(self.fps, self.width, self.height)

        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)

    def __del__(self):
        self.stream.release()

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.capture_loop, args=()).start()
        Thread(target=self.proc_loop, args=()).start()
        return self

    def capture_loop(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
                # otherwise, read the next frame from the stream
            try:
                ret, frame = self.stream.read()
                ProcQueue().put(self.frame_queue, frame)
                cv2.imshow("frame", cv2.resize(frame, None, fx=0.25, fy=0.25))
                cv2.waitKey(1)
            except Exception as e:
                logger.warn("{}".format(e))
                continue

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def proc_streaming_marked(self):
        frame = ProcQueue().get(self.result_queue)
        if frame is not None:
            blob_ret, blob_jpeg = cv2.imencode('.jpg', frame)
            # convert bytes data
            result = blob_jpeg.tobytes()
            return result
        return None

    def proc_loop(self):
        while True:
            if self.stopped:
                return
            try:
                frame = ProcQueue().get(self.frame_queue)
                if frame is None:
                    continue

                result = self.proc(frame=frame)

                ProcQueue().put(self.result_queue, result)

            except Exception as e:
                logger.warn("{}".format(e))
                continue

    @staticmethod
    def proc(frame):
        # TODO : replace gray with analyized (with using cv engine "cv")frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray


if __name__ == '__main__':
    from utils.constant import VIDEO_SRC
    vid = VideoFeed(VIDEO_SRC)
    vid.start()
