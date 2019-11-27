import sys
import cv2
import os
from utils.constant import ROOT_DIR, ROI_TRACK_AREA, ROI_CROSS_LINE, ROI_ORDER_AREA_1, ROI_ORDER_POINT_1, \
    ROI_ORDER_AREA_2, ROI_ORDER_POINT_2


class SetRoi:
    def __init__(self, fst_frame):
        self.fst_frame = fst_frame

        height, width = int(self.fst_frame.shape[0]), int(self.fst_frame.shape[1])
        self.fst_frame = cv2.resize(self.fst_frame, (width, height))

        self.landmarks = []
        self.idx = 0
        self.img = None

    def draw_contour(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.landmarks.append((x, y))
            self.idx += 1
            if len(self.landmarks) == 1:
                cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)
            else:
                (x1, y1) = self.landmarks[-2]
                cv2.line(self.img, (x1, y1), (x, y), (0, 0, 255), 2)

    def set_roi(self, roi_type):
        roi = None

        sys.stdout.write("setting the {}\n".format(roi_type))
        if roi_type in [ROI_TRACK_AREA]:
            color = (0, 255, 0)
        elif roi_type in [ROI_CROSS_LINE, ROI_ORDER_POINT_1, ROI_ORDER_POINT_2]:
            color = (255, 0, 0)
        elif roi_type in [ROI_ORDER_AREA_1, ROI_ORDER_AREA_2]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 255)

        self.img = self.fst_frame.copy()
        height, width = self.img.shape[:2]

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.draw_contour)
        while True:
            cv2.imshow('Frame', self.img)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('r'):  # reset the whole points
                sys.stdout.write("{}: -- reset --\n".format(roi_type))
                self.landmarks = []
                self.idx = 0
                self.img = self.fst_frame.copy()

            elif k == ord('b'):  # back to the point
                if len(self.landmarks) > 0:
                    self.idx -= 1
                    sys.stdout.write("{}: -- back to the point -- {}\n".format(roi_type, self.idx))
                    del self.landmarks[-1]

                    self.img = self.fst_frame.copy()
                    for idx in range(len(self.landmarks) - 1):
                        x, y = self.landmarks[idx]
                        x1, y1 = self.landmarks[idx + 1]
                        cv2.line(self.img, (x, y), (x1, y1), color, 2)

            elif k == ord('q'):  # quit and save
                sys.stdout.write("{}: -- break and return --\n".format(roi_type))
                if len(self.landmarks) > 0:
                    for idx in range(len(self.landmarks)):
                        (x, y) = self.landmarks[idx]
                        x = float(x) / float(width)
                        y = float(y) / float(height)
                        self.landmarks[idx] = [x, y]
                    res = self.landmarks
                    self.landmarks = []

                    roi = res
                    self.fst_frame = self.img.copy()
                break

        cv2.destroyAllWindows()
        return roi


if __name__ == '__main__':
    img_path = os.path.join(ROOT_DIR, "first_frame.jpg")
    _img = cv2.imread(img_path)

    set_roi = SetRoi(fst_frame=_img)

    order_area_1 = set_roi.set_roi(roi_type=ROI_ORDER_AREA_1)
    order_area_2 = set_roi.set_roi(roi_type=ROI_ORDER_AREA_2)
    roi_obj = {
        ROI_ORDER_AREA_1: order_area_1,
        ROI_ORDER_AREA_2: order_area_2,
    }

    from utils.common import save_json
    save_json(json_path="roi.json", obj=roi_obj)
