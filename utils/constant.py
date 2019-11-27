import os

_cur_dir = os.path.dirname(os.path.realpath(__file__))

# [ROOT]
ROOT_DIR = os.path.join(_cur_dir, os.pardir)

# [CONFIG]
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
INI_FILE = os.path.join(CONFIG_DIR, 'config.ini')

# config/infos
INFO_DIR = os.path.join(CONFIG_DIR, 'infos')
SECTION_INFO_FILE = os.path.join(INFO_DIR, 'section_info.json')
CAMERA_INFO_FILE = os.path.join(INFO_DIR, 'camera_info.json')
DISTORT_INFO_FILE = os.path.join(INFO_DIR, 'distort_info.json')
DEWARP_INFO_FILE = os.path.join(INFO_DIR, 'dewarp_info.json')

# config/logs
LOG_DIR = os.path.join(CONFIG_DIR, 'logs')

# config/models
MODEL_DIR = os.path.join(CONFIG_DIR, 'models')
DEWARP_MODEL = [os.path.join(MODEL_DIR, fn) for fn in ['dewarp_map1.npy', 'dewarp_map2.npy']]
DISTORT_MODEL = [os.path.join(MODEL_DIR, fn) for fn in ['distort_map1.npy', 'distort_map2.npy']]

# [SAVE]
SAVE_DIR = os.path.join(ROOT_DIR, 'data')

for _dir in [CONFIG_DIR, INFO_DIR, MODEL_DIR, LOG_DIR]:
    if not os.path.exists(_dir):
        os.makedirs(_dir)


# [KEYS]
# detects keys
KEY_FRECT = "float_rect"    # [x, y, x1, y1], range(float) = [0.0, 1.0], [0.0, 1.0]
KEY_COLOR = "color"
KEY_LABEL = "label"
KEY_CONFIDENCE = "confidence"
# Tracker Keys
KEY_RECT = "rect"           # [x, y, w, h], range(int) = [0, img_width], [0, img_height]
KEY_TID = "tracker_id"
KEY_TRACKER = "tracker"
KEY_INV_PT = "inverse_point"
KEY_INV_RECT = "inverse_rect"
KEY_PREV_RECT = "prev_rect"

# [TRACKERS]
TRK_DLIB = "DLIB"
TRK_CSRT = "CSRT"
TRK_MOSSE = "MOSSE"

# [DETECTORS]
DET_YOLO2 = "YOLO2"
DET_YOLO3 = "YOLO3"
DET_SSD = "SSD"
DET_VGG = "VGG"
DET_DARKNET2 = "DARKNET2"
DET_DARKNET3 = "DARKNET3"

KEY_UPDATE_STATE = "update_status"
KEY_DWELLING_FRAMES = "dwelling_frames"
KEY_ENTER_FRAME_POS = "entered_frame_pos"

KEY_CROSSLINE_STATE = "crossline_state"
KEY_SPEED = "speed"

KEY_FRAMES_TO_ORDER_PT_1 = "frames_to_order_pt_1"
KEY_FRAMES_AT_ORDER_PT_1 = "frames_at_order_pt_1"
KEY_FRAMES_TO_ORDER_PT_2 = "frames_to_order_pt_2"
KEY_FRAMES_AT_ORDER_PT_2 = "frames_at_order_pt_2"

KEY_FRONT_FLAG = "exist_front_vehicle"

# [ROIs]
ROI_TRACK_AREA = "track_area"
ROI_CROSS_LINE = "cross_line"
ROI_ORDER_AREA_1 = "order_area_1"
ROI_ORDER_POINT_1 = "order_point_1"
ROI_ORDER_AREA_2 = "order_area_2"
ROI_ORDER_POINT_2 = "order_point_2"

# thresholds
UPDATE_STATE_THRESH = 10

OVERLAP_THRESH = 30

ORDER_PT_RANGE_THRESH = 30

#
DAY_TIME = range(7, 20)  # 7:00am to 8:00pm
GOOD_TRACK_QUALITY = 5

#
PREFIX_SAVED_VIDEO = "SAVE_"
PREFIX_RESULT_VIDEO = "RES_"

#
SKIP1 = 1
SKIP2 = 20

#
SCALE_FACTOR = 0.5

from settings import *
