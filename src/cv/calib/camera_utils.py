import re
import sys
import json

from src.cv.calib.dewarp_utils import DewarpUtils
from src.cv.calib.distort_utils import DistortUtils

from utils.constant import CAMERA_INFO_FILE, DISTORT_INFO_FILE, DEWARP_INFO_FILE
import utils.logger as logger


def str2val(str_val):
    str_val = str_val.replace(' ', '')

    tmp = str_val
    tmp = tmp.replace(',', '0')
    tmp = tmp.replace('.', '0')

    digits = re.findall('\d+', tmp)
    if len(digits) == 0:
        return 0

    # find the max len digits
    max_di = ""
    for di in digits:
        if len(di) > len(max_di):
            max_di = di

    pos = tmp.find(max_di)
    if pos != -1:
        tmp = str_val[pos: pos + len(max_di)]
        if tmp.find('.') != -1 or tmp.find(',') != -1:
            return float(tmp)
        else:
            return int(tmp)
    else:
        return 0


def save_info(json_path, info):
    with open(json_path, 'w') as jp:
        json.dump(info, jp, indent=2)


def load_info(json_path):
    with open(json_path, 'r') as jp:
        info = json.load(jp)
        return info


def load_cam_info(json_path):
    """
        info = {
                "camera_type": "XDV360",
                "spec": {
                    "FOV": 220,
                    "focal_length": "F2.0 f=1.1mm",
                    "sensor": "8M CMOS",
                    "channels": [
                        {
                            "id": 2,
                            "name": "",
                            "fps": 60,
                            "resolution": {"width": 1440, "height": 1440}
                        }
                    ]
                }

    :param json_path:
    :return: dict
    """
    try:
        with open(json_path, 'r') as jp:
            info = json.load(jp)
            return info
    except Exception as e:
        logger.warn("{}".format(e))
        sys.exit(0)


class Camera:
    def __init__(self, debug=False, overwrite=False):
        self.debug = debug

        self.cam_info = load_info(json_path=CAMERA_INFO_FILE)
        dewarp_info = load_info(json_path=DEWARP_INFO_FILE)
        distort_info = load_info(json_path=DISTORT_INFO_FILE)

        logger.info("Configure fisheye distort.")
        self.distort = DistortUtils(info=distort_info, restore=overwrite)
        logger.info("Configure fisheye dewarp.")
        self.dewarp = DewarpUtils(info=dewarp_info, restore=overwrite)
