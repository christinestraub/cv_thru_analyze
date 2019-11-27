import cv2
import os
import math
import numpy as np
import copy
from utils.constant import MODEL_DIR, KEY_FRECT
import utils.logger as logger


class DewarpUtils:

    def __init__(self, info, restore=False):
        src_sz = info['width'], info['height']
        start_angle = info['start_angle']
        cover_angle = info['cover_angle']

        self.r_factor = 0.3  # inner_radius / outer_radius
        self.scale_factor = 0.5
        self.restore = restore

        src_w, src_h = src_sz
        logger.info("\tsrc: w x h = {} x {}".format(src_w, src_h))

        r_outer = (src_h + src_w) / 4
        r_inner = self.r_factor * r_outer
        dst_h = int((r_outer - r_inner) * self.scale_factor)
        dst_w = int((r_outer * cover_angle * (np.pi / 180.0) * 0.6) * self.scale_factor)
        # 0.75 = ratio between width / height
        logger.info("\tdst: w x h = {} x {}, inner_radius : outer_radius = {} : {}".
                    format(dst_w, dst_h, r_inner, r_outer))

        self.radius = [r_inner, r_outer]
        self.src_sz = src_sz
        self.dst_sz = [dst_w, dst_h]
        self.start_angle_rad = float(start_angle) * np.pi / 180.0
        self.end_angle_rad = float(start_angle + cover_angle) * np.pi / 180.0

        self.model = self.__build_dewarp_map(model_info=info)

        self.dst_sz = [dst_w, dst_h]

    def get_dst_sz(self):
        return self.dst_sz

    def calibrate(self, img):
        return cv2.remap(src=img, map1=self.model['map1'], map2=self.model['map2'], interpolation=cv2.INTER_LINEAR)

    def get_dewarp_model(self):
        return self.model

    def __build_dewarp_map(self, model_info):
        logger.info("\tbuild dewarp map.")
        map_x_path = os.path.join(MODEL_DIR, model_info['model_paths'][0])
        map_y_path = os.path.join(MODEL_DIR, model_info['model_paths'][1])
        if os.path.exists(map_x_path) and os.path.exists(map_y_path) and not self.restore:
            logger.info("\tload the pre-calculated dewarp maps.")
            map_x = np.load(map_x_path)
            map_y = np.load(map_y_path)
        else:
            logger.info("\tcalculated dewarp maps.")
            src_w, src_h = self.src_sz
            dst_w, dst_h = self.dst_sz
            r_in, r_out = self.radius
            start_rad = self.start_angle_rad
            delta_rad = self.end_angle_rad - self.start_angle_rad

            cx = src_w / 2
            cy = src_h / 2

            map_x = np.zeros((dst_h, dst_w), dtype=np.float32)
            map_y = np.zeros((dst_h, dst_w), dtype=np.float32)
            for y in range(dst_h):
                for x in range(dst_w):
                    r = r_out - (r_out - r_in) * float(y) / float(dst_h)
                    theta = (float(x) / float(dst_w)) * delta_rad

                    src_x = cx - r * np.cos(theta + start_rad)
                    src_y = cy - r * np.sin(theta + start_rad)

                    # flip the dewrapped frame
                    map_x.itemset((y, x), int(src_x))
                    map_y.itemset((y, x), int(src_y))

            logger.info("\tsave dewarp maps.")
            np.save(map_x_path, map_x)
            np.save(map_y_path, map_y)
        return {'map1': map_x, 'map2': map_y}

    def cvt_dewarp_pt_2_fisheye_pt(self, pt):
        [x, y] = pt
        src_w, src_h = self.src_sz
        dst_w, dst_h = self.dst_sz
        r_in, r_out = self.radius
        start_rad = self.start_angle_rad
        delta_rad = self.end_angle_rad - self.start_angle_rad

        cx, cy = src_w / 2, src_h / 2

        r = r_out - (r_out - r_in) * float(y) / float(dst_h)
        theta = (float(x) / float(dst_w)) * delta_rad

        src_x = cx - r * np.cos(theta + start_rad)
        src_y = cy - r * np.sin(theta + start_rad)
        return [int(src_x), int(src_y)]

    def cvt_dewarp_rect_2_fisheye_rect(self, rect):
        x, y, w, h = rect
        inv_pts = []
        for pt in [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]:
            inv_pt = self.cvt_dewarp_pt_2_fisheye_pt(pt=pt)
            inv_pts.append(inv_pt)

        area_sz = cv2.contourArea(contour=np.array(inv_pts).astype(np.uint))
        contour_conner1 = np.array(inv_pts).min(axis=0)
        contour_conner2 = np.array(inv_pts).max(axis=0)
        contour_cen_pt = np.average(np.array(inv_pts), axis=0)

        contour_w = contour_conner2[0] - contour_conner1[0]
        contour_h = contour_conner2[1] - contour_conner1[1]

        ratio = contour_w / contour_h
        new_rc_h = math.sqrt(area_sz / ratio)
        new_rc_w = new_rc_h * ratio

        new_rc = [contour_cen_pt[0] - new_rc_w / 2,
                  contour_cen_pt[1] - new_rc_h / 2,
                  contour_cen_pt[0] + new_rc_w / 2,
                  contour_cen_pt[1] + new_rc_h / 2]
        return new_rc

    def cvt_dewarp_dets_2_fisheye_dets(self, dets):
        src_w, src_h = self.src_sz
        dst_w, dst_h = self.dst_sz

        new_dets = []
        for _det in dets:
            new_det = copy.deepcopy(_det)

            [x1, y1, x2, y2] = new_det[KEY_FRECT] * np.array([dst_w, dst_h, dst_w, dst_h])
            [n_x1, n_y1, n_w, n_h] = self.cvt_dewarp_rect_2_fisheye_rect(rect=[x1, y1, x2 - x1, y2 - y1])
            new_frect = [n_x1, n_y1, n_w, n_h] / np.array([src_w, src_h, src_w, src_h])
            new_det[KEY_FRECT] = new_frect

            new_dets.append(new_det)
        return new_dets
