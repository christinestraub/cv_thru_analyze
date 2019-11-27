import cv2
import os
import numpy as np
import utils.logger as logger
from utils.constant import MODEL_DIR


class DistortUtils:
    def __init__(self, info, debug=False, restore=False):
        self.debug = debug

        self.scale_range_min = 0.5
        self.scale_range_max = 1.0
        self.scale_range_step = 0.05

        self.restore = restore
        self.model = self.__build_distort_map(model_info=info)

    def __build_distort_map(self, model_info):
        logger.info("\tbuild distort map.")
        map_x_path = os.path.join(MODEL_DIR, model_info['model_paths'][0])
        map_y_path = os.path.join(MODEL_DIR, model_info['model_paths'][1])
        if os.path.exists(map_x_path) and os.path.exists(map_y_path) and not self.restore:
            logger.info("\tload the pre-calculated distort maps.")
            map_x = np.load(map_x_path)
            map_y = np.load(map_y_path)
        else:
            logger.info("\tcalculate distort maps.")
            try:
                width = model_info['width']
                height = model_info['height']
                scale = model_info['scale_factor']
                k = np.array(model_info['camera_matrix'])
                d = np.array(model_info['distort_vector'])
                k_new = k.copy()
                k_new[(0, 1), (0, 1)] *= scale
                map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K=k, D=d, R=None, P=k_new, size=(width, height),
                                                                   m1type=cv2.CV_16SC2)
            except Exception as e:
                logger.error(e)
                logger.info("\tcalculate the camera distort model info")
                model_info = self.__calc_opt_model(model_info=model_info)

                width = model_info['width']
                height = model_info['height']
                scale = model_info['scale_factor']
                k = np.array(model_info['camera_matrix'])
                d = np.array(model_info['distort_vector'])
                k_new = k.copy()
                k_new[(0, 1), (0, 1)] *= scale
                map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K=k, D=d, R=None, P=k_new, size=(width, height),
                                                                   m1type=cv2.CV_16SC2)

            logger.info("\tsave distort maps.")
            np.save(map_x_path, map_x)
            np.save(map_y_path, map_y)
        return {'map1': map_x, 'map2': map_y}

    def __calc_opt_model(self, model_info):
        # Intrinsic parameters(Camera Matrix):
        width = model_info['width']
        height = model_info['height']
        cen_x = width / 2.0
        cen_y = height / 2.0

        # camera matrix
        logger.info("\tconfigure the camera matrix")
        camera_mat = [[1050, 0., cen_x],  # [1050, 0., cen_x]
                      [0., 1050, cen_y],  # [0.,  1050,    cy], origin
                      [0., 0., 1.]]
        k = np.array(camera_mat)
        logger.info(str(k))

        # distort vector
        logger.info("\tconfigure the distort vector")
        distort_vec = [0., 0., 0., 0.]
        d = np.array(distort_vec)
        logger.info(str(d))

        # find the optimize scale factor
        logger.info("\tfind the optimized scale factor in [{}, {}] with step {}".format(
            self.scale_range_min,
            self.scale_range_max,
            self.scale_range_step))
        ones = np.ones((height, width, 3), dtype=np.uint8) * 255
        _step = self.scale_range_step
        _min = self.scale_range_min
        _max = self.scale_range_max

        scale_factor = self.scale_range_min
        for scale in np.arange(_min, _max, _step):
            logger.info("\t\tscale : {}".format(scale))

            knew = k.copy()
            knew[(0, 1), (0, 1)] *= scale
            _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(K=k, D=d, R=None, P=knew, size=(width, height),
                                                               m1type=cv2.CV_16SC2)
            # camera calibration
            ones_distort = cv2.remap(src=ones, map1=_map1, map2=_map2, interpolation=cv2.INTER_LINEAR)
            ones_distort_gray = cv2.cvtColor(ones_distort, cv2.COLOR_BGR2GRAY)

            mask = cv2.threshold(ones_distort_gray, 1, 255, cv2.THRESH_BINARY)[1]

            if self.debug:
                cv2.imshow("ones_distort", ones_distort)
                cv2.imshow("mask", mask)
                cv2.waitKey(1000)

            if mask[int(cen_y)][0] == 255 and mask[int(cen_y)][-1] == 255:
                scale_factor = scale
                break

        logger.info("\tresult scale factor: {}".format(scale_factor))

        return {
            'camera_matrix': camera_mat,
            'distort_vector': distort_vec,
            'scale_factor': scale_factor,
            'width': width,
            'height': height
        }

    def calibrate(self, img):
        return cv2.remap(src=img, map1=self.model['map1'], map2=self.model['map2'],
                         interpolation=cv2.INTER_CUBIC)

    def get_map_sz(self):
        return self.model['map1'].shape[:2]

    def get_distort_model(self):
        return self.model
