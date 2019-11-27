import os
import numpy as np
from utils.constant import MODEL_DIR

# [pascal labels]
np.random.seed(10101)
labels_path = os.path.join(MODEL_DIR, 'label_names', 'pascal.names')
PASCAL_LABELS = open(labels_path).read().strip().split("\n")
PASCAL_COLORS = np.random.randint(0, 255, size=(len(PASCAL_LABELS), 3), dtype=np.uint8)

P_PERSON = [15]
P_BICK_MOTOR = [2, 14]
P_VEHICLES = [6, 7]
P_TRAIN = [19]


# [coco labels]
np.random.seed(7)
labels_path = os.path.join(MODEL_DIR, 'label_names', 'coco.names')
COCO_LABELS = open(labels_path).read().strip().split("\n")
COCO_COLORS = np.random.randint(0, 255, size=(len(COCO_LABELS), 3), dtype=np.uint8)

C_PERSON = [0]
C_VEHICLES = [2, 5, 7]
C_BICK_MOTOR = [1, 3]
C_TRAIN = [6]
C_TRAFFIC_SIGNS = [9, 11]
