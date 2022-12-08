import random

from util.utils import box_cxcywh_to_xyxy

import numpy as np

def find_valid_object_position(o_w, o_h, img_w, img_h, 
                               already_set_obj = None,
                               max_occlued = 0.5,
                               max_trunc = 0.5,):
    assert max_occlued >= 0 and max_occlued <= 1
    
    attempt = 0
    while True:
        attempt += 1
        x = random.randint(
            int(-max_trunc * o_w),
            int(img_w - o_w + max_trunc * o_w),
        )
        y = random.randint(
            int(-max_trunc * o_h),
            int(img_h - o_h + max_trunc * o_h),
        )
        if max_occlued < 1. and already_set_obj is not None:
            found = True
            bbox = np.array([x, y, o_w, o_h])
            coverage = calculate_coverage(bbox, already_set_obj)
            if np.max(coverage) < max_occlued:
                break
            if attempt >= 10:
                found = False
                break
        else:
            found = True
            break
    
    return found, x, y

def calculate_coverage(bbox, already_set_obj):
    bbox_xyxy = box_cxcywh_to_xyxy(bbox)
    al_set_obj_xyxy = box_cxcywh_to_xyxy(already_set_obj)
    
    intersection = np.maximum(0, np.minimum(bbox_xyxy[2], al_set_obj_xyxy[:, 2]) - np.maximum(bbox_xyxy[0], al_set_obj_xyxy[:, 0])) \
                    * np.maximum(0, np.minimum(bbox_xyxy[3], al_set_obj_xyxy[:, 3]) - np.maximum(bbox_xyxy[1], al_set_obj_xyxy[:, 1]))
                    
    coverage = intersection / (already_set_obj[:, 2] * already_set_obj[:, 3])
    
    return coverage

def calculate_iou(bbox, already_set_obj):
    # bbox: np.array([cx, cy, w, h])
    # already_set_obj: np.array (N, 4)
    bbox_xyxy = box_cxcywh_to_xyxy(bbox)
    al_set_obj_xyxy = box_cxcywh_to_xyxy(already_set_obj)
    
    intersection = np.maximum(0, np.minimum(bbox_xyxy[2], al_set_obj_xyxy[:, 2]) - np.maximum(bbox_xyxy[0], al_set_obj_xyxy[:, 0])) \
                    * np.maximum(0, np.minimum(bbox_xyxy[3], al_set_obj_xyxy[:, 3]) - np.maximum(bbox_xyxy[1], al_set_obj_xyxy[:, 1]))
                    
    union = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]) + \
    (al_set_obj_xyxy[:, 2] - al_set_obj_xyxy[:, 0]) * (al_set_obj_xyxy[:, 3] - al_set_obj_xyxy[:, 1]) - intersection
    
    iou = intersection / union # (N, )
    
    return iou
