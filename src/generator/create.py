from PIL import Image
import numpy as np
import os
import string
import random
import matplotlib.pyplot as plt

from util.utils import box_xyxy_to_xywh, get_bbox_from_mask

from src.image_augmentation.blendings import apply_blendings_and_paste_onto_background

from src.image_augmentation.motion_blur import LinearMotionBlur3C
from src.image_augmentation.object_position import find_valid_object_position


def create_image_anno_wrapper(
    args,
    output_dir = None,
    scale = (1, 1),
    rotation_augment = 0,
    blending_list=["none"],
    max_allowed_iou=0.5,
    ):
    """ 
    Wrapper used to pass params to workers
    """
    objects = args["objects"]
    distract_objects = args["distractors"]
    background = args["background"]
    
    # Create synthesized images, including masks and labels
    background, img_annotations = create_image_anno(objects, distract_objects, background,
        scale=scale,
        rotation_augment=rotation_augment,
        blending_list=blending_list,
        max_allowed_iou=max_allowed_iou,
    )
    
    annotation = {"annotations": img_annotations}
    rnd_img_name = create_random_name()
    img_path = os.path.join(output_dir, f"{rnd_img_name}.png")
    annotation["file_name"] = f"{rnd_img_name}.png"
    annotation["img_path"] = img_path
    background.save(img_path, "png")
    
    return annotation

def create_random_name(length = 20):
    letters = string.ascii_lowercase
    numbers = string.digits
    return ''.join(random.choice(letters) for i in range(length))

def scale_img_to_dim(img, dim):
    """
    Scale image to given dimension
    """
    # Scale to max dim = 256
    o_w, o_h = img.size
    max_dim = dim
    if o_w > o_h:
        new_w = max_dim
        new_h = int(o_h * (new_w / o_w))
    else:
        new_h = max_dim
        new_w = int(o_w * (new_h / o_h))
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    return img
    
def create_image_anno(
    objects_ann,
    distractor_ann,
    bg_ann,
    scale=(1, 1),
    rotation_augment=0,
    blending_list=["none"],
    max_allowed_iou=0.5,
):
    
    all_objects = objects_ann + distractor_ann
    assert len(all_objects) > 0
    
    ### LOAD BACKGROUND ###
    bg_path = bg_ann["img_path"]
    background= Image.open(bg_path).convert("RGBA")

    # Resize background for width = 1920
    bg_w, bg_h = background.size
    background = scale_img_to_dim(background, 1920)
    # Randomly scale down anb up
    scale_ = random.randint(640, 1920)
    background = scale_img_to_dim(background, scale_)
    background = scale_img_to_dim(background, 1920)
    bg_w, bg_h = background.size

    # new_w = 1920
    # new_h = int(bg_h * (new_w / bg_w))
    # background = background.resize((new_w, new_h), Image.ANTIALIAS)
    # bg_w, bg_h = background.size
    
    # Create img annotation dict
    already_set_obj = None
    img_annotations = []
    for obj_idx, obj_data in enumerate(all_objects):
        ann = {}
        ann["name"] = obj_data["name"]
        ann["supercategory"] = obj_data["supercategory"]
        
        ### LOAD OBJ IMAGE ###
        obj_img_path = obj_data["img_path"]
        if not os.path.exists(obj_img_path):
            continue
        obj_img = Image.open(obj_img_path).convert("RGBA")
        # Scale to max dim = 256
        obj_img = scale_img_to_dim(obj_img, 256)
        
        # Randomly scale down anb up
        scale_ = random.randint(32, 256)
        obj_img = scale_img_to_dim(obj_img, scale_)
        obj_img = scale_img_to_dim(obj_img, 256)
        
        o_w, o_h = obj_img.size
        
        ### Augment Image ###
        # Rotate
        if rotation_augment != 0:
            obj_img = obj_img.rotate(np.random.randint(-rotation_augment, rotation_augment), expand=True)
            o_w, o_h = obj_img.size
        if scale != (1, 1):
            scale_factor = np.random.uniform(scale[0], scale[1])
            obj_img = obj_img.resize((int(o_w*scale_factor), int(o_h*scale_factor)), Image.ANTIALIAS)
            o_w, o_h = obj_img.size
            
        ### Extract foreground and mask ###
        foreground = np.array(obj_img.copy())[:, :, :3]
        foreground = Image.fromarray(foreground, "RGB")
        
        mask = np.array(obj_img.copy())[:, :, 3]
        mask = Image.fromarray(mask, mode="L")
        
        background = np.array(background.copy())[:, :, :3]
        background = Image.fromarray(background, "RGB")
        
        # Get Bbox
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 100, 0, 1)
        try:
            bbox = get_bbox_from_mask(mask_array) # (x1, y1, x2, y2)
        except:
            continue
        to_w, to_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Determine position
        success, x, y = find_valid_object_position(to_w, to_h, bg_w, bg_h, already_set_obj=already_set_obj, max_occlued=max_allowed_iou)
        if not success:
            continue

        bbox[::2] += x
        bbox[1::2] += y
        bbox_cxcywh = box_xyxy_to_xywh(bbox)
        ann["bbox"] = bbox_cxcywh.tolist()
        img_annotations.append(ann)
        
        new_bbox = bbox.copy()
        if max_allowed_iou < 1:
            already_set_obj = np.concatenate((already_set_obj, new_bbox.reshape(1, 4)), axis=0) \
                if already_set_obj is not None else new_bbox.reshape(1, 4)
        
        
        # # Apply blending
        foreground, mask, background = apply_blendings_and_paste_onto_background(
            background, blending_list, foreground, mask, x, y
        )
        
        if ["motion_blur"] in blending_list:
            background = LinearMotionBlur3C(background)
        
        background.paste(foreground, (x, y), mask)
   
    return background, img_annotations

