import PIL
import numpy as np
import signal
import shutil
import copy
import random
import matplotlib.pyplot as plt
from pathlib import Path


def create_directory_structure(root_path, dataset_name, clear=False):
    """_summary_

    Parameters
    ----------
    root_path : Path
        Root path, where to create the dataset directory
    dataset_name : str
        Name of the dataset. Will be used as the name of the dataset directory
    clear : bool, optional
        If true removes already existing files in the directory, by default False

    Returns
    -------
    dataset_path : Path
    images_path : Path
    targets_path : Path
    """
    if type(root_path) == str:
        root_path = Path(root_path)
    
    dataset_path = root_path / dataset_name
    if clear and dataset_path.exists():
        # Prompt user to confirm deletion
        print("\n", "#"*10, "WARNING" , "#"*10)
        print("Clearing existing directory structure...")
        print("This will delete all files in the following directory: \n")
        print(dataset_path, "\n")
        print("#"*10, "WARNING" , "#"*10, "\n")
        print("Are you sure you want to continue? [y/n]")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(dataset_path)
        else:
            print("Aborting...")
            return
    dataset_path.mkdir(exist_ok=True)
    
    # Create directories: images, targetsW
    images_path = dataset_path / "images"
    images_path.mkdir(exist_ok=True)
    
    targets_path = dataset_path / "targets"
    targets_path.mkdir(exist_ok=True)
    
    return dataset_path, images_path, targets_path
        
def annotation_to_coco_ann(annotations_raw):
    """Converts the output of image rendering annotations to COCO format

    Parameters
    ----------
    annotations_raw : List[Dict]
        raw output of render_configurations function

    Returns
    -------
    coco_ann : Dict
        Standard COCO annotation format
    """
    
    coco_gt = {}
    categories = []
    images = []
    annotations = []
    
    
    cat_to_id = {}
    supercat_to_supid = {}
    ann_new_id = 1
    image_ann_id = 1
    for image_info in annotations_raw:
        annotations_ = image_info["annotations"]
        file_name = image_info["file_name"]
        img_path = image_info["img_path"]
        
        ### Create Image Annotation ###
        img = PIL.Image.open(img_path)
        img_w, img_h = img.width, img.height
        image_ann = {}
        image_ann["file_name"] = file_name
        image_ann["id"] = image_ann_id
        image_ann["img_path"] = img_path
        image_ann["height"] = img_h
        image_ann["width"] = img_w
        images.append(image_ann)
        image_ann_id += 1
        
        for ann in annotations_:
            cat = ann["name"]
            supercat = ann["supercategory"]
            bbox = ann["bbox"] # xywh    
            # Make box valid
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(img_w - bbox[0], bbox[2])
            bbox[3] = min(img_h - bbox[1], bbox[3])
            
            ### Create category list ###
            if cat not in cat_to_id:
                cat_to_id[cat] = len(cat_to_id) + 1
            if supercat not in supercat_to_supid:
                supercat_to_supid[supercat] = len(supercat_to_supid) + 1
            category_ann = {
                "id": cat_to_id[cat],
                "name": cat,
                "supercategory": supercat,
                "sup_id": supercat_to_supid[supercat],
            }
            categories.append(category_ann)
        
            ### Create annotation list ###
            ann_formatted = {
                "id": ann_new_id,
                "image_id": image_ann["id"],
                "category_id": cat_to_id[cat],
                "sup_id": supercat_to_supid[supercat],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(ann_formatted)
            ann_new_id += 1
            
    coco_gt["categories"] = categories
    coco_gt["images"] = images
    coco_gt["annotations"] = annotations
    
    return coco_gt
 
def coco_ann_to_vfn_ann(train_coco_ann):
    """ Convert COCO annotation to VFN annotation format
    Same as COCO but gruops annotations by category and supercategory

    Parameters
    ----------
    train_coco_ann : Dict
        COCO annotation

    Returns
    -------
    val_coco_gt : Dict
        Annotations gathered by category
    supercat_to_cat : Dict
        Annotation gathered by supercategory
    """
    
    
    coco_images = train_coco_ann["images"]
    coco_anns = train_coco_ann["annotations"]
    coco_cats = train_coco_ann["categories"]
    
    cat_id_to_cat = {it["id"]: it for it in coco_cats}
    ann_id_to_ann = {it["id"]: it for it in coco_anns}
    image_id_to_image = {it["id"]: it for it in coco_images}
    
    ### Group annotations by picture ###
    image_id_to_anns = {it["id"]: [] for it in coco_images}
    for it in coco_anns:
        image_id_to_anns[it["image_id"]].append(it)
        
    ### Create annotation file by category ###
    new_image_id = 1
    new_images = []
    new_anns = []
    for image_id, anns in image_id_to_anns.items():
        image = copy.deepcopy(image_id_to_image[image_id])
        # Group annotations by category
        cat_id_to_anns_ = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_anns_:
                cat_id_to_anns_[cat_id] = []
            cat_id_to_anns_[cat_id].append(ann)
        
        for cat_id, anns_in_cat in cat_id_to_anns_.items():
            new_image = copy.deepcopy(image)
            new_image["id"] = new_image_id
            new_images.append(new_image)
            new_image_id += 1
            
            for ann in anns_in_cat:
                new_ann = copy.deepcopy(ann)
                new_ann["image_id"] = new_image["id"]
                new_anns.append(new_ann)
                
    val_coco_gt = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco_cats,
    }
    
    ### Create annotation file by super category ###
    new_image_id = 1
    new_images = []
    new_anns = []
    for image_id, anns in image_id_to_anns.items():
        image = copy.deepcopy(image_id_to_image[image_id])
        # Group annotations by category
        sup_id_to_anns_ = {}
        for ann in anns:
            sup_id = ann["sup_id"]
            if sup_id not in sup_id_to_anns_:
                sup_id_to_anns_[sup_id] = []
            sup_id_to_anns_[sup_id].append(ann)
        
        for sup_id, anns_in_sup in sup_id_to_anns_.items():
            new_image = copy.deepcopy(image)
            new_image["id"] = new_image_id
            new_images.append(new_image)
            new_image_id += 1
            
            for ann in anns_in_sup:
                new_ann = copy.deepcopy(ann)
                new_ann["image_id"] = new_image["id"]
                new_anns.append(new_ann)
                
    supval_coco_gt = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco_cats,
    }
    
    return val_coco_gt, supval_coco_gt
       
              
def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def get_bbox_from_mask(mask_img):
    # mask_img: np.array (H, W) bool array
    # return: (x_min, y_min, x_max, y_max)
    # Get idx of first non-zero element in each row

    mask_img = mask_img.astype(np.uint8)
    mask_h, mask_w = mask_img.shape
    
    min_y_ = np.argmin(mask_img, axis=0)
    min_y_ = min_y_[min_y_ != 0]
    min_y = np.min(min_y_)
    
    min_x_ = np.argmin(mask_img, axis=1)
    min_x_ = min_x_[min_x_ != 0]
    min_x = np.min(min_x_)
    
    max_y_ = np.argmin(mask_img[::-1, :], axis=0) 
    max_y_ = max_y_[max_y_ != 0]
    max_y = mask_h - np.min(max_y_)
   
    max_x_ = np.argmin(mask_img[:, ::-1], axis=1)
    max_x_ = max_x_[max_x_ != 0]
    max_x = mask_w - np.min(max_x_)
    
    bbox = np.array([min_x, min_y, max_x, max_y])
    return bbox

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b = np.concatenate(b, axis=-1)
    return b

def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (w), (h)]
    b = np.concatenate(b, axis=-1)
    return b

def box_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = np.split(x, 4, axis=-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
            (x2 - x1), (y2 - y1)]
    b = np.concatenate(b, axis=-1)
    return b

def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = np.split(x, 4, axis=-1)
    b = [(x1), (y1),
            (x2 - x1), (y2 - y1)]
    
    b = np.concatenate(b, axis=-1)
    return b

def box_xywh_to_cxcywh(x):
    x1, y1, w, h = np.split(x, 4, axis=-1)
    b = [(x1 + 0.5 * w), (y1 + 0.5 * h),
         (w), (h)]
    b = np.concatenate(b, axis=-1)
    return b

def box_xywh_to_xyxy(x):
    x1, y1, w, h = np.split(x, 4, axis=-1)
    b = [(x1), (y1),
         (x1 + w), (y1 + h)]
    b = np.concatenate(b, axis=-1)
    return b

def PIL2array1C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        
            