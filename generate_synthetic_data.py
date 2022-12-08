from pathlib import Path
import sys

import matplotlib.pyplot as plt
import shutil
import random
import json
import PIL
import numpy as np
from src.generator.handler import PasteDatasetGenerator
from util.utils import annotation_to_coco_ann, create_directory_structure
from configs.config import Config

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def create_new_dataset(config):
    assert isinstance(config, Config)
    dataset_name = config.DATASET_NAME
    target_dir = config.TARGET_DIR
    
    dataset_dir, images_dir, targets_dir = create_directory_structure(target_dir, dataset_name, clear=True)
    print(f"Output directory: {dataset_dir}")
    
    data_generator = PasteDatasetGenerator(config=config)
   
    data_generator.generate_synthetic_dataset(
        num_imgs = config.NUM_OF_IMAGES,
        dataset_path=dataset_dir,
        make_targets=config.MAKE_TARGETS,
    )

def display_random_ds_img(coco_gt):
    """Displays a random image from the dataset

    Parameters
    ----------
    coco_gt : Dict
        COCO annotation format
    """
    img_ids = [img["id"] for img in coco_gt["images"]]
    img_id = random.choice(img_ids)
    img_ann = [img for img in coco_gt["images"] if img["id"] == img_id][0]
    img_path = img_ann["img_path"]
    
    img = PIL.Image.open(img_path)
    
    fig = plt.figure(figsize=(3, 3), dpi=300)
    # Plot image
    plt.imshow(img)
    
    # Plot bbox
    anns = [ann for ann in coco_gt["annotations"] if ann["image_id"] == img_id]
    for ann in anns:
        bbox = ann["bbox"]
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    
    plt.axis("off")
    
    return fig

if __name__ == "__main__":
    config = Config()
    
    if False:
        create_new_dataset(config)
        
    if True:
        coco_gt_path = None
        if coco_gt_path is None:
            dataset_path = Path(config.TARGET_DIR) / config.DATASET_NAME
        coco_gt = json.load(open(dataset_path / f"{config.DATASET_NAME}_coco_gt.json"))
        for i in range(4):
            fig = display_random_ds_img(coco_gt)
            plt.savefig(f"syn_ds_img_{i}.png", bbox_inches='tight', pad_inches=0, dpi=500)
            fig.clf()
    exit()
  
        
    with open(output_dir / "annotations.json", "r") as f:
        out_ann = json.load(f)
    coco_gt = annotation_to_coco_ann(out_ann)
    
    json.dump(coco_gt, open(output_dir / f"{dataset_name}_coco_gt.json", "w"))
    exit()
    
    # print(out_ann)
    # img_ann = out_ann[0]
    # img_path = img_ann["img_path"]
    # bbox = [it["bbox"] for it in img_ann["annotations"]] # cx cy w h
    
    # image = plt.imread(img_path)
    # plt.imshow(image)
    # # Plot bounding boxes
    # for it in bbox:
    #     plt.gca().add_patch(plt.Rectangle((it[0], it[1]), it[2], it[3], fill=False, edgecolor='r', linewidth=2))
        
    # plt.savefig(output_dir / "example.png")
    # exit()
   