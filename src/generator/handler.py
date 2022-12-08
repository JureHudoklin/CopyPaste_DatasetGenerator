import json
import random
import time
import os
import PIL
import numpy as np
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Dict, List

import tqdm

from configs.config import Config
from src.generator.create import create_image_anno_wrapper

from util.utils import annotation_to_coco_ann, init_worker, coco_ann_to_vfn_ann

class PasteDatasetGenerator:
    def __init__(self, config: Dict):
        self.config = config

    def generate_synthetic_dataset(self,
        num_imgs: int,
        dataset_path: Path,
        make_targets = True,
    ):
        assert isinstance(self.config, Config)
        if type(dataset_path) == str:
            dataset_path = Path(dataset_path)
            print("Warning: dataset_path should be a pathlib.Path object.")
        
        print(f"{'#' * 20} Generating data {'#' * 20}")
        start_time = time.time()
        
        # Get annotation paths
        objects_json_path = self.config.OBJECTS_ANN_PATH
        background_json_path = self.config.BACKGROUND_ANN_PATH
        distractors_json_path = self.config.DISTRACT_OBJ_ANN_PATH


        # Load annotations
        objects_data, background_data, distractor_data = self.load_relevant_data(
            objects_json_path, background_json_path, distractors_json_path,
        )
        print("Loading annotations")
        
        # Create image list for data generation
        generate_data = self._create_list_of_img_configurations(
            objects_data,
            background_data,
            distractor_data,
            num_imgs
        )
        
        print("Rendering images")
        coco_gt, val_coco_gt, supval_coco_gt = self.render_configurations(
            generate_data,
            dataset_path,
            number_of_workers=self.config.NUMBER_OF_WORKERS,
        )
        
        print("Saving annotations")
        # Save annotations
        json.dump(coco_gt, open(dataset_path / f"{self.config.DATASET_NAME}_coco_gt.json", "w"), indent=4)
        json.dump(val_coco_gt, open(dataset_path / f"{self.config.DATASET_NAME}_val_coco_gt.json", "w"), indent=4)
        json.dump(supval_coco_gt, open(dataset_path / f"{self.config.DATASET_NAME}_supval_coco_gt.json", "w"), indent=4)
        
        if make_targets:
            print("Rendering target images")
            self.render_target_images(coco_gt, objects_data, dataset_path, number_of_workers=self.config.NUMBER_OF_WORKERS)
        
        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print(f"Generation took: {elapsed:.2f} min")
        
    def _create_list_of_img_configurations(self,
        objects_data: List[Dict],
        background_data: List[Dict],
        distractor_data: List[Dict],
        num_images: int
    ):
        sup_to_objects = {}
        for obj in objects_data:
            supercategory = obj["supercategory"]
            sup_to_objects.setdefault(supercategory, []).append(obj)
            
        annotations = []
        for _ in range(num_images):
            objects = []
            distractor_objects = []
            
            ### Select objects ###
            n = min(random.randint(self.config.MIN_NO_OF_OBJECTS,
                                   self.config.MAX_NO_OF_OBJECTS),
                    len(objects_data))
            num_sup = random.randint(1, min(n, len(sup_to_objects.keys())))
            sup_keys = random.sample(list(sup_to_objects.keys()), num_sup)
            
            sup_objs = []
            for sup in sup_keys:
                sup_objs.extend(sup_to_objects[sup])
            
            idxs = random.sample(range(len(sup_objs)), n)
            objects = [sup_objs[i] for i in idxs]
        
            ### Select Distractor Objects ###
            if distractor_data is not None:
                n_dist = min(random.randint(self.config.MIN_NO_OF_DISTRACTOR_OBJECTS,
                                            self.config.MAX_NO_OF_DISTRACTOR_OBJECTS),
                             len(distractor_data))
                dist_idx = random.sample(range(len(distractor_data)), n_dist)
                distractor_objects = [distractor_data[i] for i in dist_idx]
    
            ### Select Background ###
            bg_file = random.choice(background_data)
            
            img_ann = {"objects": objects, "distractors": distractor_objects, "background": bg_file}
            annotations.append(img_ann)
            
        return annotations


    def render_configurations(self,
        generate_data,
        dataset_path: Path,
        number_of_workers: bool,
    ):  
        assert isinstance(self.config, Config)
        images_path = dataset_path / "images"
        # Run configurations
        partial_func = partial(
            create_image_anno_wrapper,
            output_dir=images_path,
            scale = (self.config.MIN_SCALE, self.config.MAX_SCALE),
            rotation_augment= self.config.MAX_DEGREES,
            blending_list=self.config.BLENDING_LIST,
            max_allowed_iou= self.config.MAX_ALLOWED_IOU,
        )
        multiprocessing = number_of_workers > 1
        if not multiprocessing:
            annotations = []
            for p in tqdm.tqdm(generate_data):
                ann = partial_func(p)
                annotations.append(ann)
        else:         
            p = Pool(number_of_workers, init_worker)
            try:
                out = []
                for ann in tqdm.tqdm(p.imap_unordered(partial_func, generate_data), total=len(generate_data)):
                    out.append(ann)
            except KeyboardInterrupt:
                print("....\nCaught KeyboardInterrupt, terminating workers")
                p.terminate()
            else:
                p.close()
            p.join()
            
            annotations = out
        
        # Convert annotations to coco format
        coco_gt = annotation_to_coco_ann(annotations)
        val_coco_gt, supval_coco_gt = coco_ann_to_vfn_ann(coco_gt)
        
        return coco_gt, val_coco_gt, supval_coco_gt

    def render_target_images(self,
                             coco_gt: Dict,
                             objects_data: List[Dict],
                             dataset_path: Path,
                             number_of_workers: int = 1,
                             ):
        targets_path = dataset_path / "targets"
        categories = coco_gt["categories"]
        
        name_to_cat = {cat["name"]: cat for cat in categories}
        # Sort objects by category
        cat_to_obj = {}
        for obj in objects_data:
            name = obj["name"]
            supercat = obj["supercategory"]
            if name not in cat_to_obj:
                cat_to_obj[name] = []
            cat_to_obj[name].append(obj)
            
        multiprocessing = number_of_workers > 1
        if not multiprocessing:
            objs = list(cat_to_obj.values())
            for obj in tqdm.tqdm(objs):
                format_cat(obj, targets_path)
        else:
            objs = list(cat_to_obj.values())
            partial_func = partial(format_cat, path=targets_path)
            p = Pool(number_of_workers, init_worker)
            try:
                for _ in tqdm.tqdm(p.imap_unordered(partial_func, objs), total=len(cat_to_obj)):
                    pass
            except KeyboardInterrupt:
                print("....\nCaught KeyboardInterrupt, terminating workers")
                p.terminate()
            else:
                p.close()
            p.join()
            
        

    def load_relevant_data(self,
        object_json_path: str,
        background_json_path: str,
        distractor_json_path = None,
    ):
        # Objects
        objects_data = load_data_from_split_file(object_json_path)
        
        # Backgrounds
        background_data = load_data_from_split_file(background_json_path)
        
        # Distractors
        distractor_data = None
        if distractor_json_path is not None:
            distractor_data = load_data_from_split_file(distractor_json_path)
        
        return objects_data, background_data, distractor_data


def load_data_from_split_file(json_file: Union[str, Path]):
    if isinstance(json_file, str):
        json_file = Path(json_file)
    assert json_file.exists(), f"File {json_file.resolve()} does not exist!"
    with json_file.open("r") as f:
        data = json.load(f)
    
    return data

def format_cat(objs, path):
    objs = sorted(objs, key=lambda x: x["file_name"])
    # Sample 100 images
    idxs = np.linspace(0, len(objs)-1, 50, dtype=int)
    objs = [objs[i] for i in idxs]
    
    # Create target images
    for obj in objs:
        name = obj["name"]
        supercat = obj["supercategory"]
        pth = path / name
        pth.mkdir(parents=True, exist_ok=True)
        
        # Find min idx of already existing images
        existing = [int(f.stem) for f in pth.glob("*.png")]
        if len(existing) > 0:
            min_idx = max(existing) + 1
        else:
            min_idx = 0
        
        img = PIL.Image.open(obj["img_path"])
        img.save(pth / f"{min_idx}.png")

