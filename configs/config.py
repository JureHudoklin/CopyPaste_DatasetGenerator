import json

class Config(object):

    DATASET_NAME = "synthetic_dataset"  # Name of your dataset
    
    # Here a folder named {dataset_name} will be created, the new dataset will be saved in this folder
    TARGET_DIR = "PUT_YOUR_PATH_HERE"
    NUM_OF_IMAGES = 10  # Number of images to generate
    MAKE_TARGETS = True

    # Path to the json file containing the images and background annotations
    # The json file should be in the following format:
    # OBJECTS_ANN --> List[{name: str, supercategory: str, file_name: str, img_path: str}, ...]
    # BACKGROUND_ANN --> List[{scene: str, file_name: str, img_path: str}, ...]
    # DISTRACT_OBJ_ANN (optional) --> List[{name: str, supercategory: str, file_name: str, img_path: str}, ...]
    OBJECTS_ANN_PATH = "PUT_YOUR_PATH_HERE"
    BACKGROUND_ANN_PATH = "PUT_YOUR_PATH_HERE"
    DISTRACT_OBJ_ANN_PATH = None # Optional

    NUMBER_OF_WORKERS = 8  # Number of workers to use for multiprocessing

    # Augmentations to apply to the images
    BLENDING_LIST = [
        #"none", # No blending
        #"gaussian", 
        #"box", 
        #"gamma_correction",
        #"illumination", 
        #"poisson",
        "poisson_fast", # Faster version of poisson blending, requires CUDA
        #"motion_blur", # Applies motion blur to the image
        "mixed", # Combination of gaussian, box, gamma_correction and illumination
    ],

    MIN_NO_OF_OBJECTS = 13 # Minimum number of objects to be placed in the image
    MAX_NO_OF_OBJECTS = 23 # Maximum number of objects to be placed in the image

    MIN_NO_OF_DISTRACTOR_OBJECTS = 0 # Minimum number of distractor objects to be placed in the image
    MAX_NO_OF_DISTRACTOR_OBJECTS = 4 # Maximum number of distractor objects to be placed in the image

    MAX_ATTEMPTS_TO_SYNTHESIZE = 20 # Maximum number of attempts to synthesize an image

    # Object images are all first scaled to the same size 256px, then they are scaled by the provided ratio
    MIN_SCALE = 0.4 
    MAX_SCALE = 1.8
    MAX_DEGREES = 30 # Maximum degrees to rotate the object
    MAX_TRUNCATION_FRACTION = 0.25
    MAX_ALLOWED_IOU = 0.3

    MINFILTER_SIZE = 3

    def __init__(self, load_path=None):
        if load_path is not None:
            self._load(load_path)

    def _load(self, path):
        print("Loading config from {}".format(path))
        with open(path, "r") as f:
            config = json.load(f)
        self.__dict__.update(config)

    def save(self, path):
        print("Saving config to {}".format(path))
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
