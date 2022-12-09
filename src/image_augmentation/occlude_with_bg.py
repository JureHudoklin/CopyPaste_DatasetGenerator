import numpy as np
import copy
from PIL import Image
from util.utils import get_bbox_from_mask

def occlude_with_bg(img, max = 0.5):
    """ Randomly deletes a portion of the image

    Parameters
    ----------
    img : PIL.Image
    max : float, optional
        Max valid section to delete, by default 0.5
    """
    
    img_arr = np.array(img)
    mask = np.array(img.copy())[:, :, 3]
    mask = Image.fromarray(mask, mode="L")
    
    # Get Bbox
    mask_array = np.array(mask)
    mask_array = np.where(mask_array > 100, 0, 1)
    
    bbox = get_bbox_from_mask(mask_array) # (x1, y1, x2, y2)
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
     # Get random section to delete
    x1 = np.random.randint(bbox[0], bbox[2])
    y1 = np.random.randint(bbox[1], bbox[3])
    x2 = np.random.randint(x1, bbox[2])
    y2 = np.random.randint(y1, bbox[3])

    for i in range(10):
        delete_area = (x2 - x1) * (y2 - y1)
        
        if delete_area / area < max:
            break
        else:
            x1 = x1 - (x1 - bbox[0]) // 2
            y1 = y1 - (y1 - bbox[1]) // 2
            x2 = x2 + (bbox[2] - x2) // 2
            y2 = y2 + (bbox[3] - y2) // 2
    
    img_arr[y1:y2, x1:x2, 3] = 0
    
    return Image.fromarray(img_arr, "RGBA")
    
    
            
    
    
    