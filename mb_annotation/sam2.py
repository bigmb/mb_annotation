## function for segment anything 2

import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .bounding_box import generate_bounding_box
import cv2

__all__ = ["show_anns","get_mask_generator","get_mask"]

def show_anns(anns, borders=True, show=True):
    """
    show the annotations
    Args:
        anns (list): list of annotations
        borders (bool): if True, show the borders of the annotations
    Returns:
        None
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
 
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 
    if show:    
        ax.imshow(img)
    return img


def get_mask_generator(sam2_checkpoint='../checkpoints/sam2_hiera_large.pt', model_cfg='sam2_hiera_l.yaml',device='cpu'):
    """
    get the mask generator
    Args:
        sam2_checkpoint (str): path to the sam2 checkpoint
        model_cfg (str): path to the model configuration
    Returns:
        mask_generator (SAM2AutomaticMaskGenerator): mask generator
    """

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device =device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    return mask_generator

def get_similarity_value(box1,box2):
    val1 = abs(box1[0]-box2[0])
    val2 = abs(box1[1]-box2[1])
    val3 = abs(box1[2]-box2[2])
    val4 = abs(box1[3]-box2[3])
    total_val = val1+val2+val3+val4
    return total_val

def get_final_similar_box(box1,box2: list):
    best_box = None
    best_val = None
    index = None
    for i in box2:
        val = get_similarity_value(box1,i)
        if best_box is None or val < best_val:
            best_box = i
            best_val = val
            index = box2.index(i)
    return best_box,index

def get_mask(image_path,bbox_value,sam2_checkpoint,model_cfg,device='cpu',show_full: bool=False,show_final: bool=False,*kwargs):
    """
    get the mask
    Args:
        image_path (str): path to the image
        bbox_value : bounding box value to get mask for
        sam2_checkpoint (str): path to the sam2 checkpoint
        model_cfg (str): path to the model configuration
        show_full (bool): if True, show the full mask
        show_final (bool): if True, show the final mask 
        **kwargs: additional arguments
    Returns:
        mask (np.array): mask
        bbox (list): bounding box
    """
    print('Getting mask')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask_generator = get_mask_generator(sam2_checkpoint,model_cfg,device)
    mask_full = mask_generator.generate(image)
    if show_full:
        show_anns(mask_full)
    print('Getting final mask')

    main_bbox = []
    for i in mask_full:
        mask_val = [i['bbox'][1],i['bbox'][0],i['bbox'][1]+i['bbox'][3],i['bbox'][0]+i['bbox'][2]]
        main_bbox.append(mask_val)

    value_list,index = get_final_similar_box(bbox_value,main_bbox)
    final_mask = mask_full[index]
    if show_final:
        show_anns(final_mask)
    return final_mask['segmentation'],final_mask['bbox']