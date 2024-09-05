## function for segment anything 2

import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
 

__all__ = ["show_anns","get_mask_generator"]


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
