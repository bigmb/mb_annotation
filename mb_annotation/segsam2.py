## function for segment anything 2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from sam2.build_sam import build_sam2_video_predictor
from os import listdir
from os.path import isfile, join
from sam2.sam2_image_predictor import SAM2ImagePredictor


__all__ = ["show_anns","get_mask_generator","get_mask_for_bbox","get_all_masks","video_predictor",
           "show_masks_image","show_box","show_points","image_predictor"]

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

def get_mask_for_bbox(image_path,bbox_value,sam2_checkpoint,model_cfg,device='cpu',show_full: bool=False,show_final: bool=False,*kwargs):
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
        main_bbox (list): main bounding box
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
        mask_val = [i['bbox'][1],i['bbox'][0],
                    (i['bbox'][3]+i['bbox'][1]),(i['bbox'][2]+i['bbox'][0])]  ## adding the bbox values to get the correct bbox for the bounding bbox function
        main_bbox.append(mask_val)


    value_list,index = get_final_similar_box(bbox_value,main_bbox)
    final_mask = mask_full[index]
    final_bbox = [final_mask['bbox'][1],final_mask['bbox'][0],
                    (final_mask['bbox'][3]+final_mask['bbox'][1]),(final_mask['bbox'][2]+final_mask['bbox'][0])]
    if show_final:
        show_anns(final_mask)
    return final_mask['segmentation'],final_bbox,main_bbox

def get_all_masks(image_path,sam2_checkpoint,model_cfg,device='cpu',*kwargs):
    """
    get all the masks
    Args:
        image_path (str): path to the image
        sam2_checkpoint (str): path to the sam2 checkpoint
        model_cfg (str): path to the model configuration
    Returns:
        masks (list): list of masks
    """
    print('Getting all masks')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask_generator = get_mask_generator(sam2_checkpoint,model_cfg,device)
    mask_full = mask_generator.generate(image)
    print('Getting final mask')
    return mask_full


def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Display a mask on a given axis.
    Args:
        mask (np.ndarray): The mask to be displayed.
        ax (matplotlib.axes.Axes): The axis on which to display the mask.
        obj_id (int, optional): Object ID for color selection. Defaults to None.
        random_color (bool, optional): If True, use a random color. Defaults to False.
    Returns:
        None
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask_image(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_masks_image(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Display masks on a given image.
    Args:
        image (np.ndarray): The image on which to display the masks.
        masks (list): List of masks to be displayed.
        scores (list): List of scores corresponding to the masks.
        point_coords (list, optional): List of point coordinates. Defaults to None.
        box_coords (list, optional): List of box coordinates. Defaults to None.
        input_labels (list, optional): List of input labels. Defaults to None.
        borders (bool, optional): If True, display the borders of the masks. Defaults to True.
    Returns:
        None
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask_image(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_points(coords, labels, ax, marker_size=200):
    """
    Display points on a given axis.
    Args:
        coords (np.ndarray): Array of point coordinates.
        labels (np.ndarray): Array of point labels (1 for positive, 0 for negative).
        ax (matplotlib.axes.Axes): The axis on which to display the points.
        marker_size (int, optional): Size of the markers. Defaults to 200.
    Returns:
        None
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """
    Display a bounding box on a given axis.
    Args:
        box (list or np.ndarray): The bounding box coordinates [x0, y0, x1, y1].
        ax (matplotlib.axes.Axes): The axis on which to display the box.
    Returns:
        None
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class video_predictor:
    """
    Segmentation anything 2 video predictor
    Args:
        model_cfg (str): Path to the model config file.
        sam2_checkpoint (str): Path to the model checkpoint file.
        device (str, optional): Device on which to run the model. Defaults to 'cpu'.
    Returns:
        None
    """
    def __init__(self,model_cfg,sam2_checkpoint,device='cpu') -> None:
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def inference_state(self,video_image_folder):
        """
        Initializing video from path of video frames folder
        """
        self.video_image_folder=video_image_folder
        self.frame_names = [f for f in listdir(self.video_image_folder) if isfile(join(self.video_image_folder, f))]
        self.frame_names = sorted(self.frame_names)
        self.joined_frame_names = [self.video_image_folder+'/'+self.frame_names[i] for i in range(len(self.frame_names))]
        self.inference_state = self.predictor.init_state(video_path=video_image_folder)

    def reset_state(self):
        """
        Reseting state to base
        """
        self.predictor.reset_state(self.inference)
    
    def predict_item(self,bbox:list[list]= None,points: list[list]=None,labels: list=None,frame_idx=0,show=True,image_loc=None,gemini_bbox=True,**kwargs):
        """
        Getting prediction from bounding box. Can add points to determine the location of segmentation
        """
        ann_obj_id=1
        if points and labels:
            points = np.array(points,dtype=np.float32)
            labels = np.array(labels,np.int32)
            print(f"points : {points}")
            print(f"labels : {labels}")
            prompts={}
            prompts[ann_obj_id] = points, labels

        if bbox:
            bbox = np.array(bbox,dtype=np.float32)
            if gemini_bbox:
                bbox = bbox[[1,0,3,2]]
            print(f"bbox : {bbox}")


        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                            inference_state=self.inference_state,
                                            frame_idx=frame_idx,
                                            obj_id=ann_obj_id,
                                            points=points,
                                            labels=labels,
                                            box=bbox,**kwargs)

        if show:
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {frame_idx}")
            if image_loc==None:
                image_loc =self.joined_frame_names[0]
            plt.imshow(Image.open(image_loc))
            try: 
                show_points(points, labels, plt.gca())
            except:
                pass
            for i, out_obj_id in enumerate(out_obj_ids):
                try:
                    show_points(*prompts[out_obj_id], plt.gca())
                except:
                    pass
                show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    def predict_video(self,vis_frame_stride=30):
        """
        Using the predicting item and propogate it through the all frames.
        """
        video_segments = {} 
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

        plt.close("all")
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(self.joined_frame_names[out_frame_idx]))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

class image_predictor:
    """
    Class for image prediction
    Args:
        model_cfg (str): Path to the model config file.
        sam2_checkpoint (str): Path to the model checkpoint file.
        device (str, optional): Device on which to run the model. Defaults to 'cpu'.
    Returns:
        None
    """
    def __init__(self,model_cfg,sam2_checkpoint,device='cpu') -> None:
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))

    def set_image(self,image):
        """
        Setting image
        Args:
            image (str): Path to the image or numpy array
        Returns:
            None
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            self.image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        else:
            self.image = image
        self.image_set = self.predictor.set_image(self.image)

    def predict_item(self,bbox:list[list]= None,points: list[list]=None,labels: list=None,show=True,gemini_bbox=True,**kwargs):
        """
        Getting prediction from bounding box or points.
        Args:
            bbox (list[list], optional): Bounding box. Defaults to None.
            points (list[list], optional): Points. Defaults to None.
            labels (list, optional): Labels. Defaults to None.
            show (bool, optional): Show the image. Defaults to True.
        Returns:
            masks, scores, logits
        """
        if points and labels:
            points = np.array(points,dtype=np.float32)
            labels = np.array(labels,np.int32)
            print(f"points : {points}")
            print(f"labels : {labels}")

        if bbox:
            bbox = np.array(bbox,dtype=np.float32)
            if gemini_bbox:
                bbox = bbox[[1,0,3,2]]
            print(f"bbox : {bbox}")

        predict_args ={}

        if points is not None:
            predict_args["point_coords"] = points
            predict_args["point_labels"] = labels

        if bbox is not None:
            predict_args["box"] = bbox
        print(predict_args)

        masks, scores, logits = self.predictor.predict(**predict_args,multimask_output=False,**kwargs)

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if show:
            plt.figure(figsize=(9, 6))
            plt.imshow(self.image)
            if points is not None:
                show_points(points, labels, plt.gca())
            if bbox is not None:
                show_box(bbox, plt.gca())
            # for i, mask in enumerate(masks):
            #     show_mask(mask, plt.gca(), random_color=True)
            show_masks_image(self.image, masks, scores, point_coords=points, box_coords=bbox, input_labels=labels)

        return masks, scores, logits
    
    
    



