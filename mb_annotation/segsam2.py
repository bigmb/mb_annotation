"""
SAM2 Module

This module provides functionality for Segment Anything 2 (SAM2) model operations,
including mask generation, video prediction, and image segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from os import listdir
from os.path import isfile, join
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
import os
from typing import List, Dict, Union, Tuple, Optional, Any

class SAM2Processor:
    """
    Main class for SAM2 model operations including mask generation and visualization.
    """
    
    def __init__(self, sam2_checkpoint: str = '../checkpoints/sam2_hiera_large.pt',
                 model_cfg: str = 'sam2_hiera_l.yaml', device: str = 'cpu'):
        """
        Initialize SAM2Processor.

        Args:
            sam2_checkpoint (str): Path to model checkpoint
            model_cfg (str): Path to model configuration
            device (str): Device to run on
        """
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.mask_generator = self._initialize_mask_generator()

    def _initialize_mask_generator(self) -> SAM2AutomaticMaskGenerator:
        """Initialize the mask generator."""
        sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, 
                         device=self.device, apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(sam2)

    def show_anns(self, anns: List[Dict], borders: bool = True, show: bool = True) -> Optional[np.ndarray]:
        """Display annotations on an image."""
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
     
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                      sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8),
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                          for contour in contours]
                cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1)
        if show:    
            ax.imshow(img)
        return img

    def get_similarity_value(self, box1: List[float], box2: List[float]) -> float:
        """Calculate similarity between two bounding boxes."""
        val1 = abs(box1[0]-box2[0])
        val2 = abs(box1[1]-box2[1])
        val3 = abs(box1[2]-box2[2])
        val4 = abs(box1[3]-box2[3])
        return val1 + val2 + val3 + val4

    def get_final_similar_box(self, box1: List[float], box2: List[List[float]]) -> Tuple[List[float], int]:
        """Find the most similar box from a list of boxes."""
        best_box = None
        best_val = None
        index = None
        for i in box2:
            val = self.get_similarity_value(box1, i)
            if best_box is None or val < best_val:
                best_box = i
                best_val = val
                index = box2.index(i)
        return best_box, index

    def get_mask_for_bbox(self, image_path: str, bbox_value: List[float],
                         show_full: bool = False, show_final: bool = False) -> Tuple[np.ndarray, List[float], List[List[float]]]:
        """Get mask for a specific bounding box."""
        print('Getting mask')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_full = self.mask_generator.generate(image)
        if show_full:
            self.show_anns(mask_full)
        print('Getting final mask')

        main_bbox = []
        for i in mask_full:
            mask_val = [i['bbox'][1], i['bbox'][0],
                       (i['bbox'][3]+i['bbox'][1]), (i['bbox'][2]+i['bbox'][0])]
            main_bbox.append(mask_val)

        value_list, index = self.get_final_similar_box(bbox_value, main_bbox)
        final_mask = mask_full[index]
        final_bbox = [final_mask['bbox'][1], final_mask['bbox'][0],
                     (final_mask['bbox'][3]+final_mask['bbox'][1]),
                     (final_mask['bbox'][2]+final_mask['bbox'][0])]
        if show_final:
            self.show_anns([final_mask])
        return final_mask['segmentation'], final_bbox, main_bbox

    def get_all_masks(self, image_path: str) -> List[Dict]:
        """Get all masks for an image."""
        print('Getting all masks')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_full = self.mask_generator.generate(image)
        print('Getting final mask')
        return mask_full

    def show_mask(self, mask: np.ndarray, ax: plt.Axes, obj_id: Optional[int] = None,
                 random_color: bool = False) -> None:
        """Display a mask on a given axis."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_masks_image(self, image: np.ndarray, masks: List[np.ndarray],
                        scores: List[float], point_coords: Optional[np.ndarray] = None,
                        box_coords: Optional[np.ndarray] = None,
                        input_labels: Optional[np.ndarray] = None,
                        borders: bool = True) -> None:
        """Display multiple masks on an image."""
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask_image(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def show_points(self, coords: np.ndarray, labels: np.ndarray,
                   ax: plt.Axes, marker_size: int = 200) -> None:
        """Display points on an axis."""
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                  marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                  marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box: Union[List, np.ndarray], ax: plt.Axes) -> None:
        """Display a bounding box on an axis."""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                                 facecolor=(0, 0, 0, 0), lw=2))

class VideoPredictor:
    """Class for video prediction using SAM2."""

    def __init__(self, model_cfg: str, sam2_checkpoint: str, device: str = 'cpu'):
        """Initialize VideoPredictor."""
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.video_image_folder = None
        self.frame_names = None
        self.joined_frame_names = None
        self.inference_state = None

    def set_inference_state(self, video_image_folder: str) -> None:
        """Initialize inference state from video frames."""
        self.video_image_folder = video_image_folder
        self.frame_names = [f for f in listdir(video_image_folder)
                          if isfile(join(video_image_folder, f))]
        self.frame_names = sorted(self.frame_names)
        self.joined_frame_names = [join(video_image_folder, frame)
                                 for frame in self.frame_names]
        self.inference_state = self.predictor.init_state(video_path=video_image_folder)

    def reset_state(self) -> None:
        """Reset the inference state."""
        self.predictor.reset_state(self.inference_state)

    def predict_item(self, bbox: Optional[List[List[float]]] = None,
                    points: Optional[List[List[float]]] = None,
                    labels: Optional[List[int]] = None,
                    frame_idx: int = 0, show: bool = True,
                    image_loc: Optional[str] = None,
                    gemini_bbox: bool = True, **kwargs) -> None:
        """Make predictions on a single frame."""
        ann_obj_id = 1
        prompts = {}

        if points is not None and labels is not None:
            points = np.array(points, dtype=np.float32)
            labels = np.array(labels, np.int32)
            prompts[ann_obj_id] = points, labels

        if bbox is not None:
            bbox = np.array(bbox, dtype=np.float32)
            if gemini_bbox:
                bbox = bbox[[1,0,3,2]]

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=bbox,
            **kwargs)

        if show:
            self._visualize_prediction(image_loc, frame_idx, points, labels,
                                     prompts, out_obj_ids, out_mask_logits)

    def predict_video(self, vis_frame_stride: int = 30) -> None:
        """Process all frames in the video."""
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        plt.close("all")
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(self.joined_frame_names[out_frame_idx]))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                SAM2Processor.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

class ImagePredictor:
    """Class for image prediction using SAM2."""

    def __init__(self, model_cfg: str, sam2_checkpoint: str, device: str = 'cpu'):
        """Initialize ImagePredictor."""
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))
        self.image = None

    def set_image(self, image: Union[str, np.ndarray]) -> None:
        """Set the image for prediction."""
        if isinstance(image, str):
            image = cv2.imread(image)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.image = image
        self.predictor.set_image(self.image)

    def predict_item(self, bbox: Optional[List[List[float]]] = None,
                    points: Optional[List[List[float]]] = None,
                    labels: Optional[List[int]] = None,
                    show: bool = True, gemini_bbox: bool = True,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions on the current image."""
        predict_args = {}

        if points is not None and labels is not None:
            points = np.array(points, dtype=np.float32)
            labels = np.array(labels, np.int32)
            predict_args["point_coords"] = points
            predict_args["point_labels"] = labels

        if bbox is not None:
            bbox = np.array(bbox, dtype=np.float32)
            if gemini_bbox:
                bbox = bbox[[1,0,3,2]]
            predict_args["box"] = bbox

        masks, scores, logits = self.predictor.predict(
            **predict_args, multimask_output=False, **kwargs)

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if show:
            self._visualize_prediction(masks, scores, points, bbox, labels)

        return masks, scores, logits

class DataProcessor:
    """Class for handling data loading and processing."""

    @staticmethod
    def load_data(file: str) -> Dict:
        """Load data from CSV file."""
        t1 = pd.read_csv(file)
        assert 'image_path' in t1.columns
        assert 'mask_path' in t1.columns
        t1_image = list(t1['image_path'])
        t1_masks = list(t1['mask_path'])
        data = {}
        for i in range(len(t1_image)):
            data[i] = {"image": t1_image[i], "annotation": t1_masks[i]}
        return data

    @staticmethod
    def read_batch(data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read a batch of data."""
        ent = data[np.random.randint(len(data))]
        Img = cv2.imread(ent["image"])[...,::-1]
        ann_map = cv2.imread(ent["annotation"])

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                           interpolation=cv2.INTER_NEAREST)

        mat_map = ann_map[:,:,0]
        ves_map = ann_map[:,:,2]
        mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)

        inds = np.unique(mat_map)[1:]
        points = []
        masks = []
        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)
            masks.append(mask)
            coords = np.argwhere(mask > 0)
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])
        return Img, np.array(masks), np.array(points), np.ones([len(masks),1])

class ModelTrainer:
    """Class for training SAM2 models."""

    def __init__(self, predictor: SAM2ImagePredictor, device: str = 'cpu'):
        """Initialize ModelTrainer."""
        self.predictor = predictor
        self.device = device

    def train(self, data: Dict, epochs: int = 10, lr: float = 1e-6,
             save_step: int = 10, save_all: bool = False) -> SAM2ImagePredictor:
        """Train the model."""
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)
        
        optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(),
                                    lr=lr, weight_decay=4e-5)
        scaler = GradScaler()
        
        os.makedirs("sam_model_checkpoints", exist_ok=True)

        mean_iou = 0
        for itr in range(epochs):
            with torch.amp.autocast(device_type=self.device):
                image, mask, input_point, input_label = DataProcessor.read_batch(data)
                if mask.shape[0] == 0:
                    continue

                self.predictor.set_image(image)

                loss = self._compute_loss(itr,mask, input_point, input_label)
                
                self.predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if itr % save_step == 0:
                    self._save_checkpoint(itr, save_all)

        return self.predictor

    def _compute_loss(self, itr : int, mask: np.ndarray, input_point: np.ndarray,
                     input_label: np.ndarray) -> torch.Tensor:
        """Compute the loss for training."""
        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None)

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0)
                           for feat_level in self.predictor._features["high_res_feats"]]
        
        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features)

        prd_masks = self.predictor._transforms.postprocess_masks(
            low_res_masks, self.predictor._orig_hw[-1])

        gt_mask = torch.tensor(mask.astype(np.float32)).to(self.device)
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) -
                   (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        
        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

        print(f"Iteration {itr}, Segmentation Loss: {seg_loss.item()}, Score Loss: {score_loss.item()}, Mean IOU: {mean_iou}")
        return seg_loss + score_loss * 0.05

    def _save_checkpoint(self, iteration: int, save_all: bool) -> None:
        """Save model checkpoint."""
        if save_all:
            path = f"./sam_model_checkpoints/sam_model_{iteration}.pt"
        else:
            path = "./sam_model_checkpoints/sam_model.pt"
        torch.save(self.predictor.model.state_dict(), path)


# Create convenience functions that use the classes
def get_mask_generator(*args, **kwargs):
    """Convenience function to get a mask generator."""
    processor = SAM2Processor(*args, **kwargs)
    return processor.mask_generator

def get_mask_for_bbox(*args, **kwargs):
    """Convenience function to get a mask for a bounding box."""
    processor = SAM2Processor()
    return processor.get_mask_for_bbox(*args, **kwargs)

def get_all_masks(*args, **kwargs):
    """Convenience function to get all masks."""
    processor = SAM2Processor()
    return processor.get_all_masks(*args, **kwargs)

def load_data(*args, **kwargs):
    """Convenience function to load data."""
    return DataProcessor.load_data(*args, **kwargs)

def read_batch(*args, **kwargs):
    """Convenience function to read a batch."""
    return DataProcessor.read_batch(*args, **kwargs)

def train_model(data: Dict, predictor: SAM2ImagePredictor, *args, **kwargs):
    """Convenience function to train a model."""
    trainer = ModelTrainer(predictor)
    return trainer.train(data, *args, **kwargs)
