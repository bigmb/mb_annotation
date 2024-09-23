## file for florence functions

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from functools import partial

__all__ = ["florence_model"]

class florence_model:
    """
    Class for florence model
    Args:
        model_name (str): Name of the model
        device (str, optional): Device on which to run the model. Defaults to 'cpu'.
    Returns:
        None
    """

    def __init__(self,model_name="microsoft/Florence-2-large",device='cpu') -> None:

        if device == 'cpu':
            device = "cpu"
        elif device == 'cuda':
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True)

    def get_task_types(self):
        """
        Function to get task types
        Args:
            None
        Returns:
            task_type (dict): List of task types
        """
        task_list =  {'captions':['<CAPTION>','<DETAILED_CAPTION>','<MORE_DETAILED_CAPTION>'],'character_recognition':['<OCR>','<OCR_WITH_REGION>'],
                      'object_detection':['<OD>','<REGION_PROPOSAL>','<DENSE_REGION_PROPOSAL>'],'segmentation':['<REGION_TO_SEGMENTATION>'],
                      'description':['<REGION_TO_CATOGORY>','<REGION_TO_DESCRIPTION>'],
                      'extra':['<PHRASE_GROUNDING>','<OPEN_VOCABULARY_DETECTION>','<REFERRING_EXPRESSION_SEGMENTATION>']}
        return task_list
    
    def define_task(self,task_type:list = ['<OD>','CAPTION']):
        """
        Function to define task
        Args:
            task_type (list): List of task types
        Returns:
            None
        """
        self.task_type = task_type

    def set_image(self,image_path:str):
        """
        Function to set image
        Args:
            image_path (str): Image path
        Returns:
            None
        """
        self.image = Image.open(image_path)

    def generate_text(self,prompt:str =None):
        """
        Function to generate text
        Args:
            prompt (str): Prompt
        Returns:
            res (str): Result
        """
        final_ans = []
        for i in self.task_type:
            if prompt:
                prompt = i + prompt
            else:
                prompt = i
        
            inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(inputs["input_ids"],pixel_values=inputs["pixel_values"],
                                             max_new_tokens=1024,early_stopping=False,do_sample=False,num_beams=3)
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=i, 
                                                                image_size=(self.image.width, self.image.height))
            final_ans.append(parsed_answer)
        return final_ans
    
    def plot_box(self,data,image=None,show=True,save_path=None):
        """
        Plot BBox on image
        Args:
            data (dict): BBox data
            image (PIL Image): Image
        Returns:
            fig (matplotlib figure): Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        if image is None:
            image = self.image
        if show:
            ax.imshow(image)

        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1),
                                 x2 - x1,
                                 y2 - y1,
                                 linewidth=2,
                                 edgecolor='lime',
                                 facecolor='none')
            ax.add_patch(rect)
            plt.text(x1,
                 y1,
                 label,
                 color='black',
                 fontsize=8,
                 bbox=dict(facecolor='lime', alpha=1))

        ax.axis('off')
        if show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)

    def draw_polygons(self, prediction,image=None, fill_mask=False,show=True,save_path=None):
        """
        Draws segmentation masks with polygons on an image.
        Args:
            image (PIL Image): The image to draw on.
            prediction (dict): The prediction containing polygons and labels.
            fill_mask (bool): Whether to fill the polygons with color.
        """
        if image is None:
            image = self.image
        draw = ImageDraw.Draw(image)
        scale = 1

        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            color = "lime"
            fill_color = "lime" if fill_mask else None

            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue

                _polygon = (_polygon * scale).reshape(-1).tolist()
                if fill_mask:
                    draw.polygon(_polygon, outline=color, fill=fill_color)
                else:
                    draw.polygon(_polygon, outline=color)
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
        if show:
            plt.imshow(image)
        if save_path is not None:
            image.save(save_path)

        return image

    def convert_to_od_format(self,data):
        """
        Convert data to od format
        Args:
            data (dict): Data
        Returns:
            od_results (dict): od results
        """
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', [])
        od_results = {'bboxes': bboxes, 'labels': labels}

        return od_results
    
    def draw_ocr_bbox(self,data,image=None,show=True,save_path=None):
        """
        Draw OCR BBox
        Args:
            data (dict): Data
            image (PIL Image): Image
            show (bool): Show
            save_path (str): Save path
        Returns:
            fig (matplotlib figure): Figure
        """
        scale = 1
        if image is None:
            image = self.image
        
        draw = ImageDraw.Draw(image)
        bboxes,labels= data['bboxes'],data['labels']

        for bbox, label in zip(bboxes, labels):
            color='lime'
            new_box= (np.array(bbox)*scale).tolist()
            draw.polygon(new_box,width=4, outline=color)
            draw.text((new_box[0]+8, new_box[0][1]+2), "{}".format(label),align='right', fill=color)

        if show:
            plt.imshow(image)
        if save_path is not None:
            image.save(save_path)
        return image
    
    def load_model(self,model_path:str):
        """
        Function to load model
        Args:
            model_path (str): Model path
        Returns:
            None
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def load_collate(self):
        """
        Function to load collate
        Args:
            None
        Returns:
            None
        """
        partial_collate = partial(my_collate_fn, processor=self.processor)
        return partial_collate

class load_florence_dataset:
    """
    Requires csv file path.
    CSV file should have:
        image path (image_path) ,
        boundingbox or suffix (bbox in the format [xmin, ymin, xmax, ymax] -should be in order of labels.) -list[list], 
        prefix (if not mentioned, default <OD> will be used), 
        labels or suffix (labels will be used as suffix after converting. if suffix is used, please make it in proper format) -list[list]
        train_type : train or validation (if not mentioned, default random 80:20 will be used)
        
    Check the test file for more details. (./test_data/florence_test.csv)   
    Args:
        csv_file_path (str): Path to the csv file
    Returns:
        None
    """

    def __init__(self,csv_file_path) -> None:
        self.df = pd.read_csv(csv_file_path)
        if 'bbox' not in self.df.columns and 'suffix' not in self.df.columns:
            raise ValueError("CSV file should have 'bbox' column or 'suffix' columns")
        if 'labels' not in self.df.columns and 'suffix' not in self.df.columns:
            raise ValueError("CSV file should have 'labels' or 'suffix' columns")
        if 'prefix' not in self.df.columns:
            self.df['prefix'] = '<OD>'
        if 'prefix' in self.df.columns:
            prefix_type_list =  ['<CAPTION>','<DETAILED_CAPTION>','<MORE_DETAILED_CAPTION>','<OCR>','<OCR_WITH_REGION>',
                                 '<OD>','<REGION_PROPOSAL>','<DENSE_REGION_PROPOSAL>','<REGION_TO_SEGMENTATION>',
                                 '<REGION_TO_CATOGORY>','<REGION_TO_DESCRIPTION>',
                                 '<PHRASE_GROUNDING>','<OPEN_VOCABULARY_DETECTION>','<REFERRING_EXPRESSION_SEGMENTATION>']
            prefix_u = list(self.df['prefix'].unique())
            assert len(prefix_u)==1,'Check prefix - multiple values'
            assert type(prefix_type_list.index(prefix_u[0]))==int,'Check prefix value. Not in index'
        
        if 'train_type' not in self.df.columns:
            self.df['train_type'] = 'train'
            val_indices = np.random.choice(self.df.index, size=int(len(self.df) * 0.2), replace=False)
            self.df.loc[val_indices, 'train_type'] = 'validation'

        final_data = []
        if 'suffix' not in self.df.columns:
            self.df['suffix'] = None
            temp_dict={}
            prefix_val = self.df.prefix.iloc[0]
            for i, row in self.df.iterrows():
                if isinstance(row['bbox'], str):
                    bbox_list = eval(row['bbox'])
                else:
                    bbox_list = row['bbox']
                    
                if isinstance(row['labels'], str):
                    try:
                        labels_list = eval(row['labels'])
                    except:
                        labels_list = [row['labels']]
                else:
                    labels_list = row['labels']
                assert len(bbox_list)==len(labels_list),f'BBox list and labels list not equal in row {i}'
                temp_str = ""
                for j in range(len(bbox_list)):
                    if isinstance(bbox_list[j],str):
                        bbox_list[j] = eval(bbox_list[j])
                    temp_str = temp_str+ f"{labels_list[j]}<loc_{int(bbox_list[j][0])}><loc_{int(bbox_list[j][1])}><loc_{int(bbox_list[j][2])}><loc_{int(bbox_list[j][3])}>"
                self.df.at[i, 'suffix'] = temp_str
                temp_dict['image'] = row['image_path']
                temp_dict['prefix'] = prefix_val
                temp_dict['suffix'] = temp_str
                final_data.append(temp_dict)
        self.final_csv = pd.DataFrame(final_data)
        self.final_csv.to_csv('./final_data_florence.csv',index=False)
        print(f'final data example : {final_data[0]}')
        print(f'final csv example: {self.final_csv.head(2)}')
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.final_csv.iloc[idx]

class dataset_data(Dataset):
    def __init__(self,df) -> None:
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # row_data = self.final_csv[self.final_csv.index==idx] ## check how to seperate train/val before goin to this step
        image_path = self.df.iloc[idx]['image_path']
        prefix = self.df.iloc[idx]['prefix']
        suffix = self.df.iloc[idx]['suffix']
        try:
            image = Image.open(image_path)
        except:
            print(f"Error opening image: {image_path}")
        return prefix, suffix, image
    
def my_collate_fn(batch,processor):
    """
    Collate function for the dataloader
    Args:
        batch (list): List of data
    Returns:
        dict: Dictionary of data
    """
    prefix = [item[0] for item in batch]
    suffix = [item[1] for item in batch]
    image = [item[2] for item in batch]
    inputs = processor(text=list(prefix), images=list(image), return_tensors="pt", padding=True).to('cpu')
    return inputs, suffix



# def collate_fn(batch):
#     """
#     Collate function for the dataloader
#     Args:
#         batch (list): List of data
#     Returns:
#         dict: Dictionary of data
#     """
#     prefix = [item[0] for item in batch]
#     suffix = [item[1] for item in batch]
#     image = [item[2] for item in batch]
#     inputs = processor(text=list(prefix), images=list(image), return_tensors="pt", padding=True).to('cpu')
#     return inputs, suffix

