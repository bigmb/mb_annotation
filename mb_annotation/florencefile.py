## file for florence functions

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
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
            if prompt == None:
                prompt = self.task_type
            else:
                prompt = self.task_type + prompt
        
            inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(inputs["input_ids"],pixel_values=inputs["pixel_values"],
                                             max_new_tokens=1024,early_stopping=False,do_sample=False,num_beams=3)
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=self.task_type, 
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
        fig, ax = plt.subplots(figsize=(15, 10))
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
        return fig

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