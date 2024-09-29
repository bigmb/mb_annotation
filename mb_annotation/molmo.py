## file for molmo functions

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
from PIL import Image
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

__all__ = ["molmo_model"]

class molmo_model:
    def __init__(self,model_name="allenai/Molmo-7B-D-0924",model_path=None,processor = None,device='cpu') -> None:
        if device == 'cpu':
            device = "cpu"
        elif device == 'cuda':
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        if model_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,
                                                              torch_dtype='auto',device_map=self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,
                                                              torch_dtype='auto',device_map=self.device)

        if processor:
            self.processor = processor
        else:
            self.processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True,
                                                              torch_dtype='auto',device_map=self.device)
            
    def run_inference(self,image,text):
        """
        Function to run inference
        Args:
            image (str): Path to the image
            text (str): Text to be predicted
        Returns:
            None
        """
        if isinstance(image,str):
            self.image = Image.open(image)
        inputs = self.processor.process(images=[self.image],text=text)

        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(inputs,
                                                GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
                                                tokenizer=self.processor.tokenizer)
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)            
        return generated_text
    def extract_points(self,text):
        pattern = r'x(\d+)\s*=\s*"([\d.]+)"\s*y(\d+)\s*=\s*"([\d.]+)"'
        matches = re.findall(pattern, text)
        coordinates = [(float(x), float(y)) for _, x, _, y in matches]
        return np.array(coordinates)
    
    def final_coordinates(self,text,plot=True,**kwargs):
        res = self.extract_points(text)
        res_updated = res * np.array(self.image.size) / 100
        if plot:
            self.plot_points(res_updated,**kwargs)
        return res_updated

    def plot_points(self,points,radius =10, thickness=-10,color=(0, 0, 255)):
        
        image_point = np.array(self.image)
        for x,y in points:
            image_point = cv2.circle(image_point, (int(x),int(y)), radius=10, color=color, thickness=thickness)
        plt.imshow(image_point)

