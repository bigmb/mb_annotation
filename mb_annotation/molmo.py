"""
Molmo Module

This module provides functionality for working with the Molmo model for image and text processing.
It includes capabilities for model initialization, inference, and coordinate extraction/plotting.
"""

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
from PIL import Image
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Any

__all__ = ["MolmoModel"]

class MolmoModel:
    """
    A class for handling Molmo model operations including inference and coordinate processing.

    This class provides functionality to:
    1. Initialize and configure Molmo models
    2. Run inference on image-text pairs
    3. Extract and process coordinate points
    4. Visualize points on images

    Attributes:
        device (str): The device to run the model on ('cpu' or 'cuda')
        model_name (str): Name of the Molmo model
        model_path (Optional[str]): Path to a local model
        model: The loaded Molmo model
        processor: The model's processor
        image: The currently loaded image
    """

    def __init__(self, 
                 model_name: str = "allenai/Molmo-7B-D-0924",
                 model_path: Optional[str] = None,
                 processor: Optional[Any] = None,
                 device: str = 'cpu') -> None:
        """
        Initialize the MolmoModel.

        Args:
            model_name (str): Name of the Molmo model to use
            model_path (Optional[str]): Path to a local model file
            processor (Optional[Any]): Custom processor for the model
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        if device == 'cpu':
            device = "cpu"
        elif device == 'cuda':
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        
        # Initialize model
        if model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map=self.device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map=self.device
            )

        # Initialize processor
        if processor:
            self.processor = processor
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map=self.device
            )
            
    def run_inference(self, image: Union[str, Image.Image], text: str) -> str:
        """
        Run inference on an image-text pair using the Molmo model.

        Args:
            image (Union[str, Image.Image]): Path to the image or PIL Image object
            text (str): Text prompt for the model

        Returns:
            str: Generated text from the model
        """
        if isinstance(image, str):
            self.image = Image.open(image)
        else:
            self.image = image

        inputs = self.processor.process(images=[self.image], text=text)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)            
        return generated_text

    def extract_points(self, text: str) -> np.ndarray:
        """
        Extract coordinate points from generated text.

        Args:
            text (str): Text containing coordinate information

        Returns:
            np.ndarray: Array of extracted coordinates
        """
        pattern = r'x(\d+)\s*=\s*"([\d.]+)"\s*y(\d+)\s*=\s*"([\d.]+)"'
        matches = re.findall(pattern, text)
        coordinates = [(float(x), float(y)) for _, x, _, y in matches]
        return np.array(coordinates)
    
    def final_coordinates(self, 
                         text: str,
                         plot: bool = True,
                         **kwargs) -> np.ndarray:
        """
        Process and optionally visualize coordinates from text.

        Args:
            text (str): Text containing coordinate information
            plot (bool): Whether to plot the coordinates on the image
            **kwargs: Additional arguments for plot_points

        Returns:
            np.ndarray: Array of processed coordinates
        """
        res = self.extract_points(text)
        res_updated = res * np.array(self.image.size) / 100
        if plot:
            self.plot_points(res_updated, **kwargs)
        return res_updated

    def plot_points(self,
                   points: np.ndarray,
                   radius: int = 10,
                   thickness: int = -10,
                   color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        """
        Plot points on the current image.

        Args:
            points (np.ndarray): Array of coordinates to plot
            radius (int): Radius of the plotted points
            thickness (int): Thickness of the point markers (-1 for filled)
            color (Tuple[int, int, int]): RGB color for the points
        """
        image_point = np.array(self.image)
        for x, y in points:
            image_point = cv2.circle(
                image_point,
                (int(x), int(y)),
                radius=radius,
                color=color,
                thickness=thickness
            )
        plt.imshow(image_point)
