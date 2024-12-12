"""
Florence Module

This module provides integration with Microsoft's Florence model for advanced image understanding
and processing. It includes functionality for model initialization, training, and inference.
"""

import torch
from PIL import Image, ImageDraw
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
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator, Optional, Union
import os

__all__ = ["FlorenceModel", "FlorenceDatasetLoader", "FlorenceDataset"]

class FlorenceModel:
    """
    A class for handling Florence model operations including inference and visualization.

    This class provides functionality to:
    1. Initialize and configure Florence models
    2. Run inference on images
    3. Visualize model predictions
    4. Train and fine-tune models

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda')
        model_name (str): Name of the Florence model
        model: The loaded Florence model
        processor: The model's processor
        peft_model: LoRA-adapted model for fine-tuning
    """

    def __init__(self, 
                 model_name: str = "microsoft/Florence-2-base",
                 finetuned_model: bool = False,
                 device: str = 'cpu') -> None:
        """
        Initialize the FlorenceModel.

        Args:
            model_name (str): Name or path of the Florence model
            finetuned_model (bool): Whether to load a finetuned model
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = self._setup_device(device)
        self.model_name = model_name
        self._initialize_model(finetuned_model)

    def _setup_device(self, device: str) -> str:
        """Set up the computation device."""
        if device == 'cuda':
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_model(self, finetuned_model: bool) -> None:
        """Initialize the model and processor."""
        if finetuned_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        else:
            self._ft_file_dir = './florence_model_cache'
            os.makedirs(self._ft_file_dir, exist_ok=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self._ft_file_dir,
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

    def get_task_types(self) -> Dict[str, List[str]]:
        """
        Get available task types for the Florence model.

        Returns:
            Dict[str, List[str]]: Dictionary of available task types and their prompts
        """
        return {
            'captions': ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>'],
            'character_recognition': ['<OCR>', '<OCR_WITH_REGION>'],
            'object_detection': ['<OD>', '<REGION_PROPOSAL>', '<DENSE_REGION_PROPOSAL>'],
            'segmentation': ['<REGION_TO_SEGMENTATION>'],
            'description': ['<REGION_TO_CATOGORY>', '<REGION_TO_DESCRIPTION>'],
            'extra': ['<PHRASE_GROUNDING>', '<OPEN_VOCABULARY_DETECTION>', 
                     '<REFERRING_EXPRESSION_SEGMENTATION>']
        }

    def define_task(self, task_type: List[str]) -> None:
        """
        Define the task type for the model.

        Args:
            task_type (List[str]): List of task types to use
        """
        self.task_type = task_type

    def set_image(self, image_path: str) -> None:
        """
        Set the image for processing.

        Args:
            image_path (str): Path to the image file
        """
        self.image = Image.open(image_path)

    def generate_text(self, image: Optional[str] = None, prompt: Optional[str] = None) -> List[Any]:
        """
        Generate text based on the image and prompt.

        Args:
            image (Optional[str]): Optional path to the image file
            prompt (Optional[str]): Optional prompt to guide text generation

        Returns:
            List[Any]: List of generated results
        """
        if isinstance(image, str):
            self.set_image(image)
        elif not self.image:
            print('Image not set. Please provide an image path or set the image.')
        final_ans = []
        for task in self.task_type:
            current_prompt = f"{task}{prompt}" if prompt else task
            inputs = self._prepare_inputs(current_prompt)
            generated_text = self._generate_and_process(inputs, task)
            final_ans.append(generated_text)
        return final_ans

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        inputs = self.processor(
            text=prompt,
            images=self.image,
            return_tensors="pt"
        ).to(self.device)
        return inputs

    def _generate_and_process(self, inputs: Dict[str, torch.Tensor], task: str) -> Any:
        """Generate text and process the output."""
        output_ids = self.model.generate(
            inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=False
        )[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(self.image.width, self.image.height)
        )


    def plot_box(self,
                 data: Dict[str, List]= None,
                 image: Optional[str] = None,
                 show: bool = True,
                 save_path: Optional[str] = None) -> None:
        """
        Plot bounding boxes on an image.

        Args:
            data (Dict[str, List]): Dictionary containing bounding boxes and labels. if not provided, it will generate from the image
            image (Optional[str]): Image path to plot on. If not provided, it will use the set image
            show (bool): Whether to display the plot
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        if isinstance(image, str):
            self.set_image(image)
        if show:
            image = self.image
            ax.imshow(image)

        if self.task_type==None:
            print('Task type not defined. Using default task type <OD>')
            self.task_type = ['<OD>']
        
        print('Task type:',self.task_type)

        if data == None:
            data = self.generate_text()           

        for task in self.task_type:
            for bbox, label in zip(data[0][task]['bboxes'], data[0][task]['labels']):
                self._draw_bbox(ax, bbox, label)

        ax.axis('off')
        if show:
            plt.show()
        if save_path:
            fig.savefig(save_path)

    def _draw_bbox(self, ax: plt.Axes, bbox: List[float], label: str) -> None:
        """Draw a single bounding box with label."""
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1,
            label,
            color='black',
            fontsize=8,
            bbox=dict(facecolor='lime', alpha=1)
        )

    def draw_polygons(self,
                     prediction: Dict[str, List],
                     image: Optional[Image.Image] = None,
                     fill_mask: bool = False,
                     show: bool = True,
                     save_path: Optional[str] = None) -> Image.Image:
        """
        Draw segmentation masks with polygons on an image.

        Args:
            prediction (Dict[str, List]): Dictionary containing polygons and labels
            image (Optional[Image.Image]): Image to draw on or str of image path
            fill_mask (bool): Whether to fill the polygons
            show (bool): Whether to display the result
            save_path (Optional[str]): Path to save the result

        Returns:
            Image.Image: Image with drawn polygons
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif not image:
            image = self.image
        draw = ImageDraw.Draw(image)

        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            self._draw_polygon(draw, polygons, label, fill_mask)

        if show:
            plt.imshow(image)
        if save_path:
            image.save(save_path)

        return image

    def _draw_polygon(self,
                     draw: ImageDraw.Draw,
                     polygons: List[List[float]],
                     label: str,
                     fill_mask: bool) -> None:
        """Draw a single polygon with label."""
        color = "lime"
        fill_color = "lime" if fill_mask else None

        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1, 2)
            if len(polygon) < 3:
                print('Invalid polygon:', polygon)
                continue

            polygon = polygon.reshape(-1).tolist()
            draw.polygon(polygon, outline=color, fill=fill_color)
            draw.text((polygon[0] + 8, polygon[1] + 2), label, fill=color)

    def load_model(self, model_path: str) -> None:
        """
        Load a pretrained model from path.

        Args:
            model_path (str): Path to the model folder
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(self.device)

    def _my_collate_fn(self,batch: List) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Collate function for the dataloader.

        Args:
            batch (list): List of data items

        Returns:
            tuple: Processed batch data
        """
        prefix = [item[0] for item in batch]
        suffix = [item[1] for item in batch]
        image = [item[2] for item in batch]
        inputs = self.processor(text=list(prefix), images=list(image), return_tensors="pt", padding=True).to('cpu')
        return inputs, suffix
    
    def _dataloader(self,dataset: Dataset, batch_size: int) -> DataLoader:
        """
        Create a DataLoader for the dataset.

        Args:
            dataset (Dataset): Input dataset
            batch_size (int): Batch size

        Returns:
            DataLoader: DataLoader instance
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self._my_collate_fn,shuffle=True)
        return dataloader

    def dataset_prepare(self,df: Optional[pd.DataFrame] ,batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare the dataset for training.
        Args:
            df (pd.DataFrame): DataFrame containing dataset information. Can be Path also.
            batch_size (int): Batch size for the data loaders
        Returns:
            DataLoader, DataLoader: Training and validation data loaders
        """
        if isinstance(df,str):
            df = pd.read_csv(df)
        train_dataset = df[df.train_type=='train'].reset_index()
        # train_dataset.drop(columns=['train_type'],inplace=True)
        # train_dataset.drop(columns=['index'],inplace=True)
        val_dataset = df[df.train_type=='validation'].reset_index()
        # val_dataset.drop(columns=['train_type'],inplace=True)
        # val_dataset.drop(columns=['index'],inplace=True)
        train_dataset_new = FlorenceDataset(train_dataset)
        val_dataset_new = FlorenceDataset(val_dataset)
        train_loader = self._dataloader(train_dataset_new,batch_size)
        val_loader = self._dataloader(val_dataset_new,batch_size)
        return train_loader, val_loader

    def _setup_training(self, learning_rate: float = 1e-6,target_modules : List = ["q_proj", "o_proj", "k_proj", "v_proj", "linear",
                          "Conv2d", "lm_head", "fc2"]) -> None:
        """
        Set up the model for training.

        Args:
            learning_rate (float): Learning rate for training
        """
        self._setup_lora(target_modules)
        for param in self.peft_model.vision_tower.parameters():
            try:
                param.requires_grad = False
            except:
                param.is_trainable = False
        self.optimizer = AdamW(self.peft_model.parameters(), lr=learning_rate)

    def _setup_lora(self,modules: List) -> None:
        """Set up LoRA configuration for model fine-tuning."""
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian"
        )
        self.peft_model = get_peft_model(self.model, config)
        self.peft_model.print_trainable_parameters()

    def train_model(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 10,
                   learning_rate: float = 1e-6,
                   target_modules: List = ["q_proj", "o_proj", "k_proj", "v_proj", "linear",
                          "Conv2d", "lm_head", "fc2"],
                    output_dir = './model_checkpoints') -> None:
        """
        Train the Florence model.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            target_modules (List): List of target modules for fine-tuning the model
            output_dir (str): Path to save the model checkpoints
        """
        self._setup_training(learning_rate, target_modules)
        num_training_steps = epochs * len(train_loader)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        for epoch in range(epochs):
            self._train_epoch(epoch, epochs, train_loader)
            self._validate_epoch(epoch, epochs, val_loader)
            self._save_checkpoint(epoch,output_dir)

    def _train_epoch(self,
                    epoch: int,
                    total_epochs: int,
                    train_loader: DataLoader) -> None:
        """Run one training epoch."""
        self.peft_model.train()
        train_loss = 0
        
        for inputs, answers in tqdm(train_loader,
                                  desc=f"Training Epoch {epoch + 1}/{total_epochs}"):
            loss = self._training_step(inputs, answers)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

    def _training_step(self, inputs: Dict[str, torch.Tensor],
                      answers: List[str]) -> float:
        """Perform one training step."""
        labels = self._prepare_labels(answers)
        outputs = self.peft_model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _prepare_labels(self, answers: List[str]) -> torch.Tensor:
        """Prepare labels for training."""
        return self.processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).input_ids.to(self.device)

    def _validate_epoch(self,
                       epoch: int,
                       total_epochs: int,
                       val_loader: DataLoader) -> None:
        """Run one validation epoch."""
        self.peft_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader,
                                      desc=f"Validation Epoch {epoch + 1}/{total_epochs}"):
                labels = self._prepare_labels(answers)
                outputs = self.peft_model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

    def _save_checkpoint(self, epoch: int,output_dir : str) -> None:
        """Save model checkpoint."""
        output_dir = f"{output_dir}/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        self.peft_model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        # shutil.copyfile(f"{self._ft_file_dir}+'/models--microsoft--Florence-2-base-ft/snapshots/", f"{output_dir}/epoch_{epoch+1}/snapshots/")



class FlorenceDatasetLoader:
    """
    Class for loading and preprocessing Florence datasets.

    This class handles:
    1. Loading data from CSV files
    2. Preprocessing image and label data
    3. Generating formatted data for model training

    Args:
        csv_file_path (str): Path to the CSV file containing dataset information
    """

    def __init__(self, csv_file_path: str,loc_to_save: str = './florence_file.csv',train_val_split: float = 0.2) -> None:
        """Initialize the dataset loader.
        Args:
            csv_file_path (str): Path to the CSV file containing dataset information. 
                                 Give prefix and suffix columns otherwise it will assume <OD> as prefix,
                                                         and check for bbox and labels columns for suffix.
                                 Made currently for object detection task.
            loc_to_save (str): Path to save the final CSV file
            train_val_split (float): Train-Validation split ratio
        """

        self.df = pd.read_csv(csv_file_path)
        self.loc_to_save = loc_to_save
        self.train_val_split = train_val_split
        self._validate_dataframe()
        self._initialize_dataframe()
        self._process_data()

    def _validate_dataframe(self) -> None:
        """Validate the input dataframe structure."""
        if 'bbox' not in self.df.columns and 'suffix' not in self.df.columns:
            raise ValueError("CSV file should have 'bbox' or 'suffix' columns")
        if 'labels' not in self.df.columns and 'suffix' not in self.df.columns:
            raise ValueError("CSV file should have 'labels' or 'suffix' columns")

    def _initialize_dataframe(self) -> None:
        """Initialize and validate dataframe columns."""
        if 'prefix' not in self.df.columns:
            self.df['prefix'] = '<OD>'
        self._validate_prefix()
        
        if 'train_type' not in self.df.columns:
            self._split_train_val()
            print(f"Train-Validation split: {self.df['train_type'].value_counts()}")
            print(f"Total images: {len(self.df)}")

    def _validate_prefix(self) -> None:
        """Validate prefix values in the dataframe."""
        prefix_type_list = [
            '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>',
            '<OCR>', '<OCR_WITH_REGION>', '<OD>', '<REGION_PROPOSAL>',
            '<DENSE_REGION_PROPOSAL>', '<REGION_TO_SEGMENTATION>',
            '<REGION_TO_CATOGORY>', '<REGION_TO_DESCRIPTION>',
            '<PHRASE_GROUNDING>', '<OPEN_VOCABULARY_DETECTION>',
            '<REFERRING_EXPRESSION_SEGMENTATION>'
        ]
        prefix_u = list(self.df['prefix'].unique())
        if len(prefix_u) != 1:
            raise ValueError('Multiple prefix values found')
        if prefix_u[0] not in prefix_type_list:
            raise ValueError('Invalid prefix value')

    def _split_train_val(self) -> None:
        """Split data into training and validation sets."""
        self.df['train_type'] = 'train'
        val_indices = np.random.choice(
            self.df.index,
            size=int(len(self.df) * self.train_val_split),
            replace=False
        )
        self.df.loc[val_indices, 'train_type'] = 'validation'

    def _process_data(self) -> None:
        """Process the dataset and create final CSV."""
        final_data = pd.DataFrame()
        if 'suffix' not in self.df.columns:
            self.df['suffix'] = None
            prefix_val = self.df.prefix.iloc[0]
            
            for idx, row in self.df.iterrows():
                processed_row = self._process_row(row, prefix_val)
                final_data = pd.concat(
                    [final_data, pd.DataFrame([processed_row])],
                    ignore_index=True
                )

        self.final_csv = final_data
        self.final_csv.to_csv(self.loc_to_save, index=False)
        print(f'Final CSV example: {self.final_csv.head(2)}')

    def _process_row(self, row: pd.Series, prefix_val: str) -> Dict[str, Any]:
        """Process a single row of data."""
        image_path = row['image_path']
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f'Image not found in path {image_path}: {e}')
            return None

        bbox_list = self._parse_list(row['bbox'])
        labels_list = self._parse_list(row['labels'])

        if len(bbox_list) != len(labels_list):
            print(f'BBox list and labels list not equal in row {row.name}')
            return None

        suffix = self._create_suffix(bbox_list, labels_list, image)
        return {
            'image': image_path,
            'prefix': prefix_val,
            'suffix': suffix,
            'train_type': row['train_type']
        }

    def _parse_list(self, value: Union[str, List]) -> List:
        """Parse string representations of lists."""
        if isinstance(value, str):
            try:
                return eval(value)
            except:
                return [value]
        return value

    def _create_suffix(self,
                      bbox_list: List[List[float]],
                      labels_list: List[str],
                      image: Image.Image) -> str:
        """Create suffix string from bounding boxes and labels."""
        suffix = ""
        for bbox, label in zip(bbox_list, labels_list):
            bbox = self._process_bbox(bbox, image)
            suffix += f"{label}<loc_{bbox[0]}><loc_{bbox[1]}><loc_{bbox[3]}><loc_{bbox[2]}>"
        return suffix

    def _process_bbox(self,
                     bbox: List[float],
                     image: Image.Image) -> List[int]:
        """Process bounding box coordinates."""
        if isinstance(bbox, str):
            bbox = eval(bbox)
        bbox = bbox.copy()
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]

        image_size = np.array([image.width, image.height, image.height, image.width])
        bbox = np.array(bbox) / image_size * 1000
        bbox = bbox.astype(int)
        bbox = [min(x, 999) for x in bbox]
        return bbox

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> pd.Series:
        """Get an item from the final CSV."""
        return self.final_csv.iloc[idx]

class FlorenceDataset(Dataset):
    """
    Dataset class for Florence model training.

    This class provides:
    1. Data loading and preprocessing for training
    2. Batch generation for model training
    3. Train/validation split handling

    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the dataset."""
        self.df = df

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, Image.Image]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            Tuple[str, str, Image.Image]: Tuple of (prefix, suffix, image)
        """
        image_path = self.df.iloc[idx]['image']
        prefix = self.df.iloc[idx]['prefix']
        suffix = self.df.iloc[idx]['suffix']
        
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            
        return prefix, suffix, image
