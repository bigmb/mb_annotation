"""
Utility Module

This module provides utility functions for environment variable management and video processing.
"""

import os
from dotenv import load_dotenv
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional

__all__ = ["UtilityManager"]

class UtilityManager:
    """
    A class providing utility functions for environment management and video processing.

    This class encapsulates utility functions for:
    1. Loading and managing environment variables
    2. Converting videos to sequences of images
    """

    def __init__(self):
        """Initialize the UtilityManager."""
        self.env_vars = None

    def load_env_file(self, file_path: str = './.env') -> Dict[str, str]:
        """
        Load environment variables from a .env file.

        Args:
            file_path (str): Path to the .env file. Defaults to './.env'.

        Returns:
            Dict[str, str]: Dictionary of loaded environment variables
        """
        load_dotenv(file_path)
        # self.env_vars = os.environ
        # return self.env_vars

    def video_to_images(self,
                       video_path: str,
                       image_save_path: str,
                       image_name: str,
                       frame_interval: int = 1,
                       duration: Optional[float] = None,
                       image_format: str = 'png') -> List[str]:
        """
        Convert a video file to a sequence of images.

        Args:
            video_path (str): Path to the video file
            image_save_path (str): Directory path to save the images
            image_name (str): Base name for the generated images
            frame_interval (int): Interval between frames to capture. Defaults to 1
            duration (Optional[float]): Maximum duration of video to process in seconds. Defaults to None
            image_format (str): Format for saved images. Defaults to 'png'

        Returns:
            List[str]: List of paths to the generated images

        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if frame_interval < 1:
            raise ValueError("frame_interval must be at least 1")

        os.makedirs(image_save_path, exist_ok=True)    
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        image_list = []
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Converting video to images") as pbar:
            while True:
                success, frame = video.read()
                if not success:
                    break

                current_time = frame_count / fps
                if duration is not None and current_time > duration:
                    print(f"Video duration reached ({duration}s)")
                    print(f"Total frames: {total_frames}")
                    print(f"Current frame: {frame_count}")
                    print(f"Current time: {current_time:.2f}s")
                    break
                
                if frame_count % frame_interval == 0:
                    output_path = os.path.join(
                        image_save_path,
                        f"{image_name}_{frame_count:04d}.{image_format}"
                    )
                    cv2.imwrite(output_path, frame)
                    image_list.append(output_path)

                frame_count += 1
                pbar.update(1)
        
        video.release()
        return image_list
