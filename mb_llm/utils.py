## extra functions

import os
from dotenv import load_dotenv
import cv2
from tqdm import tqdm

__all__ = ["load_env_file","video_to_images"]

def load_env_file(file_path='./env'):
    """
    Load environment variables from a .env file.

    Args:
        file_path (str): Path to the .env file. Defaults to '.env'.

    Returns:
        None
    """
    load_dotenv(file_path)

    # Get the loaded environment variables
    env_vars = os.environ

    return env_vars

def video_to_images(video_path,image_save_path,image_name,frame_interval=1,duration=None,image_format='png'):
    """
    Convert a video to images
    Args:
        video_path (str): Path to the video file.
        image_save_path (str): Path to save the images.
        image_name (str): Name of the images.
        image_format (str): Format of the images. Defaults to 'png'.
        frame_interval (int): Frame interval. Defaults to 1.
        duration (float): Duration of the video. Defaults to None.
    Returns:
        image_list (list): List of images.
    """
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
                print("Video duration reached")
                print(f"Total frames: {total_frames}")
                print(f"Current frame: {frame_count}")
                print(f"Current time: {current_time}")
                break
            
            if frame_count % frame_interval == 0:
                output_path = os.path.join(image_save_path, f"{image_name}_{frame_count:04d}.{image_format}")
                cv2.imwrite(output_path, frame)
                image_list.append(output_path)

            frame_count += 1
            pbar.update(1)
    
    video.release()
    
    return image_list