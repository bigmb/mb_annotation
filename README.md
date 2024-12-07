# MB LLM

MB LLM is a Python package that provides tools for image and video annotation, segmentation, object detection and Fine tuning models. It integrates several powerful models and techniques including SAM2 (Segment Anything 2), Florence, and Molmo.

## Features

- **Florence Integration**: Advanced image understanding and processing capabilities
- **Molmo Support**: Specialized molecular and material analysis features
- **SAM2 Integration**: State-of-the-art segmentation capabilities including:
  - Automatic mask generation
  - Video segmentation
  - Image prediction
  - Custom model training

## Installation

```bash
pip install mb_llm
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add your API keys and configurations to `.env`
3. Load environment variables using the provided utility:

```python
from mb_llm.utils import load_env_file
env_vars = load_env_file()
```

## Usage Examples

### Bounding Box Detection

```python
from mb_llm.bounding_box import google_model, generate_bounding_box, add_bounding_box

# Initialize model
model = google_model(api_key="your_api_key")

# Generate bounding boxes
result = generate_bounding_box(model, "path/to/image.jpg")

# Add bounding box to image
image, box = add_bounding_box("path/to/image.jpg", bounding_box, "label")
```

### Florence Model Usage

```python
from mb_llm.florencefile import florence_model

# Initialize model
model = florence_model(model_name="microsoft/Florence-2-base")

# Set image and generate text
model.set_image("path/to/image.jpg")
result = model.generate_text(prompt="Describe this image")
```

### Molmo Model Usage

```python
from mb_llm.molmo import molmo_model

# Initialize model
model = molmo_model(model_name="allenai/Molmo-7B-D-0924")

# Run inference
result = model.run_inference("path/to/image.jpg", "text prompt")
```

### SAM2 Segmentation

```python
from mb_llm.segsam2 import get_mask_generator, image_predictor

# Get mask generator
mask_generator = get_mask_generator(sam2_checkpoint="path/to/checkpoint.pt")

# Use image predictor
predictor = image_predictor(model_cfg="config.yaml", sam2_checkpoint="checkpoint.pt")
predictor.set_image("path/to/image.jpg")
masks, scores, logits = predictor.predict_item(bbox=[x0, y0, x1, y1])
```

### Video Processing

```python
from mb_llm.utils import video_to_images

# Convert video to image frames
image_list = video_to_images(
    video_path="video.mp4",
    image_save_path="frames/",
    image_name="frame",
    frame_interval=1
)
```

## Module Overview

- `florencefile.py`: Integration with Florence model for image understanding
- `molmo.py`: Specialized functions for molecular and material analysis
- `segsam2.py`: SAM2 integration for advanced segmentation tasks
- `utils.py`: Utility functions for environment setup and video processing

