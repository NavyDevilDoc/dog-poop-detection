# Project Configuration Documentation

## Overview

The `project_config.yaml` file serves as the central configuration hub for the computer vision project. It defines all settings, parameters, and metadata needed for dataset creation, annotation, training data splitting, and YOLO format compliance.

## Configuration Structure

The configuration is organized into logical sections that correspond to different phases of the machine learning pipeline:

```yaml
project:           # Project metadata and identification
annotation:        # Annotation tool settings
dataset:           # Dataset structure and class definitions
directories:       # File system organization
extraction:        # Media processing parameters
patterns:          # File format and naming conventions
splitting:         # Train/validation/test split configuration
yolo:             # YOLO format specifications
```

## Configuration Sections

### Project Metadata

```yaml
project:
  name: poop_detection
  description: Multi-class dog poop detection system
  version: 1.0.0
  created: '2025-11-01T15:19:13.336469'
```

**Fields:**
- `name`: Project identifier used in filenames and logs
- `description`: Human-readable project description
- `version`: Semantic versioning for project tracking
- `created`: ISO timestamp of project initialization

**Usage:** Referenced by tools for logging, file naming, and metadata tracking.

### Annotation Configuration

```yaml
annotation:
  auto_save: true
  default_class: poop
  validation_on_save: true
```

**Fields:**
- `auto_save`: Automatically saves annotations during labeling process
- `default_class`: Pre-selected class in annotation tools (speeds up workflow)
- `validation_on_save`: Validates annotation format before saving

**Impact:** Controls behavior of annotation tools and ensures data quality during the labeling process.

### Dataset Structure

```yaml
dataset:
  name: poop_detection_dataset
  classes:
    - poop
    - leaf
    - stick
    - rock
  class_colors:
    poop: red
    leaf: green
    stick: brown
    rock: gray
```

**Class Definitions:**
- `classes`: Ordered list of object classes (order determines class IDs in YOLO format)
- `class_colors`: Visual colors for bounding boxes in annotation tools
- `name`: Dataset directory name

**Class ID Mapping:**
```
0: poop   (red)
1: leaf   (green)
2: stick  (brown)
3: rock   (gray)
```

### Directory Structure

```yaml
directories:
  base: poop_detection_dataset
  images: images
  labels: labels
  train: train
  val: val
  test: test
  yolo_labels: yolo_labels
```

**Resulting File Structure:**
```
poop_detection_dataset/           # base
├── images/                       # Raw images
├── labels/                       # Annotation files
├── train/                        # Training split
│   ├── images/
│   └── labels/
├── val/                         # Validation split
│   ├── images/
│   └── labels/
├── test/                        # Test split (optional)
│   ├── images/
│   └── labels/
└── yolo_labels/                 # YOLO format exports
```

### Extraction Parameters

```yaml
extraction:
  target_frames_per_video: 25
  output_format: jpg
  quality: 95
  preview_samples: 6
```

**Parameters:**
- `target_frames_per_video`: Number of frames to extract from each video
- `output_format`: Image format for extracted frames (`jpg`, `png`)
- `quality`: JPEG compression quality (1-100, higher = better quality)
- `preview_samples`: Number of sample images to show in preview grid

**Usage Example:**
```python
# In image_extraction.py
TARGET_FRAMES = config.get('extraction.target_frames_per_video', 25)
QUALITY = config.get('extraction.quality', 95)
```

### File Patterns and Formats

```yaml
patterns:
  image_extensions:
    - .jpg
    - .jpeg
    - .png
  video_extensions:
    - .mp4
    - .avi
    - .mov
    - .mkv
  timestamp_format: '%Y%m%d_%H%M%S'
```

**Format Specifications:**
- `image_extensions`: Supported image file types for processing
- `video_extensions`: Supported video file types for frame extraction
- `timestamp_format`: Python `strftime` format for file naming

**Generated Filenames:**
```
extracted_frames_video1_20251101_151913/
video1_frame_0001.jpg
session_20251101_151913.json
```

### Dataset Splitting Configuration

```yaml
splitting:
  train_ratio: 0.8      # 80% for training
  val_ratio: 0.2        # 20% for validation
  test_ratio: 0.0       # 0% for testing (optional)
  random_seed: 42       # Reproducible splits
  stratified: false     # Simple random splitting
```

**Split Calculation:**
- Total images: 1000
- Training set: 800 images (80%)
- Validation set: 200 images (20%)
- Test set: 0 images (0%)

**Stratified Splitting:**
When `stratified: true`, maintains class distribution across splits:
```python
# Example with 1000 images, 40% poop, 60% other classes
train_set: 800 images (320 poop, 480 others)
val_set:   200 images (80 poop, 120 others)
```

### YOLO Format Configuration

```yaml
yolo:
  format_version: '8'           # YOLOv8 format
  coordinate_format: normalized # Normalized coordinates (0-1)
  min_annotation_rate: 0.2      # Minimum 20% of images must be annotated
```

**YOLO Format Specifications:**
- `format_version`: YOLO version compatibility (`5`, `8`, etc.)
- `coordinate_format`: Coordinate system (`normalized`, `absolute`)
- `min_annotation_rate`: Quality control threshold for dataset completion

**YOLO Label Format:**
```
# Each line: class_id center_x center_y width height
0 0.5 0.6 0.3 0.4    # poop at center-right, 30% width, 40% height
1 0.2 0.8 0.1 0.15   # leaf at bottom-left, 10% width, 15% height
```

## Usage in Python Scripts

### Loading Configuration

```python
import yaml
from pathlib import Path

def load_config():
    config_path = Path("poop_detection_dataset/project_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
```

### Accessing Configuration Values

```python
# Project info
project_name = config['project']['name']
classes = config['dataset']['classes']

# Extraction settings
frames_per_video = config['extraction']['target_frames_per_video']
image_quality = config['extraction']['quality']

# Directory paths
base_dir = Path(config['directories']['base'])
images_dir = base_dir / config['directories']['images']
labels_dir = base_dir / config['directories']['labels']

# Split ratios
train_ratio = config['splitting']['train_ratio']
val_ratio = config['splitting']['val_ratio']
```

### Dynamic Configuration Updates

```python
def update_config(config_path, updates):
    """Update configuration file with new values"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update nested values
    for key_path, value in updates.items():
        keys = key_path.split('.')
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

# Example: Update frame extraction count
update_config('project_config.yaml', {
    'extraction.target_frames_per_video': 50,
    'splitting.val_ratio': 0.15
})
```

## Configuration Best Practices

### 1. Environment-Specific Configs

```yaml
# config/development.yaml
extraction:
  target_frames_per_video: 10    # Faster for testing
  preview_samples: 3

# config/production.yaml  
extraction:
  target_frames_per_video: 100   # Higher quality dataset
  preview_samples: 6
```

### 2. Validation Schema

```python
CONFIG_SCHEMA = {
    'project': {'name': str, 'version': str},
    'dataset': {'classes': list},
    'splitting': {
        'train_ratio': float,
        'val_ratio': float,
        'test_ratio': float
    }
}

def validate_config(config):
    """Validate configuration against schema"""
    # Check split ratios sum to 1.0
    ratios = config['splitting']
    total = ratios['train_ratio'] + ratios['val_ratio'] + ratios['test_ratio']
    assert abs(total - 1.0) < 0.001, f"Split ratios must sum to 1.0, got {total}"
    
    # Check classes are defined
    assert len(config['dataset']['classes']) > 0, "Must define at least one class"
```

### 3. Configuration Inheritance

```yaml
# base_config.yaml
defaults:
  extraction:
    quality: 95
    output_format: jpg
  
# project_config.yaml (inherits from base)
extends: base_config.yaml
extraction:
  target_frames_per_video: 25   # Overrides base
  # quality: 95 inherited from base
```

## Integration Examples

### With Image Extraction

```python
# image_extraction.py
config = load_config()
TARGET_FRAMES = config['extraction']['target_frames_per_video']
DATASET_NAME = config['dataset']['name']
OUTPUT_FORMAT = config['extraction']['output_format']
```

### With Annotation Tools

```python
# annotation_tool.py
config = load_config()
CLASSES = config['dataset']['classes']
CLASS_COLORS = config['dataset']['class_colors']
AUTO_SAVE = config['annotation']['auto_save']
DEFAULT_CLASS = config['annotation']['default_class']
```

### With Dataset Splitting

```python
# split_dataset.py
config = load_config()
train_ratio = config['splitting']['train_ratio']
val_ratio = config['splitting']['val_ratio']
random_seed = config['splitting']['random_seed']
```

## Migration and Versioning

### Configuration Updates

```python
def migrate_config_v1_to_v2(config):
    """Migrate from version 1.0 to 2.0"""
    if config.get('version') == '1.0.0':
        # Add new fields
        config['yolo']['min_annotation_rate'] = 0.2
        config['project']['version'] = '2.0.0'
    return config
```

### Backward Compatibility

```python
def get_with_fallback(config, key_path, default):
    """Get config value with fallback for missing keys"""
    try:
        keys = key_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value
    except KeyError:
        return default

# Usage
frames = get_with_fallback(config, 'extraction.target_frames_per_video', 50)
```

This configuration system provides a flexible, maintainable foundation for the entire computer vision pipeline, enabling easy customization and environment-specific deployments.