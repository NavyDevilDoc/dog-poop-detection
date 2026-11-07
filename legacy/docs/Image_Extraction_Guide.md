# Image Extraction Tool Documentation

## Overview

The `image_extraction.py` script is a comprehensive tool for preprocessing media files (videos and photos) to create datasets for computer vision projects, specifically designed for object detection tasks. It automatically extracts frames from videos, processes photos, and sets up annotation-ready directory structures.

## Features

- **Automatic Media Discovery**: Scans `raw_media/New/` directory for videos and photos
- **Frame Extraction**: Extracts evenly distributed frames from videos using ratio-based sampling
- **Photo Processing**: Processes and optionally resizes photos for annotation
- **Status Tracking**: Maintains logs and status files for dataset management
- **Directory Management**: Automatically organizes media into processed/unprocessed folders
- **Annotation Setup**: Creates YOLO-compatible directory structure for labeling
- **Cleanup Tools**: Manages temporary directories and file organization

## Directory Structure

```
project_root/
├── raw_media/
│   ├── New/           # Place new videos/photos here
│   └── Processed/     # Automatically moved after processing
├── dataset_name/
│   ├── images/        # Extracted frames ready for annotation
│   ├── labels/        # YOLO format labels (created during annotation)
│   ├── .annotation_status.json    # Tracks annotation progress
│   └── .extraction_log.json       # Logs extraction sessions
└── extracted_frames_*/            # Temporary extraction directories
```

## Configuration

The script uses a `config.py` file for settings:

```python
config = {
    'extraction': {
        'target_frames_per_video': 50  # Number of frames to extract per video
    },
    'dataset': {
        'name': 'poop_detection_dataset'  # Default dataset name
    }
}
```

## Core Functions

### Media Discovery Functions

#### `scan_media_in_new()`
Automatically scans the `raw_media/New/` directory for video and photo files.

**Returns:**
- `tuple`: (video_files_list, photo_files_list)

**Supported Formats:**
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- Photos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

#### `select_media()`
Primary media selection function that:
1. First attempts auto-discovery in `New/` directory
2. Falls back to manual file selection dialog if no media found
3. Allows user to choose processing all discovered media or manual selection

### Frame Extraction Functions

#### `extract_frames_from_video(video_path, output_dir, target_frames=50)`
Extracts frames from video using ratio-based sampling for even distribution.

**Parameters:**
- `video_path`: Path to input video file
- `output_dir`: Directory to save extracted frames
- `target_frames`: Number of frames to extract (default: 50)

**Algorithm:**
- Calculates frame interval: `total_frames // target_frames`
- Extracts every nth frame to achieve target count
- Saves frames as: `{video_name}_frame_{count:04d}.jpg`

**Returns:** Number of frames successfully extracted

### Photo Processing Functions

#### `process_photos(photo_paths, output_dir)`
Processes photos for annotation readiness.

**Features:**
- Automatic resizing for images larger than 4K resolution
- Maintains aspect ratio during resizing
- Saves as JPEG with 95% quality
- Handles naming conflicts automatically

### Status Tracking Functions

#### `create_status_files(dataset_dir)`
Creates necessary tracking files if they don't exist:
- `.annotation_status.json`: Tracks which images have been annotated
- `.extraction_log.json`: Logs extraction session details

#### `log_extraction_session(dataset_dir, video_files, frames_extracted, extraction_dirs)`
Records extraction session metadata including:
- Timestamp and session ID
- Videos processed
- Total frames extracted
- Extraction directories created
- Configuration used

#### `update_image_status(dataset_dir, new_image_files, session_id)`
Marks newly extracted images as "unannotated" in the status tracking system.

### File Management Functions

#### `move_media_to_processed(media_path)`
Moves successfully processed media from `New/` to `Processed/` directory with automatic conflict resolution.

#### `setup_annotation_directories(extracted_frames_dir, dataset_name)`
Creates YOLO-compatible directory structure:
- Copies frames to `{dataset_name}/images/`
- Creates empty `{dataset_name}/labels/` directory
- Returns paths for annotation tools

#### `cleanup_extraction_directories(extraction_dirs, force=False)`
Manages temporary extraction directories with options:
1. Delete all directories
2. Keep all directories  
3. Selective deletion
4. Force cleanup (no prompts)

## Usage Examples

### Basic Usage
```bash
python image_extraction.py
```

### Workflow Example

1. **Place Media Files**
   ```
   raw_media/New/
   ├── video1.mp4
   ├── video2.mov
   └── photo1.jpg
   ```

2. **Run Extraction**
   ```bash
   python image_extraction.py
   ```

3. **Interactive Process**
   - Tool auto-discovers 3 media files
   - Confirms processing all files
   - Extracts 50 frames from each video
   - Processes photos
   - Shows preview of extracted frames

4. **Setup for Annotation**
   - Creates `poop_detection_dataset/images/` with all frames
   - Creates `poop_detection_dataset/labels/` for future annotations
   - Logs session details
   - Cleans up temporary directories

5. **Result Structure**
   ```
   poop_detection_dataset/
   ├── images/
   │   ├── video1_frame_0001.jpg
   │   ├── video1_frame_0002.jpg
   │   ├── ...
   │   └── photo_photo1.jpg
   ├── labels/                    # Ready for annotation
   ├── .annotation_status.json    # Tracking file
   └── .extraction_log.json       # Session log
   ```

## Integration with Annotation Workflow

The extracted images are designed to work seamlessly with annotation tools:

```bash
# After running image_extraction.py
python annotation_tool.py  # Next step for labeling
```

The status tracking system helps manage large datasets by tracking:
- Which images need annotation
- Which images are completed
- Extraction session metadata
- Processing history

## Error Handling

The script includes robust error handling for:
- Missing directories (auto-creates as needed)
- Corrupted video files
- File permission issues
- Naming conflicts
- Large file processing

## Performance Considerations

- **Memory Usage**: Processes one frame at a time to minimize memory footprint
- **Storage**: Temporary directories can be cleaned up to save space
- **Processing Speed**: Uses OpenCV for efficient video processing
- **File Sizes**: Automatically resizes large images to reasonable dimensions

## Dependencies

```python
import cv2                # Video/image processing
import numpy as np        # Array operations
import matplotlib.pyplot  # Preview functionality
import tkinter           # File selection dialogs
from pathlib import Path # File system operations
import json              # Status tracking
import shutil            # File operations
```

## Best Practices

1. **Organize Media**: Place new media in `raw_media/New/` for automatic processing
2. **Backup Originals**: Original files are moved to `Processed/`, not deleted
3. **Monitor Disk Space**: Clean up temporary directories regularly
4. **Batch Processing**: Process related videos together for consistent datasets
5. **Quality Control**: Use preview function to verify frame quality before annotation

This tool is the first step in a computer vision pipeline, preparing raw media for annotation and eventual model training.