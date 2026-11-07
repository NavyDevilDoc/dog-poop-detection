# Annotation Tool Documentation

## Overview

The `annotation_tool.py` script provides an interactive GUI-based annotation system for creating bounding box labels on images. Built with matplotlib, it supports multi-class object detection annotation with advanced features like zoom, pan, overlap detection, and comprehensive progress tracking.

## Features

### Core Annotation Features
- **Multi-Class Support**: Annotate multiple object classes with color-coded bounding boxes
- **Interactive GUI**: Click-and-drag interface for creating precise bounding boxes
- **Real-Time Preview**: Live preview of bounding boxes during creation
- **Overlap Detection**: Automatic detection and handling of overlapping annotations
- **Status Tracking**: Comprehensive progress tracking and session management

### Navigation & Viewing
- **Zoom & Pan**: Mouse wheel zoom with arrow key panning when zoomed
- **Pan Mode**: Toggle pan mode to disable box creation for navigation
- **Progress Indicators**: Current image position and completion statistics
- **Image Filtering**: Multiple modes for viewing different subsets of images

### Quality Control
- **Minimum Box Size**: Automatic enforcement of minimum bounding box dimensions
- **Overlap Warnings**: IoU-based overlap detection with user confirmation
- **Auto-Save**: Periodic automatic saving every 5 annotations
- **Validation**: Annotation format validation before saving

## Class Structure

### MultiClassAnnotator Class

The main annotation interface that manages the entire annotation workflow.

```python
class MultiClassAnnotator:
    def __init__(self, images_dir, labels_dir, classes, mode='new_only')
```

**Constructor Parameters:**
- `images_dir`: Directory containing images to annotate
- `labels_dir`: Directory to save annotation files
- `classes`: List of object classes (e.g., ['poop', 'leaf', 'stick', 'rock'])
- `mode`: Annotation mode ('new_only', 'all', 'resume')

## Annotation Modes

### 1. New Only Mode (Default)
```python
mode = 'new_only'
```
- Shows only unannotated images
- Ideal for fresh annotation sessions
- Skips already completed images

### 2. All Images Mode
```python
mode = 'all'
```
- Shows all images in dataset
- Allows re-annotation of completed images
- Useful for quality review and corrections

### 3. Resume Mode
```python
mode = 'resume'
```
- Continues from where previous session left off
- Shows unannotated images only
- Maintains progress continuity

## User Interface Components

### Class Selection Buttons
```
[Poop] [Leaf] [Stick] [Rock]  # Top row - class selection
```
- Dynamic color coding for each class
- Highlighted button shows current active class
- Click to switch between object classes

### Action Buttons
```
[Skip] [Archive] [Save] [Next] [Clear All] [Undo Last] [Reset Zoom] [Quit]
```

**Button Functions:**
- **Skip**: Move to next image without saving annotations
- **Archive**: Move current image to archived folder (removes from active dataset)
- **Save**: Save current annotations to JSON file
- **Next**: Save and proceed to next image
- **Clear All**: Remove all bounding boxes from current image
- **Undo Last**: Remove the most recently created bounding box
- **Reset Zoom**: Return to original zoom level
- **Quit**: Save progress and exit annotation session

### Information Displays
- **Progress Counter**: `Image 15/127 (new_only): image_name.jpg`
- **Current Class**: Shows active class with color coding
- **Zoom Level**: `Zoom: 2.3x` when zoomed in
- **Pan Mode Indicator**: Warning when pan mode is active

## Keyboard Controls

### Navigation
```
Arrow Keys    - Pan image when zoomed in
Mouse Wheel   - Zoom in/out at cursor position
P Key         - Toggle pan mode on/off
```

### Pan Mode
- **Activate**: Press 'P' key
- **Purpose**: Allows panning without creating bounding boxes
- **Visual**: Orange warning indicator when active
- **Usage**: Navigate zoomed images or examine details without accidental annotations

## Mouse Controls

### Bounding Box Creation
1. **Click and Hold**: Left mouse button at starting corner
2. **Drag**: Move to opposite corner (live preview shown)
3. **Release**: Complete bounding box creation

### Zoom Operations
- **Zoom In**: Scroll wheel up at cursor position
- **Zoom Out**: Scroll wheel down at cursor position
- **Zoom Limits**: 0.5x to 10x magnification

### Pan Operations (Pan Mode)
- **Click and Drag**: Move image view when in pan mode
- **Reset**: Use "Reset Zoom" button to return to original view

## Annotation Workflow

### Basic Workflow
```python
# 1. Start annotation tool
python annotation_tool.py

# 2. Select annotation mode
# Choose: new_only, all, or resume

# 3. For each image:
#    a. Select object class
#    b. Draw bounding boxes
#    c. Save or proceed to next

# 4. Review session summary
# Check progress and recommendations
```

### Detailed Step-by-Step Process

#### 1. Session Initialization
```
📁 Total images: 150
📋 Selected for annotation: 75 (new_only mode)  
🏷️ Classes: poop, leaf, stick, rock
```

#### 2. Image Annotation
```python
# For each image:
1. Image loads with progress indicator
2. Select target class using top buttons
3. Click and drag to create bounding box
4. Box appears with class label and color
5. Repeat for all objects in image
6. Save manually or auto-save triggers
```

#### 3. Quality Control Checks
```python
# Overlap Detection Example
⚠️ Overlap detected: 65% with poop (same class)
🔔 High overlap (65%) - please confirm using buttons
[Keep Box] [Cancel Box]  # User choice required
```

#### 4. Session Completion
```python
📋 ANNOTATION SESSION COMPLETED
⏰ Session Results:
   • Images processed this session: 25
   • Images remaining in 'new_only' mode: 50
   • Total dataset progress: 85/150 (56.7%)

💡 Recommendations:
   • Continue annotating - 65 images remaining
   • Good progress! 100+ annotations will give better results

🎯 Suggested Next Steps:
   • Continue annotation sessions
   • Target: 15 more annotations needed for basic training
```

## File Formats and Storage

### Annotation File Format (JSON)
```json
{
  "imagePath": "image_001.jpg",
  "shapes": [
    {
      "label": "poop",
      "points": [[100, 150], [200, 250]],
      "shape_type": "rectangle"
    },
    {
      "label": "leaf", 
      "points": [[300, 100], [380, 180]],
      "shape_type": "rectangle"
    }
  ]
}
```

### Status Tracking File (.annotation_status.json)
```json
{
  "image_001.jpg": {
    "status": "annotated",
    "annotated_date": "2025-11-03T14:30:15",
    "extraction_session": "session_20251103_143015"
  },
  "image_002.jpg": {
    "status": "archived",
    "archived_date": "2025-11-03T14:31:22"
  }
}
```

### Project Status File (PROJECT_STATUS.txt)
```
DOG POOP DETECTION PROJECT STATUS
Last Updated: 2025-11-03 14:35

DATASET STATUS:
├── Images Collected: 150
├── Images Annotated: 85 (56.7%)
├── Images Archived: 15
├── Images Remaining: 65
└── Ready for Training: YES

MODEL STATUS:
├── Training Data Available: YES
├── Augmentation Ready: YES
└── Production Ready: NO (Need 100+)
```

## Configuration Integration

### Loading Configuration
```python
# Classes from config
classes = config.get('dataset.classes', ['poop', 'leaf', 'stick', 'rock'])

# Colors from config  
class_colors = config.get('dataset.class_colors', {
    'poop': 'red',
    'leaf': 'green', 
    'stick': 'brown',
    'rock': 'gray'
})

# Paths from config
images_dir = config.get_dataset_path('images')
labels_dir = config.get_dataset_path('labels')
```

### Dynamic Color Generation
```python
# Automatic color assignment for any class list
def generate_dynamic_colors(classes):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    return {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}
```

## Advanced Features

### Overlap Detection Algorithm
```python
def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    # Calculate intersection area
    x_left = max(box1['x'], box2['x'])
    y_top = max(box1['y'], box2['y'])
    x_right = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y_bottom = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area  
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height'] 
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

### Auto-Save System
```python
# Triggers auto-save every 5 annotations
if self.boxes_since_save >= 5:
    self.save_annotation()
    self.boxes_since_save = 0
    print("💾 Auto-saved (5 boxes completed)")
```

### Archive System
```python
# Move low-quality images out of training set
def archive_current_image(self):
    archive_dir = self.images_dir.parent / 'archived_images'
    # Moves image to archive folder
    # Updates status tracking
    # Removes from active annotation queue
```

## Error Handling and Robustness

### Display Update Safety
```python
try:
    plt.draw()
except Exception as e:
    print(f"⚠️ Display update skipped: {e}")
    # Continues without crashing on display issues
```

### File Operation Safety
```python
# Safe file operations with fallbacks
try:
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=2)
except Exception as e:
    # Fallback to ASCII encoding
    with open(status_file, 'w', encoding='ascii', errors='replace') as f:
        json.dump(status_data, f, indent=2)
```

### Minimum Box Size Enforcement
```python
# Prevent accidentally tiny boxes
if width < 10 or height < 10:
    width = max(20, width)
    height = max(20, height)
    print(f"📦 Created minimum size box ({width}x{height})")
```

## Usage Examples

### Basic Annotation Session
```bash
# Start annotation tool
python annotation_tool.py

# Interactive prompts:
# Mode Selection:
#   1. New images only (75 images) [DEFAULT]  
#   2. All images (150 images)
#   3. Resume unannotated (75 images)
# Choice: 1

# Session begins with first unannotated image
```

### Custom Class Configuration
```python
# In config.yaml
dataset:
  classes:
    - poop
    - leaf
    - stick
    - rock
    - trash    # Added new class
  class_colors:
    poop: red
    leaf: green
    stick: brown
    rock: gray
    trash: orange  # Custom color
```

### Integration with Pipeline
```bash
# Complete annotation workflow
python image_extraction.py    # Extract frames from videos
python annotation_tool.py     # Annotate extracted images  
python yolo_conversion.py     # Convert to YOLO format
python data_augmentation.py   # Multiply dataset
python model_training.py      # Train detection model
```

## Performance and Scalability

### Memory Management
- **Single Image Loading**: Only current image kept in memory
- **Efficient Updates**: Selective redrawing of changed elements
- **Resource Cleanup**: Proper cleanup of matplotlib elements

### Large Dataset Handling
- **Filtered Views**: Only load relevant images based on mode
- **Progress Tracking**: Persistent progress across sessions
- **Batch Processing**: Auto-save prevents loss of work

### Response Time Optimization
```python
# Non-blocking GUI updates
try:
    plt.draw()
except:
    # Skip drawing errors, continue annotation
    pass

# Efficient overlap calculation
# Only check overlaps when necessary
overlaps = self.check_overlap(new_box, existing_boxes, threshold=0.1)
```

## Troubleshooting

### Common Issues

#### 1. Images Not Loading
```
❌ Images directory not found: path/to/images
```
**Solution**: Verify image directory path in configuration

#### 2. Display Issues
```
⚠️ Display update skipped: backend error
```
**Solution**: Annotation continues; use save button to preserve work

#### 3. High Memory Usage
**Solution**: Close annotation tool periodically; restart for long sessions

### Best Practices

1. **Regular Saves**: Use manual save button frequently
2. **Quality Control**: Use overlap detection to maintain annotation quality  
3. **Batch Sessions**: Annotate in focused sessions of 50-100 images
4. **Archive Poor Images**: Use archive function for low-quality images
5. **Mode Selection**: Use 'new_only' mode for efficient workflow

## Integration Points

### With Image Extraction
- Reads images from extraction output directory
- Uses extraction session metadata for tracking
- Integrates with status files created during extraction

### With YOLO Conversion  
- Produces JSON annotations compatible with YOLO conversion
- Maintains class order consistency
- Provides completion statistics for conversion readiness

### With Data Augmentation
- Annotation completeness determines augmentation readiness
- Status tracking prevents augmentation of incomplete datasets
- Quality metrics guide augmentation parameters

This annotation tool provides a professional-grade interface for creating high-quality object detection datasets with comprehensive progress tracking and quality control features.