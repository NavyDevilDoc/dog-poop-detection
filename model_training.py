from pathlib import Path
import yaml
from ultralytics import YOLO
from config import config, get_project_title


def validate_dataset_config(data_yaml_path):
    """
    Validate the dataset configuration file
    
    Args:
        data_yaml_path: Path to data.yaml file
        
    Returns:
        Boolean indicating if config is valid
    """
    
    if not Path(data_yaml_path).exists():
        print(f"❌ Config file not found: {data_yaml_path}")
        return False
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key in config: {key}")
                return False
        
        # Check if directories exist
        base_path = Path(config['path'])
        train_path = base_path / config['train']
        val_path = base_path / config['val']
        
        if not train_path.exists():
            print(f"❌ Training directory not found: {train_path}")
            return False
            
        if not val_path.exists():
            print(f"❌ Validation directory not found: {val_path}")
            return False
        
        # Count images
        train_images = len(list(train_path.glob('*.jpg')))
        val_images = len(list(val_path.glob('*.jpg')))
        
        print(f"✅ Dataset validation passed:")
        print(f"   • Training images: {train_images}")
        print(f"   • Validation images: {val_images}")
        print(f"   • Classes: {config['nc']} ({config['names']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False

def train_yolo_model(data_yaml_path, model_size='n', epochs=100, img_size=640, batch_size=16, 
                     save_dir='runs/detect/train'):
    """
    Train a YOLOv8 model on your poop detection dataset
    
    Args:
        data_yaml_path: Path to your data.yaml config file
        model_size: YOLOv8 model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        epochs: Number of training epochs
        img_size: Input image size for training
        batch_size: Training batch size
        save_dir: Directory to save training results
        
    Returns:
        Path to trained model
    """
    
    try:        
        # Select model based on size
        model_variants = {
            'n': 'yolov8n.pt',  # Nano - fastest, least accurate
            's': 'yolov8s.pt',  # Small - good balance
            'm': 'yolov8m.pt',  # Medium - more accurate
            'l': 'yolov8l.pt',  # Large - very accurate
            'x': 'yolov8x.pt'   # XLarge - most accurate, slowest
        }
        
        model_name = model_variants.get(model_size, 'yolov8n.pt')
        
        print(f"🚀 Starting YOLOv8 training...")
        print(f"   • Model: {model_name}")
        print(f"   • Dataset: {data_yaml_path}")
        print(f"   • Epochs: {epochs}")
        print(f"   • Image size: {img_size}")
        print(f"   • Batch size: {batch_size}")
        print("=" * 50)
        
        # Check for existing trained models (dynamic versioning)
        existing_models = []
        runs_dir = Path('runs/detect/train')
        
        if runs_dir.exists():
            for model_dir in runs_dir.iterdir():
                if model_dir.is_dir():
                    best_pt = model_dir / 'weights' / 'best.pt'
                    if best_pt.exists():
                        existing_models.append({
                            'path': str(best_pt),
                            'name': model_dir.name,
                            'size': f"{best_pt.stat().st_size / 1024 / 1024:.1f} MB",
                            'modified': best_pt.stat().st_mtime
                        })
        
        # Sort by modification time (newest first)
        existing_models.sort(key=lambda x: x['modified'], reverse=True)
        
        # Give user choice and determine versioning
        if existing_models:
            print(f"\n🔍 Found existing trained models:")
            for i, model in enumerate(existing_models, 1):
                print(f"   {i}. {model['name']} ({model['size']})")
            
            print(f"   0. Start fresh with pre-trained {model_name}")
            
            choice = input(f"\nSelect model to use/continue (0-{len(existing_models)}, default: 0): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(existing_models):
                selected_model = existing_models[int(choice) - 1]
                model = YOLO(selected_model['path'])
                print(f"🔄 Continuing training from: {selected_model['name']}")
                
                # Create versioned name for new training
                base_name = selected_model['name']
                if '_v' in base_name:
                    # Extract version number and increment
                    parts = base_name.split('_v')
                    version_num = int(parts[-1]) + 1
                    new_name = f"{parts[0]}_v{version_num}"
                else:
                    # First versioning
                    new_name = f"{base_name}_v2"
            else:
                model = YOLO(model_name)
                new_name = 'poop_detector_v1'
                print(f"🆕 Starting fresh training with: {model_name}")
        else:
            model = YOLO(model_name)
            new_name = 'poop_detector_v1'
            print(f"🆕 No existing models found, starting fresh with: {model_name}")


        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=new_name,
            project=save_dir,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,      # Generate training plots
            verbose=True
        )
        
        # Get path to best model
        best_model_path = Path(save_dir) / new_name / 'weights' / 'best.pt'
        
        if results:
            print("\n📊 Training Summary:")
            try:
                # Extract final metrics
                final_epoch = len(results) - 1 if hasattr(results, '__len__') else 'N/A'
                print(f"   • Epochs completed: {final_epoch}")
                print(f"   • Model saved to: {best_model_path}")
            except:
                print(f"   • Training completed successfully")
        
        print("🎉 Training completed successfully!")
        print(f"📁 Best model saved to: {best_model_path}")
        
        return str(best_model_path), results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def validate_trained_model(model_path, data_yaml_path):
    """
    Validate the trained model on the validation set
    
    Args:
        model_path: Path to trained model (.pt file)
        data_yaml_path: Path to dataset config
    """
    
    try:
        
        
        print("🔍 Validating trained model...")
        
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation
        metrics = model.val(data=data_yaml_path)
        
        # Print key metrics
        print("📊 Validation Results:")
        print(f"   • Precision: {metrics.box.mp:.3f}")
        print(f"   • Recall: {metrics.box.mr:.3f}")
        print(f"   • mAP50: {metrics.box.map50:.3f}")
        print(f"   • mAP50-95: {metrics.box.map:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def test_model_inference(model_path, test_image_path):
    """
    Test the trained model on a sample image
    
    Args:
        model_path: Path to trained model
        test_image_path: Path to test image
    """
    
    try:      
        print(f"🧪 Testing inference on: {test_image_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image_path)
        
        # Print detections
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"🎯 Found {len(boxes)} detections:")
                for i, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    print(f"   Detection {i+1}: Class={int(cls)}, Confidence={conf:.3f}")
            else:
                print("🤷 No detections found")
        
        # Save result image
        output_path = Path(test_image_path).parent / f"detected_{Path(test_image_path).name}"
        result_img = results[0].plot()
        import cv2
        cv2.imwrite(str(output_path), result_img)
        print(f"💾 Detection result saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return None

def get_recommended_training_params(num_images):
    """
    Get recommended training parameters based on dataset size
    
    Args:
        num_images: Total number of training images
        
    Returns:
        Dictionary with recommended parameters
    """
    
    if num_images < 50:
        return {
            'model_size': 'n',
            'epochs': 150,
            'batch_size': 8,
            'img_size': 416
        }
    elif num_images < 200:
        return {
            'model_size': 's', 
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640
        }
    else:
        return {
            'model_size': 'm',
            'epochs': 50,
            'batch_size': 32,
            'img_size': 640
        }

if __name__ == "__main__":
    print(get_project_title('training'))
    print("=" * 50)
    
    # Configuration
    DATASET_DIR = config.get('dataset.name', 'poop_detection_dataset')
    DATA_YAML = f"{DATASET_DIR}/data.yaml"
    
    # Step 2: Validate dataset
    if not validate_dataset_config(DATA_YAML):
        print("❌ Dataset validation failed")
        print("Make sure you've run the dataset splitter first!")
        exit(1)
    
    # Step 3: Get training parameters
    train_images_dir = Path(DATASET_DIR) / 'train' / 'images'
    num_images = len(list(train_images_dir.glob('*.jpg')))
    
    recommended_params = get_recommended_training_params(num_images)
    
    print(f"\n📋 Recommended parameters for {num_images} images:")
    for key, value in recommended_params.items():
        print(f"   • {key}: {value}")
    
    # Ask user if they want to proceed
    proceed = input("\n🚀 Start training with recommended parameters? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("Training cancelled. You can modify parameters in the script if needed.")
        exit(0)

    # Step 4: Train the model
    training_result = train_yolo_model(
        data_yaml_path=DATA_YAML,
        model_size=recommended_params['model_size'],
        epochs=recommended_params['epochs'],
        batch_size=recommended_params['batch_size'],
        img_size=recommended_params['img_size']
    )

    # Handle the tuple return
    if training_result:
        model_path, results = training_result  # Unpack the tuple
        
        if model_path and Path(model_path).exists():
            # Step 5: Validate the trained model
            validate_trained_model(model_path, DATA_YAML)
            
            # Step 6: Test inference on a sample image
            val_images_dir = Path(DATASET_DIR) / 'val' / 'images'
            sample_images = list(val_images_dir.glob('*.jpg'))
            
            if sample_images:
                print(f"\n🧪 Testing on sample image...")
                test_model_inference(model_path, str(sample_images[0]))
            
            print(f"\n🎉 Training pipeline completed!")
            print(f"📁 Your trained model: {model_path}")
            print("🚀 Ready to detect poop!")
            
        else:
            print("❌ Model file not found after training")
    else:
        print("❌ Training failed")