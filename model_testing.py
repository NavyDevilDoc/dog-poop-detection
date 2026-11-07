from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def load_trained_model(model_path):
    """
    Load your trained poop detection model
    
    Args:
        model_path: Path to your trained .pt file
        
    Returns:
        Loaded YOLO model
    """
    try:
        from ultralytics import YOLO
        
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return None
        
        print(f"🔄 Loading trained model: {model_path}")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def select_test_images():
    """
    GUI to select test images from your phone/camera
    
    Returns:
        List of selected image paths
    """
    root = tk.Tk()
    root.withdraw()
    
    print("📁 Select your test images...")
    
    # Allow multiple image selection
    file_paths = filedialog.askopenfilenames(
        title="Select Test Images (from phone/camera)",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return list(file_paths)

def detect_poop_in_image(model, image_path, confidence_threshold=0.5, use_enhanced=False, detection_mode="balanced"):
    """
    Run poop detection on a single image with optional enhancements
    
    Args:
        model: Loaded YOLO model
        image_path: Path to test image
        confidence_threshold: Minimum confidence for detections
        use_enhanced: Whether to use enhanced detection strategies
        detection_mode: Mode for enhanced detection ("conservative", "balanced", "sensitive", "multi_scale")
        
    Returns:
        Detection results and annotated image
    """
    
    try:
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        if use_enhanced:
            # Use enhanced detection
            detections, scale_info = enhanced_detection_mode(model, image_path, detection_mode)
            
            # Get annotated image for the best detection
            if detections:
                # Use the mode's confidence for final annotation
                modes = {"conservative": 0.7, "balanced": 0.5, "sensitive": 0.3, "multi_scale": 0.4}
                conf = modes.get(detection_mode, 0.5)
                results = model(image_path, conf=conf, verbose=False)
                annotated_img = results[0].plot() if results else None
            else:
                annotated_img = None
                
        else:
            # Standard detection
            results = model(image_path, conf=confidence_threshold, verbose=False)
            
            # Process results
            detections = []
            annotated_img = None
            
            for result in results:
                # Get annotated image
                annotated_img = result.plot()
                
                # Extract detection info
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = box.conf[0].item()
                        cls = box.cls[0].item()
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'confidence': conf,
                            'class': int(cls),
                            'bbox': xyxy,
                            'class_name': model.names[int(cls)] if hasattr(model, 'names') else 'object'
                        }
                        detections.append(detection)
        
        # Print results
        if detections:
            for i, detection in enumerate(detections):
                conf = detection['confidence']
                class_name = detection.get('class_name', 'object')
                print(f"   🎯 Detection {i+1}: {class_name} (confidence={conf:.3f})")
        else:
            print("   🤷 No objects detected")
        
        return detections, annotated_img
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return [], None

def batch_test_images(model, image_paths, confidence_threshold=0.5, save_results=True):
    """
    Test multiple images and generate a comprehensive report
    
    Args:
        model: Loaded YOLO model
        image_paths: List of image paths to test
        confidence_threshold: Minimum confidence threshold
        save_results: Whether to save annotated images
        
    Returns:
        Dictionary with test results
    """
    
    # Create output directory
    output_dir = Path("poop_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🧪 Testing {len(image_paths)} images...")
    print("=" * 50)
    
    results_summary = {
        'total_images': len(image_paths),
        'images_with_poop': 0,
        'images_without_poop': 0,
        'total_detections': 0,
        'results': []
    }
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n📸 Image {i}/{len(image_paths)}")
        
        # Run detection
        detections, annotated_img = detect_poop_in_image(model, image_path, confidence_threshold)
        
        # Save results
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'detections': detections,
            'poop_found': len(detections) > 0
        }
        
        results_summary['results'].append(result)
        results_summary['total_detections'] += len(detections)
        
        if len(detections) > 0:
            results_summary['images_with_poop'] += 1
            print(f"   ✅ POOP DETECTED! ({len(detections)} detection(s))")
        else:
            results_summary['images_without_poop'] += 1
            print(f"   ✅ No poop detected - clean image!")
        
        # Save annotated image
        if save_results and annotated_img is not None:
            output_path = output_dir / f"detected_{image_name}.jpg"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"   💾 Saved result: {output_path}")
    
    return results_summary

def display_results_summary(results_summary):
    """
    Display a nice summary of detection results
    
    Args:
        results_summary: Results from batch_test_images
    """
    
    print("\n" + "="*60)
    print("🎯 POOP DETECTION TEST RESULTS")
    print("="*60)
    
    print(f"📊 Summary:")
    print(f"   • Total images tested: {results_summary['total_images']}")
    print(f"   • Images with poop: {results_summary['images_with_poop']}")
    print(f"   • Images without poop: {results_summary['images_without_poop']}")
    print(f"   • Total detections: {results_summary['total_detections']}")
    
    if results_summary['total_images'] > 0:
        poop_rate = results_summary['images_with_poop'] / results_summary['total_images']
        print(f"   • Poop detection rate: {poop_rate:.1%}")
    
    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(results_summary['results'], 1):
        status = "🎯 POOP FOUND" if result['poop_found'] else "✅ CLEAN"
        confidence = ""
        if result['detections']:
            max_conf = max([d['confidence'] for d in result['detections']])
            confidence = f" (confidence: {max_conf:.3f})"
        
        print(f"   {i:2d}. {result['image_name']}: {status}{confidence}")
    
    print("="*60)
    print("🎉 Testing complete! Check 'poop_detection_results' folder for annotated images.")

def create_detection_gallery(results_summary, max_images=9):
    """
    Create a visual gallery of detection results
    
    Args:
        results_summary: Results from batch_test_images
        max_images: Maximum images to show in gallery
    """
    
    results = results_summary['results'][:max_images]
    
    # Calculate grid size
    n_images = len(results)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    # Fix axes handling - make it always a flat array
    if n_images == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= len(axes):
            break
            
        # Load and display original image
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        
        # Create title
        status = "🎯 POOP DETECTED" if result['poop_found'] else "✅ CLEAN"
        confidence = ""
        if result['detections']:
            max_conf = max([d['confidence'] for d in result['detections']])
            confidence = f"\nConfidence: {max_conf:.3f}"
        
        axes[i].set_title(f"{result['image_name']}\n{status}{confidence}", fontsize=10)
        axes[i].axis('off')
        
        # Draw bounding boxes if any
        for detection in result['detections']:
            bbox = detection['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                               linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
    
    # Hide empty subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Poop Detection Results Gallery", fontsize=16, y=1.02)
    plt.show()


def save_enhanced_detection_result(model, image_path, detections, detection_mode, confidence_threshold):
    """
    Save annotated image with enhanced detection results
    
    Args:
        model: YOLO model
        image_path: Original image path
        detections: Detection results from enhanced mode
        detection_mode: Detection mode used
        confidence_threshold: Confidence threshold used
    """
    try:
        # Create output directory
        output_dir = Path("poop_detection_results")
        output_dir.mkdir(exist_ok=True)
        
        # Run detection again to get annotated image
        results = model(image_path, conf=confidence_threshold, verbose=False)
        
        if results and results[0]:
            annotated_img = results[0].plot()
            
            # Create filename with mode info
            image_name = Path(image_path).stem
            output_filename = f"enhanced_{detection_mode}_{image_name}.jpg"
            output_path = output_dir / output_filename
            
            # Save annotated image
            cv2.imwrite(str(output_path), annotated_img)
            print(f"   💾 Saved enhanced result: {output_path}")
            
            return str(output_path)
        else:
            # If no detections, save original with "clean" annotation
            img = cv2.imread(image_path)
            image_name = Path(image_path).stem
            output_filename = f"enhanced_{detection_mode}_{image_name}_clean.jpg"
            output_path = output_dir / output_filename
            
            # Add text overlay indicating clean image
            height, width = img.shape[:2]
            cv2.putText(img, f"CLEAN - {detection_mode.upper()} MODE", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(str(output_path), img)
            print(f"   💾 Saved clean result: {output_path}")
            
            return str(output_path)
            
    except Exception as e:
        print(f"   ❌ Error saving enhanced result: {e}")
        return None


def find_latest_trained_model():
    """
    Automatically find the most recent trained model
    
    Returns:
        Path to latest model or None
    """
    
    # Common locations for trained models
    search_paths = [
        "runs/detect/train/poop_detector/weights/best.pt",
        "runs/detect/poop_detector/weights/best.pt", 
        "poop_detector/weights/best.pt",
        "best.pt"
    ]
    
    for path in search_paths:
        if Path(path).exists():
            print(f"🔍 Found trained model: {path}")
            return str(path)
    
    print("❌ No trained model found automatically")
    return None


def test_confidence_thresholds(model, test_images, thresholds=None):
    """
    Test different confidence thresholds to find optimal setting
    
    Args:
        model: Loaded YOLO model
        test_images: List of image paths
        thresholds: List of thresholds to test (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    Returns:
        Dictionary with threshold analysis results
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("🎚️ CONFIDENCE THRESHOLD OPTIMIZATION")
    print("=" * 50)
    
    threshold_results = {}
    
    for thresh in thresholds:
        total_detections = 0
        images_with_detections = 0
        detection_details = []
        
        print(f"\n🔍 Testing threshold: {thresh}")
        
        for img_path in test_images:
            results = model(img_path, conf=thresh, verbose=False)
            detections = len(results[0].boxes) if results[0].boxes else 0
            
            total_detections += detections
            if detections > 0:
                images_with_detections += 1
                # Get confidence scores for this image
                if results[0].boxes:
                    confidences = [box.conf[0].item() for box in results[0].boxes]
                    detection_details.append({
                        'image': Path(img_path).name,
                        'count': detections,
                        'confidences': confidences
                    })
        
        threshold_results[thresh] = {
            'total_detections': total_detections,
            'images_with_detections': images_with_detections,
            'detection_rate': images_with_detections / len(test_images),
            'avg_detections_per_image': total_detections / len(test_images),
            'details': detection_details
        }
        
        print(f"   📊 Results: {images_with_detections}/{len(test_images)} images, "
              f"{total_detections} total detections")
    
    # Display optimization summary
    print(f"\n🎯 THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 50)
    for thresh, results in threshold_results.items():
        print(f"Threshold {thresh:.1f}: "
              f"{results['detection_rate']:.1%} detection rate, "
              f"{results['avg_detections_per_image']:.1f} avg detections/image")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Conservative (fewer false positives): Use 0.6-0.7")
    print(f"   • Balanced (current default): Use 0.5")
    print(f"   • Sensitive (catch more poop): Use 0.3-0.4")
    print(f"   • Very sensitive (maximum detection): Use 0.1-0.2")
    
    return threshold_results

def multi_scale_detection(model, image_path, scales=None, confidence_threshold=0.5):
    """
    Run detection at multiple scales to catch objects of different sizes
    
    Args:
        model: Loaded YOLO model
        image_path: Path to image
        scales: List of image sizes to test (default: [416, 640, 832])
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        Combined detection results from all scales
    """
    if scales is None:
        scales = [416, 640, 832]  # Small, medium, large
    
    print(f"🔍 Multi-scale detection on: {Path(image_path).name}")
    
    all_detections = []
    scale_results = {}
    
    for scale in scales:
        print(f"   📏 Testing scale: {scale}px")
        
        try:
            # Run detection at this scale
            results = model(image_path, imgsz=scale, conf=confidence_threshold, verbose=False)
            
            scale_detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    detection = {
                        'confidence': box.conf[0].item(),
                        'class': int(box.cls[0].item()),
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'scale': scale
                    }
                    scale_detections.append(detection)
                    all_detections.append(detection)
            
            scale_results[scale] = scale_detections
            print(f"      🎯 Found {len(scale_detections)} detections")
            
        except Exception as e:
            print(f"      ❌ Error at scale {scale}: {e}")
            scale_results[scale] = []
    
    # Remove duplicate detections using simple overlap check
    unique_detections = remove_duplicate_detections(all_detections)
    
    print(f"   📊 Total unique detections: {len(unique_detections)}")
    
    return unique_detections, scale_results

def remove_duplicate_detections(detections, overlap_threshold=0.5):
    """
    Remove duplicate detections based on bounding box overlap
    
    Args:
        detections: List of detection dictionaries
        overlap_threshold: IoU threshold for considering detections duplicates
    
    Returns:
        List of unique detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    unique_detections = []
    
    for detection in detections:
        is_duplicate = False
        
        for existing in unique_detections:
            iou = calculate_iou(detection['bbox'], existing['bbox'])
            if iou > overlap_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_detections.append(detection)
    
    return unique_detections

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU score (0-1)
    """
    # Calculate intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def enhanced_detection_mode(model, image_path, mode="balanced"):
    """
    Run detection with different optimization strategies
    
    Args:
        model: Loaded YOLO model
        image_path: Path to image
        mode: Detection mode - "conservative", "balanced", "sensitive", or "multi_scale"
    
    Returns:
        Enhanced detection results
    """
    modes = {
        "conservative": {"conf": 0.7, "scales": [640]},
        "balanced": {"conf": 0.5, "scales": [640]}, 
        "sensitive": {"conf": 0.3, "scales": [640]},
        "multi_scale": {"conf": 0.4, "scales": [416, 640, 832]}
    }
    
    if mode not in modes:
        mode = "balanced"
    
    settings = modes[mode]
    
    print(f"🎯 Enhanced detection mode: {mode.upper()}")
    
    if len(settings["scales"]) == 1:
        # Single scale detection
        results = model(image_path, conf=settings["conf"], imgsz=settings["scales"][0], verbose=False)
        
        detections = []
        if results[0].boxes:
            for box in results[0].boxes:
                detection = {
                    'confidence': box.conf[0].item(),
                    'class': int(box.cls[0].item()),
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'mode': mode
                }
                detections.append(detection)
        
        return detections, None
    else:
        # Multi-scale detection
        return multi_scale_detection(model, image_path, settings["scales"], settings["conf"])


if __name__ == "__main__":
    print("🎯 Enhanced Poop Detection Field Tester")
    print("=" * 40)
    
    # Step 1: Load trained model
    model_path = find_latest_trained_model()
    
    if not model_path:
        print("Please specify the path to your trained model:")
        model_path = input("Model path: ").strip()
    
    model = load_trained_model(model_path)
    if not model:
        exit(1)
    
    # Step 2: Select test images
    test_images = select_test_images()
    
    if not test_images:
        print("❌ No images selected for testing")
        exit(1)
    
    print(f"📁 Selected {len(test_images)} images for testing")
    
    # Step 3: Choose testing mode
    print(f"\n🎛️ Testing Mode Options:")
    print(f"   1. Standard testing (original functionality)")
    print(f"   2. Confidence threshold optimization")
    print(f"   3. Enhanced detection modes")
    print(f"   4. Multi-scale detection analysis")
    
    mode_choice = input("Select mode (1-4, default: 1): ").strip()
    
    if mode_choice == "2":
        # Threshold optimization
        threshold_results = test_confidence_thresholds(model, test_images)
        
        # Ask if they want to test with a specific threshold
        test_specific = input("\n🎚️ Test with specific threshold? (y/n): ").lower().strip()
        if test_specific == 'y':
            confidence = input("Enter confidence threshold (0.1-0.9): ").strip()
            try:
                confidence = float(confidence)
                confidence = max(0.1, min(0.9, confidence))
                
                print(f"\n🧪 Testing with confidence {confidence}")
                results = batch_test_images(model, test_images, confidence_threshold=confidence)
                display_results_summary(results)
                
                show_gallery = input("\n🖼️ Show visual gallery? (y/n): ").lower().strip()
                if show_gallery == 'y':
                    create_detection_gallery(results)
            except:
                print("❌ Invalid threshold value")
        else:
            # NEW: Ask if they want to save results from threshold optimization
            save_results = input("\n💾 Save annotated images from optimization? (y/n): ").lower().strip()
            if save_results == 'y':
                # Test all images with the recommended sensitive threshold
                print(f"\n💾 Saving results with sensitive threshold (0.3)")
                results = batch_test_images(model, test_images, confidence_threshold=0.3)
                display_results_summary(results)
    
    elif mode_choice == "3":
        # Enhanced detection modes
        print(f"\n🎯 Enhanced Detection Modes:")
        print(f"   1. Conservative (fewer false positives)")
        print(f"   2. Balanced (default)")
        print(f"   3. Sensitive (catch more poop)")
        print(f"   4. Multi-scale (comprehensive)")
        
        enhance_choice = input("Select enhancement (1-4, default: 2): ").strip()
        modes = {"1": "conservative", "2": "balanced", "3": "sensitive", "4": "multi_scale"}
        detection_mode = modes.get(enhance_choice, "balanced")
        
        print(f"\n🧪 Testing with {detection_mode} mode")
        
        # Ask if user wants to save results
        save_results = input("💾 Save annotated images? (y/n): ").lower().strip() == 'y'
        
        # Test each image with enhanced detection
        saved_files = []
        for i, image_path in enumerate(test_images, 1):
            print(f"\n📸 Image {i}/{len(test_images)}")
            detections, scale_info = enhanced_detection_mode(model, image_path, detection_mode)
            
            if detections:
                print(f"   ✅ ENHANCED DETECTION: {len(detections)} object(s) found")
                for j, det in enumerate(detections):
                    class_name = model.names[det['class']] if hasattr(model, 'names') else 'object'
                    print(f"      {j+1}. {class_name}: {det['confidence']:.3f}")
                
                # Save result if requested
                if save_results:
                    modes_conf = {"conservative": 0.7, "balanced": 0.5, "sensitive": 0.3, "multi_scale": 0.4}
                    conf = modes_conf.get(detection_mode, 0.5)
                    saved_path = save_enhanced_detection_result(model, image_path, detections, detection_mode, conf)
                    if saved_path:
                        saved_files.append(saved_path)
            else:
                print(f"   ✅ No objects detected")
                
                # Save clean result if requested  
                if save_results:
                    modes_conf = {"conservative": 0.7, "balanced": 0.5, "sensitive": 0.3, "multi_scale": 0.4}
                    conf = modes_conf.get(detection_mode, 0.5)
                    saved_path = save_enhanced_detection_result(model, image_path, [], detection_mode, conf)
                    if saved_path:
                        saved_files.append(saved_path)
        
        # Summary of saved files
        if save_results and saved_files:
            print(f"\n💾 Saved {len(saved_files)} annotated images to 'poop_detection_results' folder")
    
    elif mode_choice == "4":
        # Multi-scale analysis
        print(f"\n🔍 Multi-scale Detection Analysis")
        
        # Ask if user wants to save results
        save_results = input("💾 Save annotated images? (y/n): ").lower().strip() == 'y'
        
        saved_files = []
        for i, image_path in enumerate(test_images, 1):
            print(f"\n📸 Image {i}/{len(test_images)}")
            detections, scale_results = multi_scale_detection(model, image_path, confidence_threshold=0.4)
            
            # Show results by scale
            for scale, scale_dets in scale_results.items():
                print(f"      Scale {scale}px: {len(scale_dets)} detections")
                
            # Save result if requested
            if save_results:
                saved_path = save_enhanced_detection_result(model, image_path, detections, "multi_scale", 0.4)
                if saved_path:
                    saved_files.append(saved_path)
        
    else:
        # Standard testing (original functionality)
        confidence = input("\n🎚️ Confidence threshold (0.1-0.9, default 0.5): ").strip()
        try:
            confidence = float(confidence) if confidence else 0.5
            confidence = max(0.1, min(0.9, confidence))
        except:
            confidence = 0.5
        
        print(f"Using confidence threshold: {confidence}")
        
        # Ask if user wants to save results (consistent with other modes)
        save_results = input("💾 Save annotated images? (y/n, default: y): ").lower().strip()
        save_results = save_results != 'n'  # Default to True unless explicitly 'n'
        
        results = batch_test_images(model, test_images, confidence_threshold=confidence, save_results=save_results)
        display_results_summary(results)
        
        show_gallery = input("\n🖼️ Show visual gallery? (y/n): ").lower().strip()
        if show_gallery == 'y':
            create_detection_gallery(results)
        
        # Summary of saved files (consistent with other modes)
        if save_results:
            saved_count = len([r for r in results['results']])  # All images get saved
            print(f"\n💾 Saved {saved_count} annotated images to 'poop_detection_results' folder")
    
    print("\n🎉 Enhanced field testing complete!")
    print("Happy optimized poop hunting! 💩🔍")