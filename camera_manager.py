import cv2
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Callable
from ultralytics import YOLO
import logging

from config import config
from models import PersonDetection

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera stream and person detection"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[YOLO] = None
        self.is_running = False
        self.detection_callback: Optional[Callable] = None
        self.profile_callback: Optional[Callable] = None
        self.last_detection_time = {}
        
        # Motion detection for optimization
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.last_frame = None
        self.motion_threshold = 500  # Minimum motion pixels to trigger detection
        self.frame_skip_counter = 0
        self.detection_skip_frames = 3  # Skip 3 frames between detections when no motion
        
        # Quality monitoring - less aggressive for performance
        self.quality_check_interval = 60  # Check quality every 60 frames (less frequent)
        self.quality_check_counter = 0
        self.poor_quality_count = 0
        self.max_poor_quality_frames = 20  # Higher threshold before reset
        self.last_quality_reset = time.time()
        self.min_reset_interval = 120  # Don't reset more than once every 2 minutes
        
    def initialize(self) -> bool:
        """Initialize camera and YOLO model with enhanced quality settings"""
        try:
            return self._initialize_camera_with_quality_settings()
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _initialize_camera_with_quality_settings(self) -> bool:
        """Initialize camera with optimal settings for RTMP or RTSP"""
        import os
        
        if config.USE_RTMP:
            # RTMP optimized settings
            logger.info("🎥 Using RTMP protocol for better streaming performance")
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'buffer_size;512000|'      # Smaller buffer for RTMP
                'max_delay;200000|'        # Low delay for real-time
                'fflags;nobuffer|'         # No buffering
                'flags;low_delay|'         # Low delay flag
                'err_detect;ignore_err|'   # Ignore decode errors
                'rw_timeout;10000000'      # 10 second timeout
            )
        else:
            # High-bitrate RTSP settings for 2560x1920@4096kbps camera
            logger.info("📡 Using high-res RTSP settings for 5MP@4096kbps camera")
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'        # TCP for high-bitrate reliability
                'buffer_size;2048000|'       # Larger buffer for 4096kbps stream
                'max_delay;1000000|'         # Higher delay tolerance for high bitrate
                'stimeout;20000000|'         # 20s timeout for large frames
                'reconnect;1|'               # Auto-reconnect
                'reconnect_streamed;1|'      # Reconnect during streaming  
                'reconnect_delay_max;3|'     # Stable reconnect for high-res
                'fflags;flush_packets|'      # Flush packets for high bitrate
                'flags;low_delay|'           # Low delay when possible
                'probesize;2097152|'         # Larger probe for high-res (2MB)
                'analyzeduration;2000000|'   # 2s analysis for high bitrate
                'err_detect;ignore_err|'     # Ignore H.264 decode errors
                'avoid_negative_ts;make_zero|'  # Fix timestamp issues
                'max_interleave_delta;500000'   # Allow interleave for high bitrate
            )
        
        # Initialize camera with protocol-appropriate settings
        self.cap = cv2.VideoCapture(config.CAMERA_URL, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error("Could not open camera stream")
            return False
        
        # Let camera use its native resolution, we'll resize in processing
        # Don't force resolution - camera will use its default
        self.cap.set(cv2.CAP_PROP_FPS, 15)             # Reduced FPS for stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Minimal buffer for real-time
        
        # Quality enhancement settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)   # 60 second connection timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # 10 second read timeout
        
        # Image quality settings
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)     # Optimal brightness
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)       # Optimal contrast  
        self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)     # Optimal saturation
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Auto exposure
        
        # Verify actual camera settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"✓ Optimized Camera initialized - Native resolution: {actual_width}x{actual_height}@{actual_fps}fps")
        
        # Load YOLO model with torch settings
        logger.info("Loading YOLO model...")
        import torch
        import os
        
        try:
            # Try loading with weights_only=False
            import functools
            original_load = torch.load
            torch.load = functools.partial(torch.load, weights_only=False)
            
            self.model = YOLO("yolov8n.pt")
            logger.info("✅ YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO with weights_only=False: {e}")
            # Restore original torch.load and try alternative
            torch.load = original_load
            try:
                # Try with environment variable
                os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
                self.model = YOLO("yolov8n.pt")
                logger.info("✅ YOLO model loaded with env override")
            except Exception as e2:
                logger.error(f"Failed to load YOLO completely: {e2}")
                return False
        finally:
            # Ensure torch.load is restored
            if 'original_load' in locals():
                torch.load = original_load
        
        logger.info("Camera and model initialized successfully")
        return True
    
    def set_detection_callback(self, callback: Callable):
        """Set callback function for when a person is detected"""
        self.detection_callback = callback
    
    def set_profile_callback(self, callback: Callable):
        """Set callback function for profile lookup during live feed"""
        self.profile_callback = callback
    
    def start_detection(self):
        """Start the detection loop in a separate thread"""
        if not self.cap or not self.model:
            logger.error("Camera not initialized")
            return
        
        self.is_running = True
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        logger.info("Detection started")
    
    def stop_detection(self):
        """Stop the detection loop"""
        self.is_running = False
        logger.info("Detection stopped")
    
    def _detection_loop(self):
        """Main detection loop"""
        consecutive_failures = 0
        max_failures = 5
        frame_skip_count = 0
        skip_every_n_frames = 3  # Process every 3rd frame for better performance
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({consecutive_failures}/{max_failures}) - attempting to reconnect...")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive failures, stopping detection")
                        break
                    
                    # Try to reconnect the camera with optimal settings
                    self.cap.release()
                    self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                    if self.cap.isOpened():
                        logger.info("Camera reconnected successfully")
                        # Apply optimal settings again
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        self.cap.set(cv2.CAP_PROP_FPS, 25)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
                        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)
                        # Format specification removed - invalid constant
                    
                    time.sleep(min(consecutive_failures, 5))  # Exponential backoff up to 5 seconds
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                
                # Validate frame integrity with comprehensive checks
                if not self._validate_frame(frame):
                    logger.warning("Received corrupted frame, skipping...")
                    self.poor_quality_count += 1
                    continue
                
                # Quality monitoring - check every N frames
                self.quality_check_counter += 1
                if self.quality_check_counter >= self.quality_check_interval:
                    self.quality_check_counter = 0
                    if not self._check_frame_quality(frame):
                        self.poor_quality_count += 1
                        logger.warning(f"Poor quality frame detected ({self.poor_quality_count}/{self.max_poor_quality_frames})")
                    else:
                        self.poor_quality_count = 0  # Reset counter on good quality
                
                # Auto-reset camera if too many poor quality frames
                if self.poor_quality_count >= self.max_poor_quality_frames:
                    current_time = time.time()
                    if current_time - self.last_quality_reset > self.min_reset_interval:
                        logger.warning("🔄 Auto-resetting camera due to poor quality")
                        self._reset_camera_connection()
                        self.last_quality_reset = current_time
                        self.poor_quality_count = 0
                        continue
                
                # Frame skipping for performance optimization
                frame_skip_count += 1
                if frame_skip_count < skip_every_n_frames:
                    continue  # Skip this frame
                frame_skip_count = 0  # Reset counter
                
                # Motion detection removed - capture every detection event for review
                
                # Log frame dimensions for debugging
                if hasattr(self, '_frame_logged') == False:
                    self._frame_logged = True
                    frame_height, frame_width = frame.shape[:2]
                    logger.info(f"🎥 Processing frames at {frame_width}x{frame_height} resolution")
                    logger.info(f"🎥 Frame received successfully, shape: {frame.shape}")
                    logger.info(f"📍 Capturing every detection event for review")
                
                # Resize frame for your 2560x1920 camera (50% scale for optimal performance)
                processing_frame = frame
                original_height, original_width = frame.shape[:2]
                
                # For your 5MP camera, resize to smaller resolution for smoothness
                if original_width > 960:
                    target_width = config.PROCESSING_WIDTH   # 960
                    target_height = config.PROCESSING_HEIGHT # 720
                    processing_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    logger.debug(f"Resized 5MP frame from {original_width}x{original_height} to {target_width}x{target_height}")
                elif original_width > 640:
                    # Fallback resize for other resolutions
                    scale = min(1280/original_width, 960/original_height)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    processing_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    logger.debug(f"Resized frame from {original_width}x{original_height} to {new_width}x{new_height}")
                
                # Run YOLO inference with optimized settings for maximum smoothness
                results = self.model(
                    processing_frame, 
                    imgsz=480,  # Smaller size for faster processing
                    conf=config.CONFIDENCE_THRESHOLD,  # Use our threshold
                    iou=0.7,  # Higher NMS threshold for speed
                    agnostic_nms=False,
                    max_det=3,  # Fewer max detections for speed
                    classes=[0, 15, 16],  # person (0), cat (15), dog (16)
                    verbose=False,
                    half=False,  # Disable FP16 for stability
                    device='cpu'  # Force CPU for stability on some systems
                )
                
                # Process detections with scaling if needed
                scale_factor = 1.0
                if original_height > 720 or original_width > 1280:
                    scale_factor = max(original_width / processing_frame.shape[1], original_height / processing_frame.shape[0])
                
                self._process_detections(results[0], frame, scale_factor)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def _validate_frame(self, frame):
        """Validate frame integrity to detect corruption"""
        if frame is None:
            return False
        
        # Check frame dimensions
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return False
        
        # Check for all-black frame (potential corruption)
        if np.count_nonzero(frame) == 0:
            return False
        
        # Check for uniform color (potential corruption)
        if np.max(frame) == np.min(frame):
            return False
        
        # Check for reasonable frame shape (3 channels for BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        
        return True
    
    def _check_frame_quality(self, frame):
        """Check frame quality metrics"""
        try:
            # Convert to grayscale for quality analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate image sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate brightness (average pixel value)
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Quality thresholds - more lenient for performance
            min_sharpness = 50     # Reduced minimum sharpness threshold
            min_brightness = 10    # Reduced minimum brightness
            max_brightness = 245   # Increased maximum brightness
            min_contrast = 10      # Reduced minimum contrast
            
            # Check quality metrics
            quality_checks = [
                sharpness > min_sharpness,
                min_brightness < brightness < max_brightness,
                contrast > min_contrast
            ]
            
            quality_passed = sum(quality_checks) >= 2  # At least 2 out of 3 checks must pass
            
            if not quality_passed:
                logger.debug(f"Quality metrics: sharpness={sharpness:.1f}, brightness={brightness:.1f}, contrast={contrast:.1f}")
            
            return quality_passed
            
        except Exception as e:
            logger.warning(f"Quality check failed: {e}")
            return True  # Default to good quality if check fails
    
    def _reset_camera_connection(self):
        """Reset camera connection for better quality"""
        try:
            logger.info("🔄 Resetting camera connection...")
            
            # Release current connection
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Wait a moment for cleanup
            time.sleep(2)
            
            # Re-initialize with enhanced settings
            if self._initialize_camera_with_quality_settings():
                logger.info("✅ Camera reset successful")
            else:
                logger.error("❌ Camera reset failed")
                
        except Exception as e:
            logger.error(f"Error resetting camera: {e}")
    
    def _process_detections(self, results, frame, scale_factor=1.0):
        """Process YOLO detection results"""
        if not results.boxes:
            return
        
        current_time = time.time()
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            
            # Process person, cat, and dog detections only
            if class_name not in ["person", "cat", "dog"]:
                continue  # Skip other detections silently
            
            confidence = box.conf.item()
            logger.info(f"{class_name.title()} detected with confidence: {confidence:.2f}")
            if confidence < config.CONFIDENCE_THRESHOLD:
                logger.info(f"Confidence {confidence:.2f} below threshold {config.CONFIDENCE_THRESHOLD}")
                continue
            
            # Get bounding box coordinates and scale back to original frame size
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if scale_factor != 1.0:
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
            bbox = (x1, y1, x2, y2)
            
            # Create a more flexible cooldown system based on overlap
            bbox_area = (x2 - x1) * (y2 - y1)
            overlapping_key = None
            
            # Check for overlapping recent detections
            for existing_key, last_time in list(self.last_detection_time.items()):
                if current_time - last_time > config.PERSON_DETECTION_COOLDOWN * 2:
                    # Clean up old entries
                    del self.last_detection_time[existing_key]
                    continue
                
                # Parse existing bbox
                try:
                    ex_x1, ex_y1, ex_x2, ex_y2 = map(int, existing_key.split('_'))
                    
                    # Calculate overlap
                    overlap_x1 = max(x1, ex_x1)
                    overlap_y1 = max(y1, ex_y1)
                    overlap_x2 = min(x2, ex_x2)
                    overlap_y2 = min(y2, ex_y2)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        overlap_ratio = overlap_area / bbox_area
                        
                        # If significant overlap and recent, skip
                        if overlap_ratio > 0.5 and current_time - last_time < config.PERSON_DETECTION_COOLDOWN:
                            overlapping_key = existing_key
                            break
                except:
                    pass
            
            if overlapping_key:
                continue
            
            # Record this detection
            bbox_key = f"{x1}_{y1}_{x2}_{y2}"
            self.last_detection_time[bbox_key] = current_time
            
            logger.info(f"🎯 {class_name.title()} detected! Confidence: {confidence:.3f}, BBox: ({x1},{y1},{x2},{y2})")
            
            # Create detection object
            detection = PersonDetection(
                id="",
                timestamp=datetime.now(),
                image_path="",  # Will be set when image is saved
                confidence=confidence,
                bbox=bbox
            )
            
            # Create full screen image with bounding box drawn
            full_screen_image = frame.copy()
            
            # Draw thick bounding box on full screen
            cv2.rectangle(full_screen_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Corner markers removed
            
            # Try to get profile information for richer overlay
            profile_info = None
            if self.profile_callback:
                try:
                    profile_info = self.profile_callback(bbox, datetime.now())
                except Exception as e:
                    logger.warning(f"Profile callback failed: {e}")
            
            # Create simple labels for live feed (no AI analysis)
            if class_id == 0:  # Person
                main_label = f"PERSON DETECTED {confidence:.1%}"
            elif class_id == 15:  # Cat
                main_label = f"CAT DETECTED {confidence:.1%}"
            elif class_id == 16:  # Dog
                main_label = f"DOG DETECTED {confidence:.1%}"
            else:
                main_label = f"{class_name.upper()} DETECTED {confidence:.1%}"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Main label
            label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(full_screen_image, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(full_screen_image, main_label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Add AI analysis info box to top right corner for stored images
            if profile_info:
                frame_height, frame_width = full_screen_image.shape[:2]
                
                # Prepare AI analysis info
                info_lines = []
                
                # Demographics
                gender = profile_info.get('gender', 'unknown')
                skin_tone = profile_info.get('skin_tone', 'unknown')
                if gender != 'unknown':
                    info_lines.append(f"Gender: {gender}")
                if skin_tone != 'unknown':
                    info_lines.append(f"Skin: {skin_tone}")
                
                # Summary
                summary = profile_info.get('summary', '')
                if summary:
                    info_lines.append(f"Summary: {summary[:40]}...")
                
                # Physical description
                physical_desc = profile_info.get('physical_description', '')
                if physical_desc:
                    info_lines.append(f"Physical: {physical_desc[:35]}...")
                
                # Draw info box in top right corner
                if info_lines:
                    box_width = 300
                    line_height = 20
                    box_height = len(info_lines) * line_height + 20
                    
                    # Position in top right corner
                    box_x = frame_width - box_width - 10
                    box_y = 10
                    
                    # Draw background box
                    cv2.rectangle(full_screen_image, (box_x, box_y), 
                                 (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
                    cv2.rectangle(full_screen_image, (box_x, box_y), 
                                 (box_x + box_width, box_y + box_height), (255, 255, 255), 2)
                    
                    # Draw info lines
                    for i, line in enumerate(info_lines):
                        text_y = box_y + 20 + (i * line_height)
                        cv2.putText(full_screen_image, line, (box_x + 10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Timestamp label
            timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(full_screen_image, (x2 - timestamp_size[0] - 10, y2 + 5), 
                         (x2, y2 + timestamp_size[1] + 15), (255, 255, 255), -1)
            cv2.putText(full_screen_image, timestamp, (x2 - timestamp_size[0] - 5, y2 + timestamp_size[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Remove center crosshairs (commented out)
            # height, width = full_screen_image.shape[:2]
            # center_x, center_y = width // 2, height // 2
            
            # Use full screen image instead of cropped
            person_image = full_screen_image
            
            # Call detection callback if set
            if self.detection_callback:
                self.detection_callback(detection, person_image)
    
    def _detect_motion(self, frame):
        """Detect motion in the current frame to optimize processing"""
        try:
            # Convert frame to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(gray)
            
            # Count non-zero pixels (motion pixels)
            motion_pixels = cv2.countNonZero(fg_mask)
            
            # Check if motion exceeds threshold
            has_motion = motion_pixels > self.motion_threshold
            
            if has_motion:
                logger.debug(f"🏃 Motion detected: {motion_pixels} pixels")
            
            return has_motion
            
        except Exception as e:
            logger.warning(f"Motion detection failed: {e}")
            return True  # Default to processing if motion detection fails
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources cleaned up")