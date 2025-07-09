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
        self.last_detection_time = {}
        
    def initialize(self) -> bool:
        """Initialize camera and YOLO model"""
        try:
            # Setup optimal FFmpeg environment for RTSP
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'buffer_size;1024000|'
                'max_delay;500000|'
                'stimeout;30000000'
            )
            
            # Initialize camera with optimized settings
            self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                logger.error("Could not open camera stream")
                return False
            
            # Optimal camera properties for stable detection
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
            self.cap.set(cv2.CAP_PROP_FPS, 25)  # Standard FPS for stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            
            # H264 codec and error resilience settings
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30 second connection timeout
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)   # 30 second read timeout
            
            # Format specification removed - invalid constant
            
            # Verify actual camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"✓ Camera initialized - Requested: 1920x1080, Actual: {actual_width}x{actual_height} @ {actual_fps}fps")
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def set_detection_callback(self, callback: Callable):
        """Set callback function for when a person is detected"""
        self.detection_callback = callback
    
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
        skip_every_n_frames = 2  # Process every 2nd frame to reduce load
        
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
                    continue
                
                # Skip frames to reduce processing load
                frame_skip_count += 1
                if frame_skip_count % skip_every_n_frames != 0:
                    continue
                
                # Log frame dimensions for debugging
                if hasattr(self, '_frame_logged') == False:
                    self._frame_logged = True
                    frame_height, frame_width = frame.shape[:2]
                    logger.info(f"🎥 Processing frames at {frame_width}x{frame_height} resolution")
                    logger.info(f"🎥 Frame received successfully, shape: {frame.shape}")
                
                # Run YOLO inference with optimized settings for stability
                results = self.model(
                    frame, 
                    imgsz=960,  # Reduced resolution for better stability
                    conf=config.CONFIDENCE_THRESHOLD,  # Use our threshold
                    iou=0.5,  # Non-max suppression threshold
                    agnostic_nms=False,
                    max_det=10,  # Reduced max detections
                    classes=[0],  # Only detect persons (class 0)
                    verbose=False
                )
                
                # Process detections
                self._process_detections(results[0], frame)
                
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
    
    def _process_detections(self, results, frame):
        """Process YOLO detection results"""
        if not results.boxes:
            return
        
        current_time = time.time()
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            
            # Only process person detections
            if class_name != "person":
                continue
            
            confidence = box.conf.item()
            logger.info(f"Person detected with confidence: {confidence:.2f}")
            if confidence < config.CONFIDENCE_THRESHOLD:
                logger.info(f"Confidence {confidence:.2f} below threshold {config.CONFIDENCE_THRESHOLD}")
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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
            
            logger.info(f"🎯 Person detected! Confidence: {confidence:.3f}, BBox: ({x1},{y1},{x2},{y2})")
            
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
            
            # Add confidence and timestamp labels
            label = f"PERSON DETECTED {confidence:.1%}"
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Main label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(full_screen_image, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(full_screen_image, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
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
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources cleaned up")