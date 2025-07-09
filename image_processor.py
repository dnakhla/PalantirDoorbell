import cv2
import os
import numpy as np
from datetime import datetime
from typing import Optional
import logging

from config import config
from models import PersonDetection

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing, saving, and enhancement"""
    
    def __init__(self):
        self.image_counter = 0
    
    def save_person_image(self, detection: PersonDetection, image: np.ndarray) -> str:
        """Save a person detection image - using original image only"""
        try:
            # Generate filename
            timestamp = detection.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"person_{timestamp}_{self.image_counter:04d}.jpg"
            filepath = os.path.join(config.CAPTURED_IMAGES_DIR, filename)
            
            # Save ONLY the original image with high quality (no enhancement)
            cv2.imwrite(filepath, image, [
                cv2.IMWRITE_JPEG_QUALITY, 95,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            
            self.image_counter += 1
            logger.info(f"✅ Saved original person image: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save person image: {e}")
            return ""
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality using OpenCV - stable and conservative approach"""
        if image.size == 0:
            return image
        
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Conservative enhancement that preserves image integrity
            if len(img_float.shape) == 3:
                # Convert to LAB color space for better luminance enhancement
                lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
                
                # Only enhance the L (luminance) channel conservatively
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:, :, 0] = clahe.apply((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
                
                # Convert back to BGR
                img_float = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
            else:
                # Grayscale enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_float = clahe.apply((img_float * 255).astype(np.uint8)) / 255.0
            
            # Gentle gamma correction for better visibility
            mean_brightness = np.mean(img_float)
            if mean_brightness < 0.4:  # Dark image
                gamma = 0.8  # Brighten slightly
            elif mean_brightness > 0.6:  # Bright image
                gamma = 1.2  # Darken slightly
            else:
                gamma = 1.0  # Keep as is
            
            if gamma != 1.0:
                img_float = np.power(np.clip(img_float, 0, 1), 1.0/gamma)
            
            # Gentle sharpening
            gaussian = cv2.GaussianBlur(img_float, (3, 3), 0.5)
            sharpened = cv2.addWeighted(img_float, 1.2, gaussian, -0.2, 0)
            
            # Ensure values are in valid range
            enhanced = np.clip(sharpened, 0, 1)
            
            # Convert back to uint8
            enhanced = (enhanced * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in image enhancement: {e}")
            # If enhancement fails, return original image
            return image
    
    def resize_image(self, image: np.ndarray, target_height: int = 300) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        if image.size == 0:
            return image
        
        height, width = image.shape[:2]
        if height <= target_height:
            return image
        
        # Calculate new width maintaining aspect ratio
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Resize with high quality interpolation
        resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    def extract_face_if_possible(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from person image if detected"""
        try:
            # Load OpenCV face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding around face
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                face_image = image[y1:y2, x1:x2]
                return face_image
            
            return None
            
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None