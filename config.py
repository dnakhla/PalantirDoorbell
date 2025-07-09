import os
from dataclasses import dataclass
from typing import Set

@dataclass
class Config:
    # Camera settings
    RTSP_URL: str = "rtsp://dnakhla%40gmail.com:vi6Z39oCPz-iYmZ@192.168.86.42:554/h264Preview_01_main"
    
    # Directories
    CAPTURED_IMAGES_DIR: str = "captured_people"
    DATABASE_DIR: str = "database"
    
    # Detection settings - More aggressive capture
    CONFIDENCE_THRESHOLD: float = 0.60  # 80% confidence threshold for highest quality detections
    PERSON_DETECTION_COOLDOWN: int = 2  # Faster capture rate
    CAPTURE_MULTIPLE_ANGLES: bool = True  # Capture multiple shots per person
    
    # Clustering settings
    SIMILARITY_THRESHOLD: float = 0.7
    MIN_IMAGES_FOR_CLUSTER: int = 2
    
    # AI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Performance settings
    MAX_PEOPLE_IN_MEMORY: int = 1000
    IMAGE_RESIZE_HEIGHT: int = 640  # Standard YOLO input size
    CAPTURE_RESOLUTION_WIDTH: int = 1920  # High-res capture
    CAPTURE_RESOLUTION_HEIGHT: int = 1080  # High-res capture
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.CAPTURED_IMAGES_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_DIR, exist_ok=True)

# Global config instance
config = Config()