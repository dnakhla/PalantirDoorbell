import os
from dataclasses import dataclass
from typing import Set

@dataclass
class Config:
    # Camera settings - Reolink RTSP format
    CAMERA_URL: str = os.getenv("CAMERA_URL", "rtsp://dnakhla%40gmail.com:vi6Z39oCPz-iYmZ@192.168.86.42:554/h264Preview_01_main")
    
    # Auto-detect protocol and optimize settings
    USE_RTMP: bool = False  # Using RTSP on port 554
    
    # RTSP URL (primary)
    RTSP_URL: str = CAMERA_URL
    
    # Directories
    CAPTURED_IMAGES_DIR: str = "captured_people"
    DATABASE_DIR: str = "database"
    
    # Detection settings - Optimized for smoothness
    CONFIDENCE_THRESHOLD: float = 0.6  # Lower threshold for more responsive detection
    PERSON_DETECTION_COOLDOWN: int = 3  # Slightly longer cooldown for smoother processing
    CAPTURE_MULTIPLE_ANGLES: bool = True  # Capture multiple shots per person
    
    # Clustering settings
    SIMILARITY_THRESHOLD: float = 0.92  # Higher threshold for more precise person matching
    MIN_IMAGES_FOR_CLUSTER: int = 2
    
    # AI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Performance settings - optimized for smoothness over quality
    MAX_PEOPLE_IN_MEMORY: int = 20  # Further reduced for smooth processing
    IMAGE_RESIZE_HEIGHT: int = 480  # Smaller input for faster processing
    
    # Camera native resolution (for reference)
    CAMERA_NATIVE_WIDTH: int = 2560   # Your camera's native width
    CAMERA_NATIVE_HEIGHT: int = 1920  # Your camera's native height
    CAMERA_BITRATE: int = 4096        # 4096 kbps bitrate
    
    # Processing resolution (scaled down for maximum smoothness)
    PROCESSING_WIDTH: int = 960       # Further reduced for smoothness
    PROCESSING_HEIGHT: int = 720      # Further reduced for smoothness
    
    # AI Analysis limits for performance
    MAX_IMAGES_FOR_SEQUENCE_ANALYSIS: int = 10  # Limit sequence analysis to recent 10 images
    MIN_TIME_BETWEEN_REANALYSIS: int = 300  # 5 minutes between re-analysis (seconds)
    MAX_IMAGES_PER_PROFILE: int = 50  # Keep only 50 most recent images per person
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.CAPTURED_IMAGES_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_DIR, exist_ok=True)

# Global config instance
config = Config()