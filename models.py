from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class PersonDetection:
    """Represents a single person detection event"""
    id: str
    timestamp: datetime
    image_path: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class PersonProfile:
    """Represents a clustered person profile"""
    id: str
    name: Optional[str]
    description: str
    images: List[str]
    detections: List[PersonDetection]
    first_seen: datetime
    last_seen: datetime
    total_appearances: int
    ai_analysis: Optional[str] = None
    last_analysis_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_detection(self, detection: PersonDetection):
        """Add a new detection to this profile"""
        self.detections.append(detection)
        self.images.append(detection.image_path)
        self.last_seen = detection.timestamp
        self.total_appearances += 1
    
    def get_summary(self) -> dict:
        """Get a summary of this person's profile"""
        return {
            "id": self.id,
            "name": self.name or "Unknown",
            "description": self.description,
            "total_appearances": self.total_appearances,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "image_count": len(self.images),
            "ai_analysis": self.ai_analysis
        }