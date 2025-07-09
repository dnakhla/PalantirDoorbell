import numpy as np
import cv2
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from config import config
from models import PersonDetection, PersonProfile

logger = logging.getLogger(__name__)

class PersonClustering:
    """Handles person clustering using image embeddings and similarity matching"""
    
    def __init__(self):
        self.profiles: Dict[str, PersonProfile] = {}
        self.feature_extractor = None
        self.embeddings_cache = {}
        self.load_profiles()
    
    def load_profiles(self):
        """Load existing person profiles from disk"""
        try:
            profiles_file = os.path.join(config.DATABASE_DIR, "person_profiles.pkl")
            if os.path.exists(profiles_file):
                with open(profiles_file, 'rb') as f:
                    self.profiles = pickle.load(f)
                logger.info(f"Loaded {len(self.profiles)} person profiles")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
    
    def save_profiles(self):
        """Save person profiles to disk"""
        try:
            profiles_file = os.path.join(config.DATABASE_DIR, "person_profiles.pkl")
            with open(profiles_file, 'wb') as f:
                pickle.dump(self.profiles, f)
            logger.info(f"Saved {len(self.profiles)} person profiles")
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from an image for similarity comparison"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            image_resized = cv2.resize(image_rgb, (128, 256))
            
            # Simple feature extraction using histogram of oriented gradients
            features = self._extract_hog_features(image_resized)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # Compute HOG features
        features = hog.compute(gray)
        
        # Normalize features
        features = features.flatten()
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def find_similar_person(self, detection: PersonDetection, image_path: str) -> Optional[str]:
        """Find existing person profile that matches the detection"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        # Extract features for new detection
        new_features = self.extract_features(image_path)
        if new_features is None:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        # Compare with existing profiles
        for profile_id, profile in self.profiles.items():
            if not profile.images:
                continue
            
            # Calculate similarity with existing images in profile
            profile_similarities = []
            
            for existing_image_path in profile.images[-3:]:  # Check last 3 images
                if not os.path.exists(existing_image_path):
                    continue
                
                # Get cached features or extract new ones
                if existing_image_path in self.embeddings_cache:
                    existing_features = self.embeddings_cache[existing_image_path]
                else:
                    existing_features = self.extract_features(existing_image_path)
                    if existing_features is not None:
                        self.embeddings_cache[existing_image_path] = existing_features
                
                if existing_features is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        new_features.reshape(1, -1), 
                        existing_features.reshape(1, -1)
                    )[0][0]
                    profile_similarities.append(similarity)
            
            if profile_similarities:
                avg_similarity = np.mean(profile_similarities)
                if avg_similarity > best_similarity and avg_similarity > config.SIMILARITY_THRESHOLD:
                    best_similarity = avg_similarity
                    best_match_id = profile_id
        
        if best_match_id:
            logger.info(f"Found matching person: {best_match_id} (similarity: {best_similarity:.3f})")
        
        return best_match_id
    
    def create_new_profile(self, detection: PersonDetection, image_path: str, description: str = "") -> PersonProfile:
        """Create a new person profile"""
        profile = PersonProfile(
            id="",
            name=None,
            description=description,
            images=[image_path],
            detections=[detection],
            first_seen=detection.timestamp,
            last_seen=detection.timestamp,
            total_appearances=1
        )
        
        self.profiles[profile.id] = profile
        logger.info(f"Created new person profile: {profile.id}")
        return profile
    
    def add_detection_to_profile(self, profile_id: str, detection: PersonDetection, image_path: str):
        """Add a new detection to existing profile"""
        if profile_id not in self.profiles:
            logger.error(f"Profile {profile_id} not found")
            return
        
        profile = self.profiles[profile_id]
        profile.add_detection(detection)
        
        # Update image path in detection
        detection.image_path = image_path
        
        logger.info(f"Added detection to profile {profile_id}")
        
        # Return the updated profile for potential re-analysis
        return profile
    
    def get_all_profiles(self) -> List[PersonProfile]:
        """Get all person profiles"""
        return list(self.profiles.values())
    
    def get_profile(self, profile_id: str) -> Optional[PersonProfile]:
        """Get specific person profile"""
        return self.profiles.get(profile_id)
    
    def cleanup_old_profiles(self, days_old: int = 30):
        """Remove profiles that haven't been seen for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        profiles_to_remove = []
        for profile_id, profile in self.profiles.items():
            if profile.last_seen < cutoff_date:
                profiles_to_remove.append(profile_id)
        
        for profile_id in profiles_to_remove:
            del self.profiles[profile_id]
            logger.info(f"Removed old profile: {profile_id}")
        
        if profiles_to_remove:
            self.save_profiles()
    
    def get_statistics(self) -> Dict:
        """Get clustering statistics"""
        if not self.profiles:
            return {}
        
        total_profiles = len(self.profiles)
        total_detections = sum(profile.total_appearances for profile in self.profiles.values())
        
        # Calculate time range
        all_timestamps = []
        for profile in self.profiles.values():
            all_timestamps.extend([d.timestamp for d in profile.detections])
        
        if all_timestamps:
            earliest = min(all_timestamps)
            latest = max(all_timestamps)
            time_range = latest - earliest
        else:
            time_range = timedelta(0)
        
        return {
            "total_profiles": total_profiles,
            "total_detections": total_detections,
            "time_range_days": time_range.days,
            "avg_detections_per_profile": total_detections / total_profiles if total_profiles > 0 else 0
        }