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
        """Hybrid approach: HOG similarity first, then time-based fallback"""
        if not image_path or not os.path.exists(image_path):
            return self.find_recent_person(detection)
        
        # Step 1: Try HOG similarity matching
        hog_match = self._find_by_similarity(image_path)
        if hog_match:
            logger.info(f"✅ HOG similarity match: {hog_match}")
            return hog_match
        
        # Step 2: Fallback to time-based grouping
        time_match = self.find_recent_person(detection)
        if time_match:
            logger.info(f"⏰ Time-based fallback match: {time_match}")
            return time_match
        
        logger.info(f"🆕 No match found - will create new profile")
        return None
    
    def _find_by_similarity(self, image_path: str) -> Optional[str]:
        """Find existing person using HOG similarity with GIF-friendly constraints"""
        # Extract features for new detection
        new_features = self.extract_features(image_path)
        if new_features is None:
            return None
        
        best_match_id = None
        best_similarity = 0
        max_images_per_session = 15  # Same limit as time-based grouping
        
        # Compare with existing profiles (only recent ones for speed)
        for profile_id, profile in self.profiles.items():
            if not profile.images:
                continue
            
            # Only check profiles from last hour for performance
            if (datetime.now() - profile.last_seen).total_seconds() > 3600:
                continue
            
            # Check if profile has room for more images (GIF-friendly limit)
            if len(profile.images) >= max_images_per_session:
                logger.info(f"📹 HOG skip: {profile_id} has {len(profile.images)} images (max: {max_images_per_session})")
                continue
            
            # Calculate similarity with last few images
            profile_similarities = []
            
            for existing_image_path in profile.images[-3:]:
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
            logger.info(f"🎯 HOG match found: {best_match_id} (similarity: {best_similarity:.3f})")
        
        return best_match_id
    
    def find_recent_person(self, detection: PersonDetection) -> Optional[str]:
        """Find a person profile based on recent time proximity with GIF-friendly grouping"""
        current_time = detection.timestamp
        time_window_seconds = 60  # Tight 1-minute window for GIF creation
        max_images_per_session = 15  # Maximum images per GIF session
        
        # Look for profiles with recent activity
        for profile_id, profile in self.profiles.items():
            if not profile.detections:
                continue
            
            # Check if last detection was within the time window
            last_detection_time = profile.last_seen
            time_diff = (current_time - last_detection_time).total_seconds()
            
            # Check if we're within the tight time window
            if time_diff <= time_window_seconds:
                # Check if we haven't exceeded the max images for this session
                if len(profile.images) < max_images_per_session:
                    logger.info(f"⏰ Recent visit found: {profile_id} (last seen {time_diff:.1f} seconds ago, {len(profile.images)} images)")
                    return profile_id
                else:
                    logger.info(f"📹 Session full: {profile_id} has {len(profile.images)} images (max: {max_images_per_session})")
                    # Don't return this profile - let it create a new one for the next session
                    continue
            
            # Check for session gap - if there's a gap of more than 2 minutes, start new session
            elif time_diff > 120:  # 2 minutes gap indicates new visit
                continue
        
        # No recent profile found within tight constraints
        return None
    
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
        
        # Cleanup old images if we exceed the limit
        if len(profile.images) > config.MAX_IMAGES_PER_PROFILE:
            # Remove oldest images and detections
            excess_count = len(profile.images) - config.MAX_IMAGES_PER_PROFILE
            
            # Delete old image files
            for old_image in profile.images[:excess_count]:
                try:
                    if os.path.exists(old_image):
                        os.remove(old_image)
                        logger.info(f"🗑️ Deleted old image: {old_image}")
                except Exception as e:
                    logger.warning(f"Failed to delete old image {old_image}: {e}")
            
            # Keep only recent images and detections
            profile.images = profile.images[excess_count:]
            profile.detections = profile.detections[excess_count:]
            
            logger.info(f"📉 Trimmed profile {profile_id} to {len(profile.images)} images")
        
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