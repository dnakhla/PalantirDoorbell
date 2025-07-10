import logging
import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Optional

from camera_manager import CameraManager
from image_processor import ImageProcessor
from ai_analyzer import AIAnalyzer  # Needed for profile analysis
from person_clustering import PersonClustering
from config import config
from models import PersonDetection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('doorbell_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DoorbellAI:
    """Main application class that coordinates all components"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.image_processor = ImageProcessor()
        self.ai_analyzer = AIAnalyzer()  # Needed for profile analysis
        self.person_clustering = PersonClustering()
        self.is_running = False
        
        # Set up camera detection callback
        self.camera_manager.set_detection_callback(self.on_person_detected)
        
        # Set up profile lookup callback for overlay
        self.camera_manager.set_profile_callback(self.get_profile_for_overlay)
    
    async def start(self):
        """Start the DoorbellAI system"""
        logger.info("Starting DoorbellAI system...")
        
        # Initialize camera
        if not self.camera_manager.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        # Start detection
        self.camera_manager.start_detection()
        self.is_running = True
        
        logger.info("DoorbellAI system started successfully")
        return True
    
    def stop(self):
        """Stop the DoorbellAI system"""
        logger.info("Stopping DoorbellAI system...")
        self.is_running = False
        
        # Stop camera detection
        self.camera_manager.stop_detection()
        
        # Save person profiles
        self.person_clustering.save_profiles()
        
        # Cleanup camera resources
        self.camera_manager.cleanup()
        
        logger.info("DoorbellAI system stopped")
    
    def on_person_detected(self, detection: PersonDetection, person_image):
        """Callback when a person is detected"""
        try:
            logger.info(f"Person detected at {detection.timestamp} with confidence {detection.confidence:.2f}")
            
            # Save the person image
            image_path = self.image_processor.save_person_image(detection, person_image)
            if not image_path:
                logger.error("Failed to save person image")
                return
            
            # Update detection with image path
            detection.image_path = image_path
            
            # Try to find existing person profile using hybrid approach (HOG + time)
            matching_profile_id = self.person_clustering.find_similar_person(detection, image_path)
            
            if matching_profile_id:
                # Add to existing time-based group
                updated_profile = self.person_clustering.add_detection_to_profile(matching_profile_id, detection, image_path)
                logger.info(f"📸 Added to existing profile: {matching_profile_id} (total: {len(updated_profile.images)} images)")
                logger.info(f"🤝 Hybrid grouping - same person or visit session")
            else:
                # Create new visit profile with AI analysis
                logger.info(f"🧠 Running AI analysis on new person...")
                ai_analysis = self.ai_analyzer.comprehensive_person_analysis(image_path)
                
                # Check if person was rejected during pre-screening
                if "error" in ai_analysis:
                    logger.warning(f"❌ Person rejected during AI analysis: {ai_analysis.get('reason', 'Unknown')}")
                    # Delete the saved image since person wasn't identifiable
                    try:
                        os.remove(image_path)
                        logger.info(f"🗑️ Deleted unidentifiable person image: {image_path}")
                    except:
                        pass
                    return
                
                # Extract analysis data
                nickname_data = ai_analysis.get("nickname_and_features", {})
                nickname = nickname_data.get("nickname", f"Visit {detection.timestamp.strftime('%m/%d %H:%M')}")
                description = nickname_data.get("overall_impression", f"Person detected at {detection.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if isinstance(description, dict):
                    description = str(description)
                
                profile = self.person_clustering.create_new_profile(detection, image_path, description)
                profile.name = nickname
                
                # Store comprehensive analysis in profile
                profile.ai_analysis = ai_analysis
                
                logger.info(f"✅ Created new profile: {profile.id} with AI analysis")
                logger.info(f"📈 Analysis includes: {list(ai_analysis.keys())}")
                logger.info(f"👤 Profile: {nickname} - {description[:50]}...")
            
            # Save profiles periodically
            self.person_clustering.save_profiles()
            
            # Note: Real-time updates will be handled by UI polling and WebSocket fallback
            
        except Exception as e:
            logger.error(f"Error processing person detection: {e}")
    
    def get_profile_for_overlay(self, bbox, timestamp):
        """Get profile information for live feed overlay"""
        try:
            # Look for recent profiles that might match this detection
            current_time = timestamp
            time_window_minutes = 5  # Increased window for better matching
            
            best_match = None
            best_similarity = 0
            
            for profile_id, profile in self.person_clustering.profiles.items():
                if not profile.detections:
                    continue
                
                # Check if this profile has recent activity
                last_detection_time = profile.last_seen
                time_diff = (current_time - last_detection_time).total_seconds() / 60
                
                if time_diff <= time_window_minutes:
                    # Check if bounding box is similar to recent detections
                    for recent_detection in profile.detections[-5:]:  # Check more recent detections
                        similarity = self._bbox_similarity(bbox, recent_detection.bbox)
                        if similarity > best_similarity and similarity > 0.4:  # Lower threshold for more matches
                            best_similarity = similarity
                            best_match = profile
            
            if best_match:
                # Return profile info for overlay
                analysis = getattr(best_match, 'ai_analysis', {}).get('comprehensive_analysis', {})
                logger.info(f"🎯 Overlay match found: {best_match.id} (similarity: {best_similarity:.3f})")
                logger.info(f"📋 Analysis data: {list(analysis.keys())}")
                
                return {
                    'name': best_match.name,
                    'description': best_match.description,
                    'nickname': analysis.get('nickname', best_match.name),
                    'gender': analysis.get('gender', 'unknown'),
                    'skin_tone': analysis.get('skin_tone', 'unknown'),
                    'physical_description': analysis.get('physical_description', ''),
                    'summary': analysis.get('summary', ''),
                    'notable_features': analysis.get('notable_features', [])
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting profile for overlay: {e}")
            return None
    
    def _bbox_similarity(self, bbox1, bbox2):
        """Calculate similarity between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                return intersection / union if union > 0 else 0
            
            return 0
        except:
            return 0
    
    def get_statistics(self):
        """Get system statistics"""
        stats = self.person_clustering.get_statistics()
        stats["system_status"] = "running" if self.is_running else "stopped"
        stats["timestamp"] = datetime.now().isoformat()
        return stats
    
    def get_all_profiles(self):
        """Get all person profiles"""
        return self.person_clustering.get_all_profiles()
    
    def get_profile(self, profile_id: str):
        """Get specific person profile"""
        return self.person_clustering.get_profile(profile_id)

# Global app instance
app_instance: Optional[DoorbellAI] = None

async def main():
    """Main application entry point"""
    global app_instance
    
    # Create app instance
    app_instance = DoorbellAI()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if app_instance:
            app_instance.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the system
    if not await app_instance.start():
        logger.error("Failed to start DoorbellAI")
        return
    
    # Keep the main thread alive
    try:
        while app_instance.is_running:
            await asyncio.sleep(1)
            
            # Periodic maintenance
            if datetime.now().minute == 0:  # Every hour
                app_instance.person_clustering.cleanup_old_profiles()
                
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        app_instance.stop()

if __name__ == "__main__":
    asyncio.run(main())