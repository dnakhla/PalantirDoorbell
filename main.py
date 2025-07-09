import logging
import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Optional

from camera_manager import CameraManager
from image_processor import ImageProcessor
from ai_analyzer import AIAnalyzer
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
        self.ai_analyzer = AIAnalyzer()
        self.person_clustering = PersonClustering()
        self.is_running = False
        
        # Set up camera detection callback
        self.camera_manager.set_detection_callback(self.on_person_detected)
    
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
            
            # Try to find existing person profile
            matching_profile_id = self.person_clustering.find_similar_person(detection, image_path)
            
            if matching_profile_id:
                # Add to existing profile
                updated_profile = self.person_clustering.add_detection_to_profile(matching_profile_id, detection, image_path)
                logger.info(f"Added detection to existing profile: {matching_profile_id}")
                
                # Re-analyze with all images if person has multiple images
                if updated_profile and len(updated_profile.images) >= 2:
                    logger.info(f"🔄 Re-analyzing person with {len(updated_profile.images)} images...")
                    timestamps = [det.timestamp.isoformat() for det in updated_profile.detections]
                    sequence_analysis = self.ai_analyzer.analyze_person_sequence(updated_profile.images, timestamps)
                    
                    if "comprehensive_analysis" in sequence_analysis:
                        # Update profile with new analysis
                        updated_profile.ai_analysis = sequence_analysis
                        comp_analysis = sequence_analysis["comprehensive_analysis"]
                        updated_profile.name = comp_analysis.get("nickname", updated_profile.name)
                        updated_profile.description = comp_analysis.get("summary", updated_profile.description)
                        logger.info(f"✅ Updated profile with sequence analysis: {comp_analysis.get('nickname', 'Unknown')}")
                    else:
                        logger.warning("❌ Sequence analysis failed")
            else:
                # Create new profile with comprehensive AI analysis
                logger.info(f"🧠 Running comprehensive AI analysis on new person...")
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
                
                # Extract nickname and description for profile creation
                nickname_data = ai_analysis.get("nickname_and_features", {})
                nickname = nickname_data.get("nickname", "Unknown Person")
                description = nickname_data.get("overall_impression", "Person detected")
                
                if isinstance(description, dict):
                    description = str(description)
                
                profile = self.person_clustering.create_new_profile(detection, image_path, description)
                
                # Set the nickname as the profile name
                profile.name = nickname
                
                # Store comprehensive analysis in profile
                if hasattr(profile, 'ai_analysis'):
                    profile.ai_analysis = ai_analysis
                
                logger.info(f"✅ Created new profile: {profile.id} with comprehensive AI analysis")
                logger.info(f"📊 Analysis includes: {list(ai_analysis.keys())}")
            
            # Save profiles periodically
            self.person_clustering.save_profiles()
            
            # Note: Real-time updates will be handled by UI polling and WebSocket fallback
            
        except Exception as e:
            logger.error(f"Error processing person detection: {e}")
    
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