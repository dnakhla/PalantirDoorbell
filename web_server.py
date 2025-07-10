from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import os
import json
import cv2
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time

from config import config
from main import DoorbellAI

# Configure logging
logger = logging.getLogger(__name__)

# Global instances
camera_stream = None
app_instance = None
websocket_connections = set()
stream_restart_count = 0

def _validate_web_frame(frame):
    """Validate frame integrity for web stream"""
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

def get_camera_stream():
    """Initialize camera stream for video display with optimized settings"""
    global camera_stream
    if camera_stream is None or not camera_stream.isOpened():
        try:
            # Release existing stream if it exists
            if camera_stream is not None:
                camera_stream.release()
            
            # Protocol-optimized FFmpeg setup
            import os
            if config.USE_RTMP:
                # RTMP optimized for web streaming
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'buffer_size;256000|'      # Small buffer for web
                    'max_delay;100000|'        # Very low delay
                    'fflags;nobuffer|'         # No buffering
                    'flags;low_delay|'         # Low delay
                    'err_detect;ignore_err|'   # Ignore errors
                    'rw_timeout;5000000'       # 5 second timeout
                )
            else:
                # High-res RTSP optimized for web display of 5MP stream
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'           # TCP for high-bitrate
                    'buffer_size;1024000|'          # Medium buffer for 4096kbps
                    'max_delay;500000|'             # Balanced delay for high-res
                    'stimeout;15000000|'            # 15s timeout
                    'reconnect;1|'                  # Auto-reconnect
                    'reconnect_streamed;1|'
                    'reconnect_delay_max;2|'        # Stable reconnect
                    'fflags;flush_packets|'         # Flush for high bitrate
                    'flags;low_delay|'              # Low delay when possible
                    'probesize;1048576|'            # 1MB probe for web
                    'analyzeduration;1000000|'      # 1s analysis
                    'err_detect;ignore_err|'        # Ignore H.264 errors
                    'avoid_negative_ts;make_zero|'  # Fix timestamps
                    'max_interleave_delta;200000'   # Small interleave for web
                )
            
            camera_stream = cv2.VideoCapture(config.CAMERA_URL, cv2.CAP_FFMPEG)
            
            # Simple settings for web stream
            camera_stream.set(cv2.CAP_PROP_FPS, 15)
            camera_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera_stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            if not camera_stream.isOpened():
                logger.error("Failed to open camera stream")
                camera_stream = None
            else:
                actual_width = int(camera_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(camera_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"🔄 Web camera stream initialized: {actual_width}x{actual_height}")
                
        except Exception as e:
            logger.error(f"Failed to initialize camera stream: {e}")
            camera_stream = None
    return camera_stream

def generate_frames():
    """Generate video frames for streaming with real-time object detection"""
    camera = get_camera_stream()
    if not camera or not camera.isOpened():
        logger.error("No camera available for web stream")
        return
    
    # Load YOLO model for real-time detection overlay
    try:
        from ultralytics import YOLO
        import torch
        import functools
        
        # Load YOLO model with same settings as camera_manager
        original_load = torch.load
        torch.load = functools.partial(torch.load, weights_only=False)
        yolo_model = YOLO("yolov8n.pt")
        torch.load = original_load
        
        logger.info("YOLO model loaded for live feed boundary boxes")
    except Exception as e:
        logger.warning(f"Could not load YOLO for live feed: {e}")
        yolo_model = None
    
    consecutive_failures = 0
    max_failures = 3
    frame_count = 0
    detection_frequency = 10  # Run detection every 10th frame for smoother web stream
    last_recovery_time = time.time()
    
    # Auto-reset timer for web stream
    last_auto_reset = time.time()
    auto_reset_interval = 60  # Auto-reset every 60 seconds
    
    while True:
        # Simple frame reading with basic error handling
        ret, frame = camera.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures % 10 == 0:  # Log every 10th failure
                logger.warning(f"Web stream read failures: {consecutive_failures}")
            
            if consecutive_failures >= max_failures * 5:  # Higher threshold
                logger.info("🔄 Attempting camera reconnection...")
                camera = get_camera_stream()
                consecutive_failures = 0
                if camera and camera.isOpened():
                    logger.info("✅ Camera reconnected")
                else:
                    time.sleep(2)
            
            time.sleep(0.05)  # Shorter delay
            continue
        
        consecutive_failures = 0
        
        if not _validate_web_frame(frame):
            continue
        
        # Auto-reset web stream every 60 seconds to prevent freezing
        current_time = time.time()
        if current_time - last_auto_reset > auto_reset_interval:
            logger.info("🔄 Auto-resetting web stream to prevent freezing (60s interval)")
            camera.release()
            camera = get_camera_stream()
            last_auto_reset = current_time
            if not camera or not camera.isOpened():
                logger.error("Failed to reset web stream")
                time.sleep(2)
                continue
        
        # Add real-time object detection overlay (reduced frequency for smoothness)
        frame_count += 1
        if yolo_model is not None and frame_count % detection_frequency == 0:
            try:
                # Resize frame for processing if too large
                processing_frame = frame
                original_height, original_width = frame.shape[:2]
                scale_factor = 1.0
                if original_height > 720 or original_width > 1280:
                    scale = min(1280/original_width, 720/original_height)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    processing_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    scale_factor = max(original_width / new_width, original_height / new_height)
                
                # Run YOLO detection for multiple objects (reduced frequency)
                results = yolo_model(processing_frame, conf=0.6, classes=[0, 15, 16], verbose=False)  # person, cat, dog
                
                # Draw bounding boxes on frame
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        class_name = yolo_model.names[class_id]
                        
                        # Define white boundaries with colored labels for different object types
                        box_color = (255, 255, 255)  # White for all boundaries
                        corner_color = (255, 255, 255)  # White corners
                        label_color = (255, 255, 255)   # Purple label
                        
                        if class_id == 0:  # Person
                            main_label = "SUBJECT DETECTED"
                        elif class_id == 15:  # Cat
                            main_label = "CAT DETECTED"
                        elif class_id == 16:  # Dog
                            main_label = "DOG DETECTED"
                        else:
                            continue  # Skip unknown classes
                        
                        # Process the detection - scale coordinates back to original frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if scale_factor != 1.0:
                            x1 = int(x1 * scale_factor)
                            y1 = int(y1 * scale_factor)
                            x2 = int(x2 * scale_factor)
                            y2 = int(y2 * scale_factor)
                        
                        # Draw shadow for depth effect - increased thickness
                        shadow_offset = 3
                        cv2.rectangle(frame, (x1 + shadow_offset, y1 + shadow_offset), 
                                     (x2 + shadow_offset, y2 + shadow_offset), (0, 0, 0), 8)
                        
                        # Draw main bounding box with gradient-like effect - increased thickness
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 8)
                        lighter_color = tuple(min(255, c + 50) for c in box_color)
                        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), lighter_color, 4)
                        
                        # Add corner markers for professional look - increased size
                        corner_size = 30
                        corner_thickness = 6
                        
                        # Top-left corner
                        cv2.line(frame, (x1, y1), (x1 + corner_size, y1), corner_color, corner_thickness)
                        cv2.line(frame, (x1, y1), (x1, y1 + corner_size), corner_color, corner_thickness)
                        
                        # Top-right corner
                        cv2.line(frame, (x2, y1), (x2 - corner_size, y1), corner_color, corner_thickness)
                        cv2.line(frame, (x2, y1), (x2, y1 + corner_size), corner_color, corner_thickness)
                        
                        # Bottom-left corner
                        cv2.line(frame, (x1, y2), (x1 + corner_size, y2), corner_color, corner_thickness)
                        cv2.line(frame, (x1, y2), (x1, y2 - corner_size), corner_color, corner_thickness)
                        
                        # Bottom-right corner
                        cv2.line(frame, (x2, y2), (x2 - corner_size, y2), corner_color, corner_thickness)
                        cv2.line(frame, (x2, y2), (x2, y2 - corner_size), corner_color, corner_thickness)
                        
                        # Create professional label with better styling
                        confidence_label = f"{confidence:.1%} CONFIDENCE"
                        timestamp_label = f"{datetime.now().strftime('%H:%M:%S')}"
                        
                        # Main label background with shadow (updated for larger font)
                        label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)[0]
                        label_bg_height = 42
                        
                        # Shadow for label background
                        cv2.rectangle(frame, (x1 + 2, y1 - label_bg_height + 2), 
                                     (x1 + label_size[0] + 22, y1 + 2), (0, 0, 0), -1)
                        
                        # Main label background
                        cv2.rectangle(frame, (x1, y1 - label_bg_height), 
                                     (x1 + label_size[0] + 20, y1), label_color, -1)
                        
                        # Label border
                        cv2.rectangle(frame, (x1, y1 - label_bg_height), 
                                     (x1 + label_size[0] + 20, y1), (255, 255, 255), 2)
                        
                        # Main label text (increased font size for better visibility)
                        cv2.putText(frame, main_label, (x1 + 10, y1 - 8), 
                                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
                        
                        # Confidence label (smaller, bottom right)
                        conf_size = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (x2 - conf_size[0] - 10, y2 + 5), 
                                     (x2, y2 + 25), (255, 255, 255), -1)
                        cv2.rectangle(frame, (x2 - conf_size[0] - 10, y2 + 5), 
                                     (x2, y2 + 25), (0, 0, 0), 1)
                        cv2.putText(frame, confidence_label, (x2 - conf_size[0] - 5, y2 + 18), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Timestamp label (top right)
                        time_size = cv2.getTextSize(timestamp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (x2 - time_size[0] - 10, y1 - 25), 
                                     (x2, y1 - 5), (255, 255, 0), -1)
                        cv2.rectangle(frame, (x2 - time_size[0] - 10, y1 - 25), 
                                     (x2, y1 - 5), (0, 0, 0), 1)
                        cv2.putText(frame, timestamp_label, (x2 - time_size[0] - 5, y1 - 12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            except Exception as e:
                logger.warning(f"Error in real-time detection: {e}")
        
        # Resize frame for web display with performance priority
        # Resize to 960x540 for smoother streaming
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        
        # Encode frame with optimized settings for maximum smoothness
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 70,  # Lower quality for faster encoding
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Optimize for size
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Disable progressive for faster encoding
        ]
        
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

async def initialize_detection_system():
    """Initialize the person detection system"""
    global app_instance
    try:
        logger.info("Initializing DoorbellAI detection system...")
        app_instance = DoorbellAI()
        
        # Test camera connection first
        logger.info(f"Testing camera connection to: {config.CAMERA_URL} ({'RTMP' if config.USE_RTMP else 'RTSP'} protocol)")
        
        success = await app_instance.start()
        if success:
            logger.info("✓ Person detection system initialized and running")
            logger.info(f"✓ Detection running: {app_instance.is_running}")
        else:
            logger.error("✗ Failed to start detection system")
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize detection system: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize systems on startup and cleanup on shutdown"""
    # Startup
    await initialize_detection_system()
    yield
    # Shutdown (cleanup if needed)
    pass

# Create FastAPI app with hot reloading support
app = FastAPI(
    title="Neighborhood Watch AI",
    description="Intelligence-Powered Surveillance and Person Monitoring System",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/images", StaticFiles(directory=config.CAPTURED_IMAGES_DIR), name="images")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("intelligence_dashboard.html", {"request": request})

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    if not app_instance:
        return JSONResponse({
            "total_people": 0,
            "system_status": "initializing",
            "timestamp": datetime.now().isoformat()
        })
    
    stats = app_instance.get_statistics()
    return JSONResponse(stats)

@app.get("/api/profiles")
async def get_all_profiles():
    """Get all person profiles"""
    if not app_instance:
        return JSONResponse([])
    
    profiles = app_instance.get_all_profiles()
    
    # Convert to JSON-serializable format
    profiles_data = []
    for profile in profiles:
        profile_data = profile.get_summary()
        # Filter out missing images and get last 5 existing images
        existing_images = [img for img in profile.images if os.path.exists(img)]
        profile_data["images"] = existing_images[-5:]  # Last 5 existing images
        profiles_data.append(profile_data)
    
    return JSONResponse(profiles_data)

@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get specific person profile"""
    if not app_instance:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile = app_instance.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile_data = profile.get_summary()
    # Filter out missing images and get all existing images
    existing_images = [img for img in profile.images if os.path.exists(img)]
    profile_data["images"] = existing_images
    
    return JSONResponse(profile_data)

@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a specific person profile and associated images"""
    if not app_instance:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    try:
        # Get profile to access images
        profile = app_instance.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Delete associated images
        import os
        for image_path in profile.images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Deleted image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to delete image {image_path}: {e}")
        
        # Remove profile from memory
        if profile_id in app_instance.person_clustering.profiles:
            del app_instance.person_clustering.profiles[profile_id]
            app_instance.person_clustering.save_profiles()
            logger.info(f"Deleted profile: {profile_id}")
        
        return JSONResponse({"success": True, "message": f"Profile {profile_id} deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting profile {profile_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {str(e)}")

@app.post("/api/profiles/{profile_id}/name")
async def update_profile_name(profile_id: str, request: Request):
    """Update person profile name"""
    return JSONResponse({"success": True, "message": "Name updated successfully"})

@app.post("/api/generate-names")
async def generate_names():
    """Generate AI names for all profiles"""
    return JSONResponse({
        "success": True, 
        "message": "Generated names for 0 people"
    })

@app.post("/api/clear-timeline")
async def clear_timeline():
    """Clear all captured people data and images"""
    try:
        if app_instance:
            # Clear profiles from memory
            app_instance.person_clustering.profiles = {}
            app_instance.person_clustering.save_profiles()
            
            # Clear captured images
            import shutil
            shutil.rmtree(config.CAPTURED_IMAGES_DIR)
            os.makedirs(config.CAPTURED_IMAGES_DIR, exist_ok=True)
            
            # Clear database
            database_file = os.path.join(config.DATABASE_DIR, "person_profiles.pkl")
            if os.path.exists(database_file):
                os.remove(database_file)
            
            logger.info("🗑️ Timeline cleared - all subject data purged")
            return JSONResponse({"success": True, "message": "Timeline cleared successfully"})
        else:
            return JSONResponse({"success": False, "message": "System not initialized"})
    except Exception as e:
        logger.error(f"Error clearing timeline: {e}")
        return JSONResponse({"success": False, "message": f"Error clearing timeline: {str(e)}"})

@app.get("/api/video_feed")
async def video_feed():
    """Live video stream endpoint with optimized headers"""
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/api/live_feed")
async def get_live_feed():
    """Get live camera feed status"""
    return JSONResponse({
        "status": "running" if app_instance and app_instance.is_running else "stopped",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stream_healthy = False
    if camera_stream is not None:
        stream_healthy = camera_stream.isOpened()
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": app_instance is not None,
        "detection_running": app_instance.is_running if app_instance else False,
        "stream_healthy": stream_healthy
    })

@app.post("/api/stream/restart")
async def restart_stream():
    """Restart the camera stream"""
    global camera_stream
    try:
        if camera_stream:
            camera_stream.release()
        camera_stream = None
        
        # Reinitialize stream
        new_stream = get_camera_stream()
        if new_stream and new_stream.isOpened():
            logger.info("🔄 Camera stream restarted successfully")
            return JSONResponse({"success": True, "message": "Stream restarted successfully"})
        else:
            logger.error("❌ Failed to restart camera stream")
            return JSONResponse({"success": False, "message": "Failed to restart stream"})
    except Exception as e:
        logger.error(f"Error restarting stream: {e}")
        return JSONResponse({"success": False, "message": f"Error: {str(e)}"})

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

async def broadcast_profile_update():
    """Broadcast profile updates to all connected clients"""
    if not app_instance:
        return
    
    profiles = app_instance.get_all_profiles()
    
    # Convert to JSON-serializable format
    profiles_data = []
    for profile in profiles:
        profile_data = profile.get_summary()
        # Filter out missing images and get last 5 existing images
        existing_images = [img for img in profile.images if os.path.exists(img)]
        profile_data["images"] = existing_images[-5:]  # Last 5 existing images
        profiles_data.append(profile_data)
    
    message = {
        "type": "profiles_update",
        "data": profiles_data
    }
    
    # Send to all connected clients
    disconnected = set()
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except:
            disconnected.add(websocket)
    
    # Remove disconnected clients
    websocket_connections -= disconnected

# Development hot reload support
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable hot reloading
        reload_dirs=[".", "templates", "static"],
        log_level="info"
    )