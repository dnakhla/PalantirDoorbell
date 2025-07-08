import cv2
import time
import asyncio
import threading
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import whisper
import subprocess
import base64
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict, deque
import glob
from pathlib import Path
from typing import Dict, List, Optional

# --- Configuration ---
RTSP_URL = "rtsp://dnakhla%40gmail.com:vi6Z39oCPz-iYmZ@192.168.86.42:554/h264Preview_01_main"
LOG_FILE = "detections.log.parquet"
ANALYSIS_LOG_FILE = "analysis.log.parquet"
PEOPLE_DB_FILE = "people_database.parquet"
INSIGHTS_DB_FILE = "insights_database.parquet"
TRANSCRIPTION_LOG_FILE = "transcriptions.log"
LOG_COOLDOWN_SECONDS = 5  # Cooldown per object class to avoid spamming the log
CAPTURED_IMAGES_DIR = "captured_subjects"
AUDIO_RECORDINGS_DIR = "audio_recordings"
IMAGE_COOLDOWN_SECONDS = 15  # Cooldown per track_id for saving new images
TEMPORAL_GROUPING_WINDOW = 120  # Seconds - captures within this window likely same person
MAX_CAPTURES_PER_SUBJECT = 10  # Maximum captures per subject type
MOVEMENT_THRESHOLD = 50  # Minimum pixels movement to trigger capture
VISION_ANALYSIS_INTERVAL = 60  # Seconds between vision analysis
MAX_TRANSCRIPTION_HISTORY = 50  # Maximum number of transcriptions to keep
AUDIO_SEGMENT_DURATION = 10  # Duration of each audio segment in seconds
MAX_AUDIO_FILES = 100  # Maximum number of audio files to keep

# Security-relevant objects to INCLUDE in detection and capture
SECURITY_OBJECTS = {
    'person', 'dog', 'cat', 'bird',  # Living beings
    'backpack', 'handbag', 'suitcase',  # Suspicious containers
    'knife', 'scissors',  # Potential weapons
    'bottle', 'wine glass', 'cup',  # Alcohol/substances
    'cell phone', 'laptop',  # Electronics (theft targets)
    'bicycle', 'motorcycle'  # Quick escape vehicles
}

# All other YOLO classes will be excluded - only security-relevant objects are processed

# --- Global Variables & Settings ---
app = FastAPI()

# Create directories early to avoid static file mounting errors
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
os.makedirs(AUDIO_RECORDINGS_DIR, exist_ok=True)

last_frame = None
last_detections = []
last_transcription = ""
cumulative_transcription = ""  # Append all transcriptions here
frame_lock = threading.Lock()
transcription_lock = threading.Lock()
last_log_time = {}
last_image_capture_time = {}
last_positions = {}  # Track previous positions for movement detection
transcription_history = deque(maxlen=MAX_TRANSCRIPTION_HISTORY)
audio_files_list = deque(maxlen=MAX_AUDIO_FILES)  # Track saved audio files
subject_patterns = defaultdict(list)  # Track patterns for each subject type
people_groups = {}  # Track identified people groups: {person_id: {'images': [], 'description': '', 'last_seen': timestamp}}
last_vision_analysis = 0
current_insights = {
    "subject_analysis": "",
    "patterns": "",
    "audio_highlights": "",
    "timestamp": 0
}

# Fixed settings as per user request
settings = {
    "confidence_threshold": 0.5,
    "grayscale": False, # Still kept for potential future use, but not exposed in UI
}

# --- Logging ---
def setup_log_file():
    """Create the log file with a header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        # Define schema for the Parquet file
        schema = pa.schema([
            pa.field("timestamp", pa.float64()),
            pa.field("class", pa.string()),
            pa.field("confidence", pa.float64()),
            pa.field("track_id", pa.int64()),
            pa.field("image_path", pa.string())
        ])
        # Create an empty table and write to Parquet
        table = pa.Table.from_arrays([[], [], [], [], []], schema=schema)
        pq.write_table(table, LOG_FILE)

def setup_analysis_log_file():
    """Create the analysis log file with a header if it doesn't exist."""
    if not os.path.exists(ANALYSIS_LOG_FILE):
        # Define schema for the analysis Parquet file
        schema = pa.schema([
            pa.field("timestamp", pa.float64()),
            pa.field("hour", pa.int64()),
            pa.field("analysis_type", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("images_analyzed", pa.string()),  # JSON string of image paths
            pa.field("people_count", pa.int64()),
            pa.field("other_detections", pa.string())  # JSON string of other detection counts
        ])
        # Create an empty table and write to Parquet
        table = pa.Table.from_arrays([[], [], [], [], [], [], []], schema=schema)
        pq.write_table(table, ANALYSIS_LOG_FILE)

def setup_people_database():
    """Create the people database file if it doesn't exist."""
    if not os.path.exists(PEOPLE_DB_FILE):
        # Define schema for the people database
        schema = pa.schema([
            pa.field("person_id", pa.string()),
            pa.field("assigned_name", pa.string()),  # LLM-generated name
            pa.field("persona", pa.string()),  # LLM-generated persona
            pa.field("first_seen", pa.float64()),
            pa.field("last_seen", pa.float64()),
            pa.field("description", pa.string()),
            pa.field("appearance_features", pa.string()),  # JSON string of features
            pa.field("visit_count", pa.int64()),
            pa.field("sample_images", pa.string()),  # JSON string of image paths
            pa.field("persona_generated", pa.bool_())  # Whether persona has been generated
        ])
        # Create an empty table and write to Parquet
        table = pa.Table.from_arrays([[], [], [], [], [], [], [], [], [], []], schema=schema)
        pq.write_table(table, PEOPLE_DB_FILE)

def setup_insights_database():
    """Create the insights database file if it doesn't exist."""
    if not os.path.exists(INSIGHTS_DB_FILE):
        # Define schema for the insights database
        schema = pa.schema([
            pa.field("timestamp", pa.float64()),
            pa.field("hour", pa.int64()),  # Hour since epoch for grouping
            pa.field("date_hour", pa.string()),  # Human readable hour
            pa.field("insight_type", pa.string()),  # 'hourly', 'person_specific', 'general'
            pa.field("person_id", pa.string()),  # For person-specific insights
            pa.field("subject_analysis", pa.string()),
            pa.field("patterns", pa.string()),
            pa.field("audio_highlights", pa.string()),
            pa.field("people_activity", pa.string()),  # JSON string of people activities
            pa.field("security_alerts", pa.string()),  # JSON string of security observations
            pa.field("summary", pa.string())  # Overall summary for the hour/person
        ])
        # Create an empty table and write to Parquet
        table = pa.Table.from_arrays([[], [], [], [], [], [], [], [], [], [], []], schema=schema)
        pq.write_table(table, INSIGHTS_DB_FILE)

def log_detection(class_name, confidence, track_id, image_path=None):
    """Log a single detection event to the Parquet file with a cooldown."""
    current_time = time.time()
    # Use a combination of class_name and track_id for cooldown to log unique instances
    log_key = f"{class_name}_{track_id}"
    if log_key not in last_log_time or (current_time - last_log_time[log_key]) > LOG_COOLDOWN_SECONDS:
        # Create a DataFrame for the new entry
        new_data = pd.DataFrame({
            "timestamp": [current_time],
            "class": [class_name],
            "confidence": [confidence],
            "track_id": [track_id],
            "image_path": [image_path]
        })
        # Append to Parquet file
        table = pa.Table.from_pandas(new_data)
        # Use append_table to add rows to existing Parquet file
        with pq.ParquetWriter(LOG_FILE, table.schema) as writer:
            writer.write_table(table)
        last_log_time[log_key] = current_time

def log_analysis(summary, images_analyzed, people_count=0, other_detections=None):
    """Log analysis results to the analysis Parquet file."""
    try:
        current_time = time.time()
        current_hour = int(current_time // 3600)  # Hour since epoch
        
        # Convert lists/dicts to JSON strings
        images_json = json.dumps(images_analyzed) if images_analyzed else "[]"
        other_json = json.dumps(other_detections) if other_detections else "{}"
        
        # Create a DataFrame for the new entry
        new_data = pd.DataFrame({
            "timestamp": [current_time],
            "hour": [current_hour],
            "analysis_type": ["background_vision"],
            "summary": [summary],
            "images_analyzed": [images_json],
            "people_count": [people_count],
            "other_detections": [other_json]
        })
        
        # Append to Parquet file
        table = pa.Table.from_pandas(new_data)
        
        # Check if file exists and append
        if os.path.exists(ANALYSIS_LOG_FILE):
            existing_table = pq.read_table(ANALYSIS_LOG_FILE)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, ANALYSIS_LOG_FILE)
        else:
            pq.write_table(table, ANALYSIS_LOG_FILE)
            
    except Exception as e:
        print(f"Error logging analysis: {e}")

def save_insights_to_database(subject_analysis, patterns, audio_highlights, people_activity=None, security_alerts=None):
    """Save insights to the database for hour-by-hour and person-by-person tracking."""
    try:
        current_time = time.time()
        current_hour = int(current_time // 3600)  # Hour since epoch
        date_hour = time.strftime("%Y-%m-%d %H:00", time.localtime(current_time))
        
        # Create hourly insight entry
        hourly_summary = f"Activity from {date_hour}: {subject_analysis}; {patterns}; {audio_highlights}"
        
        hourly_data = {
            "timestamp": current_time,
            "hour": current_hour,
            "date_hour": date_hour,
            "insight_type": "hourly",
            "person_id": "",
            "subject_analysis": subject_analysis,
            "patterns": patterns,
            "audio_highlights": audio_highlights,
            "people_activity": json.dumps(people_activity) if people_activity else "{}",
            "security_alerts": json.dumps(security_alerts) if security_alerts else "{}",
            "summary": hourly_summary
        }
        
        # Save hourly insights
        hourly_df = pd.DataFrame([hourly_data])
        hourly_table = pa.Table.from_pandas(hourly_df)
        
        # Append to database
        if os.path.exists(INSIGHTS_DB_FILE):
            existing_table = pq.read_table(INSIGHTS_DB_FILE)
            combined_table = pa.concat_tables([existing_table, hourly_table])
            pq.write_table(combined_table, INSIGHTS_DB_FILE)
        else:
            pq.write_table(hourly_table, INSIGHTS_DB_FILE)
        
        # Create person-specific insights if we have people data
        if people_activity and isinstance(people_activity, list):
            for person_data in people_activity:
                if isinstance(person_data, dict) and 'tracked_id' in person_data:
                    person_id = person_data['tracked_id']
                    person_activity = person_data.get('activity', '')
                    person_description = person_data.get('description', '')
                    
                    person_summary = f"Person {person_id} activity at {date_hour}: {person_activity}. Description: {person_description}"
                    
                    person_insight_data = {
                        "timestamp": current_time,
                        "hour": current_hour,
                        "date_hour": date_hour,
                        "insight_type": "person_specific",
                        "person_id": person_id,
                        "subject_analysis": f"Person {person_id}: {person_description}",
                        "patterns": f"Activity: {person_activity}",
                        "audio_highlights": audio_highlights,
                        "people_activity": json.dumps([person_data]),
                        "security_alerts": json.dumps(security_alerts) if security_alerts else "{}",
                        "summary": person_summary
                    }
                    
                    person_df = pd.DataFrame([person_insight_data])
                    person_table = pa.Table.from_pandas(person_df)
                    
                    # Append person insight
                    existing_table = pq.read_table(INSIGHTS_DB_FILE)
                    combined_table = pa.concat_tables([existing_table, person_table])
                    pq.write_table(combined_table, INSIGHTS_DB_FILE)
        
    except Exception as e:
        print(f"Error saving insights to database: {e}")

# --- Image Capture ---
def save_captured_image(frame, bbox, track_id, class_name):
    """Saves a cropped image of the detected object, organized by subject type with limits."""
    current_time = time.time()
    image_key = f"{class_name}_{track_id}"

    # Check movement before capturing
    if not has_moved(track_id, bbox):
        return None

    if image_key not in last_image_capture_time or \
       (current_time - last_image_capture_time[image_key]) > IMAGE_COOLDOWN_SECONDS:

        # Create subject-specific directory
        subject_dir = os.path.join(CAPTURED_IMAGES_DIR, class_name)
        os.makedirs(subject_dir, exist_ok=True)

        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bounding box coordinates are within frame dimensions
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            cropped_img = frame[y1:y2, x1:x2]
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            filename = f"{track_id}_{timestamp_str}.jpg"
            filepath = os.path.join(subject_dir, filename)
            
            # Clean up old captures if we exceed the limit
            cleanup_old_captures(subject_dir, class_name)
            
            cv2.imwrite(filepath, cropped_img)
            last_image_capture_time[image_key] = current_time
            return filepath
    return None

def has_moved(track_id, bbox):
    """Check if the object has moved enough to warrant a new capture."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    if track_id not in last_positions:
        last_positions[track_id] = (center_x, center_y)
        return True  # First detection, always capture
    
    prev_x, prev_y = last_positions[track_id]
    distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
    
    if distance > MOVEMENT_THRESHOLD:
        last_positions[track_id] = (center_x, center_y)
        return True
    
    return False

def cleanup_old_captures(subject_dir, class_name):
    """Remove oldest captures if we exceed MAX_CAPTURES_PER_SUBJECT."""
    try:
        files = []
        for filename in os.listdir(subject_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(subject_dir, filename)
                files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove oldest files if we exceed the limit
        while len(files) >= MAX_CAPTURES_PER_SUBJECT:
            oldest_file = files.pop(0)
            try:
                os.remove(oldest_file[0])
                print(f"Removed old capture: {oldest_file[0]}")
            except OSError as e:
                print(f"Error removing file {oldest_file[0]}: {e}")
    except Exception as e:
        print(f"Error cleaning up captures for {class_name}: {e}")

# --- LLM Vision Analysis ---
def encode_image_to_base64(image_path):
    """Encode image to base64 for LLM vision analysis."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def analyze_subjects_with_vision():
    """Analyze captured subjects using GPT-4.1-mini vision capabilities for enhanced people tracking."""
    try:
        # Check if we have an API key for background analysis
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return analyze_subjects_fallback()
        
        # Get recent captures, focusing on people
        recent_captures = []
        people_images = []
        
        if os.path.exists(CAPTURED_IMAGES_DIR):
            for class_name in os.listdir(CAPTURED_IMAGES_DIR):
                subject_dir = os.path.join(CAPTURED_IMAGES_DIR, class_name)
                if os.path.isdir(subject_dir):
                    # Get most recent captures
                    files = []
                    for filename in os.listdir(subject_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            filepath = os.path.join(subject_dir, filename)
                            files.append((filepath, os.path.getmtime(filepath)))
                    
                    if files:
                        # Sort by modification time (newest first)
                        files.sort(key=lambda x: x[1], reverse=True)
                        recent_captures.append((class_name, files[0][0]))
                        
                        # Collect people images for enhanced analysis
                        if class_name == 'person':
                            people_images.extend([f[0] for f in files[:3]])  # Last 3 person images
        
        if not recent_captures:
            return "No recent captures available for analysis."
        
        # Enhanced analysis for people using GPT-4.1-mini
        analysis_summary = ""
        if people_images:
            try:
                people_analysis = analyze_people_with_gpt(people_images, api_key)
                if people_analysis:
                    analysis_summary = f"People activity outside home in 19123: {people_analysis}"
            except Exception as e:
                print(f"Error in GPT people analysis: {e}")
        
        # Update pattern data and generate basic counts
        analysis_parts = []
        for class_name, image_path in recent_captures:
            # Record pattern data
            current_time = time.time()
            subject_patterns[class_name].append({
                'timestamp': current_time,
                'path': image_path
            })
            
            # Keep only recent patterns (last hour)
            cutoff_time = current_time - 3600
            subject_patterns[class_name] = [
                p for p in subject_patterns[class_name] 
                if p['timestamp'] > cutoff_time
            ]
            
            count = len(subject_patterns[class_name])
            if class_name != 'person':  # People summary handled above
                analysis_parts.append(f"{class_name}: {count} detections")
        
        # Combine enhanced people analysis with other detections
        if analysis_summary:
            if analysis_parts:
                return f"{analysis_summary}; Other: {'; '.join(analysis_parts)}"
            else:
                return analysis_summary
        else:
            return "; ".join(analysis_parts) if analysis_parts else "No significant activity detected"
        
    except Exception as e:
        print(f"Error in vision analysis: {e}")
        return analyze_subjects_fallback()

def analyze_people_with_gpt(image_paths, api_key):
    """Use GPT-4.1-mini to analyze people images and provide identification and tracking."""
    if not image_paths:
        return None
    
    # Get recent and grouped people data
    people_analysis = identify_and_group_people(image_paths, api_key)
    if people_analysis:
        # Log the analysis
        log_people_analysis(people_analysis, image_paths)
        return people_analysis.get('summary', 'No analysis available')
    
    return None

def group_images_by_time(image_paths):
    """Group images by temporal proximity - images captured within TEMPORAL_GROUPING_WINDOW are likely the same person."""
    if not image_paths:
        return []
    
    # Get image timestamps and sort by time (newest first)
    images_with_timestamps = []
    for img_path in image_paths:
        try:
            timestamp = os.path.getmtime(img_path)
            images_with_timestamps.append((img_path, timestamp))
        except:
            continue
    
    # Sort by timestamp (newest first)
    images_with_timestamps.sort(key=lambda x: x[1], reverse=True)
    
    # Group images within temporal windows
    groups = []
    current_group = []
    last_timestamp = None
    
    for img_path, timestamp in images_with_timestamps:
        if last_timestamp is None or (last_timestamp - timestamp) <= TEMPORAL_GROUPING_WINDOW:
            # Same temporal group
            current_group.append(img_path)
            last_timestamp = timestamp
        else:
            # Start new group
            if current_group:
                groups.append(current_group)
            current_group = [img_path]
            last_timestamp = timestamp
    
    # Add final group
    if current_group:
        groups.append(current_group)
    
    return groups

def identify_and_group_people(image_paths, api_key):
    """Advanced people identification and grouping using GPT-4.1-mini with comprehensive context."""
    if not image_paths:
        return None
    
    # Smart temporal grouping - group images by time clusters
    temporal_groups = group_images_by_time(image_paths)
    
    # Select representative images from each temporal group (max 5 total)
    selected_images = []
    for group in temporal_groups[:5]:  # Max 5 groups
        selected_images.append(group[0])  # Take first (most recent) from each group
    
    if not selected_images:
        return None
    
    # Get current context for analysis
    current_time = time.time()
    current_date = time.strftime("%A, %B %d, %Y", time.localtime(current_time))
    current_time_str = time.strftime("%I:%M %p", time.localtime(current_time))
    day_of_week = time.strftime("%A", time.localtime(current_time))
    
    # Get image timestamps for temporal context
    image_contexts = []
    for i, img_path in enumerate(selected_images, 1):
        try:
            img_timestamp = os.path.getmtime(img_path)
            img_time = time.strftime("%I:%M %p", time.localtime(img_timestamp))
            img_contexts.append(f"Image {i}: captured at {img_time}")
        except:
            img_contexts.append(f"Image {i}: timestamp unknown")
    
    image_timing_context = "; ".join(img_contexts)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"SECURITY CAMERA INTELLIGENCE ANALYSIS\n"
                           f"Location: Residential doorbell camera, zip code 19123\n"
                           f"Current Date/Time: {current_date} at {current_time_str}\n"
                           f"Day Context: {day_of_week}\n"
                           f"Image Timing: {image_timing_context}\n\n"
                           f"ANALYSIS REQUIREMENTS:\n"
                           f"Analyze these security camera captures for comprehensive intelligence gathering.\n"
                           f"For each unique person detected:\n"
                           f"1) Assign ID (Person A, B, C, etc.)\n"
                           f"2) Detailed physical description (clothing, build, distinguishing features, accessories)\n"
                           f"3) Estimated demographics (age range, gender) if clearly visible\n"
                           f"4) Behavioral analysis (posture, movement, actions, intent)\n"
                           f"5) Temporal patterns (time of day significance, duration of presence)\n"
                           f"6) Security assessment (threat level, suspicious indicators, normal activity)\n"
                           f"7) Objects carried or interacted with\n"
                           f"8) Relationship to property (resident, visitor, delivery, suspicious)\n"
                           f"9) Environmental context (weather impact on behavior/clothing)\n"
                           f"10) Cross-reference with time patterns (school hours, work hours, etc.)\n\n"
                           f"FORMAT: {{\"people\": [{{\"id\": \"Person A\", \"description\": \"detailed physical\", \"demographics\": \"age/gender\", \"behavior\": \"actions observed\", \"timing\": \"temporal significance\", \"security_assessment\": \"threat/normal\", \"objects\": \"items carried\", \"relationship\": \"resident/visitor/delivery/suspicious\", \"images\": [1,3], \"intelligence_notes\": \"additional observations\"}}], \"scene_analysis\": \"overall scene intelligence\", \"security_summary\": \"key security insights\", \"recommendations\": \"suggested actions or monitoring\"}}"
                }
            ]
        }
    ]
    
    # Add images to the message with numbering
    for i, img_path in enumerate(selected_images, 1):
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4.1-mini",
        "messages": messages,
        "max_tokens": 800,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    # Parse JSON response
                    analysis_data = json.loads(content)
                    return process_people_identification(analysis_data, selected_images)
                except json.JSONDecodeError:
                    # Fallback to text summary
                    return {"summary": content, "people": []}
        else:
            print(f"GPT API Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"GPT people identification failed: {e}")
    
    return None

def process_people_identification(analysis_data, image_paths):
    """Process the GPT analysis to update people database and tracking."""
    global people_groups
    current_time = time.time()
    
    if "people" not in analysis_data:
        return analysis_data
    
    # Process each identified person
    for person_data in analysis_data["people"]:
        person_id = person_data.get("id", "Unknown")
        description = person_data.get("description", "")
        features = person_data.get("features", "")
        activity = person_data.get("activity", "")
        image_indices = person_data.get("images", [])
        
        # Get image paths for this person
        person_images = [image_paths[i-1] for i in image_indices if 0 < i <= len(image_paths)]
        
        # Check if this person matches any existing entries
        existing_person_id = find_matching_person(features, description)
        
        if existing_person_id:
            # Update existing person
            people_groups[existing_person_id]['last_seen'] = current_time
            people_groups[existing_person_id]['visit_count'] += 1
            people_groups[existing_person_id]['images'].extend(person_images)
            # Keep only recent images (last 10)
            people_groups[existing_person_id]['images'] = people_groups[existing_person_id]['images'][-10:]
        else:
            # Create new person entry
            new_person_id = f"Person_{len(people_groups) + 1}"
            people_groups[new_person_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'description': description,
                'features': features,
                'activity': activity,
                'visit_count': 1,
                'images': person_images
            }
        
        # Update the person ID in the analysis
        person_data["tracked_id"] = existing_person_id or new_person_id
    
    # Update people database
    save_people_to_database()
    
    return analysis_data

def find_matching_person(features, description):
    """Find if this person matches any existing tracked person."""
    # Simple keyword matching - could be enhanced with embedding similarity
    for person_id, data in people_groups.items():
        existing_features = data.get('features', '').lower()
        existing_desc = data.get('description', '').lower()
        
        # Check for matching clothing or distinctive features
        if features.lower() in existing_features or description.lower() in existing_desc:
            return person_id
    
    return None

def save_people_to_database():
    """Save current people groups to the database."""
    try:
        # Prepare data for saving
        data_rows = []
        for person_id, data in people_groups.items():
            data_rows.append({
                "person_id": person_id,
                "assigned_name": data.get('assigned_name', ''),
                "persona": data.get('persona', ''),
                "first_seen": data.get('first_seen', 0),
                "last_seen": data.get('last_seen', 0),
                "description": data.get('description', ''),
                "appearance_features": data.get('features', ''),
                "visit_count": data.get('visit_count', 1),
                "sample_images": json.dumps(data.get('images', [])[-3:]),  # Keep last 3 images
                "persona_generated": data.get('persona_generated', False)
            })
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, PEOPLE_DB_FILE)
            
    except Exception as e:
        print(f"Error saving people database: {e}")

def log_people_analysis(analysis_data, image_paths):
    """Log people analysis results."""
    try:
        summary = analysis_data.get('summary', 'People activity detected')
        people_count = len(analysis_data.get('people', []))
        
        # Create detailed log entry
        log_analysis(
            summary=summary,
            images_analyzed=image_paths,
            people_count=people_count,
            other_detections={"people_details": analysis_data.get('people', [])}
        )
        
    except Exception as e:
        print(f"Error logging people analysis: {e}")

def analyze_subjects_fallback():
    """Fallback analysis when GPT is not available."""
    try:
        analysis_parts = []
        current_time = time.time()
        cutoff_time = current_time - 3600
        
        for class_name, detections in subject_patterns.items():
            recent_detections = [d for d in detections if d['timestamp'] > cutoff_time]
            if recent_detections:
                count = len(recent_detections)
                analysis_parts.append(f"{class_name}: {count} detections in the last hour outside home in 19123")
        
        return "; ".join(analysis_parts) if analysis_parts else "No recent activity detected"
    except Exception as e:
        return f"Analysis error: {str(e)}"

def get_hourly_summaries(hours_back=24):
    """Generate hourly activity summaries from the analysis database."""
    try:
        summaries = []
        current_time = time.time()
        
        # Check if analysis log exists
        if not os.path.exists(ANALYSIS_LOG_FILE):
            return []
        
        df = pd.read_parquet(ANALYSIS_LOG_FILE)
        if df.empty:
            return []
        
        # Convert timestamps to datetime for grouping
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour_group'] = df['datetime'].dt.floor('H')
        
        # Get data for the specified time range
        cutoff_time = current_time - (hours_back * 3600)
        recent_df = df[df['timestamp'] >= cutoff_time]
        
        if recent_df.empty:
            return []
        
        # Group by hour and summarize
        hourly_groups = recent_df.groupby('hour_group')
        
        for hour, group in hourly_groups:
            hour_summary = {
                'hour': hour.strftime('%Y-%m-%d %H:00'),
                'timestamp': hour.timestamp(),
                'total_analyses': len(group),
                'people_detected': group['people_count'].sum(),
                'summaries': group['summary'].tolist(),
                'unique_people': 0
            }
            
            # Extract unique people information
            try:
                people_details = []
                for other_det in group['other_detections']:
                    if other_det and other_det != "{}":
                        details = json.loads(other_det)
                        if 'people_details' in details:
                            people_details.extend(details['people_details'])
                
                unique_people_ids = set()
                for person in people_details:
                    if 'tracked_id' in person:
                        unique_people_ids.add(person['tracked_id'])
                
                hour_summary['unique_people'] = len(unique_people_ids)
                hour_summary['people_activities'] = [p.get('activity', '') for p in people_details if p.get('activity')]
                
            except Exception as e:
                print(f"Error processing people details for hour {hour}: {e}")
            
            summaries.append(hour_summary)
        
        # Sort by timestamp (most recent first)
        summaries.sort(key=lambda x: x['timestamp'], reverse=True)
        return summaries
        
    except Exception as e:
        print(f"Error generating hourly summaries: {e}")
        return []

def get_activity_timeline(hours_back=12):
    """Generate a combined timeline of detections and analysis."""
    try:
        timeline = []
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        # Add detection events
        if os.path.exists(LOG_FILE):
            df = pd.read_parquet(LOG_FILE)
            recent_detections = df[df['timestamp'] >= cutoff_time]
            
            for _, row in recent_detections.iterrows():
                timeline.append({
                    'timestamp': row['timestamp'],
                    'type': 'detection',
                    'class': row['class'],
                    'track_id': row['track_id'],
                    'confidence': row['confidence'],
                    'description': f"{row['class']} detected (ID: {row['track_id']})"
                })
        
        # Add analysis events
        if os.path.exists(ANALYSIS_LOG_FILE):
            df = pd.read_parquet(ANALYSIS_LOG_FILE)
            recent_analyses = df[df['timestamp'] >= cutoff_time]
            
            for _, row in recent_analyses.iterrows():
                timeline.append({
                    'timestamp': row['timestamp'],
                    'type': 'analysis',
                    'people_count': row['people_count'],
                    'summary': row['summary'],
                    'description': f"AI Analysis: {row['summary'][:100]}..."
                })
        
        # Sort by timestamp (most recent first)
        timeline.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return timeline[:50]  # Return last 50 events
        
    except Exception as e:
        print(f"Error generating activity timeline: {e}")
        return []

def analyze_patterns():
    """Analyze patterns in subject detections."""
    try:
        patterns = []
        current_time = time.time()
        
        for class_name, detections in subject_patterns.items():
            if len(detections) > 1:
                # Calculate frequency
                recent_detections = [d for d in detections if current_time - d['timestamp'] < 3600]
                if len(recent_detections) > 2:
                    avg_interval = (recent_detections[-1]['timestamp'] - recent_detections[0]['timestamp']) / len(recent_detections)
                    if avg_interval < 300:  # Less than 5 minutes
                        patterns.append(f"Frequent {class_name} activity (every {avg_interval:.1f}s) outside home in 19123")
                    elif len(recent_detections) >= 5:
                        patterns.append(f"Regular {class_name} presence ({len(recent_detections)} detections/hour) outside home in 19123")
        
        return "; ".join(patterns) if patterns else "No significant patterns detected"
        
    except Exception as e:
        print(f"Error in pattern analysis: {e}")
        return f"Pattern analysis error: {str(e)}"

def analyze_audio_highlights():
    """Analyze audio transcription for highlights."""
    try:
        if not transcription_history:
            return "No audio transcriptions available"
        
        # Get recent transcriptions
        recent_transcriptions = list(transcription_history)[-10:]  # Last 10 transcriptions
        
        if not recent_transcriptions:
            return "No recent audio activity"
        
        # Simple keyword analysis (placeholder for more sophisticated LLM analysis)
        keywords = ['delivery', 'package', 'knock', 'door', 'bell', 'help', 'hello', 'car', 'truck', 'emergency']
        highlights = []
        
        for transcript in recent_transcriptions:
            if isinstance(transcript, dict) and 'text' in transcript:
                text = transcript['text'].lower()
                for keyword in keywords:
                    if keyword in text:
                        highlights.append(f"'{keyword}' mentioned")
                        break
        
        if highlights:
            return "; ".join(set(highlights))  # Remove duplicates
        else:
            return "No significant audio highlights detected"
            
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        return f"Audio analysis error: {str(e)}"

def vision_analysis_thread():
    """Background thread for periodic vision analysis."""
    global last_vision_analysis, current_insights
    
    while True:
        try:
            current_time = time.time()
            
            if current_time - last_vision_analysis >= VISION_ANALYSIS_INTERVAL:
                print("Running vision analysis...")
                
                # Perform analyses
                subject_analysis = analyze_subjects_with_vision()
                patterns = analyze_patterns()
                audio_highlights = analyze_audio_highlights()
                
                # Extract people activity data for insights
                people_activity_data = extract_people_activity_for_insights()
                security_alerts = extract_security_alerts(subject_analysis, patterns, audio_highlights)
                
                # Update insights
                current_insights.update({
                    "subject_analysis": subject_analysis,
                    "patterns": patterns,
                    "audio_highlights": audio_highlights,
                    "timestamp": current_time
                })
                
                # Save insights to database for stateful tracking
                save_insights_to_database(
                    subject_analysis=subject_analysis,
                    patterns=patterns,
                    audio_highlights=audio_highlights,
                    people_activity=people_activity_data,
                    security_alerts=security_alerts
                )
                
                # Auto-generate personas for frequent visitors
                auto_generate_personas_for_frequent_visitors()
                
                last_vision_analysis = current_time
                print(f"Vision analysis completed and saved: {subject_analysis}")
                
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print(f"Error in vision analysis thread: {e}")
            time.sleep(30)  # Wait longer on error

def extract_people_activity_for_insights():
    """Extract people activity data from the global people_groups for insights."""
    try:
        people_activity = []
        current_time = time.time()
        
        for person_id, person_data in people_groups.items():
            last_seen = person_data.get('last_seen', 0)
            # Only include people seen in the last hour
            if current_time - last_seen <= 3600:
                activity_entry = {
                    'tracked_id': person_id,
                    'description': person_data.get('description', ''),
                    'activity': person_data.get('activity', ''),
                    'last_seen': last_seen,
                    'visit_count': person_data.get('visit_count', 0),
                    'features': person_data.get('features', '')
                }
                people_activity.append(activity_entry)
        
        return people_activity
    except Exception as e:
        print(f"Error extracting people activity: {e}")
        return []

def extract_security_alerts(subject_analysis, patterns, audio_highlights):
    """Extract security-relevant alerts from the analysis."""
    try:
        alerts = []
        
        # Check for security keywords in analysis
        security_keywords = ['suspicious', 'unusual', 'unknown', 'loitering', 'emergency', 'help', 'break', 'theft']
        
        # Check subject analysis
        for keyword in security_keywords:
            if keyword.lower() in subject_analysis.lower():
                alerts.append({
                    'type': 'subject_alert',
                    'keyword': keyword,
                    'description': f"Security keyword '{keyword}' detected in subject analysis",
                    'source': 'subject_analysis'
                })
        
        # Check patterns for unusual activity
        if 'frequent' in patterns.lower() or 'regular' in patterns.lower():
            alerts.append({
                'type': 'pattern_alert',
                'description': "Frequent or regular activity pattern detected",
                'source': 'patterns'
            })
        
        # Check audio highlights
        for keyword in security_keywords:
            if keyword.lower() in audio_highlights.lower():
                alerts.append({
                    'type': 'audio_alert',
                    'keyword': keyword,
                    'description': f"Security keyword '{keyword}' detected in audio",
                    'source': 'audio_highlights'
                })
        
        return alerts
    except Exception as e:
        print(f"Error extracting security alerts: {e}")
        return []

def generate_persona_for_person(person_id, person_data, api_key):
    """Generate a memorable name and persona for a frequently seen person using GPT-4.1-mini."""
    try:
        # Get person's data
        description = person_data.get('description', '')
        features = person_data.get('features', '')
        activity = person_data.get('activity', '')
        visit_count = person_data.get('visit_count', 0)
        images = person_data.get('images', [])
        
        # Only generate personas for people seen multiple times
        if visit_count < 2:
            return None
        
        # Create a prompt for persona generation
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"A person has been detected {visit_count} times at a doorbell camera outside a home in zip code 19123. "
                               f"Physical description: {description}. "
                               f"Features: {features}. "
                               f"Typical activity: {activity}. "
                               f"Based on their appearance and behavior, create a memorable, friendly name and brief persona. "
                               f"Consider common visitor types like delivery drivers, neighbors, maintenance workers, family, etc. "
                               f"Format as JSON: {{\"name\": \"FirstName LastName or Role\", \"persona\": \"Brief 1-2 sentence description of who they likely are\", \"category\": \"neighbor/delivery/family/worker/visitor/unknown\"}}"
                    }
                ]
            }
        ]
        
        # Add sample images if available
        sample_images = images[-2:] if len(images) >= 2 else images  # Last 2 images
        for img_path in sample_images:
            if os.path.exists(img_path):
                base64_image = encode_image_to_base64(img_path)
                if base64_image:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1-mini",
            "messages": messages,
            "max_tokens": 300,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    persona_data = json.loads(content)
                    return persona_data
                except json.JSONDecodeError:
                    print(f"Failed to parse persona JSON for {person_id}: {content}")
        else:
            print(f"Persona generation API Error {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"Error generating persona for {person_id}: {e}")
    
    return None

def auto_generate_personas_for_frequent_visitors():
    """Automatically generate personas for people who have been seen multiple times."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return
        
        personas_generated = 0
        
        for person_id, person_data in people_groups.items():
            visit_count = person_data.get('visit_count', 0)
            
            # Generate persona for people seen 3+ times who don't have one yet
            if visit_count >= 3:
                # Check if persona already exists in database
                has_persona = check_if_persona_exists(person_id)
                
                if not has_persona:
                    print(f"Generating persona for frequent visitor {person_id} (seen {visit_count} times)...")
                    persona_data = generate_persona_for_person(person_id, person_data, api_key)
                    
                    if persona_data:
                        # Update the person's data with persona
                        person_data['assigned_name'] = persona_data.get('name', f'Visitor {person_id}')
                        person_data['persona'] = persona_data.get('persona', 'Regular visitor')
                        person_data['category'] = persona_data.get('category', 'visitor')
                        person_data['persona_generated'] = True
                        
                        print(f"Generated persona: {person_data['assigned_name']} - {person_data['persona']}")
                        personas_generated += 1
        
        if personas_generated > 0:
            # Save updated people data
            save_people_to_database()
            print(f"Generated {personas_generated} new personas for frequent visitors")
    
    except Exception as e:
        print(f"Error in auto persona generation: {e}")

def check_if_persona_exists(person_id):
    """Check if a person already has a generated persona in the database."""
    try:
        if os.path.exists(PEOPLE_DB_FILE):
            df = pd.read_parquet(PEOPLE_DB_FILE)
            person_row = df[df['person_id'] == person_id]
            if not person_row.empty:
                return person_row.iloc[0].get('persona_generated', False)
    except Exception as e:
        print(f"Error checking persona existence for {person_id}: {e}")
    return False

# --- Video Processing Thread ---
def video_processing_thread():
    global last_frame, last_detections

    print("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Stream opened successfully.")
    model = YOLO("yolov8n.mlpackage")
    print("Model loaded.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: Failed to grab frame.")
            time.sleep(1)
            continue

        # Apply fixed settings
        if settings["grayscale"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Use model.track() for integrated tracking
        results = model.track(frame, imgsz=640, verbose=False, conf=settings["confidence_threshold"], persist=True)
        
        boxes = None
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()  # Get boxes and move to CPU for numpy operations
        else:
            boxes = None

        current_detections = []
        if boxes is not None and boxes.id is not None:  # Ensure track IDs exist
            for i in range(len(boxes.xyxy)):
                class_id = int(boxes.cls[i])
                confidence = boxes.conf[i]
                tracker_id = int(boxes.id[i])
                class_name = model.names[class_id]
                bbox = boxes.xyxy[i]

                # Only process security-relevant objects
                if class_name not in SECURITY_OBJECTS:
                    continue

                current_detections.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "track_id": tracker_id
                })

                image_path = save_captured_image(frame, bbox, tracker_id, class_name)

                # Log the detection (now includes image_path)
                log_detection(class_name, float(confidence), tracker_id, image_path)

                # Draw detections with enhanced styling
                x1, y1, x2, y2 = map(int, bbox)
                label = f"{class_name.upper()} #{tracker_id} ({float(confidence):.1%})"
                
                # Choose colors based on object type
                if class_name == 'person':
                    box_color = (0, 255, 255)    # Yellow
                    text_color = (0, 255, 255)
                elif class_name in ['dog', 'cat']:
                    box_color = (255, 165, 0)    # Orange
                    text_color = (255, 165, 0)
                else:
                    box_color = (0, 255, 0)      # Green
                    text_color = (0, 255, 0)
                
                # Draw thicker, more prominent box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                
                # Add corner accents for better visibility
                corner_length = min(30, (x2-x1)//4, (y2-y1)//4)
                # Top-left corner
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, 6)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), box_color, 6)
                # Top-right corner
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), box_color, 6)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), box_color, 6)
                # Bottom-left corner
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), box_color, 6)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), box_color, 6)
                # Bottom-right corner
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), box_color, 6)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, 6)
                
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                bg_y1 = max(0, y1 - text_height - 15)
                bg_y2 = y1 - 5
                cv2.rectangle(frame, (x1, bg_y1), (x1 + text_width + 10, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, bg_y1), (x1 + text_width + 10, bg_y2), text_color, 2)
                
                # Draw text with better font and positioning
                cv2.putText(frame, label, (x1 + 5, y1 - 10), font, font_scale, text_color, thickness)

        # Update shared variables
        with frame_lock:
            _, buffer = cv2.imencode('.jpg', frame)
            last_frame = buffer.tobytes()
            last_detections = current_detections

# --- Audio Processing Thread ---
def audio_processing_thread():
    global last_transcription, cumulative_transcription, audio_files_list
    print("Loading Whisper model...")
    audio_model = whisper.load_model("base")
    print("Whisper model loaded.")

    # Create audio recordings directory
    os.makedirs(AUDIO_RECORDINGS_DIR, exist_ok=True)

    while True:
        process = None
        try:
            current_time = time.time()
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            audio_filename = f"audio_{timestamp_str}.wav"
            audio_filepath = os.path.join(AUDIO_RECORDINGS_DIR, audio_filename)
            
            command = [
                'ffmpeg',
                '-y',
                '-i', RTSP_URL,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-t', str(AUDIO_SEGMENT_DURATION),
                audio_filepath
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
                time.sleep(1) 
                continue

            if os.path.exists(audio_filepath):
                try:
                    result = audio_model.transcribe(audio_filepath)
                    transcribed_text = result["text"].strip()
                    
                    if transcribed_text and len(transcribed_text) > 3:  # Only save meaningful transcriptions
                        with transcription_lock:
                            last_transcription = transcribed_text
                            
                            # Append to cumulative transcription with timestamp
                            timestamp_formatted = time.strftime("%H:%M:%S", time.localtime(current_time))
                            cumulative_transcription += f"[{timestamp_formatted}] {transcribed_text}\n"
                            
                            # Add to transcription history
                            transcription_entry = {
                                'text': transcribed_text,
                                'timestamp': current_time,
                                'audio_file': audio_filename
                            }
                            transcription_history.append(transcription_entry)
                            
                            # Track audio file
                            audio_files_list.append({
                                'filename': audio_filename,
                                'filepath': audio_filepath,
                                'timestamp': current_time,
                                'duration': AUDIO_SEGMENT_DURATION,
                                'has_transcription': True,
                                'transcription': transcribed_text
                            })
                            
                            # Save transcription to log file
                            save_transcription_to_log(transcribed_text, current_time, audio_filename)
                            
                        print(f"Transcription [{timestamp_formatted}]: {transcribed_text}")
                    else:
                        # Delete audio file if no meaningful transcription
                        try:
                            os.remove(audio_filepath)
                            print(f"Deleted audio file with no transcription: {audio_filename}")
                        except OSError as e:
                            print(f"Error deleting audio file {audio_filename}: {e}")
                        
                    # Clean up old audio files if we exceed the limit
                    cleanup_old_audio_files()
                    
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    # Delete audio file if transcription fails
                    try:
                        os.remove(audio_filepath)
                        print(f"Deleted audio file due to transcription error: {audio_filename}")
                    except OSError as delete_error:
                        print(f"Error deleting failed audio file {audio_filename}: {delete_error}")
            else:
                print(f"Warning: {audio_filepath} not found after FFmpeg process.")

        except Exception as e:
            print(f"Error starting FFmpeg process: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait()
        
        # Wait a bit before next recording
        time.sleep(2)

def save_transcription_to_log(text, timestamp, audio_filename):
    """Save transcription to the log file."""
    try:
        timestamp_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        log_entry = f"[{timestamp_formatted}] ({audio_filename}) {text}\n"
        
        with open(TRANSCRIPTION_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error saving transcription to log: {e}")

def cleanup_old_audio_files():
    """Remove old audio files if we exceed the maximum limit."""
    try:
        # Get all audio files in the directory
        audio_files = []
        for filename in os.listdir(AUDIO_RECORDINGS_DIR):
            if filename.endswith('.wav'):
                filepath = os.path.join(AUDIO_RECORDINGS_DIR, filename)
                audio_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        audio_files.sort(key=lambda x: x[1])
        
        # Remove oldest files if we exceed the limit
        while len(audio_files) > MAX_AUDIO_FILES:
            oldest_file = audio_files.pop(0)
            try:
                os.remove(oldest_file[0])
                print(f"Removed old audio file: {oldest_file[0]}")
            except OSError as e:
                print(f"Error removing audio file {oldest_file[0]}: {e}")
                
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

# --- FastAPI Endpoints ---
app.mount("/captured_subjects", StaticFiles(directory=CAPTURED_IMAGES_DIR), name="captured_subjects")
app.mount("/audio_recordings", StaticFiles(directory=AUDIO_RECORDINGS_DIR), name="audio_recordings")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>DoorbellAI - Smart Security Dashboard</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                    background: #f8f9fa;
                    color: #212529;
                    line-height: 1.5;
                    font-size: 14px;
                }
                
                .container {
                    display: grid;
                    grid-template-areas:
                        "header header header"
                        "main main sidebar"
                        "analysis analysis sidebar";
                    grid-template-columns: 1fr 1fr 300px;
                    grid-template-rows: auto 1fr auto;
                    min-height: 100vh;
                    gap: 16px;
                    padding: 16px;
                    max-width: 1400px;
                    margin: 0 auto;
                }
                
                .header {
                    grid-area: header;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 16px;
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                }
                
                .header h1 {
                    font-size: 18px;
                    font-weight: 600;
                    color: #495057;
                }
                
                .status {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                    color: #6c757d;
                }
                
                .status-dot {
                    width: 8px;
                    height: 8px;
                    background: #28a745;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }
                
                .main-content {
                    grid-area: main;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                    align-items: start;
                }
                
                .sidebar {
                    grid-area: sidebar;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                
                .analysis-section {
                    grid-area: analysis;
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                }
                
                .card {
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                }
                
                .card-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 12px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #e9ecef;
                }
                
                .card-title {
                    font-size: 14px;
                    font-weight: 600;
                    color: #495057;
                    margin: 0;
                }
                
                .video-container {
                    position: relative;
                    width: 100%;
                    aspect-ratio: 16/9;
                    background: #000;
                    border-radius: 6px;
                    overflow: hidden;
                }
                
                .video-feed {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 12px;
                }
                
                .stat-item {
                    text-align: center;
                    padding: 8px;
                    background: #f8f9fa;
                    border-radius: 4px;
                    border: 1px solid #e9ecef;
                }
                
                .stat-value {
                    font-size: 18px;
                    font-weight: 600;
                    color: #495057;
                    display: block;
                }
                
                .stat-label {
                    font-size: 12px;
                    color: #6c757d;
                    margin-top: 4px;
                }
                
                .image-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
                    gap: 8px;
                }
                
                .image-item {
                    position: relative;
                    aspect-ratio: 1;
                    border-radius: 4px;
                    overflow: hidden;
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                }
                
                .image-item img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                
                .image-item::after {
                    content: attr(data-label);
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    font-size: 10px;
                    padding: 2px 4px;
                    text-align: center;
                }
                
                .transcription-box {
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 12px;
                    font-size: 13px;
                    line-height: 1.4;
                    min-height: 60px;
                    max-height: 120px;
                    overflow-y: auto;
                }
                
                .btn {
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    padding: 6px 12px;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    background: white;
                    color: #495057;
                    text-decoration: none;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .btn:hover {
                    background: #f8f9fa;
                    border-color: #adb5bd;
                }
                
                .btn-primary {
                    background: #007bff;
                    color: white;
                    border-color: #007bff;
                }
                
                .btn-primary:hover {
                    background: #0056b3;
                    border-color: #0056b3;
                }
                
                .btn-danger {
                    background: #dc3545;
                    color: white;
                    border-color: #dc3545;
                }
                
                .btn-danger:hover {
                    background: #c82333;
                    border-color: #c82333;
                }
                
                .form-group {
                    margin-bottom: 12px;
                }
                
                .form-label {
                    display: block;
                    font-size: 12px;
                    font-weight: 500;
                    color: #495057;
                    margin-bottom: 4px;
                }
                
                .form-control {
                    width: 100%;
                    padding: 6px 8px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    font-size: 13px;
                    background: white;
                    color: #495057;
                }
                
                .form-control:focus {
                    outline: none;
                    border-color: #80bdff;
                    box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
                }
                
                .group-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 12px;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    margin-bottom: 8px;
                    background: #f8f9fa;
                }
                
                .group-info {
                    flex: 1;
                }
                
                .group-name {
                    font-weight: 500;
                    color: #495057;
                    font-size: 13px;
                }
                
                .group-count {
                    font-size: 11px;
                    color: #6c757d;
                }
                
                .analysis-result {
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 12px;
                    margin-bottom: 12px;
                }
                
                .analysis-title {
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 8px;
                    font-size: 13px;
                }
                
                .analysis-text {
                    font-size: 12px;
                    line-height: 1.4;
                    color: #6c757d;
                }
                
                .insights-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 12px;
                    margin-bottom: 16px;
                }
                
                .insight-card {
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 12px;
                }
                
                .insight-title {
                    font-size: 12px;
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 8px;
                }
                
                .insight-content {
                    font-size: 12px;
                    color: #6c757d;
                    line-height: 1.3;
                }
                
                .empty-state {
                    text-align: center;
                    color: #6c757d;
                    font-size: 12px;
                    padding: 24px;
                    font-style: italic;
                }
                
                .activity-content, .people-content {
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .activity-item, .person-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 12px;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    margin-bottom: 8px;
                    background: #f8f9fa;
                    font-size: 12px;
                }
                
                .activity-time, .person-visits {
                    color: #6c757d;
                    font-size: 10px;
                }
                
                .person-name {
                    font-weight: 600;
                    color: #495057;
                }
                
                .audio-section {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .audio-stats {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 8px;
                }
                
                .audio-controls {
                    display: flex;
                    gap: 8px;
                }
                
                .charts-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 16px;
                    margin-top: 16px;
                }
                
                .chart-container {
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                }
                
                .chart-container h3 {
                    margin: 0 0 12px 0;
                    font-size: 14px;
                    font-weight: 600;
                    color: #495057;
                }
                
                .chart-container canvas {
                    width: 100% !important;
                    height: 200px !important;
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
                
                @media (max-width: 768px) {
                    .container {
                        grid-template-areas:
                            "header"
                            "main"
                            "sidebar"
                            "analysis";
                        grid-template-columns: 1fr;
                        gap: 12px;
                        padding: 12px;
                    }
                    
                    .main-content {
                        grid-template-columns: 1fr;
                    }
                    
                    .header h1 {
                        font-size: 16px;
                    }
                    
                    .image-grid {
                        grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
                    }
                    
                    .insights-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>DoorbellAI</h1>
                    <div class="status">
                        <span class="status-dot"></span>
                        <span>Live</span>
                    </div>
                </div>

                <div class="main-content">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Live Feed</h2>
                        </div>
                        <div class="video-container">
                            <img src="/video_feed" alt="Live Feed" class="video-feed">
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Recent Activity</h2>
                        </div>
                        <div id="activityContainer" class="activity-content">
                            <div class="empty-state">Loading recent activity...</div>
                        </div>
                    </div>
                </div>

                <div class="sidebar">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Audio Monitoring</h2>
                            <div id="audioStatus" class="status">
                                <span class="status-dot"></span>
                                <span>Listening</span>
                            </div>
                        </div>
                        <div class="audio-section">
                            <div class="audio-stats">
                                <div class="stat-item">
                                    <span id="audioFileCount" class="stat-value">0</span>
                                    <div class="stat-label">Audio Files</div>
                                </div>
                                <div class="stat-item">
                                    <span id="transcriptionCount" class="stat-value">0</span>
                                    <div class="stat-label">Transcriptions</div>
                                </div>
                            </div>
                            <div id="transcriptionText" class="transcription-box">
                                Listening for audio...
                            </div>
                            <div class="audio-controls">
                                <button id="viewAudioList" class="btn btn-primary">View Audio Files</button>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Recent Captures</h2>
                            <button id="cleanupButton" class="btn btn-danger">Clear</button>
                        </div>
                        <div id="galleryGrid" class="image-grid">
                            <div class="empty-state">No captures yet...</div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">People Tracking</h2>
                        </div>
                        <div id="peopleContainer" class="people-content">
                            <div class="empty-state">Loading people data...</div>
                        </div>
                    </div>
                </div>

                <div class="analysis-section">
                    <div class="card-header">
                        <h2 class="card-title">AI Insights</h2>
                        <button id="refreshInsights" class="btn">Refresh</button>
                    </div>
                    
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-title">Subject Analysis</div>
                            <div id="subjectAnalysis" class="insight-content">Analyzing...</div>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">Patterns</div>
                            <div id="patternsAnalysis" class="insight-content">Analyzing...</div>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">Audio Highlights</div>
                            <div id="audioHighlights" class="insight-content">Analyzing...</div>
                        </div>
                    </div>

                    <div id="analyticsCharts">
                        <div class="charts-grid">
                            <div class="chart-container">
                                <h3>Detection Timeline</h3>
                                <canvas id="timelineChart" width="400" height="200"></canvas>
                            </div>
                            <div class="chart-container">
                                <h3>People Activity</h3>
                                <canvas id="peopleChart" width="400" height="200"></canvas>
                            </div>
                            <div class="chart-container">
                                <h3>Security Objects</h3>
                                <canvas id="objectsChart" width="400" height="200"></canvas>
                            </div>
                            <div class="chart-container">
                                <h3>Hourly Patterns</h3>
                                <canvas id="hourlyChart" width="400" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                // DOM elements
                const transcriptionText = document.getElementById('transcriptionText');
                const galleryGrid = document.getElementById('galleryGrid');
                const activityContainer = document.getElementById('activityContainer');
                const peopleContainer = document.getElementById('peopleContainer');
                const cleanupButton = document.getElementById('cleanupButton');
                const subjectAnalysis = document.getElementById('subjectAnalysis');
                const patternsAnalysis = document.getElementById('patternsAnalysis');
                const audioHighlights = document.getElementById('audioHighlights');
                const audioFileCount = document.getElementById('audioFileCount');
                const transcriptionCount = document.getElementById('transcriptionCount');
                const viewAudioList = document.getElementById('viewAudioList');

                // Object type to emoji mapping
                const objectEmojis = {
                    'person': '👤', 'dog': '🐕', 'cat': '🐱', 'car': '🚗', 'truck': '🚚',
                    'bus': '🚌', 'motorcycle': '🏍️', 'bicycle': '🚲', 'bird': '🐦'
                };

                // Cleanup functionality
                async function cleanupCapturedSubjects() {
                    if (confirm('Delete all captured images?')) {
                        try {
                            cleanupButton.disabled = true;
                            cleanupButton.textContent = 'Clearing...';
                            
                            const response = await fetch('/cleanup', { method: 'POST' });
                            const result = await response.json();
                            
                            if (result.status === 'success') {
                                galleryGrid.innerHTML = '<div class="empty-state">No captures yet...</div>';
                                cleanupButton.textContent = 'Clear';
                                cleanupButton.disabled = false;
                                loadImageGroups();
                            } else {
                                alert('Error: ' + result.message);
                                cleanupButton.textContent = 'Clear';
                                cleanupButton.disabled = false;
                            }
                        } catch (error) {
                            console.error('Error during cleanup:', error);
                            alert('Error during cleanup: ' + error.message);
                            cleanupButton.textContent = 'Clear';
                            cleanupButton.disabled = false;
                        }
                    }
                }

                // Fetch and display recent activity
                async function fetchRecentActivity() {
                    try {
                        const response = await fetch('/insights/hourly?hours_back=6');
                        const data = await response.json();
                        
                        if (!data.hourly_insights || data.hourly_insights.length === 0) {
                            activityContainer.innerHTML = '<div class="empty-state">No recent activity...</div>';
                            return;
                        }
                        
                        let html = '';
                        for (const insight of data.hourly_insights.slice(0, 5)) {
                            const time = new Date(insight.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                            const summary = insight.summary.length > 60 ? insight.summary.substring(0, 60) + '...' : insight.summary;
                            html += `
                                <div class="activity-item">
                                    <div>
                                        <div>${summary}</div>
                                        <div class="activity-time">${time}</div>
                                    </div>
                                </div>
                            `;
                        }
                        activityContainer.innerHTML = html;
                    } catch (error) {
                        console.error('Error fetching recent activity:', error);
                        activityContainer.innerHTML = '<div class="empty-state">Error loading activity</div>';
                    }
                }

                // Fetch and display people tracking
                async function fetchPeopleTracking() {
                    try {
                        const response = await fetch('/people/with-personas');
                        const data = await response.json();
                        
                        if (!data.people || data.people.length === 0) {
                            peopleContainer.innerHTML = '<div class="empty-state">No people tracked yet...</div>';
                            return;
                        }
                        
                        let html = '';
                        for (const person of data.people.slice(0, 5)) {
                            const name = person.assigned_name || person.person_id;
                            const visits = person.visit_count;
                            const lastSeen = new Date(person.last_seen * 1000).toLocaleDateString();
                            html += `
                                <div class="person-item">
                                    <div>
                                        <div class="person-name">${name}</div>
                                        <div class="person-visits">${visits} visits • Last: ${lastSeen}</div>
                                    </div>
                                </div>
                            `;
                        }
                        peopleContainer.innerHTML = html;
                    } catch (error) {
                        console.error('Error fetching people tracking:', error);
                        peopleContainer.innerHTML = '<div class="empty-state">Error loading people data</div>';
                    }
                }

                // Fetch and display transcription with audio stats
                async function fetchTranscription() {
                    try {
                        const response = await fetch('/transcription');
                        const data = await response.json();
                        const text = data.last_transcription;
                        transcriptionText.textContent = text || 'Listening for audio...';
                        
                        // Update cumulative transcription length as proxy for activity
                        const totalLength = data.total_length || 0;
                        transcriptionCount.textContent = Math.floor(totalLength / 100); // Rough count
                        
                    } catch (error) {
                        console.error('Error fetching transcription:', error);
                    }
                }

                // Fetch and display audio statistics
                async function fetchAudioStats() {
                    try {
                        const response = await fetch('/audio/list');
                        const data = await response.json();
                        
                        if (data.audio_files) {
                            audioFileCount.textContent = data.total_files || 0;
                        }
                    } catch (error) {
                        console.error('Error fetching audio stats:', error);
                    }
                }

                // Show audio files list
                async function showAudioList() {
                    try {
                        const response = await fetch('/audio/list');
                        const data = await response.json();
                        
                        if (data.audio_files && data.audio_files.length > 0) {
                            let audioListHtml = '<div style="max-height: 400px; overflow-y: auto; padding: 10px;">';
                            audioListHtml += '<h3>Audio Recordings</h3>';
                            
                            for (const audio of data.audio_files.slice(0, 20)) {
                                const time = audio.formatted_time;
                                const transcription = audio.transcription || 'No transcription';
                                audioListHtml += `
                                    <div style="margin-bottom: 10px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                                        <div><strong>${time}</strong></div>
                                        <div style="font-size: 12px; color: #666;">${transcription}</div>
                                        <audio controls style="width: 100%; margin-top: 5px;">
                                            <source src="${audio.url}" type="audio/wav">
                                            Your browser does not support audio playback.
                                        </audio>
                                    </div>
                                `;
                            }
                            audioListHtml += '</div>';
                            
                            // Create a modal-like display
                            const modal = document.createElement('div');
                            modal.style.cssText = `
                                position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                                background: rgba(0,0,0,0.8); z-index: 1000; display: flex; 
                                align-items: center; justify-content: center;
                            `;
                            
                            const content = document.createElement('div');
                            content.style.cssText = `
                                background: white; border-radius: 8px; max-width: 80%; 
                                max-height: 80%; position: relative;
                            `;
                            
                            content.innerHTML = audioListHtml + 
                                '<button onclick="this.closest(\'.modal\').remove()" style="position: absolute; top: 10px; right: 10px; background: #dc3545; color: white; border: none; border-radius: 4px; padding: 5px 10px; cursor: pointer;">Close</button>';
                            
                            modal.className = 'modal';
                            modal.appendChild(content);
                            document.body.appendChild(modal);
                            
                            // Close on background click
                            modal.onclick = (e) => {
                                if (e.target === modal) modal.remove();
                            };
                        } else {
                            alert('No audio recordings available');
                        }
                    } catch (error) {
                        console.error('Error fetching audio list:', error);
                        alert('Error loading audio files');
                    }
                }

                // Fetch and display gallery
                async function fetchGallery() {
                    try {
                        const response = await fetch('/gallery');
                        const images = await response.json();
                        
                        if (images.length === 0) {
                            galleryGrid.innerHTML = '<div class="empty-state">No captures yet...</div>';
                            return;
                        }
                        
                        galleryGrid.innerHTML = '';
                        images.slice(0, 12).forEach(img => {
                            const div = document.createElement('div');
                            div.className = 'image-item';
                            div.setAttribute('data-label', img.class_name);
                            
                            const imgElement = document.createElement('img');
                            imgElement.src = `/captured_subjects/${img.filename}`;
                            imgElement.alt = `${img.class_name} (ID: ${img.track_id})`;
                            
                            div.appendChild(imgElement);
                            galleryGrid.appendChild(div);
                        });
                    } catch (error) {
                        console.error('Error fetching gallery:', error);
                    }
                }

                // Fetch and display insights
                async function fetchInsightsData() {
                    try {
                        const response = await fetch('/insights');
                        const data = await response.json();
                        
                        subjectAnalysis.textContent = data.subject_analysis || 'No analysis available';
                        patternsAnalysis.textContent = data.patterns || 'No patterns detected';
                        audioHighlights.textContent = data.audio_highlights || 'No audio highlights';
                    } catch (error) {
                        console.error('Error fetching insights:', error);
                        subjectAnalysis.textContent = 'Error loading analysis';
                        patternsAnalysis.textContent = 'Error loading patterns';
                        audioHighlights.textContent = 'Error loading highlights';
                    }
                }

                // Chart variables
                let timelineChart, peopleChart, objectsChart, hourlyChart;

                // Initialize charts
                function initializeCharts() {
                    const chartOptions = {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    fontSize: 10,
                                    boxWidth: 12
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: '#f0f0f0'
                                },
                                ticks: {
                                    fontSize: 10
                                }
                            },
                            x: {
                                grid: {
                                    color: '#f0f0f0'
                                },
                                ticks: {
                                    fontSize: 10,
                                    maxRotation: 45
                                }
                            }
                        }
                    };

                    // Detection Timeline Chart
                    timelineChart = new Chart(document.getElementById('timelineChart'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Detections per Hour',
                                data: [],
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                borderWidth: 2,
                                fill: true
                            }]
                        },
                        options: chartOptions
                    });

                    // People Activity Chart
                    peopleChart = new Chart(document.getElementById('peopleChart'), {
                        type: 'doughnut',
                        data: {
                            labels: [],
                            datasets: [{
                                data: [],
                                backgroundColor: ['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        fontSize: 10,
                                        boxWidth: 12
                                    }
                                }
                            }
                        }
                    });

                    // Security Objects Chart
                    objectsChart = new Chart(document.getElementById('objectsChart'), {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Object Detections',
                                data: [],
                                backgroundColor: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1']
                            }]
                        },
                        options: chartOptions
                    });

                    // Hourly Patterns Chart
                    hourlyChart = new Chart(document.getElementById('hourlyChart'), {
                        type: 'bar',
                        data: {
                            labels: ['12 AM', '2 AM', '4 AM', '6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM'],
                            datasets: [{
                                label: 'Activity Level',
                                data: [],
                                backgroundColor: 'rgba(0, 123, 255, 0.6)',
                                borderColor: '#007bff',
                                borderWidth: 1
                            }]
                        },
                        options: chartOptions
                    });
                }

                // Update charts with real data
                async function updateCharts() {
                    try {
                        // Fetch analytics data
                        const response = await fetch('/analytics');
                        const data = await response.json();

                        // Update Timeline Chart
                        if (data.hourly_summaries && data.hourly_summaries.length > 0) {
                            const last24Hours = data.hourly_summaries.slice(0, 24);
                            const timeLabels = last24Hours.map(h => {
                                const date = new Date(h.timestamp * 1000);
                                return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                            }).reverse();
                            const timeData = last24Hours.map(h => h.total_analyses || 0).reverse();
                            
                            timelineChart.data.labels = timeLabels;
                            timelineChart.data.datasets[0].data = timeData;
                            timelineChart.update();
                        }

                        // Update People Chart
                        if (data.people_database && data.people_database.people) {
                            const peopleData = data.people_database.people.slice(0, 6);
                            const peopleLabels = peopleData.map(p => p.assigned_name || p.person_id);
                            const peopleValues = peopleData.map(p => p.visit_count || 0);
                            
                            peopleChart.data.labels = peopleLabels;
                            peopleChart.data.datasets[0].data = peopleValues;
                            peopleChart.update();
                        }

                        // Update Objects Chart
                        if (data.detection_summary) {
                            const objectLabels = Object.keys(data.detection_summary);
                            const objectValues = Object.values(data.detection_summary);
                            
                            objectsChart.data.labels = objectLabels;
                            objectsChart.data.datasets[0].data = objectValues;
                            objectsChart.update();
                        }

                        // Update Hourly Patterns Chart
                        if (data.hourly_stats) {
                            const hourlyData = new Array(12).fill(0);
                            // Aggregate data by 2-hour blocks
                            Object.entries(data.hourly_stats).forEach(([hour, stats]) => {
                                const hourNum = parseInt(hour);
                                const blockIndex = Math.floor(hourNum / 2);
                                if (blockIndex < 12) {
                                    hourlyData[blockIndex] += Object.values(stats).reduce((a, b) => a + b, 0);
                                }
                            });
                            
                            hourlyChart.data.datasets[0].data = hourlyData;
                            hourlyChart.update();
                        }

                    } catch (error) {
                        console.error('Error updating charts:', error);
                    }
                }

                // Event listeners
                cleanupButton.addEventListener('click', cleanupCapturedSubjects);
                viewAudioList.addEventListener('click', showAudioList);

                // Initialize
                initializeCharts();
                fetchRecentActivity();
                fetchPeopleTracking();
                fetchTranscription();
                fetchAudioStats();
                fetchGallery();
                fetchInsightsData();
                updateCharts();

                // Set up intervals
                setInterval(fetchRecentActivity, 30000);  // Every 30 seconds
                setInterval(fetchPeopleTracking, 60000);  // Every minute
                setInterval(fetchTranscription, 3000);
                setInterval(fetchAudioStats, 10000);
                setInterval(fetchGallery, 5000);
                setInterval(fetchInsightsData, 10000);
                setInterval(updateCharts, 30000);  // Update charts every 30 seconds
            </script>
        </body>
    </html>
    """

@app.post("/settings")
async def update_settings(confidence: float = Form(...), grayscale: bool = Form(False)):
    global settings
    # The form sends 'on' for a checked box, so we checks for its existence.
    settings["confidence_threshold"] = confidence
    settings["grayscale"] = grayscale
    return {"status": "success", "settings": settings}


async def generate_video_stream():
    while True:
        with frame_lock:
            if last_frame is None:
                await asyncio.sleep(0.1)
                continue
            frame_bytes = last_frame
        
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        await asyncio.sleep(1/30)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def events():
    with frame_lock:
        # Count all detected object types
        counts = {}
        for detection in last_detections:
            class_name = detection["class"]
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return {
            "detections": last_detections,
            "timestamp": time.time(),
            "counts": counts
        }

@app.get("/transcription")
def get_transcription():
    global last_transcription, cumulative_transcription
    with transcription_lock:
        return {
            "last_transcription": last_transcription,
            "cumulative_transcription": cumulative_transcription,
            "total_length": len(cumulative_transcription),
            "has_content": bool(cumulative_transcription.strip())
        }

@app.get("/gallery")
async def get_gallery_images():
    images_data = []
    
    # Iterate through subject directories
    for class_name in os.listdir(CAPTURED_IMAGES_DIR):
        subject_dir = os.path.join(CAPTURED_IMAGES_DIR, class_name)
        if os.path.isdir(subject_dir):
            for filename in os.listdir(subject_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        track_id = parts[0]
                        timestamp_str = parts[1].split('.')[0] # Remove .jpg
                        # Reconstruct timestamp from filename for sorting/display
                        try:
                            timestamp = time.mktime(time.strptime(timestamp_str, "%Y%m%d_%H%M%S"))
                        except ValueError:
                            timestamp = os.path.getmtime(os.path.join(subject_dir, filename))

                        images_data.append({
                            "filename": f"{class_name}/{filename}",
                            "class_name": class_name,
                            "track_id": track_id,
                            "timestamp": timestamp
                        })
    
    # Sort by timestamp, newest first
    images_data.sort(key=lambda x: x["timestamp"], reverse=True)
    return images_data

@app.post("/cleanup")
async def cleanup_captured_subjects():
    """Delete all captured subject images and clear tracking data."""
    try:
        import shutil
        
        # Remove all captured subjects
        if os.path.exists(CAPTURED_IMAGES_DIR):
            shutil.rmtree(CAPTURED_IMAGES_DIR)
            
        # Recreate the directory
        os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
        
        # Clear tracking data
        global last_image_capture_time, last_positions, subject_patterns
        last_image_capture_time.clear()
        last_positions.clear()
        subject_patterns.clear()
        
        return {"status": "success", "message": "All captured subjects have been cleared"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to cleanup: {str(e)}"}

@app.get("/insights")
async def get_insights():
    """Get LLM vision analysis insights."""
    global current_insights
    
    # Add timestamp formatting
    if current_insights["timestamp"]:
        formatted_time = datetime.fromtimestamp(current_insights["timestamp"]).strftime("%H:%M:%S")
    else:
        formatted_time = "Never"
    
    return {
        "subject_analysis": current_insights["subject_analysis"],
        "patterns": current_insights["patterns"],
        "audio_highlights": current_insights["audio_highlights"],
        "last_updated": formatted_time,
        "timestamp": current_insights["timestamp"]
    }

@app.get("/analytics")
async def get_analytics():
    """Get detailed analytics data for dashboard."""
    try:
        analytics = {
            "detection_summary": {},
            "hourly_stats": {},
            "recent_detections": [],
            "transcription_summary": {},
            "hourly_summaries": [],
            "people_database": {},
            "activity_timeline": []
        }
        
        # Read detection logs
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_parquet(LOG_FILE)
                if not df.empty:
                    # Detection summary
                    analytics["detection_summary"] = df["class"].value_counts().to_dict()
                    
                    # Hourly stats
                    df["hour"] = pd.to_datetime(df["timestamp"], unit='s').dt.hour
                    hourly_counts = df.groupby(["hour", "class"]).size().unstack(fill_value=0)
                    analytics["hourly_stats"] = hourly_counts.to_dict()
                    
                    # Recent detections (last 100)
                    recent = df.tail(100).to_dict('records')
                    for record in recent:
                        record["formatted_time"] = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    analytics["recent_detections"] = recent
                    
            except Exception as e:
                print(f"Error reading detection logs: {e}")
        
        # Transcription summary
        if transcription_history:
            word_counts = {}
            for trans in transcription_history:
                if isinstance(trans, dict) and 'text' in trans:
                    words = trans['text'].lower().split()
                    for word in words:
                        word = word.strip('.,!?;:')
                        if len(word) > 3:  # Only count meaningful words
                            word_counts[word] = word_counts.get(word, 0) + 1
            
            # Top 10 most common words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            analytics["transcription_summary"] = dict(sorted_words)
        
        # Add hourly summaries
        analytics["hourly_summaries"] = get_hourly_summaries(24)
        
        # Add people database information
        if os.path.exists(PEOPLE_DB_FILE):
            try:
                people_df = pd.read_parquet(PEOPLE_DB_FILE)
                if not people_df.empty:
                    analytics["people_database"] = {
                        "total_people": len(people_df),
                        "people": people_df.to_dict('records')
                    }
            except Exception as e:
                print(f"Error reading people database: {e}")
        
        # Add activity timeline (combines detection and analysis data)
        analytics["activity_timeline"] = get_activity_timeline()
        
        return analytics
        
    except Exception as e:
        return {"error": f"Analytics error: {str(e)}"}

@app.get("/export/csv")
async def export_detections_csv():
    """Export detection data as CSV."""
    try:
        if not os.path.exists(LOG_FILE):
            raise HTTPException(status_code=404, detail="No detection data available")
            
        df = pd.read_parquet(LOG_FILE)
        if df.empty:
            raise HTTPException(status_code=404, detail="No detection data available")
        
        # Add formatted timestamp
        df["formatted_time"] = pd.to_datetime(df["timestamp"], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create CSV content
        csv_content = df.to_csv(index=False)
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=detections.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/data/detections")
async def get_detection_data(request: Request):
    """Get paginated detection data for data table."""
    try:
        page = int(request.query_params.get("page", 1))
        per_page = int(request.query_params.get("per_page", 50))
        sort_by = request.query_params.get("sort_by", "timestamp")
        sort_order = request.query_params.get("sort_order", "desc")
        
        if not os.path.exists(LOG_FILE):
            return {"data": [], "total": 0, "page": page, "per_page": per_page}
            
        df = pd.read_parquet(LOG_FILE)
        if df.empty:
            return {"data": [], "total": 0, "page": page, "per_page": per_page}
        
        # Add formatted timestamp
        df["formatted_time"] = pd.to_datetime(df["timestamp"], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort data
        ascending = sort_order == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Paginate
        total = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = df.iloc[start_idx:end_idx]
        
        return {
            "data": page_data.to_dict('records'),
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
        
    except Exception as e:
        return {"error": f"Data retrieval error: {str(e)}", "data": [], "total": 0}

@app.get("/api-key-default")
async def get_default_api_key():
    """Get default API key from environment variable."""
    return {"api_key": os.getenv("OPENAI_API_KEY", "")}

# --- Image Grouping and LLM Analysis ---
class ImageGroupAnalyzer:
    def __init__(self, images_dir: str = CAPTURED_IMAGES_DIR):
        self.images_dir = Path(images_dir)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def get_image_groups(self) -> Dict[str, List[str]]:
        """Group images by their subdirectory (category)."""
        groups = {}
        
        if not self.images_dir.exists():
            return groups
            
        for category_dir in self.images_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                images = []
                
                for ext in self.supported_formats:
                    pattern = str(category_dir / f"*{ext}")
                    images.extend(glob.glob(pattern, recursive=False))
                
                if images:
                    groups[category] = sorted(images, key=lambda x: os.path.getmtime(x), reverse=True)
                    
        return groups
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for API transmission."""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return None
    
    def analyze_group_with_openai(self, category: str, image_paths: List[str], 
                                 api_key: str, max_images: int = 5) -> Dict:
        """Send grouped images to OpenAI's vision API for analysis."""
        selected_images = image_paths[:max_images]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze these {category} images captured by a doorbell camera outside a home in zip code 19123. "
                               f"Describe what you see, any patterns, notable features, "
                               f"and provide insights about the detections. "
                               f"Focus on security-relevant observations and consider the residential context."
                    }
                ]
            }
        ]
        
        # Add images to the message
        for img_path in selected_images:
            base64_image = self.encode_image_to_base64(img_path)
            if base64_image:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1-mini",
            "messages": messages,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"OpenAI API Error {response.status_code}: {response.text}")
                return {"error": f"API Error {response.status_code}"}
                
        except Exception as e:
            print(f"OpenAI Request failed: {e}")
            return {"error": str(e)}

@app.get("/image-groups")
async def get_image_groups():
    """Get grouped images by category."""
    try:
        analyzer = ImageGroupAnalyzer()
        groups = analyzer.get_image_groups()
        
        # Format the response for the UI
        formatted_groups = {}
        for category, images in groups.items():
            formatted_groups[category] = {
                "count": len(images),
                "images": [
                    {
                        "path": img.replace(CAPTURED_IMAGES_DIR + "/", ""),
                        "filename": os.path.basename(img),
                        "timestamp": os.path.getmtime(img)
                    } for img in images
                ]
            }
        
        return {"groups": formatted_groups, "total_categories": len(groups)}
    
    except Exception as e:
        return {"error": f"Error grouping images: {str(e)}", "groups": {}}

@app.post("/analyze-group")
async def analyze_image_group(request: Request):
    """Analyze a group of images with LLM."""
    try:
        data = await request.json()
        category = data.get("category")
        api_key = data.get("api_key")
        provider = data.get("provider", "openai")
        max_images = data.get("max_images", 5)
        
        if not category or not api_key:
            raise HTTPException(status_code=400, detail="Category and API key are required")
        
        analyzer = ImageGroupAnalyzer()
        groups = analyzer.get_image_groups()
        
        if category not in groups:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
        
        image_paths = groups[category]
        
        if provider.lower() == "openai":
            result = analyzer.analyze_group_with_openai(category, image_paths, api_key, max_images)
        else:
            raise HTTPException(status_code=400, detail="Only OpenAI provider is currently supported")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract the analysis text
        analysis_text = ""
        if "choices" in result and len(result["choices"]) > 0:
            analysis_text = result["choices"][0]["message"]["content"]
        
        return {
            "category": category,
            "images_analyzed": min(len(image_paths), max_images),
            "total_images": len(image_paths),
            "analysis": analysis_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze-all-groups")
async def analyze_all_image_groups(request: Request):
    """Analyze all image groups with LLM."""
    try:
        data = await request.json()
        api_key = data.get("api_key")
        provider = data.get("provider", "openai")
        max_images = data.get("max_images", 5)
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        analyzer = ImageGroupAnalyzer()
        groups = analyzer.get_image_groups()
        
        if not groups:
            return {"analyses": {}, "summary": "No image groups found"}
        
        analyses = {}
        for category, image_paths in groups.items():
            if provider.lower() == "openai":
                result = analyzer.analyze_group_with_openai(category, image_paths, api_key, max_images)
            else:
                result = {"error": "Only OpenAI provider is currently supported"}
            
            if "error" not in result and "choices" in result and len(result["choices"]) > 0:
                analyses[category] = {
                    "analysis": result["choices"][0]["message"]["content"],
                    "images_analyzed": min(len(image_paths), max_images),
                    "total_images": len(image_paths)
                }
            else:
                analyses[category] = {
                    "error": result.get("error", "Unknown error"),
                    "images_analyzed": 0,
                    "total_images": len(image_paths)
                }
        
        # Generate summary
        total_categories = len(groups)
        total_images = sum(len(images) for images in groups.values())
        successful_analyses = sum(1 for analysis in analyses.values() if "error" not in analysis)
        
        summary = f"Analyzed {successful_analyses}/{total_categories} categories with {total_images} total images"
        
        return {
            "analyses": analyses,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "total_categories": total_categories,
            "total_images": total_images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/insights/hourly")
async def get_hourly_insights(hours_back: int = 24):
    """Get hour-by-hour insights from the database."""
    try:
        if not os.path.exists(INSIGHTS_DB_FILE):
            return {"hourly_insights": [], "total_hours": 0}
        
        df = pd.read_parquet(INSIGHTS_DB_FILE)
        if df.empty:
            return {"hourly_insights": [], "total_hours": 0}
        
        # Filter for hourly insights in the specified time range
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        hourly_df = df[
            (df['insight_type'] == 'hourly') & 
            (df['timestamp'] >= cutoff_time)
        ].sort_values('timestamp', ascending=False)
        
        hourly_insights = []
        for _, row in hourly_df.iterrows():
            insight = {
                "date_hour": row['date_hour'],
                "timestamp": row['timestamp'],
                "subject_analysis": row['subject_analysis'],
                "patterns": row['patterns'],
                "audio_highlights": row['audio_highlights'],
                "people_activity": json.loads(row['people_activity']) if row['people_activity'] else [],
                "security_alerts": json.loads(row['security_alerts']) if row['security_alerts'] else [],
                "summary": row['summary']
            }
            hourly_insights.append(insight)
        
        return {
            "hourly_insights": hourly_insights,
            "total_hours": len(hourly_insights),
            "time_range_hours": hours_back
        }
        
    except Exception as e:
        return {"error": f"Error retrieving hourly insights: {str(e)}", "hourly_insights": []}

@app.get("/insights/people")
async def get_people_insights(hours_back: int = 24):
    """Get person-by-person insights from the database."""
    try:
        if not os.path.exists(INSIGHTS_DB_FILE):
            return {"people_insights": [], "total_people": 0}
        
        df = pd.read_parquet(INSIGHTS_DB_FILE)
        if df.empty:
            return {"people_insights": [], "total_people": 0}
        
        # Filter for person-specific insights in the specified time range
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        people_df = df[
            (df['insight_type'] == 'person_specific') & 
            (df['timestamp'] >= cutoff_time)
        ].sort_values(['person_id', 'timestamp'], ascending=[True, False])
        
        # Group by person
        people_insights = {}
        for _, row in people_df.iterrows():
            person_id = row['person_id']
            if person_id not in people_insights:
                people_insights[person_id] = {
                    "person_id": person_id,
                    "activities": [],
                    "total_sightings": 0,
                    "first_seen": None,
                    "last_seen": None
                }
            
            activity = {
                "date_hour": row['date_hour'],
                "timestamp": row['timestamp'],
                "description": row['subject_analysis'],
                "activity": row['patterns'],
                "audio_context": row['audio_highlights'],
                "summary": row['summary']
            }
            
            people_insights[person_id]["activities"].append(activity)
            people_insights[person_id]["total_sightings"] += 1
            
            # Update first and last seen
            if people_insights[person_id]["first_seen"] is None or row['timestamp'] < people_insights[person_id]["first_seen"]:
                people_insights[person_id]["first_seen"] = row['timestamp']
            if people_insights[person_id]["last_seen"] is None or row['timestamp'] > people_insights[person_id]["last_seen"]:
                people_insights[person_id]["last_seen"] = row['timestamp']
        
        # Convert to list and add formatted times
        people_list = []
        for person_data in people_insights.values():
            if person_data["first_seen"]:
                person_data["first_seen_formatted"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(person_data["first_seen"]))
            if person_data["last_seen"]:
                person_data["last_seen_formatted"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(person_data["last_seen"]))
            people_list.append(person_data)
        
        return {
            "people_insights": people_list,
            "total_people": len(people_list),
            "time_range_hours": hours_back
        }
        
    except Exception as e:
        return {"error": f"Error retrieving people insights: {str(e)}", "people_insights": []}

@app.get("/audio/list")
async def get_audio_files():
    """Get list of available audio recordings."""
    try:
        audio_list = []
        
        # Convert deque to list for JSON serialization
        for audio_info in list(audio_files_list):
            audio_data = {
                "filename": audio_info["filename"],
                "timestamp": audio_info["timestamp"],
                "duration": audio_info["duration"],
                "has_transcription": audio_info["has_transcription"],
                "transcription": audio_info["transcription"],
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(audio_info["timestamp"])),
                "url": f"/audio_recordings/{audio_info['filename']}"
            }
            audio_list.append(audio_data)
        
        # Sort by timestamp (newest first)
        audio_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "audio_files": audio_list,
            "total_files": len(audio_list)
        }
        
    except Exception as e:
        return {"error": f"Error retrieving audio files: {str(e)}", "audio_files": []}

@app.post("/people/generate-personas")
async def generate_personas_for_all():
    """Manually trigger persona generation for all frequent visitors."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key not configured", "personas_generated": 0}
        
        personas_generated = 0
        results = []
        
        for person_id, person_data in people_groups.items():
            visit_count = person_data.get('visit_count', 0)
            
            # Generate persona for people seen 2+ times
            if visit_count >= 2:
                persona_data = generate_persona_for_person(person_id, person_data, api_key)
                
                if persona_data:
                    # Update the person's data with persona
                    person_data['assigned_name'] = persona_data.get('name', f'Visitor {person_id}')
                    person_data['persona'] = persona_data.get('persona', 'Regular visitor')
                    person_data['category'] = persona_data.get('category', 'visitor')
                    person_data['persona_generated'] = True
                    
                    results.append({
                        "person_id": person_id,
                        "assigned_name": person_data['assigned_name'],
                        "persona": person_data['persona'],
                        "category": person_data['category'],
                        "visit_count": visit_count
                    })
                    personas_generated += 1
        
        if personas_generated > 0:
            save_people_to_database()
        
        return {
            "personas_generated": personas_generated,
            "results": results,
            "message": f"Generated {personas_generated} personas for frequent visitors"
        }
        
    except Exception as e:
        return {"error": f"Error generating personas: {str(e)}", "personas_generated": 0}

@app.get("/people/with-personas")
async def get_people_with_personas():
    """Get all tracked people with their assigned names and personas."""
    try:
        people_with_personas = []
        
        # Get from current memory
        for person_id, person_data in people_groups.items():
            person_info = {
                "person_id": person_id,
                "assigned_name": person_data.get('assigned_name', f'Person {person_id}'),
                "persona": person_data.get('persona', 'No persona generated yet'),
                "category": person_data.get('category', 'unknown'),
                "visit_count": person_data.get('visit_count', 0),
                "last_seen": person_data.get('last_seen', 0),
                "last_seen_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(person_data.get('last_seen', 0))),
                "persona_generated": person_data.get('persona_generated', False),
                "description": person_data.get('description', ''),
                "activity": person_data.get('activity', '')
            }
            people_with_personas.append(person_info)
        
        # Also load from database if available
        if os.path.exists(PEOPLE_DB_FILE):
            try:
                df = pd.read_parquet(PEOPLE_DB_FILE)
                db_people = []
                for _, row in df.iterrows():
                    db_person = {
                        "person_id": row['person_id'],
                        "assigned_name": row.get('assigned_name', f"Person {row['person_id']}"),
                        "persona": row.get('persona', 'No persona generated yet'),
                        "visit_count": row.get('visit_count', 0),
                        "last_seen": row.get('last_seen', 0),
                        "last_seen_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row.get('last_seen', 0))),
                        "persona_generated": row.get('persona_generated', False),
                        "description": row.get('description', ''),
                        "from_database": True
                    }
                    db_people.append(db_person)
                
                # Merge with current memory data
                memory_ids = {p['person_id'] for p in people_with_personas}
                for db_person in db_people:
                    if db_person['person_id'] not in memory_ids:
                        people_with_personas.append(db_person)
                        
            except Exception as e:
                print(f"Error loading people from database: {e}")
        
        # Sort by visit count (most frequent first)
        people_with_personas.sort(key=lambda x: x['visit_count'], reverse=True)
        
        return {
            "people": people_with_personas,
            "total_people": len(people_with_personas),
            "with_personas": len([p for p in people_with_personas if p['persona_generated']]),
            "frequent_visitors": len([p for p in people_with_personas if p['visit_count'] >= 3])
        }
        
    except Exception as e:
        return {"error": f"Error retrieving people with personas: {str(e)}", "people": []}


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn

    setup_log_file()
    setup_analysis_log_file()
    setup_people_database()
    setup_insights_database()

    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    audio_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    audio_thread.start()
    vision_thread = threading.Thread(target=vision_analysis_thread, daemon=True)
    vision_thread.start()

    print("Starting FastAPI server with LLM vision analysis. Open http://localhost:8000 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)