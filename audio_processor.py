import whisper
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
from collections import deque

from config import config

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio recording and transcription for correlating with person detections"""
    
    def __init__(self):
        self.whisper_model = None
        self.audio_buffer = deque(maxlen=100)  # Store recent audio segments
        self.transcription_cache = {}
        self.is_recording = False
        self.load_whisper_model()
    
    def load_whisper_model(self):
        """Load Whisper model for audio transcription"""
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
    
    def start_recording(self):
        """Start continuous audio recording and transcription"""
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return False
        
        self.is_recording = True
        recording_thread = threading.Thread(target=self._recording_loop)
        recording_thread.daemon = True
        recording_thread.start()
        logger.info("Audio recording started")
        return True
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        logger.info("Audio recording stopped")
    
    def _recording_loop(self):
        """Main audio recording loop"""
        while self.is_recording:
            try:
                # Record audio segment
                audio_data = self._record_audio_segment()
                if audio_data is not None:
                    # Transcribe audio
                    transcript = self._transcribe_audio(audio_data)
                    if transcript:
                        timestamp = datetime.now()
                        self.audio_buffer.append({
                            "timestamp": timestamp,
                            "transcript": transcript,
                            "duration": 10  # seconds
                        })
                        logger.info(f"Audio transcribed: {transcript[:50]}...")
                
                time.sleep(5)  # Wait before next recording
                
            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
                time.sleep(1)
    
    def _record_audio_segment(self) -> Optional[np.ndarray]:
        """Record a 10-second audio segment"""
        try:
            import pyaudio
            import wave
            
            # Audio recording parameters
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            RECORD_SECONDS = 10
            
            # Initialize pyaudio
            audio = pyaudio.PyAudio()
            
            # Open audio stream
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Record audio
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            return None
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        try:
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_data)
            transcript = result["text"].strip()
            
            # Filter out empty or very short transcripts
            if len(transcript) < 10:
                return None
                
            return transcript
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return None
    
    def get_recent_audio_for_person(self, person_timestamp: datetime, window_seconds: int = 60) -> List[Dict]:
        """Get audio transcripts that occurred around the time a person was detected"""
        results = []
        
        # Search for audio within the time window
        for audio_entry in self.audio_buffer:
            time_diff = abs((audio_entry["timestamp"] - person_timestamp).total_seconds())
            if time_diff <= window_seconds:
                results.append({
                    "timestamp": audio_entry["timestamp"],
                    "transcript": audio_entry["transcript"],
                    "time_difference": time_diff
                })
        
        # Sort by time difference
        results.sort(key=lambda x: x["time_difference"])
        return results
    
    def get_audio_summary_for_profile(self, profile) -> Dict:
        """Get a summary of audio associated with a person profile"""
        all_audio = []
        
        # Collect audio for all detections in the profile
        for detection in profile.detections:
            audio_entries = self.get_recent_audio_for_person(detection.timestamp)
            all_audio.extend(audio_entries)
        
        if not all_audio:
            return {"summary": "No audio recorded", "transcript_count": 0}
        
        # Remove duplicates and sort
        unique_audio = {}
        for entry in all_audio:
            key = entry["transcript"]
            if key not in unique_audio or entry["time_difference"] < unique_audio[key]["time_difference"]:
                unique_audio[key] = entry
        
        sorted_audio = sorted(unique_audio.values(), key=lambda x: x["timestamp"])
        
        # Generate summary
        transcript_texts = [entry["transcript"] for entry in sorted_audio]
        combined_transcript = " ".join(transcript_texts)
        
        return {
            "summary": combined_transcript[:500] + "..." if len(combined_transcript) > 500 else combined_transcript,
            "transcript_count": len(sorted_audio),
            "first_audio": sorted_audio[0]["timestamp"].isoformat() if sorted_audio else None,
            "last_audio": sorted_audio[-1]["timestamp"].isoformat() if sorted_audio else None,
            "full_transcripts": transcript_texts
        }
    
    def cleanup_old_audio(self, hours_old: int = 24):
        """Remove audio entries older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        # Filter out old entries
        recent_audio = deque(maxlen=100)
        for entry in self.audio_buffer:
            if entry["timestamp"] > cutoff_time:
                recent_audio.append(entry)
        
        self.audio_buffer = recent_audio
        logger.info(f"Cleaned up old audio entries, kept {len(self.audio_buffer)} recent entries")