import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import os

from models import PersonDetection, PersonProfile
from config import config

logger = logging.getLogger(__name__)

@dataclass
class DailyPersonEntry:
    """Single person entry in daily log"""
    person_id: str
    first_seen: datetime
    last_seen: datetime
    total_detections: int
    confidence_scores: List[float]
    images: List[str]
    ai_descriptions: List[str]
    behavioral_notes: str
    security_assessment: str
    patterns_detected: List[str]
    interactions: List[str]
    estimated_purpose: str
    risk_level: str

@dataclass 
class DailyLog:
    """Complete daily log of all people detected"""
    date: date
    total_people: int
    total_detections: int
    peak_activity_hours: List[int]
    people_entries: List[DailyPersonEntry]
    daily_summary: str
    security_alerts: List[str]
    patterns_identified: List[str]
    intelligence_insights: List[str]

class DailyLogger:
    """Handles comprehensive daily logging and intelligence analysis"""
    
    def __init__(self):
        self.logs_dir = os.path.join(config.DATABASE_DIR, "daily_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.current_log: Dict[str, DailyPersonEntry] = {}
        
    def log_person_detection(self, profile: PersonProfile, detection: PersonDetection, 
                           ai_analysis: Dict[str, Any]):
        """Log a person detection with comprehensive analysis"""
        today = date.today()
        person_key = f"{profile.id}_{today}"
        
        if person_key not in self.current_log:
            # Create new daily entry for this person
            self.current_log[person_key] = DailyPersonEntry(
                person_id=profile.id,
                first_seen=detection.timestamp,
                last_seen=detection.timestamp,
                total_detections=1,
                confidence_scores=[detection.confidence],
                images=[detection.image_path],
                ai_descriptions=[ai_analysis.get('description', '')],
                behavioral_notes="",
                security_assessment="",
                patterns_detected=[],
                interactions=[],
                estimated_purpose="",
                risk_level="low"
            )
        else:
            # Update existing entry
            entry = self.current_log[person_key]
            entry.last_seen = detection.timestamp
            entry.total_detections += 1
            entry.confidence_scores.append(detection.confidence)
            entry.images.append(detection.image_path)
            if ai_analysis.get('description'):
                entry.ai_descriptions.append(ai_analysis['description'])
    
    def add_behavioral_analysis(self, person_id: str, analysis: Dict[str, Any]):
        """Add behavioral analysis to person's daily entry"""
        today = date.today()
        person_key = f"{person_id}_{today}"
        
        if person_key in self.current_log:
            entry = self.current_log[person_key]
            entry.behavioral_notes = analysis.get('behavioral_patterns', '')
            entry.security_assessment = analysis.get('security_assessment', '')
            entry.patterns_detected = analysis.get('patterns', [])
            entry.estimated_purpose = analysis.get('likely_purpose', '')
            entry.risk_level = analysis.get('risk_level', 'low')
    
    def add_interaction_note(self, person_id: str, interaction: str):
        """Add interaction note (e.g., audio transcription correlation)"""
        today = date.today()
        person_key = f"{person_id}_{today}"
        
        if person_key in self.current_log:
            self.current_log[person_key].interactions.append(interaction)
    
    def finalize_daily_log(self, ai_analyzer) -> DailyLog:
        """Generate comprehensive daily intelligence report"""
        today = date.today()
        
        # Convert current log entries to list
        people_entries = list(self.current_log.values())
        
        # Calculate statistics
        total_people = len(people_entries)
        total_detections = sum(entry.total_detections for entry in people_entries)
        
        # Analyze peak activity hours
        all_timestamps = []
        for entry in people_entries:
            all_timestamps.append(entry.first_seen.hour)
            all_timestamps.append(entry.last_seen.hour)
        
        peak_hours = self._calculate_peak_hours(all_timestamps)
        
        # Generate AI-powered daily summary and insights
        daily_summary = self._generate_daily_summary(people_entries, ai_analyzer)
        security_alerts = self._identify_security_alerts(people_entries)
        patterns = self._identify_daily_patterns(people_entries, ai_analyzer)
        intelligence_insights = self._generate_intelligence_insights(people_entries, ai_analyzer)
        
        # Create daily log
        daily_log = DailyLog(
            date=today,
            total_people=total_people,
            total_detections=total_detections,
            peak_activity_hours=peak_hours,
            people_entries=people_entries,
            daily_summary=daily_summary,
            security_alerts=security_alerts,
            patterns_identified=patterns,
            intelligence_insights=intelligence_insights
        )
        
        # Save to file
        self._save_daily_log(daily_log)
        
        # Clear current log for new day
        self.current_log.clear()
        
        return daily_log
    
    def _calculate_peak_hours(self, timestamps: List[int]) -> List[int]:
        """Calculate peak activity hours"""
        if not timestamps:
            return []
        
        hour_counts = {}
        for hour in timestamps:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Return top 3 hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]
    
    def _generate_daily_summary(self, entries: List[DailyPersonEntry], ai_analyzer) -> str:
        """Generate AI-powered daily summary"""
        if not entries or not ai_analyzer.api_key:
            return "No significant activity today."
        
        try:
            # Prepare summary data
            summary_data = {
                "total_people": len(entries),
                "total_detections": sum(e.total_detections for e in entries),
                "time_range": f"{min(e.first_seen for e in entries)} to {max(e.last_seen for e in entries)}",
                "descriptions": [desc for e in entries for desc in e.ai_descriptions[:2]],  # First 2 descriptions per person
                "purposes": [e.estimated_purpose for e in entries if e.estimated_purpose]
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": f"""Analyze this day's doorbell camera activity and provide a comprehensive daily intelligence summary:

Data: {json.dumps(summary_data, default=str)}

Generate a detailed summary covering:
- Overall activity level and patterns
- Types of visitors and their purposes
- Notable behaviors or incidents
- Time-based patterns
- Security observations
- Key insights about foot traffic

Keep it factual, analytical, and intelligence-focused. Write 2-3 paragraphs."""
                }],
                "max_tokens": 500
            }
            
            import requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ai_analyzer.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return "Daily summary generation failed."
                
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            return "Daily summary unavailable due to error."
    
    def _identify_security_alerts(self, entries: List[DailyPersonEntry]) -> List[str]:
        """Identify security alerts from daily activity"""
        alerts = []
        
        for entry in entries:
            # High risk individuals
            if entry.risk_level.lower() in ['high', 'critical']:
                alerts.append(f"High-risk individual detected: {entry.person_id} - {entry.security_assessment}")
            
            # Unusual hours (very early or very late)
            if entry.first_seen.hour < 6 or entry.first_seen.hour > 22:
                alerts.append(f"Unusual hour activity: Person {entry.person_id} detected at {entry.first_seen.hour}:00")
            
            # Multiple visits in short time
            if entry.total_detections > 5:
                alerts.append(f"High frequency visits: Person {entry.person_id} detected {entry.total_detections} times")
            
            # Suspicious patterns
            if any('suspicious' in pattern.lower() for pattern in entry.patterns_detected):
                alerts.append(f"Suspicious behavior pattern detected for person {entry.person_id}")
        
        return alerts
    
    def _identify_daily_patterns(self, entries: List[DailyPersonEntry], ai_analyzer) -> List[str]:
        """Identify patterns in daily activity using AI"""
        if not entries or not ai_analyzer.api_key:
            return []
        
        try:
            # Prepare pattern analysis data
            pattern_data = []
            for entry in entries:
                pattern_data.append({
                    "person_id": entry.person_id,
                    "visit_times": [entry.first_seen.hour, entry.last_seen.hour],
                    "total_detections": entry.total_detections,
                    "purposes": entry.estimated_purpose,
                    "descriptions": entry.ai_descriptions[:2]
                })
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": f"""Analyze this doorbell camera data to identify meaningful patterns:

Data: {json.dumps(pattern_data, default=str)}

Identify patterns such as:
- Delivery schedules and patterns
- Neighbor routines and walking patterns  
- Peak activity times
- Recurring visitors
- Service worker schedules
- Weekend vs weekday differences
- Unusual or irregular patterns

Return as a JSON array of pattern descriptions."""
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            import requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ai_analyzer.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = json.loads(result["choices"][0]["message"]["content"])
                return content.get("patterns", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return []
    
    def _generate_intelligence_insights(self, entries: List[DailyPersonEntry], ai_analyzer) -> List[str]:
        """Generate intelligence insights using AI"""
        if not entries or not ai_analyzer.api_key:
            return []
        
        try:
            # Prepare intelligence data
            intel_data = {
                "total_unique_people": len(entries),
                "security_assessments": [e.security_assessment for e in entries if e.security_assessment],
                "behavioral_notes": [e.behavioral_notes for e in entries if e.behavioral_notes],
                "estimated_purposes": [e.estimated_purpose for e in entries if e.estimated_purpose],
                "time_patterns": [(e.first_seen.hour, e.last_seen.hour) for e in entries],
                "frequency_data": [e.total_detections for e in entries]
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": f"""Generate actionable intelligence insights from this doorbell surveillance data:

Data: {json.dumps(intel_data, default=str)}

Provide insights about:
- Neighborhood security assessment
- Foot traffic intelligence 
- Behavioral pattern analysis
- Risk assessments and recommendations
- Operational security insights
- Predictive observations
- Surveillance value assessment

Return as JSON array of insights."""
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }
            
            import requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ai_analyzer.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = json.loads(result["choices"][0]["message"]["content"])
                return content.get("insights", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating intelligence insights: {e}")
            return []
    
    def _save_daily_log(self, daily_log: DailyLog):
        """Save daily log to file"""
        filename = f"daily_log_{daily_log.date.strftime('%Y_%m_%d')}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        try:
            # Convert to dict for JSON serialization
            log_dict = asdict(daily_log)
            
            # Convert datetime objects to strings
            for entry in log_dict['people_entries']:
                entry['first_seen'] = entry['first_seen'].isoformat()
                entry['last_seen'] = entry['last_seen'].isoformat()
            
            log_dict['date'] = daily_log.date.isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(log_dict, f, indent=2, default=str)
            
            logger.info(f"Daily log saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving daily log: {e}")
    
    def get_daily_log(self, target_date: date) -> DailyLog:
        """Retrieve daily log for specific date"""
        filename = f"daily_log_{target_date.strftime('%Y_%m_%d')}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to DailyLog object
            # Note: This is a simplified version - you might want to implement full deserialization
            return data
            
        except FileNotFoundError:
            logger.warning(f"No daily log found for {target_date}")
            return None
        except Exception as e:
            logger.error(f"Error loading daily log: {e}")
            return None
    
    def get_intelligence_summary(self, days: int = 7) -> Dict[str, Any]:
        """Generate intelligence summary for recent days"""
        # This would analyze multiple daily logs to identify longer-term patterns
        # Implementation would depend on specific intelligence requirements
        pass