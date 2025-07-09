import base64
import requests
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import cv2

from config import config
from models import PersonDetection, PersonProfile

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Handles AI-powered image analysis and description generation"""
    
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        if not self.api_key:
            logger.warning("OpenAI API key not found. AI descriptions will be disabled.")
    
    def analyze_person_sequence(self, image_paths: List[str], timestamps: List[str]) -> Dict[str, Any]:
        """Analyze all images of a person together with temporal context"""
        if not self.api_key:
            logger.error("❌ AI analysis unavailable - no OpenAI API key configured")
            return {"error": "AI analysis unavailable (no API key)"}
        
        if not image_paths:
            return {"error": "No images provided"}
        
        logger.info(f"🧠 Starting sequence analysis for {len(image_paths)} images")
        
        # Encode all images
        encoded_images = []
        for image_path in image_paths:
            encoded_image = self._encode_image(image_path)
            if encoded_image:
                encoded_images.append(encoded_image)
        
        if not encoded_images:
            return {"error": "Failed to process images"}
        
        # Create message content with all images
        content = [
            {
                "type": "text",
                "text": f"""This person was detected by my doorbell security camera in Philadelphia (19123) across {len(encoded_images)} images. This is a residential doorbell camera monitoring the front door/entrance area.

CONTEXT: This camera typically captures:
- Package deliveries (Amazon, UPS, FedEx, USPS)
- Visitors and guests approaching the door
- Homeowners and family members coming/going
- Neighbors walking by on sidewalk
- Service providers (maintenance, contractors, etc.)
- Solicitors and door-to-door sales people

Analyze all images together and provide a concise assessment. Return JSON:

{{
  "nickname": "short identifier based on appearance (e.g., 'UPS Driver', 'Blue Jacket Walker', 'Neighbor')",
  "description": "concise physical description and clothing",
  "gender": "male/female/unknown based on visual appearance",
  "skin_tone": "light/medium/dark/unknown based on visible skin",
  "activity": "what they're doing (delivering packages, visiting, walking by, approaching door, etc.)",
  "threat_level": "low/medium/high (consider normal door activity vs suspicious behavior)",
  "pattern": "observed behavior pattern across images (repeated visits, delivery pattern, etc.)",
  "summary": "one sentence assessment considering typical doorbell camera context"
}}

Focus on what you can observe across all images. Be factual and concise. Consider this is normal residential door activity unless clearly suspicious."""
            }
        ]
        
        # Add all images to the message
        for i, encoded_image in enumerate(encoded_images):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": content
            }],
            "response_format": {"type": "json_object"},
            "max_tokens": 300
        }
        
        try:
            logger.info("🏷️ Calling OpenAI API for sequence analysis...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result["choices"][0]["message"]["content"]
                logger.info(f"🔍 Raw Sequence Analysis Response: {raw_content}")
                analysis_result = json.loads(raw_content)
                logger.info(f"✅ Sequence Analysis: {analysis_result}")
                return {"comprehensive_analysis": analysis_result}
            else:
                logger.error(f"❌ OpenAI API Error in sequence analysis: {response.status_code} - {response.text}")
                return {"error": "Sequence analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return {"error": "Sequence analysis error"}

    def comprehensive_person_analysis(self, image_path: str) -> Dict[str, Any]:
        """Run multiple comprehensive AI analyses on a person"""
        if not self.api_key:
            logger.error("❌ AI analysis unavailable - no OpenAI API key configured")
            return {"error": "AI analysis unavailable (no API key)"}
        
        logger.info(f"🧠 Starting comprehensive AI analysis for image: {image_path}")
        
        # First, pre-screen to verify person is identifiable
        logger.info("🔍 Pre-screening: Checking if person is clearly identifiable...")
        person_check = self._verify_person_identifiable(image_path)
        
        if not person_check.get("identifiable", False):
            logger.warning(f"❌ Person not clearly identifiable: {person_check.get('reason', 'Unknown')}")
            return {"error": "Person not clearly identifiable", "reason": person_check.get('reason', 'Unknown')}
        
        logger.info(f"✅ Person verified as identifiable: {person_check.get('description', 'Clear person detected')}")
        
        analyses = {}
        
        # Run simplified analysis with 2 API calls
        logger.info("📊 Running 2 AI analysis prompts:")
        analyses["comprehensive_analysis"] = self._generate_comprehensive_analysis(image_path)
        
        logger.info(f"✅ Comprehensive analysis complete with {len(analyses)} analyses")
        return analyses
    
    def _verify_person_identifiable(self, image_path: str) -> Dict[str, Any]:
        """Pre-screen image to verify person is clearly identifiable"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"identifiable": False, "reason": "Failed to process image"}
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Look at this image and determine if there is a clearly identifiable person present. Return JSON:

{
  "identifiable": true/false,
  "confidence": "high/medium/low",
  "reason": "explanation of why person is/isn't identifiable",
  "description": "brief description of what you see"
}

Requirements for "identifiable" (BE LENIENT - enhanced image processing):
- Person should be visible in the image (don't worry about size)
- Human form should be distinguishable (even if dark, blurry, or partially obscured)
- Focus on overall human presence rather than perfect clarity
- Enhanced image processing has been applied to improve quality
- If you can reasonably identify it as a person, mark as identifiable"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 200
            }
            
            logger.info("🔍 Calling OpenAI API for person identifiability check...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result["choices"][0]["message"]["content"]
                logger.info(f"🔍 Raw Person Check Response: {raw_content}")
                check_result = json.loads(raw_content)
                logger.info(f"✅ Person Identifiability Check: {check_result}")
                return check_result
            else:
                logger.error(f"❌ OpenAI API Error in person check: {response.status_code} - {response.text}")
                return {"identifiable": False, "reason": "API error during person check"}
                
        except Exception as e:
            logger.error(f"Error in person identifiability check: {e}")
            return {"identifiable": False, "reason": "Error during person check"}
    
    def _generate_comprehensive_analysis(self, image_path: str) -> Dict[str, Any]:
        """Generate comprehensive person analysis in a single API call"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """This is a doorbell camera image from my house in Philadelphia (19123). Analyze what you see in this specific image and provide a concise, factual description. Return JSON:

{
  "nickname": "short, memorable name based on visual appearance (e.g., 'Blue Jacket', 'Tall Person', 'Backpack Walker')",
  "physical_description": "what you can see: height, build, hair, clothing colors",
  "clothing_style": "specific items visible: colors, types of clothing, accessories",
  "gender": "male/female/unknown based on visible appearance",
  "skin_tone": "light/medium/dark/unknown based on visible skin",
  "purpose": "likely reason based on what's visible (delivery, walking, visiting, etc.)",
  "demeanor": "body language and posture in this image",
  "notable_features": [
    "distinct visual detail 1",
    "distinct visual detail 2", 
    "distinct visual detail 3"
  ],
  "summary": "One concise sentence describing what you see in this image"
}

Focus only on what's visible in this specific image. Be factual and concise. Don't speculate about personality or make assumptions beyond what you can see."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            logger.info("🏷️ Calling OpenAI API for comprehensive analysis...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result["choices"][0]["message"]["content"]
                logger.info(f"🔍 Raw Comprehensive Analysis Response: {raw_content}")
                analysis_result = json.loads(raw_content)
                logger.info(f"✅ Comprehensive Analysis: {analysis_result}")
                return analysis_result
            else:
                logger.error(f"❌ OpenAI API Error in comprehensive analysis: {response.status_code} - {response.text}")
                return {"error": "Comprehensive analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": "Comprehensive analysis error"}
    
    def _analyze_physical_description(self, image_path: str) -> Dict[str, Any]:
        """Detailed physical description analysis"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Provide a detailed physical description of this person for surveillance purposes. Return as JSON:

{
  "height_estimate": "tall/medium/short",
  "build": "thin/average/stocky/heavy",
  "hair_color": "color",
  "hair_style": "description",
  "age_range": "estimated age range",
  "gender": "apparent gender",
  "race_ethnicity": "apparent ethnicity",
  "distinguishing_features": ["feature1", "feature2"],
  "facial_features": "description",
  "visible_tattoos_scars": ["any visible marks"],
  "gait_posture": "walking style/posture",
  "overall_appearance": "general description"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            logger.info("🤖 Calling OpenAI API for physical description analysis...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result["choices"][0]["message"]["content"]
                logger.info(f"🔍 Raw OpenAI Response: {raw_content}")
                analysis_result = json.loads(raw_content)
                logger.info(f"✅ OpenAI Physical Description Response: {analysis_result}")
                return analysis_result
            else:
                logger.error(f"❌ OpenAI API Error: {response.status_code} - {response.text}")
                return {"error": "Physical description analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in physical description analysis: {e}")
            return {"error": "Physical description analysis error"}
    
    def _analyze_behavior_and_posture(self, image_path: str) -> Dict[str, Any]:
        """Analyze behavioral cues and body language"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this person's behavior, body language, and posture for surveillance intelligence. Return as JSON:

{
  "body_language": "description of posture and stance",
  "facial_expression": "emotional state/expression",
  "hand_position": "what they're doing with hands",
  "walking_direction": "which way they're moving",
  "attention_focus": "where they're looking/what they're focused on",
  "confidence_level": "confident/nervous/casual/suspicious",
  "urgency_level": "rushed/casual/leisurely",
  "interaction_with_environment": "how they're interacting with surroundings",
  "suspicious_indicators": ["any concerning behaviors"],
  "normal_indicators": ["typical behaviors"],
  "behavioral_assessment": "overall behavioral analysis"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Behavioral analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return {"error": "Behavioral analysis error"}
    
    def _analyze_security_profile(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive security profiling"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Conduct a security profile assessment of this person for neighborhood watch purposes. Return as JSON:

{
  "threat_level": "LOW/MEDIUM/HIGH/CRITICAL",
  "risk_factors": ["factor1", "factor2"],
  "positive_indicators": ["indicator1", "indicator2"],
  "professional_assessment": "delivery/service worker/resident/visitor/unknown",
  "legitimacy_score": 0.85,
  "red_flags": ["any concerning elements"],
  "green_flags": ["reassuring elements"],
  "recommended_monitoring": "none/standard/elevated/high",
  "follow_up_actions": ["recommended actions"],
  "confidence_in_assessment": 0.90,
  "security_notes": "detailed security assessment"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Security profiling failed"}
                
        except Exception as e:
            logger.error(f"Error in security profiling: {e}")
            return {"error": "Security profiling error"}
    
    def _estimate_visit_purpose(self, image_path: str) -> Dict[str, Any]:
        """Estimate purpose of visit"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Estimate why this person is in the area and their likely purpose. Return as JSON:

{
  "primary_purpose": "delivery/mail/service_call/visiting/walking_dog/jogging/suspicious/residential",
  "confidence_level": 0.85,
  "supporting_evidence": ["evidence1", "evidence2"],
  "alternative_purposes": ["other possibility1", "other possibility2"],
  "time_appropriateness": "appropriate/questionable/suspicious for time of day",
  "expected_duration": "brief/short/extended stay",
  "repeat_visitor_likelihood": "high/medium/low",
  "business_legitimate": "yes/no/uncertain",
  "purpose_assessment": "detailed explanation of estimated purpose"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Purpose estimation failed"}
                
        except Exception as e:
            logger.error(f"Error in purpose estimation: {e}")
            return {"error": "Purpose estimation error"}
    
    def _analyze_clothing_and_items(self, image_path: str) -> Dict[str, Any]:
        """Analyze clothing and carried items"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this person's clothing and any items they're carrying. Return as JSON:

{
  "clothing_style": "casual/formal/work_uniform/athletic/other",
  "clothing_colors": ["color1", "color2"],
  "clothing_condition": "new/worn/dirty/professional",
  "uniform_indicators": ["any uniform elements"],
  "carried_items": ["item1", "item2"],
  "bags_containers": ["type of bags/containers"],
  "tools_equipment": ["any visible tools"],
  "suspicious_items": ["anything concerning"],
  "professional_items": ["work-related items"],
  "personal_items": ["personal belongings"],
  "clothing_analysis": "detailed clothing assessment"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Clothing analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in clothing analysis: {e}")
            return {"error": "Clothing analysis error"}
    
    def _estimate_demographics(self, image_path: str) -> Dict[str, Any]:
        """Estimate demographic information"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Estimate demographic information about this person for identification purposes. Return as JSON:

{
  "age_estimate": "specific age range",
  "gender_presentation": "apparent gender",
  "ethnicity_estimate": "apparent ethnicity/background",
  "socioeconomic_indicators": ["indicators of social/economic status"],
  "education_indicators": ["any indicators of education level"],
  "health_fitness": "apparent health/fitness level",
  "geographic_origin": "any indicators of where they might be from",
  "lifestyle_indicators": ["lifestyle clues"],
  "demographic_confidence": 0.75,
  "demographic_notes": "overall demographic assessment"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Demographic estimation failed"}
                
        except Exception as e:
            logger.error(f"Error in demographic estimation: {e}")
            return {"error": "Demographic estimation error"}
    
    def _assess_threat_level(self, image_path: str) -> Dict[str, Any]:
        """Assess potential threat level"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Assess the potential threat level of this person for neighborhood security. Return as JSON:

{
  "immediate_threat": "NONE/LOW/MEDIUM/HIGH/CRITICAL",
  "long_term_concern": "NONE/LOW/MEDIUM/HIGH",
  "threatening_behaviors": ["any concerning behaviors"],
  "reassuring_behaviors": ["positive behaviors"],
  "weapon_indicators": ["any potential weapon indicators"],
  "escape_route_awareness": "do they seem aware of exits/routes",
  "surveillance_awareness": "do they seem aware of being watched",
  "criminal_indicators": ["potential criminal activity indicators"],
  "innocent_indicators": ["indicators of legitimate presence"],
  "recommended_response": "monitor/ignore/investigate/alert_authorities",
  "threat_assessment": "detailed threat analysis"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 350
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Threat assessment failed"}
                
        except Exception as e:
            logger.error(f"Error in threat assessment: {e}")
            return {"error": "Threat assessment error"}
    
    def _generate_intelligence_notes(self, image_path: str) -> Dict[str, Any]:
        """Generate comprehensive intelligence notes"""
        try:
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return {"error": "Failed to process image"}
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Generate comprehensive intelligence notes about this person for neighborhood watch records. Return as JSON:

{
  "intelligence_summary": "concise intelligence summary",
  "notable_features": ["distinctive characteristics"],
  "behavioral_patterns": ["observable patterns"],
  "risk_assessment": "overall risk evaluation",
  "monitoring_recommendations": ["how to monitor this person"],
  "follow_up_questions": ["what else should be investigated"],
  "correlation_opportunities": ["what to correlate this with"],
  "alerting_criteria": ["when to escalate monitoring"],
  "profile_confidence": 0.80,
  "intelligence_value": "HIGH/MEDIUM/LOW intelligence value",
  "actionable_insights": ["specific actionable information"],
  "investigation_leads": ["potential leads to follow"],
  "comprehensive_notes": "detailed intelligence assessment"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
            else:
                return {"error": "Intelligence notes generation failed"}
                
        except Exception as e:
            logger.error(f"Error generating intelligence notes: {e}")
            return {"error": "Intelligence notes generation error"}

    def describe_person(self, image_path: str) -> str:
        """Generate AI description of a person from their image"""
        if not self.api_key:
            return "AI description unavailable (no API key)"
        
        try:
            # Load and encode image
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return "Failed to process image"
            
            # Prepare OpenAI request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this person and provide a detailed profile. Return your response as JSON with the following structure: {\"name\": \"suggested_name\", \"description\": \"detailed_description\", \"characteristics\": [\"trait1\", \"trait2\"], \"likely_purpose\": \"reason_for_visit\", \"clothing\": \"clothing_description\", \"items\": [\"item1\", \"item2\"], \"estimated_age\": \"age_range\", \"gender\": \"apparent_gender\", \"posture\": \"body_language\", \"context_clues\": [\"clue1\", \"clue2\"]}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }
            
            # Make API request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    # Parse JSON response
                    json_result = json.loads(content)
                    logger.info(f"Generated AI profile for {image_path}")
                    return json_result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {content}")
                    return {"description": content, "name": "Person"}
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return {"description": "AI description failed", "name": "Person"}
                
        except Exception as e:
            logger.error(f"Error generating AI description: {e}")
            return "AI description error"
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for OpenAI API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def analyze_group_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze a group of images to identify patterns and consistency"""
        if not self.api_key or not image_paths:
            return {"analysis": "Group analysis unavailable"}
        
        try:
            # Encode multiple images
            encoded_images = []
            for path in image_paths[:5]:  # Limit to 5 images for API efficiency
                encoded = self._encode_image(path)
                if encoded:
                    encoded_images.append(encoded)
            
            if not encoded_images:
                return {"analysis": "No valid images for group analysis"}
            
            # Create content with multiple images
            content = [
                {
                    "type": "text",
                    "text": """Analyze this group of images showing the same person across different visits. Provide a comprehensive analysis in JSON format:

{
  "identity_consistency": "How consistent is this person's appearance across visits?",
  "clothing_patterns": ["pattern1", "pattern2"],
  "behavioral_observations": ["behavior1", "behavior2"],
  "temporal_changes": "Any changes noticed over time",
  "confidence_score": 0.95,
  "distinguishing_features": ["feature1", "feature2"],
  "likely_role": "delivery_person/neighbor/visitor/etc",
  "clustering_quality": "high/medium/low"
}"""
                }
            ]
            
            # Add all images
            for encoded_image in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 600
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                return {"analysis": "Group analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in group analysis: {e}")
            return {"analysis": "Group analysis error"}
    
    def analyze_person_patterns(self, profile: PersonProfile) -> Dict[str, Any]:
        """Analyze patterns in a person's appearances using AI"""
        if not profile.detections:
            return {}
        
        patterns = {
            "visit_times": [],
            "confidence_scores": [],
            "time_of_day_pattern": {},
            "frequency_analysis": {},
            "behavioral_patterns": "",
            "security_assessment": "",
            "group_analysis": {}
        }
        
        # Extract visit times
        for detection in profile.detections:
            patterns["visit_times"].append(detection.timestamp)
            patterns["confidence_scores"].append(detection.confidence)
        
        # Analyze time of day patterns
        hour_counts = {}
        day_counts = {}
        for detection in profile.detections:
            hour = detection.timestamp.hour
            day = detection.timestamp.strftime("%A")
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
            day_counts[day] = day_counts.get(day, 0) + 1
        
        patterns["time_of_day_pattern"] = hour_counts
        patterns["day_of_week_pattern"] = day_counts
        
        # Calculate average confidence
        if patterns["confidence_scores"]:
            patterns["avg_confidence"] = sum(patterns["confidence_scores"]) / len(patterns["confidence_scores"])
        
        # Generate AI-powered analyses
        patterns["behavioral_patterns"] = self._analyze_behavioral_patterns_advanced(profile)
        patterns["security_assessment"] = self._assess_security_risk_advanced(profile)
        patterns["group_analysis"] = self.analyze_group_images(profile.images)
        
        return patterns
    
    def _analyze_behavioral_patterns_advanced(self, profile: PersonProfile) -> Dict[str, Any]:
        """Advanced AI behavioral pattern analysis with context"""
        if not self.api_key or not profile.detections:
            return {"analysis": "Pattern analysis unavailable"}
        
        try:
            # Prepare contextual data
            timeline_context = {
                "total_visits": profile.total_appearances,
                "time_span_days": (profile.last_seen - profile.first_seen).days,
                "visit_frequency": profile.total_appearances / max(1, (profile.last_seen - profile.first_seen).days),
                "times_of_day": [d.timestamp.hour for d in profile.detections],
                "days_of_week": [d.timestamp.strftime("%A") for d in profile.detections],
                "confidence_levels": [d.confidence for d in profile.detections],
                "description": profile.description
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Analyze this person's behavioral patterns and provide insights in JSON format:

Context: {json.dumps(timeline_context, default=str)}

Provide analysis as:
{{
  "routine_type": "regular/irregular/sporadic",
  "likely_purpose": "delivery/resident/visitor/suspicious/maintenance",
  "time_preferences": "morning/afternoon/evening/night",
  "behavior_consistency": "high/medium/low",
  "predicted_next_visit": "time_estimate",
  "risk_indicators": ["indicator1", "indicator2"],
  "notable_patterns": ["pattern1", "pattern2"],
  "confidence_in_clustering": 0.95
}}"""
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                return {"analysis": "Behavioral analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in advanced behavioral analysis: {e}")
            return {"analysis": "Behavioral analysis error"}
    
    def _analyze_behavioral_patterns(self, profile: PersonProfile) -> str:
        """Use AI to analyze behavioral patterns"""
        if not self.api_key or not profile.detections:
            return "Pattern analysis unavailable"
        
        try:
            # Prepare timeline data
            timeline_data = []
            for detection in profile.detections:
                timeline_data.append({
                    "timestamp": detection.timestamp.isoformat(),
                    "day": detection.timestamp.strftime("%A"),
                    "hour": detection.timestamp.hour,
                    "confidence": detection.confidence
                })
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Analyze this person's behavioral patterns based on their detection timeline:
                        
Person Profile:
- Total appearances: {profile.total_appearances}
- First seen: {profile.first_seen}
- Last seen: {profile.last_seen}
- Description: {profile.description}

Detection Timeline:
{json.dumps(timeline_data, indent=2)}

Provide insights on:
1. Visit frequency patterns
2. Time-of-day preferences
3. Behavioral consistency
4. Any notable irregularities
5. Possible routine identification

Keep response concise but insightful."""
                    }
                ],
                "max_tokens": 400
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return "Pattern analysis failed"
                
        except Exception as e:
            logger.error(f"Error analyzing behavioral patterns: {e}")
            return "Pattern analysis error"
    
    def _assess_security_risk_advanced(self, profile: PersonProfile) -> Dict[str, Any]:
        """Advanced AI security risk assessment with multiple factors"""
        if not self.api_key or not profile.detections:
            return {"assessment": "Security assessment unavailable"}
        
        try:
            # Comprehensive risk factors
            risk_context = {
                "total_visits": profile.total_appearances,
                "visit_frequency": profile.total_appearances / max(1, (profile.last_seen - profile.first_seen).days),
                "night_visits": sum(1 for d in profile.detections if d.timestamp.hour < 6 or d.timestamp.hour > 22),
                "unusual_hours": sum(1 for d in profile.detections if d.timestamp.hour in [0, 1, 2, 3, 4, 23]),
                "weekend_visits": sum(1 for d in profile.detections if d.timestamp.weekday() >= 5),
                "description": profile.description,
                "detection_confidence": [d.confidence for d in profile.detections],
                "time_span": (profile.last_seen - profile.first_seen).days,
                "recent_visits": sum(1 for d in profile.detections if (datetime.now() - d.timestamp).days < 7)
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Conduct a comprehensive security risk assessment for this person based on their behavioral data:

Risk Context: {json.dumps(risk_context, default=str)}

Provide assessment in JSON format:
{{
  "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
  "risk_score": 0.75,
  "primary_concerns": ["concern1", "concern2"],
  "behavioral_flags": ["flag1", "flag2"],
  "recommended_actions": ["action1", "action2"],
  "monitoring_priority": "low/medium/high",
  "threat_indicators": ["indicator1", "indicator2"],
  "legitimacy_score": 0.85,
  "reasoning": "detailed explanation of assessment"
}}"""
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                return {"assessment": "Security assessment failed"}
                
        except Exception as e:
            logger.error(f"Error in advanced security assessment: {e}")
            return {"assessment": "Security assessment error"}
    
    def _assess_security_risk(self, profile: PersonProfile) -> str:
        """Assess security risk level using AI"""
        if not self.api_key or not profile.detections:
            return "Security assessment unavailable"
        
        try:
            # Calculate visit frequency
            total_days = (profile.last_seen - profile.first_seen).days + 1
            visit_frequency = profile.total_appearances / total_days if total_days > 0 else 0
            
            # Analyze time patterns
            night_visits = sum(1 for d in profile.detections if d.timestamp.hour < 6 or d.timestamp.hour > 22)
            night_percentage = (night_visits / len(profile.detections)) * 100 if profile.detections else 0
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Assess the security risk level for this person based on their behavior:

Person Profile:
- Description: {profile.description}
- Total appearances: {profile.total_appearances}
- Visit frequency: {visit_frequency:.2f} visits per day
- Night visits: {night_visits} ({night_percentage:.1f}%)
- Time span: {total_days} days
- First seen: {profile.first_seen}
- Last seen: {profile.last_seen}

Provide:
1. Risk level (LOW/MEDIUM/HIGH)
2. Risk factors identified
3. Recommended actions
4. Confidence in assessment

Be objective and focus on behavioral indicators."""
                    }
                ],
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return "Security assessment failed"
                
        except Exception as e:
            logger.error(f"Error assessing security risk: {e}")
            return "Security assessment error"
    
    def generate_person_name(self, description: str) -> str:
        """Generate a simple name for a person based on their description"""
        if not self.api_key or not description:
            return self._generate_simple_name(description)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Based on this person description, generate a simple, memorable name (1-2 words max):

Description: {description}

Examples:
- "Mail Carrier" for postal worker
- "Dog Walker" for person with dog
- "Delivery Guy" for delivery person
- "Jogger" for person running
- "Neighbor" for regular visitor

Return just the name, nothing else."""
                    }
                ],
                "max_tokens": 20
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                name = result["choices"][0]["message"]["content"].strip()
                return name if len(name) < 50 else self._generate_simple_name(description)
            else:
                return self._generate_simple_name(description)
                
        except Exception as e:
            logger.error(f"Error generating name: {e}")
            return self._generate_simple_name(description)
    
    def _generate_simple_name(self, description: str) -> str:
        """Generate a simple name based on description keywords"""
        if not description:
            return "Person"
        
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['delivery', 'package', 'ups', 'fedex', 'amazon']):
            return "Delivery Person"
        if any(word in desc_lower for word in ['mail', 'postal', 'usps']):
            return "Mail Carrier"
        if any(word in desc_lower for word in ['dog', 'leash', 'walking']):
            return "Dog Walker"
        if any(word in desc_lower for word in ['child', 'kid', 'young']):
            return "Child"
        if any(word in desc_lower for word in ['elderly', 'older', 'senior']):
            return "Elderly Person"
        if any(word in desc_lower for word in ['uniform', 'service', 'worker']):
            return "Service Worker"
        if any(word in desc_lower for word in ['jogging', 'running', 'exercise']):
            return "Jogger"
        if any(word in desc_lower for word in ['woman', 'female', 'lady']):
            return "Woman"
        if any(word in desc_lower for word in ['man', 'male', 'guy']):
            return "Man"
        
        return "Person"
    
    def generate_profile_summary(self, profile: PersonProfile) -> str:
        """Generate a comprehensive summary of a person's profile"""
        if not profile.detections:
            return "No detection data available"
        
        patterns = self.analyze_person_patterns(profile)
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Person ID: {profile.id}")
        summary_parts.append(f"Total appearances: {profile.total_appearances}")
        summary_parts.append(f"First seen: {profile.first_seen.strftime('%Y-%m-%d %H:%M')}")
        summary_parts.append(f"Last seen: {profile.last_seen.strftime('%Y-%m-%d %H:%M')}")
        
        # Confidence analysis
        if "avg_confidence" in patterns:
            summary_parts.append(f"Average detection confidence: {patterns['avg_confidence']:.2f}")
        
        # Time patterns
        if patterns["time_of_day_pattern"]:
            most_common_hour = max(patterns["time_of_day_pattern"], key=patterns["time_of_day_pattern"].get)
            summary_parts.append(f"Most commonly seen at hour: {most_common_hour}:00")
        
        # Description
        if profile.description:
            summary_parts.append(f"Description: {profile.description}")
        
        return "\n".join(summary_parts)