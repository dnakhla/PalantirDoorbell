#!/usr/bin/env python3
"""
Generate custom images for Neighborhood Watch AI using Replicate API
"""

import os
import requests
import json
import time
from typing import Dict, Any

def generate_image(prompt: str, filename: str, style: str = "realistic") -> str:
    """Generate an image using Replicate API"""
    
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        return None
    
    payload = {
        "input": {
            "size": "1024x1024",
            "style": style,
            "prompt": prompt,
            "aspect_ratio": "Not set"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    
    try:
        print(f"Generating image: {filename}")
        print(f"Prompt: {prompt}")
        
        response = requests.post(
            "https://api.replicate.com/v1/models/recraft-ai/recraft-v3-svg/predictions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and result["output"]:
                image_url = result["output"]
                
                # Download the image
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    os.makedirs("static/images", exist_ok=True)
                    filepath = f"static/images/{filename}"
                    
                    with open(filepath, "wb") as f:
                        f.write(img_response.content)
                    
                    print(f"✓ Image saved: {filepath}")
                    return filepath
                else:
                    print(f"Failed to download image: {img_response.status_code}")
            else:
                print(f"No output in response: {result}")
        else:
            print(f"API request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error generating image: {e}")
    
    return None

def main():
    """Generate all images needed for the landing page"""
    
    images_to_generate = [
        {
            "prompt": "Modern AI surveillance camera with digital interface overlay showing person detection, futuristic neighborhood security system, high-tech monitoring display with facial recognition markers, professional security equipment, blue and white color scheme",
            "filename": "hero_surveillance.png",
            "style": "any"
        },
        {
            "prompt": "Split screen showing traditional passive security camera vs AI-powered intelligent surveillance system with data analysis, person profiling interface, pattern recognition display, before and after comparison, technology upgrade visualization",
            "filename": "before_after.png", 
            "style": "any"
        },
        {
            "prompt": "Professional security monitoring dashboard with multiple person profiles, AI analysis results, threat assessment interface, intelligence reports, surveillance command center aesthetic, dark blue interface",
            "filename": "intelligence_dashboard.png",
            "style": "any"
        },
        {
            "prompt": "Neighborhood street view with invisible AI analysis overlay showing person detection zones, walking patterns, security assessment markers, suburban setting with intelligent monitoring visualization",
            "filename": "neighborhood_monitoring.png",
            "style": "any"
        },
        {
            "prompt": "AI brain analyzing security camera footage, machine learning algorithms processing person detection data, neural network visualization, intelligence analysis concept art, purple and blue gradient",
            "filename": "ai_analysis.png",
            "style": "line_art"
        },
        {
            "prompt": "Daily intelligence report document with person profiles, security alerts, pattern analysis charts, professional surveillance reporting interface, clean document design",
            "filename": "daily_reports.png",
            "style": "any"
        }
    ]
    
    print("🏘️ Generating images for Neighborhood Watch AI...")
    print("=" * 50)
    
    successful_images = []
    
    for image_config in images_to_generate:
        result = generate_image(
            image_config["prompt"],
            image_config["filename"],
            image_config["style"]
        )
        
        if result:
            successful_images.append(result)
        
        # Small delay between requests
        time.sleep(2)
        print()
    
    print("=" * 50)
    print(f"✓ Generated {len(successful_images)} images successfully")
    
    if successful_images:
        print("\nGenerated images:")
        for img in successful_images:
            print(f"  - {img}")
        
        print("\nNext steps:")
        print("1. Update index.html to use these custom images")
        print("2. Add proper alt text and captions")
        print("3. Optimize image loading and responsiveness")
    
    return successful_images

if __name__ == "__main__":
    main()