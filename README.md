# 🎯 WatchPost AI

**Intelligence-powered surveillance system that transforms any doorbell camera into a comprehensive neighborhood monitoring solution.** Automatically detects, profiles, and analyzes everyone who passes by using advanced AI vision and behavioral analysis.

## Overview

WatchPost AI represents **Void Intelligence** - extracting actionable intelligence from surveillance data that would otherwise go unused. Your doorbell sees everything but remembers nothing. Until now. WatchPost AI captures, profiles and analyses every person who passes by, turning fleeting moments into persistent intelligence.

## Key Features

### 🔍 **Comprehensive Person Analysis**
- **8 AI Analysis Types**: Physical description, behavioral assessment, security profiling, purpose estimation, clothing analysis, demographics, threat assessment, and intelligence notes
- **Multiple LLM Prompts**: Each person triggers extensive AI analysis with specialized prompts
- **Real-time Processing**: Instant analysis as people are detected
- **55% Confidence Threshold**: Reliable detection with minimal false positives

### 📊 **Daily Intelligence Logging**
- **Complete Daily Reports**: Comprehensive logs of all people detected each day
- **Pattern Recognition**: AI identifies behavioral patterns and routines
- **Security Alerts**: Automatic flagging of suspicious activities
- **Intelligence Insights**: Actionable neighborhood intelligence

### 🎯 **Advanced Detection Pipeline**
- **YOLO v8 Object Detection**: Real-time person detection
- **OpenAI GPT-4 Vision**: Comprehensive visual analysis
- **Face Recognition**: Person clustering and identification
- **Behavioral Analysis**: Pattern recognition and threat assessment
- **Persistent Storage**: All encounters stored and analyzed

### 🖥️ **Professional CCTV Interface**
- **Large Live Feed**: 500px high-resolution real-time camera stream
- **Surveillance Aesthetics**: Professional security monitoring interface
- **Date-based Filtering**: Browse historical data by date range
- **Image Gallery**: Easy browsing of all captured images
- **Technical Dashboard**: Real-time system status and capabilities

## Technical Architecture

```
├── main.py                 # Main detection system orchestrator
├── camera_manager.py       # RTSP stream processing & YOLO detection
├── ai_analyzer.py         # 8 comprehensive AI analyses per person
├── daily_logger.py        # Intelligence logging and reporting
├── person_clustering.py    # Person clustering and profiling
├── image_processor.py      # Image enhancement and storage
├── web_server.py          # Surveillance web interface
├── models.py              # Data models and structures
└── config.py              # System configuration
```

## AI Analysis Pipeline

For every person detected, the system runs comprehensive analysis:

1. **Physical Description** - Height, build, distinguishing features, appearance
2. **Behavioral Assessment** - Body language, posture, attention focus, movement patterns
3. **Security Profiling** - Threat level, risk factors, legitimacy assessment
4. **Purpose Estimation** - Why they're there, expected duration, intent analysis
5. **Clothing Analysis** - Style, items carried, uniform indicators, brand identification
6. **Demographics** - Age, gender, socioeconomic indicators, ethnicity
7. **Threat Assessment** - Immediate/long-term threat evaluation, risk scoring
8. **Intelligence Notes** - Comprehensive surveillance notes and recommendations

## Technical Stack

### Core Processing
- **RTSP Stream**: H.264 video protocol processing
- **AI Detection**: YOLO v8 real-time object detection
- **Vision Analysis**: OpenAI GPT-4 Vision API
- **Video Processing**: FFmpeg stream handling
- **Face Recognition**: Clustering & biometric analysis
- **Intelligence**: Multi-prompt behavioral assessment
- **Storage**: Persistent profile database

### Performance Specifications
- **Detection Confidence**: 55% threshold for reliable results
- **Processing Speed**: Real-time analysis with 2-second cooldown
- **Image Quality**: High-resolution capture (1920x1080)
- **Analysis Depth**: 8 comprehensive prompts per person
- **Storage**: Persistent SQLite database with image gallery

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WatchPostAI.git
cd WatchPostAI
```

2. Create a virtual environment:
```bash
python -m venv ai_cam
source ai_cam/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

5. Configure your camera RTSP URL in `config.py`

## Usage

### Quick Start
```bash
python web_server.py
```
Visit `http://localhost:8000` for the surveillance interface

### Full System
```bash
python main.py  # Detection system
python web_server.py  # Web interface (separate terminal)
```

## Intelligence Features

### Person Monitoring
- **Comprehensive Profiling**: 8 AI analyses per person
- **Security Assessment**: Risk levels and threat indicators
- **Behavioral Patterns**: Routine identification and anomaly detection
- **Visual Intelligence**: Detailed appearance and clothing analysis
- **Historical Tracking**: Complete timeline of all encounters

### Daily Intelligence Reports
- **Activity Summaries**: AI-generated daily reports
- **Pattern Recognition**: Routine identification across time
- **Security Alerts**: Automatic suspicious activity flagging
- **Intelligence Insights**: Actionable neighborhood intelligence
- **Peak Activity Analysis**: Busiest times and patterns

### Surveillance Interface
- **Professional CCTV Styling**: Authentic surveillance monitor aesthetic
- **Large Live Feed**: 500px high-resolution video monitoring
- **Image Gallery**: Easy browsing of all captured images
- **Date Filtering**: Browse historical data by date range
- **Technical Dashboard**: Real-time system capabilities display

## Configuration

### Detection Settings
```python
CONFIDENCE_THRESHOLD = 0.55   # 55% confidence for reliable detection
PERSON_DETECTION_COOLDOWN = 2 # 2-second capture rate
CAPTURE_MULTIPLE_ANGLES = True # Multiple shots per person
```

### AI Analysis Configuration
- Runs 8 different specialized prompts per person
- Generates 400-500 tokens of analysis per prompt
- Creates comprehensive intelligence profiles
- Persistent storage with image gallery

## Privacy & Security

- **Local Processing**: All analysis happens on your device
- **Secure Storage**: Encrypted local database storage
- **API Security**: Only images sent to OpenAI for analysis
- **No Cloud Storage**: All data remains on your network
- **Compliance Ready**: Built for privacy law compliance

## Use Cases

### Neighborhood Security
- **Suspicious Activity Detection**: AI identifies concerning behaviors
- **Pattern Recognition**: Unusual timing or frequency alerts
- **Threat Assessment**: Risk evaluation for each person
- **Security Intelligence**: Comprehensive neighborhood profiling

### Community Intelligence
- **Visitor Patterns**: Understanding who visits when
- **Service Schedules**: Delivery and service timing patterns
- **Foot Traffic Analysis**: Peak activity identification
- **Routine Mapping**: Regular walker and jogger identification

### Business Intelligence
- **Customer Analytics**: Understanding visitor demographics
- **Security Monitoring**: Professional-grade threat assessment
- **Operational Insights**: Peak activity and pattern analysis
- **Compliance Documentation**: Comprehensive activity logging

## Intelligence Value

Transform your doorbell camera from passive recording to active intelligence:

- **8x AI Analysis**: Comprehensive person profiling per detection
- **Daily Intelligence**: Actionable neighborhood insights and reports
- **Pattern Recognition**: Behavioral analysis and routine predictions
- **Security Intelligence**: Professional-grade threat assessment
- **Historical Analysis**: Complete timeline of all encounters
- **Technical Excellence**: RTSP, YOLO, OpenAI, FFmpeg integration

---

*WatchPost AI: Intelligence-Powered Surveillance for the Modern World*

**Developed by Daniel Nakhla - AI/ML Subject Matter Expert**