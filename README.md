# DoorbellAI 🏠🤖

An intelligent doorbell camera system powered by AI that provides real-time object detection, people tracking, audio transcription, and automated security insights.

## 🌟 Features

### 🎥 **Real-Time Video Processing**
- Live RTSP video stream processing
- YOLOv8 object detection with tracking
- Optimized for Apple Silicon with CoreML
- Smart filtering of irrelevant objects

### 👥 **Advanced People Tracking**
- Individual person identification and tracking
- Automatic persona generation using GPT-4.1-mini
- Visit frequency and pattern analysis
- Memorable names for frequent visitors (e.g., "Mike the Mail Carrier")

### 🎧 **Audio Intelligence** 
- Continuous audio recording and transcription
- Automatic deletion of silent/empty audio files
- Cumulative transcription logging
- Audio playback for meaningful segments

### 🧠 **AI-Powered Insights**
- Hour-by-hour activity summaries
- Person-by-person behavioral analysis
- Security alert detection
- Location-aware analysis (configurable zip code)

### 📊 **Comprehensive Logging**
- Parquet-based data storage for efficient analysis
- Detection logs with confidence scores and tracking IDs
- People database with visit history
- Insights database with temporal organization

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- RTSP-compatible camera
- OpenAI API key (optional, for advanced AI features)
- FFmpeg for audio processing

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DoorbellAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv ai_cam
   source ai_cam/bin/activate  # On Windows: ai_cam\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"  # Optional but recommended
   ```

5. **Configure camera settings**
   Edit `server.py` and update the `RTSP_URL` with your camera's stream:
   ```python
   RTSP_URL = "rtsp://username:password@camera-ip:port/stream"
   ```

6. **Run the application**
   ```bash
   python server.py
   ```

7. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8000`

## 🏗️ Architecture

### Core Components

- **Video Processing**: Real-time object detection using YOLOv8
- **Audio Processing**: Continuous transcription using OpenAI Whisper
- **AI Analysis**: Background analysis using GPT-4.1-mini for insights
- **Data Storage**: Parquet files for efficient data management
- **Web Interface**: FastAPI-based dashboard with real-time updates

### File Structure
```
DoorbellAI/
├── server.py                 # Main application
├── requirements.txt          # Python dependencies
├── yolov8n.mlpackage/        # Optimized YOLO model
├── captured_subjects/        # Organized detection images
├── audio_recordings/         # Audio segments with transcriptions
├── detections.log.parquet    # Detection data
├── people_database.parquet   # People tracking data
├── insights_database.parquet # AI insights and summaries
└── transcriptions.log        # Text transcription log
```

## 📡 API Endpoints

### Real-Time Data
- `GET /` - Web dashboard
- `GET /video_feed` - Live video stream
- `GET /transcription` - Current transcription data
- `GET /events` - Current detections
- `GET /gallery` - Recent captured images

### Analytics & Insights
- `GET /insights/hourly?hours_back=24` - Hour-by-hour activity analysis
- `GET /insights/people?hours_back=24` - Person-specific insights
- `GET /people/with-personas` - People with AI-generated names
- `GET /analytics` - Comprehensive analytics dashboard
- `GET /audio/list` - Available audio recordings

### Management
- `POST /people/generate-personas` - Generate AI personas for frequent visitors
- `POST /cleanup` - Clear captured images and tracking data

## ⚙️ Configuration

### Key Settings (in server.py)
```python
# Camera Configuration
RTSP_URL = "your-rtsp-stream-url"

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5
EXCLUDED_OBJECTS = {'car', 'truck', 'bicycle', 'skateboard', ...}

# Storage Limits
MAX_CAPTURES_PER_SUBJECT = 10
MAX_AUDIO_FILES = 100
MAX_TRANSCRIPTION_HISTORY = 50

# Analysis Settings
VISION_ANALYSIS_INTERVAL = 60  # seconds
AUDIO_SEGMENT_DURATION = 10    # seconds
```

### Location Context
Update the zip code in the AI analysis prompts to match your location:
```python
# Search for "19123" in server.py and replace with your zip code
```

## 🎯 Use Cases

### 🏠 **Home Security**
- Monitor visitors and deliveries
- Track family member arrivals/departures
- Identify suspicious activity patterns
- Audio alerts for emergencies

### 📦 **Package Management**
- Automatic delivery driver recognition
- Package delivery notifications
- Tracking delivery patterns and times

### 👨‍👩‍👧‍👦 **Family Monitoring**
- Kids arriving home from school
- Elder care monitoring
- Pet activity tracking
- Visitor identification

### 🏢 **Business Applications**
- Customer visit patterns
- Staff arrival tracking
- Security incident analysis
- Automated access logging

## 🛡️ Security & Privacy

- **Local Processing**: All video processing happens locally
- **Configurable AI**: OpenAI integration is optional
- **Data Control**: All data stored locally in your environment
- **Access Control**: No external data sharing by default

## 🔧 Troubleshooting

### Common Issues

**Camera Connection**
```bash
# Test RTSP stream
ffplay "your-rtsp-url"
```

**Model Loading**
```bash
# Verify YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Audio Issues**
```bash
# Test audio extraction
ffmpeg -i "your-rtsp-url" -vn -acodec pcm_s16le -ar 16000 -ac 1 -t 5 test_audio.wav
```

### Performance Optimization

- **GPU Support**: Install CUDA-compatible PyTorch for GPU acceleration
- **Model Size**: Use smaller YOLO models (yolov8n) for faster processing
- **Resolution**: Lower camera resolution for better performance
- **Intervals**: Increase analysis intervals for lower CPU usage

## 📈 Monitoring & Analytics

### Key Metrics
- Detection accuracy and confidence scores
- People visit frequency and patterns
- Audio transcription quality
- System performance and resource usage

### Data Export
- CSV export of detection data
- Parquet files for advanced analysis
- Audio recordings with timestamps
- Comprehensive activity timelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **OpenAI**: Whisper for audio transcription and GPT for AI insights
- **FastAPI**: Web framework for the dashboard
- **OpenCV**: Computer vision processing

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for smart home security**
