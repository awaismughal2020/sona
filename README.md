# SONA - Modular AI Web Assistant

**SONA: The Mobile Assistant** - friendly, efficient, and always helpful AI-powered assistant with full AI service integration.

## Features

- **Text Chat**: Natural language conversation with intelligent intent detection
- **Voice Input**: Real-time audio recording and speech-to-text processing
- **Web Search**: Real-time information retrieval using SerpAPI
- **Image Generation**: AI-powered image descriptions using Gemini
- **Modular Architecture**: Swappable AI models without code changes
- **Docker Ready**: Full containerization with Docker Compose
- **Cloud Deploy**: Ready for AWS EC2 deployment

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SONA AI Assistant                        │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)     │  Backend (FastAPI)              │
│  ├── Chat Interface       │  ├── API Endpoints              │
│  ├── Voice Input          │  ├── Audio Processing           │
│  └── Real-time UI         │  └── Error Handling             │
├─────────────────────────────────────────────────────────────┤
│                AI Orchestrator                              │
│  ├── Speech-to-Text (Whisper API)                           │
│  ├── Intent Detection (OpenAI GPT)                          │
│  ├── Web Search (SerpAPI)                                   │
│  └── Image Generation (Gemini)                              │
├─────────────────────────────────────────────────────────────┤
│  Configuration  │  Utilities  │  Validation  │  Logging     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Required API Keys (see [API Keys](#api-keys) section)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sona-ai-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see API Keys section)
   ```

5. **Run the application**
   ```bash
   # Development mode (both frontend and backend)
   python main.py --mode dev
   
   # Or run separately:
   python main.py --mode backend    # API server only
   python main.py --mode frontend   # Streamlit UI only
   ```

6. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   # Copy environment file
   cp .env.example .env
   # Edit .env with your API keys
   
   # Build and start services
   docker-compose up --build
   ```

2. **Access the application**
   - Frontend: http://localhost:8501
   - Backend: http://localhost:8000

## 🔑 API Keys

### Required API Keys

Create accounts and obtain API keys from:

1. **OpenAI API** (Required)
   - Used for: Speech-to-text (Whisper), Intent detection (GPT)
   - Get key: https://platform.openai.com/api-keys
   - Set in `.env`: `OPENAI_API_KEY=your_key_here`

2. **Google Gemini API** (Required)
   - Used for: Image generation and descriptions
   - Get key: https://ai.google.dev/
   - Set in `.env`: `GEMINI_API_KEY=your_key_here`

3. **SerpAPI** (Required)
   - Used for: Web search functionality
   - Get key: https://serpapi.com/
   - Set in `.env`: `SERP_API_KEY=your_key_here`

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
SERP_API_KEY=your-serp-api-key-here

# Model Configuration
SPEECH_TO_TEXT_MODEL=whisper
INTENT_DETECTION_MODEL=openai
IMAGE_GENERATION_MODEL=gemini
WEB_SEARCH_MODEL=serp

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
STREAMLIT_PORT=8501
```

## Project Structure

```
sona-ai-assistant/
├── ai/                     # AI Services Module
│   ├── orchestrator.py     # Main AI coordinator
│   ├── speech_to_text/     # Speech recognition services
│   ├── intent_detection/   # Intent classification
│   ├── image_generation/   # Image generation services
│   └── web_search/         # Web search services
├── backend/                # FastAPI Backend
│   ├── app.py              # Main application
│   └── middleware/         # Error handling, logging
├── ui/                     # Streamlit Frontend
│   ├── streamlit_app.py    # Main UI application
│   └── components/         # UI components
├── config/                 # Configuration management
├── utils/                  # Utility modules
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service orchestration
└── .env.example            # Environment template
```

## Model Switching

SONA supports runtime model switching without code changes:

### Available Models

- **Speech-to-Text**: Whisper (OpenAI API)
- **Intent Detection**: OpenAI GPT-3.5/4
- **Image Generation**: Google Gemini
- **Web Search**: SerpAPI

### Switch Models via API

```bash
curl -X POST http://localhost:8000/api/v1/switch-model \
  -F "service_type=speech_to_text" \
  -F "model_type=whisper"
```

### Switch Models via Configuration

Update `.env` file:
```bash
SPEECH_TO_TEXT_MODEL=whisper
INTENT_DETECTION_MODEL=openai
IMAGE_GENERATION_MODEL=gemini
WEB_SEARCH_MODEL=serp
```

## Voice Features

### Real-time Recording
- Browser-based microphone access
- Multiple duration options (5s, 10s, 15s, custom)
- Audio quality monitoring
- Device selection support

### Audio File Upload
- Supported formats: WAV, MP3, M4A, FLAC
- Maximum file size: 10MB
- Automatic preprocessing and validation

### Audio System Requirements

Install audio dependencies:
```bash
pip install sounddevice numpy
```

For system audio support:
- **macOS**: Built-in (Core Audio)
- **Windows**: DirectSound/WASAPI
- **Linux**: ALSA/PulseAudio

## API Endpoints

### Core Endpoints

- `GET /` - Application info and status
- `GET /health` - Health check with service status
- `POST /api/v1/chat` - Text chat processing
- `POST /api/v1/upload-audio` - Audio file processing
- `GET /api/v1/models` - Available AI models
- `POST /api/v1/switch-model` - Runtime model switching

### Example Usage

**Text Chat:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -F "message=What's the price of Bitcoin?" \
  -F "session_id=session_123"
```

**Audio Upload:**
```bash
curl -X POST http://localhost:8000/api/v1/upload-audio \
  -F "audio_file=@recording.wav" \
  -F "session_id=session_123"
```

## 🚀 Deployment

### Local Development
```bash
python main.py --mode dev
```

### Docker Deployment
```bash
docker-compose up --build
```

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Ubuntu 20.04 LTS
   - t3.medium or larger
   - Security groups: 80, 443, 8000, 8501

2. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

3. **Deploy SONA**
   ```bash
   git clone <repository-url>
   cd sona-ai-assistant
   cp .env.example .env
   # Configure .env with API keys
   docker-compose up -d
   ```

4. **Configure Reverse Proxy (Optional)**
   ```bash
   # Install Nginx
   sudo apt install nginx
   
   # Configure reverse proxy for port 80/443
   # Point to localhost:8501 for frontend
   # Point to localhost:8000 for API
   ```

### Production Checklist

- [ ] Set `DEBUG=false` in `.env`
- [ ] Configure proper logging levels
- [ ] Set up SSL certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategies
- [ ] Set resource limits in docker-compose.yml

## 🔧 Configuration

### Environment Variables

| Variable         | Description           | Default  | Required  |
|------------------|-----------------------|----------|-----------|
| `OPENAI_API_KEY` | OpenAI API key        | -        | Yes       |
| `GEMINI_API_KEY` | Google Gemini API key | -        | Yes       |
| `SERP_API_KEY`   | SerpAPI key           | -        | Yes       |
| `DEBUG`          | Debug mode            | false    | No        |
| `LOG_LEVEL`      | Logging level         | INFO     | No        |
| `BACKEND_HOST`   | Backend host          | 0.0.0.0  | No        |
| `BACKEND_PORT`   | Backend port          | 8000     | No        |
| `STREAMLIT_PORT` | Frontend port         | 8501     | No        |

### Model Configuration Options

```bash
# Speech-to-Text Options
SPEECH_TO_TEXT_MODEL=whisper          # Only option currently

# Intent Detection Options  
INTENT_DETECTION_MODEL=openai         # Only option currently

# Image Generation Options
IMAGE_GENERATION_MODEL=gemini          # Only option currently

# Web Search Options
WEB_SEARCH_MODEL=serp                  # Only option currently
```

## Troubleshooting

### Common Issues

**API Keys Not Working:**
- Verify keys are correct in `.env`
- Check API key permissions and quotas
- Ensure environment is loaded: `source venv/bin/activate`

**Audio Recording Issues:**
- Check microphone permissions in browser
- Install audio dependencies: `pip install sounddevice`
- Verify audio devices: `python audio_diagnostic.py`

**Docker Issues:**
- Ensure Docker daemon is running
- Check port conflicts: `docker ps`
- View logs: `docker-compose logs -f`

**Import Errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (3.11+ required)

### Debug Mode

Enable debug logging:
```bash
DEBUG=true python main.py --mode dev
```

Check logs:
```bash
# Docker logs
docker-compose logs -f sona-backend
docker-compose logs -f sona-frontend

# Local logs
tail -f logs/sona.log
```

## Monitoring

### Health Checks

- Backend: `GET /health`
- Service status monitoring
- API response time tracking
- Error rate monitoring

### Logging

- Structured JSON logging
- Multiple log levels (DEBUG, INFO, WARN, ERROR)
- Rotating log files
- Centralized error handling

## 📄 License

This project is proprietary. Commercial use is prohibited without prior written consent.



## 🔗 Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/docs)
- [SerpAPI Documentation](https://serpapi.com/search-api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

