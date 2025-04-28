# Vision-Language QA System

A production-ready Vision-Language Question Answering system supporting multiple models, real-time streaming, and advanced features.

## Features

- 🚀 Real-time streaming responses
- 🔄 Multiple model support (BLIP-2, OFA, LLaVA)
- 🖼️ Robust image preprocessing
- 🎯 Customizable prompt engineering
- 🔒 API token authentication
- 🐳 Dockerized deployment
- 💡 Explanation mode for detailed answers

## Tech Stack

- **Frontend**:

  - Streamlit for interactive UI
  - Real-time streaming responses
  - Responsive layout with columns
  - File upload and image preview
  - Dynamic answer streaming

- **Backend**:

  - FastAPI for high-performance API
  - Uvicorn ASGI server
  - CORS middleware support
  - API key authentication
  - Streaming response support

- **AI/ML**:

  - HuggingFace Transformers
  - BLIP-2 (Salesforce/blip2-opt-2.7b)
  - OFA (OFA-Sys/OFA-base)
  - LLaVA (llava-hf/llava-1.5-7b)
  - PyTorch for model inference
  - CUDA support for GPU acceleration

- **Image Processing**:

  - torchvision for image transformations
  - PIL (Python Imaging Library)
  - Image validation and preprocessing
  - Automatic resizing and normalization

- **Security**:

  - API key authentication
  - Environment variable management
  - Secure token handling

- **Deployment**:
  - Docker containerization
  - Python 3.9 base image
  - Environment variable configuration
  - Port mapping and exposure

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
4. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

The API documentation is available at `/docs` when running the server.

## Docker Deployment

```bash
docker build -t vision-qa .
docker run -p 8000:8000 vision-qa
```
