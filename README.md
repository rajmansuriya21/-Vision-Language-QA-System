# Advanced Vision-Language QA System

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

- **Frontend**: Streamlit/Next.js
- **Backend**: FastAPI
- **Models**: HuggingFace Transformers (BLIP-2, OFA, LLaVA)
- **Image Processing**: torchvision, PIL
- **Deployment**: Docker, HuggingFace Spaces/Render

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

## License

MIT
