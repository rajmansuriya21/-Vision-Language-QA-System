from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from typing import Optional, List
import asyncio
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from .models import ModelManager
from .auth import verify_api_key
from .image_processing import preprocess_image
from .prompt_templates import get_prompt_template

load_dotenv()

app = FastAPI(title="Vision-Language QA API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize model manager
model_manager = ModelManager()

class QuestionRequest(BaseModel):
    question: str
    model_name: str = "blip2"
    explain: bool = False
    style: str = "detailed"

@app.post("/api/ask")
async def ask_question(
    image: UploadFile = File(...),
    question: str = None,
    model_name: str = "blip2",
    explain: bool = False,
    style: str = "detailed",
    api_key: str = Depends(verify_api_key)
):
    try:
        # Read and preprocess image
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content))
        processed_image = preprocess_image(pil_image)
        
        # Get model and generate response
        model = model_manager.get_model(model_name)
        prompt = get_prompt_template(style, explain)
        
        async def generate_response():
            async for token in model.generate_stream(processed_image, question, prompt):
                yield f"data: {token}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return {"models": model_manager.list_available_models()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 