import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import fastapi.responses

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Google Gemini API Settings
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
GEMINI_TEXT_URL = f'https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent'
GEMINI_IMAGE_URL = f'https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent'

API_TOKEN = os.getenv('API_TOKEN', 'defaultapitoken')

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail={"error": "Invalid API Token"})

class ChatRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": request.prompt}]}
        ]
    }
    try:
        response = requests.post(
            GEMINI_TEXT_URL + f'?key={GEMINI_API_KEY}',
            json=payload
        )
        result = response.json()
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=result)
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/generate-image", dependencies=[Depends(verify_api_key)])
async def generate_image(request: ImageRequest):
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": request.prompt}]}
        ]
    }
    try:
        response = requests.post(
            GEMINI_IMAGE_URL + f'?key={GEMINI_API_KEY}',
            json=payload
        )
        result = response.json()
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=result)
        # For Gemini multimodal/image response: check if image content exists
        parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
        img_or_desc = parts[0] if parts else {}
        return {"response": img_or_desc}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.get("/")
def root():
    return {"status": "ok"}
