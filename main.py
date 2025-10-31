import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import fastapi.responses

# --- 1. 환경 변수 로드 ---
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
API_TOKEN = os.getenv('API_TOKEN', 'defaultapitoken')

# --- 2. API URL (v1beta 사용) ---
# 모델 이름을 URL에서 제거했습니다.
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


app = FastAPI()

# --- 3. CORS (보안) 설정 (최종) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail={"error": "Invalid API Token"})

class ChatRequest(BaseModel):
    prompt: str

# --- 4. 채팅 API (/chat) ---
# 'gemini-pro' (텍스트 모델)를 직접 호출합니다.
@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    
    # 텍스트 전용 URL
    TEXT_URL = f'{BASE_URL}/gemini-pro:generateContent'
    
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": request.prompt}]}
        ]
    }
    try:
        response = requests.post(
            TEXT_URL + f'?key={GEMINI_API_KEY}',
            json=payload
        )
        result = response.json()
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=result)
        
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

# --- 5. 이미지 API (/generate-image) ---
# 'gemini-1.5-flash' (이미지 모델)를 직접 호출합니다.
@app.post("/generate-image", dependencies=[Depends(verify_api_key)])
async def generate_image(request: Request):
    
    # 이미지 전용 URL
    IMAGE_URL = f'{BASE_URL}/gemini-1.5-flash:generateContent'

    try:
        data = await request.json()
        payload = data.get('payload')

        if not payload:
            raise HTTPException(status_code=400, detail={"error": "Payload not found"})

        response = requests.post(
            IMAGE_URL + f'?key={GEMINI_API_KEY}',
            json=payload
        )
        
        result = response.json()
        
        if response.status_code != 200:
            error_detail = result.get("error", {"message": "Unknown Google API error"})
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        return result

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
    # '버전'을 추가해서 최신 코드가 적용됐는지 확인합니다.
    return {"status": "ok", "version": "final_split_model"}





