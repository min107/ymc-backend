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

# CORS (모든 요청 허용 - ca.html이 파일로 열려도 작동)
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

# --- 이 부분이 수정되었습니다 ---
# 기존의 ImageRequest(BaseModel)는 ca.html이 보내는 복잡한 payload와 맞지 않아 제거하고,
# Request를 직접 받도록 @app.post("/generate-image") 함수를 수정했습니다.

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

# --- 여기가 핵심 수정 부분입니다 ---
# ca.html에서 보낸 복잡한 'payload'를 그대로 받아서 처리하도록 수정했습니다.
@app.post("/generate-image", dependencies=[Depends(verify_api_key)])
async def generate_image(request: Request):
    try:
        # ca.html이 보낸 JSON 본문({ payload: { ... } })을 그대로 받습니다.
        data = await request.json()
        payload = data.get('payload')

        if not payload:
            raise HTTPException(status_code=400, detail={"error": "Payload not found in request body"})

        # 받은 payload를 그대로 Google API로 전송합니다.
        response = requests.post(
            GEMINI_IMAGE_URL + f'?key={GEMINI_API_KEY}',
            json=payload
        )
        
        result = response.json()
        
        if response.status_code != 200:
            # Google API에서 에러가 발생한 경우, 그 내용을 ca.html로 전달합니다.
            error_detail = result.get("error", {"message": "Unknown error from Google API"})
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        # Google API의 성공 결과를 ca.html로 그대로 반환합니다.
        return result

    except HTTPException as he:
        # 우리가 발생시킨 HTTP 예외는 그대로 전달
        raise he
    except Exception as e:
        # 그 외 서버 내부 오류 처리
        raise HTTPException(status_code=500, detail={"error": str(e)})
# --- 수정 끝 ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.get("/")
def root():
    return {"status": "ok"}