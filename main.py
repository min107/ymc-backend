import os
import requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import fastapi.responses

# 1. .env 로드
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN", "defaultapitoken")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 가 없습니다. Render 환경변수에 추가하세요.")

V1BETA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-KEY", "Authorization"],
)

api_key_header = APIKeyHeader(name="X-API-KEY")


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail={"error": "Invalid API Token"})


class ChatRequest(BaseModel):
    prompt: str


def list_models_v1beta() -> list[dict]:
    """내 계정에서 실제로 보이는 모델 목록을 가져온다."""
    url = f"{V1BETA_BASE_URL}?key={GEMINI_API_KEY}"
    r = requests.get(url, timeout=30)
    data = r.json()
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=data)
    return data.get("models", [])


def pick_text_model(models: list[dict]) -> str:
    """
    목록에서 generateContent 지원하는 텍스트 모델 하나 고른다.
    이름을 우리가 추측하는 게 아니라 실제로 열린 걸 쓴다.
    """
    for m in models:
        name = m.get("name")  # 예: "models/gemini-1.5-flash"
        supported = m.get("supportedGenerationMethods", [])
        # 텍스트 생성 지원하는 것만
        if "generateContent" in supported:
            return name
    # 하나도 없으면 에러
    raise HTTPException(
        status_code=500,
        detail={"error": "이 계정에서 generateContent 를 지원하는 v1beta 모델을 찾지 못했습니다."},
    )


def call_google_generate(model_name: str, payload: dict) -> dict:
    """선택된 모델 이름으로 generateContent 호출"""
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
    r = requests.post(url + f"?key={GEMINI_API_KEY}", json=payload, timeout=30)
    data = r.json()
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=data.get("error", data))
    return data


@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    # 1) 실제로 내 계정에 어떤 모델이 열려있는지 본다
    models = list_models_v1beta()

    # 2) 그중에서 generateContent 되는 거 하나 고른다
    model_name = pick_text_model(models)
    # model_name 은 이렇게 생겼을 거야: "models/gemini-1.5-flash" 처럼 전체 경로

    # 3) 이제 그걸로 생성
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}],
            }
        ]
    }

    result = call_google_generate(model_name, payload)

    # 4) 안전 파싱
    candidates = result.get("candidates", [])
    if not candidates:
        return {"response": ""}

    parts = candidates[0].get("content", {}).get("parts", [])
    text = parts[0].get("text", "") if parts else ""

    return {"response": text, "model_used": model_name}


# 디버깅용: 어떤 모델이 보이는지 바로 보기
@app.get("/models/v1beta", dependencies=[Depends(verify_api_key)])
async def models_v1beta():
    return {"models": list_models_v1beta()}


@app.get("/")
def root():
    return {"status": "ok", "version": "auto-pick-v1beta"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )










