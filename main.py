import os
import requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import fastapi.responses

# ============================================================
# 1. 환경 변수 로드 (.env 에서 가져오기)
# ============================================================
# Render 구조에 따라 경로가 다를 수 있으니, 없으면 그냥 현재 디렉토리의 .env 사용
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN", "defaultapitoken")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 가 설정되어 있지 않습니다. .env 에 추가하세요.")

# ============================================================
# 2. Google Generative Language API 기본 URL
#    👉 너 환경에서는 v1에서 계속 404가 났으므로
#       '기본값'을 v1beta로 두고 시작한다.
# ============================================================
V1BETA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# 텍스트: v1beta + gemini-1.5-pro
TEXT_MODEL_NAME = "gemini-1.5-pro"
# 스트리밍: v1beta + gemini-1.5-pro:streamGenerateContent
STREAM_MODEL_NAME = "gemini-1.5-pro"
# 이미지/멀티모달: v1beta + gemini-1.5-flash
IMAGE_MODEL_NAME = "gemini-1.5-flash"

# ============================================================
# 3. FastAPI 앱 & CORS
# ============================================================
app = FastAPI()

# 프런트에서 X-API-KEY 로 보내고 있으니 그대로 맞춘다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 도메인으로 제한!
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-KEY", "Authorization"],
)

api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail={"error": "Invalid API Token"})


# ============================================================
# 4. 요청 모델
# ============================================================
class ChatRequest(BaseModel):
    prompt: str


# ============================================================
# 5. 공통 호출 함수
# ============================================================
def call_google_api(url: str, payload: dict) -> dict:
    """Google Generative Language API 호출 공통 함수"""
    resp = requests.post(
        url + f"?key={GEMINI_API_KEY}",
        json=payload,
        timeout=30,
    )
    # 구글은 항상 JSON 로 내려오므로 바로 파싱
    data = resp.json()
    if resp.status_code != 200:
        # 프런트에서 바로 볼 수 있게 detail 에 구글 에러 넣기
        raise HTTPException(status_code=resp.status_code, detail=data.get("error", data))
    return data


# ============================================================
# 6. 일반 채팅 (/chat)
#    👉 v1beta + gemini-1.5-pro 로 '항상' 호출
#    👉 네가 올린 HTML이랑 바로 붙도록 응답 {"response": "..."} 로 통일
# ============================================================
@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    url = f"{V1BETA_BASE_URL}/{TEXT_MODEL_NAME}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}]
            }
        ]
    }

    result = call_google_api(url, payload)

    # 안전 파싱
    candidates = result.get("candidates", [])
    if not candidates:
        return {"response": ""}

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text = parts[0].get("text", "") if parts else ""

    return {"response": text}


# ============================================================
# 7. 스트리밍 채팅 (/chat/stream)
#    👉 나중에 프런트에서 쓰라고 같이 넣어둠
# ============================================================
from fastapi.responses import StreamingResponse

@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    stream_url = f"{V1BETA_BASE_URL}/{STREAM_MODEL_NAME}:streamGenerateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}]
            }
        ]
    }

    def event_stream():
        with requests.post(
            stream_url + f"?key={GEMINI_API_KEY}",
            json=payload,
            stream=True,
        ) as r:
            # 스트림에서도 에러일 수 있으니 먼저 체크
            if r.status_code != 200:
                try:
                    err_json = r.json()
                except Exception:
                    err_json = {"error": "stream error"}
                yield f"data: {err_json}\n\n"
                return

            # 정상일 때는 data: ... 줄 단위로 그대로 밀어준다
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data:"):
                    yield line.decode("utf-8") + "\n\n"

            # 끝 표시
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# 8. 이미지 / 멀티모달 (/generate-image)
#    👉 이건 네가 앞에서 만든 구조 그대로 살림
# ============================================================
@app.post("/generate-image", dependencies=[Depends(verify_api_key)])
async def generate_image(request: Request):
    body = await request.json()
    payload = body.get("payload")
    if not payload:
        raise HTTPException(status_code=400, detail={"error": "payload is required"})

    url = f"{V1BETA_BASE_URL}/{IMAGE_MODEL_NAME}:generateContent"
    result = call_google_api(url, payload)
    return result


# ============================================================
# 9. 모델 리스트 확인 (디버깅용)
#    👉 여기서 실제로 뜨는 이름을 보면, 나중에 모델 바꿀 때 404 없이 바로 알 수 있음
# ============================================================
@app.get("/models/v1beta", dependencies=[Depends(verify_api_key)])
async def list_models_v1beta():
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url + f"?key={GEMINI_API_KEY}", timeout=30)
    data = resp.json()
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=data)
    return data


# ============================================================
# 10. 공통 에러 핸들러
# ============================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ============================================================
# 11. 루트
# ============================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "beta-only",
    }










