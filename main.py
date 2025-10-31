import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import fastapi.responses

# ============================================================
# 1. 환경 변수 로드
# ============================================================
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN", "defaultapitoken")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 가 .env 에 설정되어 있지 않습니다.")

# ============================================================
# 2. Google Generative Language API Base URLs
#    - 텍스트는 v1 + 최신 이름으로 시도
#    - 스트리밍은 v1beta 를 쓰는 게 일반적이니 v1beta로
# ============================================================
V1_BASE_URL = "https://generativelanguage.googleapis.com/v1/models"
V1BETA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ============================================================
# 3. FastAPI 앱 + CORS
# ============================================================
app = FastAPI()

# 프런트에서도 이 이름으로 보내라고 생각하면 됨: X-API-KEY
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포에서는 특정 도메인으로 제한
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
# 5. 공통: 구글 API POST
# ============================================================
def call_google_api(url: str, payload: dict) -> dict:
    """공통 POST. 200 아니면 구글이 준 에러를 그대로 던짐."""
    resp = requests.post(
        url + f"?key={GEMINI_API_KEY}",
        json=payload,
        timeout=30,
    )
    data = resp.json()
    if resp.status_code != 200:
        # detail 에 구글 에러를 심어서 바로 프런트에서 볼 수 있게
        raise HTTPException(status_code=resp.status_code, detail=data.get("error", data))
    return data


# ============================================================
# 6. 일반 채팅 (한 번에 응답)
# ============================================================
@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    일반 텍스트 생성. 응답을 한 번에 받아서 내려준다.
    v1 + gemini-1.5-pro
    """
    TEXT_URL = f"{V1_BASE_URL}/gemini-1.5-pro:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}],
            }
        ]
    }

    result = call_google_api(TEXT_URL, payload)

    # 안전 파싱
    candidates = result.get("candidates", [])
    if not candidates:
        return {"response": ""}

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text = parts[0].get("text", "") if parts else ""

    return {"response": text}


# ============================================================
# 7. 스트리밍 채팅 (SSE)
# ============================================================
@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    """
    스트리밍으로 토큰을 바로바로 내려보내는 엔드포인트.
    - v1beta + streamGenerateContent 사용
    - 응답은 text/event-stream (SSE)
    프런트에서는:
        const es = new EventSource('/chat/stream');
    이런 식이 아니라,
        fetch('/chat/stream', { method: 'POST', body: ... })
    로 받으려면 ReadableStream 파싱 필요.
    """

    # 스트리밍은 v1beta로
    STREAM_URL = f"{V1BETA_BASE_URL}/gemini-1.5-pro:streamGenerateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}],
            }
        ]
    }

    def event_stream():
        # requests 의 stream=True 로 한 줄씩 읽어온다
        with requests.post(
            STREAM_URL + f"?key={GEMINI_API_KEY}",
            json=payload,
            stream=True,
        ) as r:
            if r.status_code != 200:
                # 스트림이기도 하지만 에러면 한 번만 내려보내고 끝낸다
                try:
                    err_json = r.json()
                except Exception:
                    err_json = {"error": "stream error"}
                # SSE 형식으로 에러 내려보냄
                yield f"data: {err_json}\n\n"
                return

            # 구글은 data: ... 형태로 줄줄이 내려보냄
            for line in r.iter_lines():
                if not line:
                    continue
                # 보통 b'data: {...}' 이렇게 옴
                if line.startswith(b"data:"):
                    # 그대로 클라이언트로 밀어준다
                    # text로 바꾸고 끝에 \n\n(SSE 규칙) 붙여줌
                    yield line.decode("utf-8") + "\n\n"

            # 끝났다는 표시 (선택)
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# 8. 이미지/멀티모달
# ============================================================
@app.post("/generate-image", dependencies=[Depends(verify_api_key)])
async def generate_image(request: Request):
    """
    프런트가 payload 전체를 넘겨주면 그걸 그대로 구글에 던진다.
    v1beta + gemini-1.5-flash
    """
    IMAGE_URL = f"{V1BETA_BASE_URL}/gemini-1.5-flash:generateContent"
    body = await request.json()
    payload = body.get("payload")
    if not payload:
        raise HTTPException(status_code=400, detail={"error": "payload is required"})

    result = call_google_api(IMAGE_URL, payload)
    return result


# ============================================================
# 9. 모델 목록 확인용 (디버깅)
# ============================================================
@app.get("/models/v1", dependencies=[Depends(verify_api_key)])
async def list_models_v1():
    url = "https://generativelanguage.googleapis.com/v1/models"
    resp = requests.get(url + f"?key={GEMINI_API_KEY}", timeout=30)
    data = resp.json()
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=data)
    return data


@app.get("/models/v1beta", dependencies=[Depends(verify_api_key)])
async def list_models_v1beta():
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url + f"?key={GEMINI_API_KEY}", timeout=30)
    data = resp.json()
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=data)
    return data


# ============================================================
# 10. 공통 HTTPException 핸들러
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
        "version": "with_streaming",
    }








