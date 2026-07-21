"""
카카오 i 오픈빌더 AI 자동응답 스킬 서버 (FastAPI + Google Gemini)
Render 환경변수: GEMINI_API_KEY
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash-latest:generateContent"
)

SYSTEM_PROMPT = """당신은 한국기술교육대학교 일학습병행 공동훈련센터 지원단의 원스탑 상담센터 상담 안내 챗봇입니다.
- 정중하고 간결하게 답변합니다 (200자 이내).
- 정확한 정보가 없으면 '관할 공단 지부·지사 혹은 담당 컨설턴트 문의'라고 답합니다.
- 질문 의도를 파악해 필요한 안내를 제공합니다."""


async def call_gemini(user_utterance: str) -> str:
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": f"{SYSTEM_PROMPT}\n\n사용자 질문: {user_utterance}"}]}
        ],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300},
    }
    async with httpx.AsyncClient(timeout=3.0) as client:
        r = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def trim_to_limit(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_simple_text_reply(text: str):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": trim_to_limit(text)}}
            ],
            "quickReplies": [
                {"label": "📅 훈련 일정 안내", "action": "message",
                 "messageText": "훈련 일정 알려줘"},
                {"label": "📞 상담 연결", "action": "message",
                 "messageText": "상담사 연결해줘"},
            ],
        },
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "kakao-ai-skill"}


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content=build_simple_text_reply("요청을 해석할 수 없습니다. 다시 시도해 주세요."),
        )

    user_request = body.get("userRequest", {})
    utterance = user_request.get("utterance", "").strip()

    if not utterance:
        return JSONResponse(
            content=build_simple_text_reply("안녕하세요! 무엇을 도와드릴까요?")
        )

    try:
        ai_reply = await call_gemini(utterance)
    except Exception:
        ai_reply = (
            "지금 답변이 지연되고 있어요. 잠시 후 다시 시도하시거나, "
            "센터 대표번호로 문의 부탁드립니다."
        )

    return JSONResponse(content=build_simple_text_reply(ai_reply))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
