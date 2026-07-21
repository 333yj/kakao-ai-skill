"""
카카오 i 오픈빌더 — OpenRouter 무료 모델 연동 (검색 기준일: 2026-07-21)
- OpenAI 호환 /v1/chat/completions 엔드포인트
- 무료 모델은 OpenRouter 대시보드에서 :free 마크 확인
- pip install openai 또는 표준 라이브러리 그대로 동작
"""
import os, json, re, time, urllib.request, urllib.error
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── 설정 ────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")     # ⚠️ 새로 발급
# 무료 모델 후보 3개 — 가용성 변동 있음, 점진적 fallback
OPENROUTER_MODEL_HINT = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-chat-v3:free",
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

KNOWLEDGE_FILE = Path(os.getenv("KNOWLEDGE_FILE", "/opt/render/project/src/knowledge.json"))

# ── 지식 코퍼스 로드 ───────────────────────────────────
def _load_knowledge() -> dict:
    if not KNOWLEDGE_FILE.exists():
        return {"sources": [], "chunks": [], "ai_router": {}, "default_response": {}}
    return json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))

KNOWLEDGE = _load_knowledge()
CORPUS    = KNOWLEDGE.get("chunks", [])
ROUTER    = KNOWLEDGE.get("ai_router", {})
DEFAULT   = KNOWLEDGE.get("default_response", {})

CACHE: dict = {}
CACHE_TTL = 300

# ── 시스템 프롬프트 (fallback 포함) ────────────────────
SYSTEM_PROMPT_FALLBACK = (
    "당신은 한국기술교육대학교 일학습병행 공동훈련센터 지원단의 상담 안내 챗봇입니다. "
    "한국어로, 200자 이내, 정중·간결하게 답변하세요. "
    "참고 문서에 없는 내용은 추정하지 말고 '1:1 상담연결 필요'로 답하세요."
)

def build_system_prompt() -> str:
    sp = ROUTER.get("system_prompt", "").strip()
    if not sp:
        return SYSTEM_PROMPT_FALLBACK
    extra = ROUTER.get("guardrails", "").strip()
    return sp + ("\n\n[운영 규칙]\n" + extra if extra else "")

# ── RAG: 코퍼스 검색 ───────────────────────────────────
def search_corpus(utterance: str, top_k: int = 3):
    if not CORPUS:
        return []
    tokens = [t for t in re.split(r"\s+", utterance) if len(t) >= 2]
    scored = []
    for c in CORPUS:
        head = c.get("text", "")[:150]
        text = c.get("text", "")
        score = sum(head.count(t) * 2 + text.count(t) for t in tokens)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

# ── OpenRouter 호출 (OpenAI 호환) ──────────────────────
def call_openrouter(utterance: str, chunks: list) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 없습니다.")

    context_text = "\n".join(f"[참고{i+1}]\n{c['text'][:600]}" for i, c in enumerate(chunks))

    user_prompt = (
        "다음 참고 문서 발췌를 근거로 사용자 질문에 답변하세요.\n"
        "문서에 없는 내용은 추정하지 말고 담당 컨설턴트 안내로 응답하세요.\n\n"
        f"{context_text}\n\n[질문]\n{utterance}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://kakao-ai-skill.onrender.com",  # OpenRouter 권장
        "X-Title":       "Kakao AI Skill",
    }

    last_err = None
    # 무료 모델 다운 시 자동 fallback
    for model in [OPENROUTER_MODEL_HINT] + [m for m in OPENROUTER_MODELS if m != OPENROUTER_MODEL_HINT]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens":  400,
        }
        try:
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            txt = (data["choices"][0]["message"]["content"] or "").strip()
            if txt:
                return txt
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.reason}"
            if e.code in (429, 500, 503):
                time.sleep(0.4)
                continue
        except Exception as e:
            last_err = str(e)
            break

    raise RuntimeError(f"OpenRouter 호출 실패: {last_err}")

# ── 응답 빌더 ──────────────────────────────────────────
def trim(t: str, limit: int = 480) -> str:
    return t if len(t) <= limit else t[: limit - 1] + "…"

def build_simple_text(t: str):
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": trim(t)}}]}}

# ── 앱 ─────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def health():
    return {
        "status": "ok",
        "provider": "openrouter-free",
        "model_hint": OPENROUTER_MODEL_HINT,
        "chunks_loaded": len(CORPUS),
        "cache_size": len(CACHE),
        "key_set": bool(OPENROUTER_API_KEY),
    }

@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content=build_simple_text("요청 처리 중 오류가 발생했어요."))

    utt = (body.get("userRequest") or {}).get("utterance", "").strip()
    if not utt:
        return JSONResponse(content=build_simple_text("안녕하세요! AI 자동 상담봇입니다. 무엇을 도와드릴까요?"))

    cache_key = utt.lower()
    if cache_key in CACHE and time.time() - CACHE[cache_key]["t"] < CACHE_TTL:
        return JSONResponse(content=build_simple_text(CACHE[cache_key]["text"]))

    chunks = search_corpus(utt, top_k=3)
    try:
        ans = call_openrouter(utt, chunks)
    except Exception:
        if chunks:
            ans = chunks[0]["text"][:280] + "\n\n자세한 내용은 1:1 상담을 통해 문의해 주세요."
        else:
            ans = (DEFAULT.get("intro", "자세한 내용은 1:1 상담을 통해 문의해 주세요."))

    CACHE[cache_key] = {"t": time.time(), "text": ans}
    return JSONResponse(content=build_simple_text(ans))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
