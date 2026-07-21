"""
카카오 i 오픈빌더 — Thin-pointer + LLM (search 기준일 2026-07-21)
- quick_faq 키워드 매칭 우선 (즉시 응답, 0.1s)
- 매칭 실패 시 OpenAI gpt-4o-mini 호출 (system_prompt는 knowledge.json에서)
- 가드레일·200자 제한·상담연결 fallback 포함
- 오픈소스 SDK: openai 1.x (검증된 httpx==0.27.2 핀)
"""
import os, json, re, time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

# ── 설정 ────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 비용/속도 균형, 한국어 OK
KNOWLEDGE_FILE   = Path(os.getenv("KNOWLEDGE_FILE",
                                  "/opt/render/project/src/knowledge.json"))
CHAR_LIMIT       = 200          # 시스템 프롬프트 명시 200자 제한
LLM_TIMEOUT_SEC  = 4.0          # 카톡 응답 권장 5초 이내
LLM_MAX_TOKENS   = 300          # 한국어 약 200자 + 안전마진

client = AsyncOpenAI(api_key=OPENAI_API_KEY,
                     timeout=LLM_TIMEOUT_SEC,
                     max_retries=1)  # 1회 자동 재시도

# ── knowledge.json 로드 ────────────────────────────────
def _load_knowledge() -> dict:
    if not KNOWLEDGE_FILE.exists():
        return {"sources": [], "quick_faq": [], "ai_router": {}, "default_response": {}}
    try:
        return json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": [], "quick_faq": [], "ai_router": {}, "default_response": {}}

KNOWLEDGE = _load_knowledge()
QUICK_FAQ = KNOWLEDGE.get("quick_faq", [])
ROUTER    = KNOWLEDGE.get("ai_router", {})
DEFAULT   = KNOWLEDGE.get("default_response", {})

# 캐시(질문 → 답변, 5분 TTL)
CACHE: dict = {}
CACHE_TTL = 300

# ── 시스템 프롬프트 빌더 ───────────────────────────────
SYSTEM_PROMPT_DEFAULT = (
    "당신은 일학습병행 안내 AI입니다. "
    "사용자 질문에 200자 이내, 정중·간결하게 한국어로 답변하세요."
)

GUARDRAILS_DEFAULT = (
    "법률 자문·세무 자문·노동 분쟁 해석은 '한국산업인력공단 본부 또는 "
    "원스탑 상담센터 전문상담 필요'로 안내. 근거 없는 추측 금지."
)

def build_system_prompt() -> str:
    """ai_router.system_prompt + guardrails 조합"""
    sp = (ROUTER.get("system_prompt") or SYSTEM_PROMPT_DEFAULT).strip()
    gr = (ROUTER.get("guardrails")    or GUARDRAILS_DEFAULT).strip()
    return f"{sp}\n\n[운영 규칙 - 반드시 준수]\n{gr}\n\n- 답변 길이: {CHAR_LIMIT}자 이내\n- 200자 초과 시 truncate\n- 모호하거나 확인이 어려운 경우: 1:1 상담 연결 안내"

# ── 한 줄 빠른 매칭(키워드 부분 일치) ──────────────────
def match_quick_faq(utterance: str):
    norm = utterance.strip()
    # 1) 정확 매칭
    for q in QUICK_FAQ:
        if norm in (q.get("keywords") or []):
            return q
    # 2) 부분 매칭
    for q in QUICK_FAQ:
        for kw in (q.get("keywords") or []):
            if kw and kw in norm:
                return q
    return None

# ── OpenAI 호출 ────────────────────────────────────────
async def call_llm(utterance: str, faq_hint: str = "") -> str:
    """system 프롬프트는 JSON에서, user는 발화 + FAQ 힌트로 구성"""
    system_prompt = build_system_prompt()

    user_text = utterance
    if faq_hint:
        user_text = (
            f"[참고 - 같은 의도의 FAQ 답변 힌트]\n{faq_hint}\n\n"
            f"[사용자 질문]\n{utterance}\n\n"
            "[답변 응답 조건]\n"
            f"- 200자 이내 한국어 정중·간결 답변\n"
            f"- 위에 FAQ 힌트가 있으니 그 결을 따라 짧고 정확하게 작성\n"
            f"- 1:1 상담 연결 필요 안내: 정확한 안내가 어려울 때 1회만 포함\n"
        )

    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.4,        # 검색으로 확인, OpenAI 1.x 유효 파라미터
        max_tokens=LLM_MAX_TOKENS,  # gpt-4o-mini에서 유효 (검색 검증)
    )
    text = (resp.choices[0].message.content or "").strip()
    return truncate(text, CHAR_LIMIT + 60)  # 200자 가드레일 강제

def truncate(text: str, hard: int = CHAR_LIMIT) -> str:
    if len(text) <= hard:
        return text
    return text[: hard - 1] + "…"

def build_simple_text(t: str):
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": t}}]}}

# ── App ─────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def health():
    """health: mode + 상태 노출"""
    has_key = bool(OPENAI_API_KEY)
    return {
        "status": "ok",
        "service": "kakao-ai-skill",
        "mode": "thin-pointer + LLM-fallback",
        "model": OPENAI_MODEL if has_key else None,
        "quick_faq_count": len(QUICK_FAQ),
        "system_prompt_chars": len(build_system_prompt()),
        "openai_key_set": has_key,
    }

@app.post("/api/chat")
async def chat(request: Request):
    """메인 스킬 엔드포인트"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content=build_simple_text("요청 처리 중 오류가 발생했어요."))

    utt = (body.get("userRequest") or {}).get("utterance", "").strip()
    if not utt:
        return JSONResponse(content=build_simple_text("안녕하세요! 무엇을 도와드릴까요?"))

    # 캐시 check
    cache_key = utt.lower()
    if cache_key in CACHE and time.time() - CACHE[cache_key]["t"] < CACHE_TTL:
        return JSONResponse(content=build_simple_text(CACHE[cache_key]["text"]))

    # 1) quick_faq 정확/부분 매칭 (즉시, LLM 미호출)
    faq_match = match_quick_faq(utt)
    if faq_match and faq_match.get("answer"):
        ans = faq_match["answer"]
        CACHE[cache_key] = {"t": time.time(), "text": ans}
        return JSONResponse(content=build_simple_text(ans))

    # 2) LLM 호출 (system_prompt ← knowledge.json)
    #    faq_hint: 가장 잘 어울리는 1개 답변을 user 메시지에 함께 전달 (답변 일관성 ↑)
    faq_hint = faq_match["answer"] if faq_match else ""

    if not OPENAI_API_KEY:
        return JSONResponse(content=build_simple_text(
            DEFAULT.get("intro", "1:1 상담 연결 안내: 잠시 후 다시 시도하시거나 1:1 상담 연결을 통해 문의해 주세요.")
        ))

    try:
        ans = await call_llm(utt, faq_hint=faq_hint)
    except Exception:
        # LLM 실패 시: 캐스케이드 fallback
        return JSONResponse(content=build_simple_text(
            DEFAULT.get("intro", "안내를 정리 중이에요. 잠시 후 다시 시도하시거나 1:1 상담 연결을 통해 문의해 주세요.")
        ))

    # 200자 가드레일 강제 트림
    ans = truncate(ans, CHAR_LIMIT)

    CACHE[cache_key] = {"t": time.time(), "text": ans}
    return JSONResponse(content=build_simple_text(ans))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
