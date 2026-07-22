"""
main.py — 카카오 i 오픈빌더 스킬 서버
일학습병행 원스탑 상담 AI (공동훈련센터 지원단)

3-Stage 응답 흐름
  1) quick_faq 키워드 매칭  → 즉시 multi-bubble 응답 (LLM 호출 X)
  2) OpenAI gpt-4o-mini 자유 답변 → multi-bubble 응답
  3) default_response (긴 안내 + 공식/지원단 자료 + footer) → multi-bubble 응답

설계 원칙
  - import·startup 시점에 어떤 사유로도 raise 하지 않음 (Render Exited 1 회피)
  - OPENAI_API_KEY가 비어있으면 LLM 분기는 safe-default로 폴백
  - 출력은 카카오톡 simpleText 990자 한도 내에서 multi-bubble로 분할해 누락 없이 표시
  - 모든 응답 끝에 [공식 참고 자료] / [지원단 참고자료] 자동 추가 (사용자 발화 무관)
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

# ── 선택 의존성: httpx 없으면 OpenAI 분기는 즉시 폴백 ──────────────
try:
    import httpx  # noqa: F401
    _HAS_HTTPX = True
except ImportError:
    sys.stderr.write("[chat] httpx missing → LLM 분기 비활성\n")
    _HAS_HTTPX = False

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로/파일 자동 탐색 — Render /opt/render/project/src 구조 대응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_SEARCH_PATHS = [
    os.environ.get("KNOWLEDGE_FILE"),
    str(Path(__file__).resolve().parent / "knowledge.json"),
    str(Path(__file__).resolve().parent.parent / "knowledge.json"),
    "/opt/render/project/src/knowledge.json",
    "/opt/render/project/knowledge.json",
    "./knowledge.json",
    "./app/knowledge.json",
]


def _pick_knowledge_path():
    for p in _SEARCH_PATHS:
        if p and os.path.exists(p):
            return p
    return None


_SAFE_DEFAULT = {
    "ai_router": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 400,
        "temperature": 0.4,
        "timeout_sec": 4.0,
        "system_prompt": (
            "당신은 일학습병행 원스탑 상담 AI입니다. 한국어 300자 이내. "
            "모르면 1:1 상담 연결 안내."
        ),
        "guardrails": "법률·세무·노동 분쟁 해석은 본부/원스탑 안내.",
    },
    "quick_faq": [
        {
            "id": "qa00",
            "keywords": ["일학습병행", "뭐예요", "정의", "제도"],
            "answer": (
                "일학습병행은 기업이 청년 등을 학습근로자로 채용해 NCS 기반의 훈련을 제공하는 제도입니다. "
                "(안전 모드 — knowledge.json 미적용 상태)"
            ),
        }
    ],
    "default_response": {
        "intro": (
            "안녕하세요. 일학습병행 원스탑 상담 AI입니다.\n\n"
            "지식 베이스를 불러오는 과정 중 지연이 발생하고 있습니다.\n"
        ),
        "extra_resources": [],
        "footer_messages": [
            "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."
        ],
    },
}


def _load_knowledge():
    path = _pick_knowledge_path()
    if not path:
        sys.stderr.write(
            "[chat] knowledge.json NOT FOUND in any search path → safe default 적용\n"
        )
        return _SAFE_DEFAULT
    try:
        with open(path, encoding="utf-8") as f:
            data = json.loads(f.read())
        sys.stderr.write(f"[chat] knowledge.json loaded OK: {path}\n")
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        sys.stderr.write(
            f"[chat] knowledge.json parse FAIL: {path} "
            f"→ {type(e).__name__}: {e} → safe default 적용\n"
        )
        return _SAFE_DEFAULT


# 한 줄: 모듈이 죽지 않는다.
_KNOWLEDGE = _load_knowledge()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 환경 변수 — 모두 안전 디폴트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = (
    _KNOWLEDGE.get("ai_router", {}).get("model")
    or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
)
OPENAI_TEMPERATURE = float(
    _KNOWLEDGE.get("ai_router", {}).get("temperature", 0.4)
)
OPENAI_MAX_TOKENS = int(
    _KNOWLEDGE.get("ai_router", {}).get("max_tokens", 400)
)
OPENAI_TIMEOUT = float(
    _KNOWLEDGE.get("ai_router", {}).get("timeout_sec")
    or os.environ.get("OPENAI_TIMEOUT", "4.0")
)
SYSTEM_PROMPT = (
    _KNOWLEDGE.get("ai_router", {}).get("system_prompt")
    or "일학습병행 안내 AI. 한국어 300자 이내."
)
GUARDRAILS = (
    _KNOWLEDGE.get("ai_router", {}).get("guardrails")
    or "법률·세무·노동 분쟁은 본부/원스탑 안내."
)

CACHE_TTL_SEC = int(os.environ.get("CACHE_TTL_SEC", "300"))
_CACHE: dict = {}

if not OPENAI_API_KEY:
    sys.stderr.write(
        "[chat] WARN: OPENAI_API_KEY env 미설정 → LLM 분기는 safe_default로 폴백\n"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 응답 빌더 — 자동 줄번호 prefix 없음, multi-bubble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _trim(text: str, limit: int = 990) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _join_bubble(*chunks: str) -> str:
    parts = [c.strip() for c in chunks if c and c.strip()]
    return _trim("\n\n".join(parts), 990)


def _build_extra_bubble(default_response: dict) -> str:
    """[공식 참고 자료] + [지원단 참고자료] 한 bubble로."""
    sections = default_response.get("extra_resources") or []
    if not sections:
        return ""
    lines: list[str] = []
    for sec in sections:
        header = sec.get("header", "").strip()
        if header:
            lines.append(header)
        for item in sec.get("items", []):
            label = item.get("label", "").strip()
            url = item.get("url", "").strip()
            if label and url:
                lines.append(f"• {label}\n  {url}")
            elif url:
                lines.append(f"• {url}")
    return _trim("\n".join(lines), 990)


def _build_footer_bubble(default_response: dict) -> str:
    footers = default_response.get("footer_messages") or []
    return _trim("\n".join(footers), 990) if footers else ""


def build_multi_bubble_reply(intro: str, default_response: dict):
    """intro 텍스트 + 공식/지원단 bubble + footer bubble (있으면)"""
    intro_clean = (intro or "").strip()
    extra = _build_extra_bubble(default_response)
    footer = _build_footer_bubble(default_response)

    outputs: list[dict] = []
    if intro_clean:
        outputs.append({"simpleText": {"text": _trim(intro_clean, 990)}})
    if extra:
        outputs.append({"simpleText": {"text": extra}})
    if footer:
        outputs.append({"simpleText": {"text": footer}})
    if not outputs:
        outputs.append(
            {"simpleText": {"text": "안녕하세요. 일학습병행 원스탑 상담 AI입니다."}}
        )
    return {"version": "2.0", "template": {"outputs": outputs}}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 키워드 매칭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _match_faq(utterance: str, faq_list):
    if not utterance or not faq_list:
        return None
    text = utterance.strip()
    for item in faq_list:
        for kw in item.get("keywords", []):
            if kw and kw in text:
                return item
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI 호출 — 예외는 모두 내부에서 흡수, 호출자엔 False/null 반환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def _call_openai(utterance: str) -> str:
    if not OPENAI_API_KEY or not _HAS_HTTPX:
        return ""
    body = {
        "model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + GUARDRAILS},
            {"role": "user", "content": utterance},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )
            if r.status_code != 200:
                sys.stderr.write(
                    f"[chat] openai http {r.status_code}: {r.text[:200]}\n"
                )
                return ""
            data = r.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        sys.stderr.write(f"[chat] openai fail {type(e).__name__}: {e}\n")
        return ""
    except Exception as e:
        sys.stderr.write(f"[chat] openai unexpect {type(e).__name__}: {e}\n")
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 앱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI()


@app.get("/")
def root():
    return {
        "service": "kakao-ai-skill",
        "status": "ok",
        "openai_key_set": bool(OPENAI_API_KEY),
        "faq_count": len(_KNOWLEDGE.get("quick_faq", [])),
    }


@app.get("/favicon.ico")
def favicon():
    return {}


@app.post("/api/chat")
async def chat(req: Request):
    try:
        payload = await req.json()
    except Exception:
        payload = {}

    utterance = (
        (payload.get("userRequest") or {}).get("utterance")
        or payload.get("utterance")
        or ""
    ).strip()

    # ── 캐시 (TTL, 동일 발화 단축) ────────────────────────
    now = time.time()
    if CACHE_TTL_SEC > 0 and utterance in _CACHE:
        ts, val = _CACHE[utterance]
        if now - ts < CACHE_TTL_SEC:
            return JSONResponse(content=val)

    faq_list = _KNOWLEDGE.get("quick_faq", [])
    default_response = _KNOWLEDGE.get(
        "default_response",
        _SAFE_DEFAULT["default_response"],
    )

    intro_text = ""

    # ── Stage 1: FAQ 매칭 ────────────────────────────────
    hit = _match_faq(utterance, faq_list)
    if hit:
        sys.stderr.write(f"[chat] faq-hit id={hit.get('id')} utterance={utterance[:20]!r}\n")
        intro_text = hit.get("answer", "")
    else:
        # ── Stage 2: OpenAI 자유 답변 ──────────────────────
        llm_answer = ""
        if OPENAI_API_KEY and _HAS_HTTPX:
            llm_answer = await _call_openai(utterance)
        if llm_answer:
            sys.stderr.write(f"[chat] llm-ok len={len(llm_answer)} utterance={utterance[:20]!r}\n")
            intro_text = llm_answer
        else:
            # ── Stage 3: default ──────────────────────────
            if not OPENAI_API_KEY:
                sys.stderr.write("[chat] llm-fail OPENAI_API_KEY not set → default\n")
            else:
                sys.stderr.write("[chat] llm-fail (timeout/4xx) → default\n")
            intro_text = default_response.get("intro", _SAFE_DEFAULT["default_response"]["intro"])

    body = build_multi_bubble_reply(intro_text, default_response)

    if CACHE_TTL_SEC > 0 and utterance:
        _CACHE[utterance] = (now, body)

    return JSONResponse(content=body)


# Render가 자동으로 `uvicorn main:app --host 0.0.0.0 --port 10000`로 띄움
