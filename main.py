"""
main.py — 카카오 i 오픈빌더 스킬 서버 (OpenAI 전용, FAQ 없음)
한국기술교육대학교 일학습병행 공동훈련센터 지원단 / 원스탑 상담 AI

설계
  - 모든 사용자 발화를 즉시 OpenAI Chat Completions API 로 전달
  - FAQ · 정적 lookup · knowledge.json · 매칭 로직 일체 없음
  - 응답은 카톡 simpleText 990자 한도 내에서 multi-bubble
        bubble[0] = OpenAI 답변
        bubble[1] = [공식 참고 자료] (법령 2건)
        bubble[2] = [지원단 참고자료] (가이드 1건) + footer "1:1 상담 연결"
  - import · startup 시점에 어떤 사유로도 raise 하지 않음
        → Render "Exited with status 1" 재발 방지
  - OPENAI_API_KEY env 비어 있으면 safe 멘트로 자동 폴백
  - timeout = 4.0s, 캐시 TTL = 300s, 동기 클라이언트(SDK 내부 threadpool)

Render 빌드 로그 기준 호환 버전
  fastapi 0.115.0
  uvicorn[standard] 0.32.0
  httpx 0.27.2
  openai 1.50.0
"""

import os
import sys
import time
from typing import Optional

# ── 외부 의존성 — 누락되더라도 서버는 살아남도록 ─────────────
try:
    from openai import (
        OpenAI,
        APIError,
        APITimeoutError,
        RateLimitError,
        AuthenticationError,
    )
    _HAS_OPENAI = True
except ImportError:
    sys.stderr.write("[chat] openai 패키지 누락 → safe 멘트로 폴백\n")
    _HAS_OPENAI = False

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ── 환경 변수 (전부 안전 디폴트) ─────────────────────────────
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL       = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.4"))
OPENAI_MAX_TOKENS  = int(os.environ.get("OPENAI_MAX_TOKENS", "500"))
OPENAI_TIMEOUT     = float(os.environ.get("OPENAI_TIMEOUT", "4.0"))
CACHE_TTL_SEC      = int(os.environ.get("CACHE_TTL_SEC", "300"))

SYSTEM_PROMPT = (
    "당신은 한국기술교육대학교 일학습병행 공동훈련센터 지원단의 '원스탑 상담 AI'입니다.\n"
    "역할: 일학습병행 제도(학습기업·학습근로자·기업현장교사·내부평가·외부평가·실시신고·운영규칙)에 대한 1차 안내.\n"
    "원칙:\n"
    "  - 한국어, 정중·간결하게 답변 (500자 이내).\n"
    "  - 근거 있는 제도 안내만, 추측·단정 금지.\n"
    "  - 학습기업·학습근로자·공동훈련센터·내부평가·지원금 등 인접 주제는 종합적으로 1차 안내.\n"
    "  - 정확한 제도 해석·법령 자문·세무·노동 분쟁은 '한국산업인력공단 본부 또는 원스탑 상담센터 전문상담'으로 안내.\n"
    "  - 민감정보(전화번호·주소·계좌)는 절대 생성 금지.\n"
    "  - 모든 답변 끝에 '자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다.' 한 줄을 포함."
)

_CACHE: dict = {}


# ── OpenAI 클라이언트 lazy-init (key 없으면 생성 안 함) ────────
_client: Optional["OpenAI"] = None

def _get_client():
    global _client
    if not OPENAI_API_KEY or not _HAS_OPENAI:
        return None
    if _client is None:
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)
            sys.stderr.write(f"[chat] OpenAI client ready model={OPENAI_MODEL}\n")
        except Exception as e:
            sys.stderr.write(f"[chat] OpenAI client init fail: {type(e).__name__}: {e}\n")
            _client = None
    return _client


# ── OpenAI 호출 — 모든 예외는 내부에서 흡수 ─────────────────
def _call_openai(utterance: str) -> str:
    client = _get_client()
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": utterance.strip()},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except AuthenticationError as e:
        sys.stderr.write(f"[chat] openai auth fail: {e}\n")
        return ""
    except RateLimitError as e:
        sys.stderr.write(f"[chat] openai rate-limit: {e}\n")
        return ""
    except APITimeoutError as e:
        sys.stderr.write(f"[chat] openai timeout: {e}\n")
        return ""
    except APIError as e:
        sys.stderr.write(f"[chat] openai api error: {e}\n")
        return ""
    except Exception as e:
        sys.stderr.write(f"[chat] openai unexpect {type(e).__name__}: {e}\n")
        return ""


# ── Multi-bubble 응답 빌더 ─────────────────────────────────
OFFICIAL_RESOURCES = [
    {
        "label": "산업현장 일학습병행 지원에 관한 법률",
        "url":   "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=210288",
    },
    {
        "label": "일학습병행 운영규칙 (고시 제2025-99호)",
        "url":   "https://www.law.go.kr/LSW/admRulLsInfoP.do?admRulSeq=2100000272052",
    },
]
SUPPORT_RESOURCES = [
    {
        "label": "공동훈련센터 전담자 업무가이드",
        "url":   "https://www.swlc.or.kr/resources/_Etc/ebook/211012/main.html",
    },
]
FOOTER_MSG = "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."


def _trim(text: str, limit: int = 990) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _build_resource_bubble(header: str, items: list) -> str:
    lines = [header]
    for it in items:
        lines.append(f"• {it['label']}\n  {it['url']}")
    return _trim("\n".join(lines), 990)


def build_multi_bubble(answer: str) -> dict:
    outputs = [
        {"simpleText": {"text": _trim(answer, 990)}},
        {"simpleText": {"text": _build_resource_bubble("[공식 참고 자료]", OFFICIAL_RESOURCES)}},
        {
            "simpleText": {
                "text": _build_resource_bubble("[지원단 참고자료]", SUPPORT_RESOURCES) + "\n\n" + FOOTER_MSG
            }
        },
    ]
    return {"version": "2.0", "template": {"outputs": outputs}}


# ── FastAPI ────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
def root():
    return {
        "service":      "kakao-ai-skill-openai",
        "status":       "ok",
        "openai_key":   bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
    }


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

    # ── 캐시 (동일 발화 TTL 단축) ──────────────────────────
    now = time.time()
    if CACHE_TTL_SEC > 0 and utterance in _CACHE:
        ts, body = _CACHE[utterance]
        if now - ts < CACHE_TTL_SEC:
            return JSONResponse(content=body)

    answer = ""
    if OPENAI_API_KEY and _HAS_OPENAI and utterance:
        sys.stderr.write(
            f"[chat] openai-req model={OPENAI_MODEL} len={len(utterance)} text={utterance[:30]!r}\n"
        )
        answer = _call_openai(utterance)
        if answer:
            sys.stderr.write(f"[chat] openai-ok len={len(answer)}\n")
        else:
            sys.stderr.write("[chat] openai-fail → safe 멘트로 fallback\n")
    else:
        if not OPENAI_API_KEY:
            sys.stderr.write("[chat] OPENAI_API_KEY 미설정 → safe 멘트로 fallback\n")
        elif not _HAS_OPENAI:
            sys.stderr.write("[chat] openai 패키지 누락 → safe 멘트로 fallback\n")
        elif not utterance:
            sys.stderr.write("[chat] 빈 발화 → safe 멘트로 fallback\n")

    if not answer:
        answer = (
            "안녕하세요. 일학습병행 원스탑 상담 AI입니다.\n\n"
            "OpenAI 서비스가 일시적으로 응답하지 않아 자유 답변을 표시하지 못했습니다.\n"
            "아래 공식 자료와 1:1 상담 연결을 통해 정확한 안내를 받으실 수 있습니다."
        )

    body = build_multi_bubble(answer)

    if CACHE_TTL_SEC > 0 and utterance:
        _CACHE[utterance] = (now, body)

    return JSONResponse(content=body)
