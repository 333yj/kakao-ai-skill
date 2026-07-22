"""
main.py — 카카오 i 오픈빌더 스킬 서버
한국기술교육대학교 일학습병행 공동훈련센터 지원단 / 원스탑 상담 AI

설계
  - 모든 사용자 발화를 즉시 OpenAI Chat Completions API 로 전달
  - OpenAI 호환 base_url (OPENAI_BASE_URL env) 자동 인식
        → 비우면 공식 OpenAI, 채우면 Groq/Gemini 호환 라우터로 라우팅
  - 응답은 카톡 simpleText 990자 한도 내에서 multi-bubble
        bubble[0] = OpenAI 답변 (줄바꿈 보정 적용)
        bubble[1] = [공식 참고 자료] (법령 2건)
        bubble[2] = [지원단 참고자료] (가이드 1건) + footer "1:1 상담 연결"
  - import · startup 시점에 어떤 사유로도 raise 하지 않음
        → Render "Exited with status 1" 재발 방지
  - OPENAI_API_KEY env 비어 있으면 safe 멘트로 자동 폴백
  - timeout = 4.0s, 캐시 TTL = 300s

Render 빌드 로그 기준 호환 버전
  fastapi 0.115.0
  uvicorn[standard] 0.32.0
  httpx 0.27.2
  openai 1.50.0
"""

import os
import sys
import re
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
OPENAI_BASE_URL    = os.environ.get("OPENAI_BASE_URL", "").strip()  # OpenAI 호환 라우터(Groq/Gemini)
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS  = int(os.environ.get("OPENAI_MAX_TOKENS", "700"))
OPENAI_TIMEOUT     = float(os.environ.get("OPENAI_TIMEOUT", "4.0"))
CACHE_TTL_SEC      = int(os.environ.get("CACHE_TTL_SEC", "300"))
# 줄바꿈 보정 on/off (기본 ON). 끄려면 ANSWER_NEWLINES=0
ANSWER_NEWLINES    = os.environ.get("ANSWER_NEWLINES", "1").strip() not in ("0", "false", "False", "")

# ── SYSTEM_PROMPT — 한국기술대 일학습병행 원스탑 상담 AI ─────
SYSTEM_PROMPT = (
    "당신은 한국기술교육대학교 일학습병행 공동훈련센터 지원단의 '원스탑 상담 AI'입니다.\n"
    "역할: 일학습병행 제도(학습기업·학습근로자·기업현장교사·내부평가·외부평가·실시신고·운영규칙 등)에 대한 1차 안내.\n"
    "원칙:\n"
    "  - 한국어, 정중·간결하게 답변 (500자 이내).\n"
    "  - 근거 있는 제도 안내만, 추측·단정 금지.\n"
    "  - 학습기업·학습근로자·공동훈련센터·내부평가·지원금 등 인접 주제는 종합적으로 1차 안내.\n"
    "  - 정확한 제도 해석·법령 자문·세무·노동 분쟁은 '한국산업인력공단 일학습지원국 또는 원스탑 상담센터 전문상담'으로 안내.\n"
    "  - 민감정보(전화번호·주소·계좌)는 절대 생성 금지.\n"
    "  - 모든 답변 끝에 '자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다.' 한 줄을 포함.\n"
    "\n"
    "[법령·제도 기준 — 정확한 명칭 사용]\n"
    "  - 상위 법률: 「산업현장 일학습병행 지원에 관한 법률」(시행령 2026.2.1. 대통령령 제35578호)\n"
    "  - 하위 고시: 「일학습병행 운영규정」(고용노동부 고시 제2025-99호, 2025.10.22 시행, 2025.12.29 일부개정)\n"
    "  - 운영 주체: 고용노동부(정책·감독), 한국산업인력공단(HRD-Net, 학습기업 선정·운영, 학습근로자 관리), 공동훈련센터(내부평가·운영 지원)\n"
    "  - 사용자 시스템: HRD-Net (학습관리·평가 등록·지원금 집행의 단일 창구)\n"
    "\n"
    "[핵심 용어 정의 — 답변에서 일관 사용]\n"
    "  - 학습기업: 법률 제13조에 따라 고용노동부 지정을 받아 일학습병행 과정을 운영하는 기업\n"
    "  - 학습근로자: 「근로기준법」 제2조제1항제1호의 근로자이며 일학습병행 과정에 참여하는 자. 청년 위주이며 2026년부터 재직자(입사 3년 이내) 유형 추가 운영\n"
    "  - 기업현장교사: 학습근로자의 실무역량 향상과 NCS 기반 현장훈련을 담당하는 기업 소속 현장교사 (구분: 일반/승급/갱신 심사, 기본·심화 이수)\n"
    "  - 내부평가: 학습기업/공동훈련센터 자체 시행, 필수 능력단위 개수 70% 이상 Pass 필요, HRD-Net 학습관리 10일 이내 등록\n"
    "  - 외부평가: 한국산업인력공단 시행 국가 시험. 능력단위별 계획 훈련시간 80% 이상 이수 + 내부평가 Pass가 응시 요건\n"
    "  - 공동훈련센터: 한국기술대 등 대학 부설 형태로, 중소기업 학습근로자 공동훈련 + 내부평가 주관\n"
    "\n"
    "[정확성 원칙 — 환각 방어]\n"
    "  - 수치·금액·지원단가·기간은 절대 임의로 생성 금지. '운영규정 별표와 HRD-Net 매뉴얼 확인 필요'로 안내하고 1:1 상담 연결로 안내한다.\n"
    "  - 법령·운영규칙의 명칭·날짜·발령번호는 위 명시 값 그대로 사용. 인용하지 않은 조항을 만들어내지 않는다.\n"
    "  - '보통 ~원', '보통 6개월' 같은 평균치/추측 수치 금지.\n"
    "  - 실존하지 않는 법령 조항 번호, 가짜 URL, 가짜 상담 전화번호 생성 금지.\n"
    "  - 동일 주제를 NCS 자격과정, 내일배움카드, K-디지털 트레이닝 등 다른 제도와 혼동하지 않는다. 인접 제도를 다루면 한 줄로만 구분 표시.\n"
    "  - 절차형 답변이면 1단계/2단계/3단계 순으로 실제 가능한 절차만 나열. 행정 절차 사이에 추측 단계 끼워넣기 금지.\n"
    "\n"
    "[가독성 — 줄바꿈 규칙 — 반드시 준수]\n"
    "  - 답변 내부에 적절한 줄바꿈을 포함해 카톡 simpleText 에서 한눈에 읽히도록 작성한다.\n"
    "  - 권장 구조(각 블록 사이 빈 줄 \\n\\n):\n"
    "      인사/확인 한 문장\n"
    "      \\n\\n\n"
    "      본문 2~4문장. 절차형이면 '1단계 → 2단계 → 3단계' 형식, 단계 사이 \\n\n"
    "      \\n\\n\n"
    "      마무리: 1:1 상담 안내 1줄\n"
    "  - 같은 종류 항목을 연속으로 나열할 때는 항목 사이를 단일 줄바꿈(\\n)으로 끊고, 다른 종류(주제 전환) 사이에는 빈 줄(\\n\\n) 사용.\n"
    "  - 글머리 기호(-, *, •) 자유 사용 가능. 마크다운(#, **)은 절대 사용 금지 (카톡에서 그대로 노출됨).\n"
    "  - 한 답변 안에 빈 줄(\\n\\n) 최소 2회, 단일 줄바꿈(\\n) 최소 1회 포함 보장.\n"
    "  - 본문 500~700자 이내 유지.\n"
    "\n"
    "[경계 — 즉시 1:1 이관]\n"
    "  - 개별 학습근로자 신상(급여·노동조건·해고·산재) 자문\n"
    "  - 특정 기업 회계·세무 자문\n"
    "  - 분쟁 조정·심판·소송 관련 의견\n"
    "  - 위 항목은 즉시 '한국산업인력공단 일학습지원국 또는 원스탑 상담센터 전문상담'으로 안내하고 추가 답변 시도 금지\n"
)

_CACHE: dict = {}


# ── 줄바꿈 보정기 — LLM이 한 줄로 뱉어도 자동으로 정리 ───────
def _beautify_answer(text: str) -> str:
    if not text:
        return text

    s = text.replace("\r", "")

    # 1) "1) ... 2) ... 3) ..." 같은 나열을 단일 \n 으로 끊기
    s = re.sub(r"(?<=[^.])(\s*)([1-9]\)\s)", r"\n\2", s)
    s = re.sub(r"(?<=\S)(\s+)([1-9]\.\s)", r"\n\2", s)
    s = re.sub(r"(?<=\S)(\s+)([①②③④⑤])", r"\n\2", s)

    # 2) 글머리 기호(•, -, *)가 어색하게 붙어 있으면 단일 \n 으로
    s = re.sub(r"(?<=[가-힣\.])([ \t]+)([•\-\*]\s)", r"\n\2", s)

    # 3) 문장 종결 직후 줄바꿈 보장
    s = re.sub(r"([\.!\?])([ \t]+)(?=[가-힣A-Za-z0-9])", r"\1\n", s)

    # 4) 3칸 이상 연속 줄바꿈 → \n\n 으로 압축
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 5) 줄 시작/끝 공백 정리, 빈 줄은 최대 1번까지
    lines = [ln.rstrip() for ln in s.split("\n")]
    cleaned, blank_streak = [], 0
    for ln in lines:
        if ln.strip() == "":
            blank_streak += 1
            if blank_streak <= 1:
                cleaned.append("")
        else:
            blank_streak = 0
            cleaned.append(ln)
    s = "\n".join(cleaned).strip()

    # 6) 마무리 멘트 직전 \n\n 보장 ("자세한 상담은 1:1 상담 연결" 시그니처)
    sig = "자세한 상담은 1:1 상담 연결"
    if sig in s and not re.search(r"\n\n[^\n]*" + re.escape(sig), s):
        s = re.sub(r"\s+" + re.escape(sig), "\n\n" + sig, s)

    return s


# ── OpenAI 클라이언트 lazy-init (OpenAI 호환 base_url 지원) ──
_client: Optional["OpenAI"] = None

def _get_client():
    global _client
    if not OPENAI_API_KEY or not _HAS_OPENAI:
        return None
    if _client is None:
        try:
            _client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL or None,
                timeout=OPENAI_TIMEOUT,
            )
            sys.stderr.write(
                f"[chat] OpenAI client ready model={OPENAI_MODEL} "
                f"base_url={'default' if not OPENAI_BASE_URL else OPENAI_BASE_URL}\n"
            )
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
        text = (resp.choices[0].message.content or "").strip()
        if ANSWER_NEWLINES:
            text = _beautify_answer(text)
        return text
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
        "label": "일학습병행 운영규정 (고시 제2025-99호)",
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
        "service":         "kakao-ai-skill-openai",
        "status":          "ok",
        "openai_key":      bool(OPENAI_API_KEY),
        "openai_model":    OPENAI_MODEL,
        "openai_base_url": OPENAI_BASE_URL or "(default api.openai.com)",
        "answer_newlines": ANSWER_NEWLINES,
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

    now = time.time()
    if CACHE_TTL_SEC > 0 and utterance in _CACHE:
        ts, body = _CACHE[utterance]
        if now - ts < CACHE_TTL_SEC:
            return JSONResponse(content=body)

    answer = ""
    if OPENAI_API_KEY and _HAS_OPENAI and utterance:
        sys.stderr.write(
            f"[chat] openai-req model={OPENAI_MODEL} "
            f"base_url={'default' if not OPENAI_BASE_URL else OPENAI_BASE_URL} "
            f"len={len(utterance)} text={utterance[:30]!r}\n"
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
            "현재 1:1 상담 연결이 지연되어 자유 답변을 표시하지 못했습니다.\n"
            "다음 방법을 통해 정확한 안내를 받으실 수 있습니다.\n\n"
            "1) 한국산업인력공단 일학습병행 상담 (1644-8000, hrdkorea.or.kr)\n"
            "2) 한국기술교육대학교 일학습병행 공동훈련센터 원스탑 상담\n\n"
            "잠시 후 다시 시도해 주세요."
        )
    else:
        if ANSWER_NEWLINES:
            answer = _beautify_answer(answer)

    body = build_multi_bubble(answer)

    if CACHE_TTL_SEC > 0 and utterance:
        _CACHE[utterance] = (now, body)

    return JSONResponse(content=body)
