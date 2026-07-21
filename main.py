"""카카오 i 오픈빌더 AI 자동응답 스킬 서버 (FastAPI) — thin pointer 모드
검색 기준일: 2026-07-21"""
import os, sys, json, re
from pathlib import Path
from fastapi import FastAPI, Request

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.json"
app = FastAPI()


def log(msg: str) -> None:
    print(f"[kakao-ai-skill] {msg}", file=sys.stderr, flush=True)


# ───────── 부팅 시 1회 로드 ─────────
try:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
    SOURCES = KB.get("sources", []) or []
    QUICK_FAQ = KB.get("quick_faq", []) or []
    DEFAULT_RESP = KB.get("default_response", {}) or {}
    log(f"✅ Loaded thin KB: sources={len(SOURCES)}, faq={len(QUICK_FAQ)}")
except Exception as e:
    log(f"❌ Failed to load knowledge.json: {e}")
    SOURCES, QUICK_FAQ, DEFAULT_RESP = [], [], {}


# ───────── 응답 텍스트 빌더 ─────────
def _build_text(answer_body: str) -> str:
    """FAQ 본문 + 참고 자료 footer + 1:1 상담 안내 멘트"""
    parts = [answer_body.strip()]

    if SOURCES:
        parts.append("\n\n[공식 참고 자료]")
        for s in SOURCES:
            parts.append(f"• {s['doc']}\n  {s['url']}")

    for m in DEFAULT_RESP.get("footer_messages", []):
        parts.append(m)

    return "\n".join(parts).strip()[:990]


def _default_response() -> str:
    intro = DEFAULT_RESP.get(
        "intro",
        "안녕하세요. 일학습병행 원스탑 상담 AI입니다. 공식 자료 안내로 도움을 드리겠습니다.",
    )
    return _build_text(intro)


def build_simple_text_reply(text: str) -> dict:
    """카카오 공식 응답 포맷(version 2.0) — simpleText 만 (버튼 없음)"""
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": (text or "")[:990]}}],
        },
    }


# ───────── FAQ 검색 ─────────
def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def find_quick_answer(query: str):
    q = normalize(query)
    if len(q) < 2:
        return None
    best = None
    best_score = 0
    for qa in QUICK_FAQ:
        score = 0
        for kw in qa.get("keywords", []):
            nkw = normalize(kw)
            if nkw and nkw in q:
                score += 2 if len(nkw) >= 3 else 1
        if score > best_score:
            best_score = score
            best = qa
    return best if best_score >= 1 else None


# ───────── 헬스체크 ─────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "thin-pointer",
        "sources": len(SOURCES),
        "quick_faq_count": len(QUICK_FAQ),
    }


# ───────── 메인 엔드포인트 ─────────
@app.post("/api/chat")
async def chat(req: Request):
    try:
        body = await req.json()
    except Exception:
        return build_simple_text_reply("답변 제공이 어렵습니다. 1:1 상담 문의 부탁드립니다.")

    user_msg = ((body.get("userRequest") or {}).get("utterance") or "").strip()

    if not user_msg:
        return build_simple_text_reply(
            "안녕하세요. 일학습병행 원스탑 상담 AI입니다. 무엇을 도와드릴까요?"
        )

    try:
        log(f"Q: {user_msg}")
        qa = find_quick_answer(user_msg)
        if qa:
            log(f"✅ FAQ hit: {qa['id']}")
            return build_simple_text_reply(_build_text(qa["answer"]))
        else:
            log(f"ℹ️ No FAQ hit → default response")
            return build_simple_text_reply(_default_response())
    except Exception as e:
        log(f"❌ TOP-LEVEL: {type(e).__name__}: {e}")
        return build_simple_text_reply(_default_response())
