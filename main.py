"""
카카오 i 오픈빌더 — LLM-우선 자유 답변 스킬 (multi-bubble)
검색 기준일: 2026-07-21
"""
import os, json, re, time, sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

# ── 설정 ────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL           = "https://api.openai.com/v1/chat/completions"
KNOWLEDGE_FILE       = Path(os.getenv("KNOWLEDGE_FILE",
                                      "/opt/render/project/src/knowledge.json"))

CACHE_TTL_SEC        = int(os.getenv("CACHE_TTL_SEC", "300"))
OPENAI_TIMEOUT       = float(os.getenv("OPENAI_TIMEOUT", "4.0"))
SIMPLE_BUBBLE_CHARS  = 990
ALWAYS_TRY_LLM       = os.getenv("ALWAYS_TRY_LLM", "1") == "1"   # 키 있으면 절대 멈추지 않음

def _load_knowledge() -> dict:
    if not KNOWLEDGE_FILE.exists():
        return {"sources": [], "quick_faq": [], "ai_router": {},
                "default_response": {"intro":"","extra_resources":[],"footer_messages":[]}}
    return json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))

KNOWLEDGE = _load_knowledge()
ROUTER    = KNOWLEDGE.get("ai_router")       or {}
QUICK_FAQ = KNOWLEDGE.get("quick_faq")        or []
DEFAULT   = KNOWLEDGE.get("default_response") or {}
SOURCES   = KNOWLEDGE.get("sources")          or []

OPENAI_MODEL  = ROUTER.get("model") or OPENAI_MODEL_DEFAULT
SYSTEM_PROMPT = ROUTER.get("system_prompt") or (
    "당신은 일학습병행 상담 안내 챗봇입니다.\n"
    "- 한국어로 정중·간결하게 답변 (500자 이내).\n"
    "- 정확한 정보가 없으면 '1:1 상담 연결 안내'로 응답.\n"
    "- 매칭된 FAQ가 있으면 그 결을 따라 동일 톤·동일 길이로 작성."
)
GUARDRAILS  = ROUTER.get("guardrails") or (
    "법률·세무·노동 분쟁 해석은 '한국산업인력공단 일학습지원국 또는 원스탑 상담센터 "
    "전문상담' 안내. 근거 없는 추측·판단 금지."
)
TEMPERATURE = float(ROUTER.get("temperature", 0.4))
MAX_TOKENS  = int(ROUTER.get("max_tokens", 700))   # 200자 안 → 700 토큰으로 확장
TIMEOUT_SEC = float(ROUTER.get("timeout_sec", OPENAI_TIMEOUT))

CACHE: dict = {}

NORM_RX = re.compile(r"[\s\.\,\!\?\·\:\;\"\'\(\)\[\]\/]+")
def norm(s: str) -> str:
    return NORM_RX.sub("", s or "").strip().lower()

def score_entry(utt_norm, entry):
    total, exact = 0, 0
    for kw in (entry.get("keywords") or []):
        k = norm(kw)
        if not k: continue
        if k == utt_norm: total += 100; exact += 1
        elif k in utt_norm: total += len(k) * 2
    return total, exact

def match_quick_faq(utterance):
    u = norm(utterance)
    if not u: return None
    best, bt, be = None, 0, -1
    for q in QUICK_FAQ:
        t, e = score_entry(u, q)
        if (e, t) > (be, bt): best, bt, be = q, t, e
    return best if bt > 0 else None

async def call_llm(user_utterance, faq_hint="") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY env empty")
    system_text = (SYSTEM_PROMPT or "").strip()
    if (GUARDRAILS or "").strip():
        system_text += "\n\n[운영 규칙]\n" + GUARDRAILS.strip()
    user_text = user_utterance
    if faq_hint:
        user_text = (
            "[참고 FAQ 힌트]\n"+faq_hint+"\n\n[사용자 질문]\n"+user_utterance+"\n\n"
            "[응답 조건]\n- 한국어 500자 이내\n- FAQ 결을 따라 정확·간결\n"
            "- 필요 시 1:1 상담 연결 안내 1회"
        )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type":  "application/json"}
    payload = {"model": OPENAI_MODEL,
               "messages":[{"role":"system","content":system_text},
                           {"role":"user","content":user_text}],
               "max_tokens":MAX_TOKENS, "temperature":TEMPERATURE}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
            r = await client.post(OPENAI_URL, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise RuntimeError(f"OPENAI timeout after {TIMEOUT_SEC}s")
    except Exception as e:
        raise RuntimeError(f"OPENAI http error: {type(e).__name__}: {e}")
    if r.status_code == 401:
        raise RuntimeError("OPENAI 401 Unauthorized — 키 무효/만료/조직 미승인")
    if r.status_code == 429:
        raise RuntimeError("OPENAI 429 quota/rate — 한도 또는 분당 초과")
    if r.status_code >= 500:
        raise RuntimeError(f"OPENAI {r.status_code} server error")
    if r.status_code >= 400:
        raise RuntimeError(f"OPENAI {r.status_code} client error: {r.text[:200]}")
    try:
        data = r.json()
    except Exception:
        raise RuntimeError("OPENAI returned non-JSON")
    text = (data.get("choices",[{}])[0].get("message",{}).get("content") or "").strip()
    if not text:
        raise RuntimeError("OPENAI returned empty content")
    return text

def _trim_to_limit(text: str, limit: int = SIMPLE_BUBBLE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"

def _simple_bubble(text: str) -> dict:
    return {"simpleText": {"text": _trim_to_limit(text or "", SIMPLE_BUBBLE_CHARS)}}

def build_simple_text_reply(text: str) -> dict:
    return {"version":"2.0","template":{"outputs":[_simple_bubble(text)]}}

def build_multi_bubble_reply(bubbles: list) -> dict:
    outs = []
    for b in (bubbles or []):
        b = (b or "").strip()
        if b: outs.append(_simple_bubble(b))
    if not outs:
        outs = [_simple_bubble("안녕하세요! 무엇을 도와드릴까요?")]
    return {"version":"2.0","template":{"outputs":outs}}

def _build_extra_sections() -> str:
    blocks = []
    for sec in (DEFAULT.get("extra_resources") or []):
        header = (sec.get("header") or "").strip()
        lines  = []
        if header: lines.append(header)
        for it in (sec.get("items") or []):
            lab = (it.get("label") or "").strip()
            url = (it.get("url")   or "").strip()
            if lab and url:
                lines.append(f"• {lab}"); lines.append(f"  {url}")
        if lines: blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()

def _join_default_bubbles() -> list:
    intro   = (DEFAULT.get("intro") or "").strip()
    extra   = _build_extra_sections()
    footers = "\n".join(DEFAULT.get("footer_messages") or []).strip()
    bubbles = []
    if intro:   bubbles.append(intro)
    if extra:   bubbles.append(extra)
    if footers: bubbles.append(footers)
    return bubbles or ["안녕하세요! 무엇을 도와드릴까요?"]

def answers_bubbles_for(answer: str) -> list:
    bubbles = [_trim_to_limit(answer, SIMPLE_BUBBLE_CHARS)]
    extra = _build_extra_sections()
    footers = "\n".join(DEFAULT.get("footer_messages") or []).strip()
    if extra:   bubbles.append(extra)
    if footers: bubbles.append(footers)
    return bubbles

def cache_get(utt_norm):
    hit = CACHE.get(utt_norm)
    if hit and time.time() - hit["t"] < CACHE_TTL_SEC:
        return hit["text"]
    return None

def cache_set(utt_norm, text):
    CACHE[utt_norm] = {"t": time.time(), "text": text}

app = FastAPI()

@app.get("/")
async def health():
    return {"status":"ok","service":"kakao-ai-skill",
            "openai_model":OPENAI_MODEL if OPENAI_API_KEY else None,
            "openai_key_set":bool(OPENAI_API_KEY),
            "schema":KNOWLEDGE.get("schema"),
            "version":KNOWLEDGE.get("version"),
            "quick_faq_count":len(QUICK_FAQ),
            "sources_count":len(SOURCES),
            "default_intro_len":len((DEFAULT.get("intro") or "").strip()),
            "extra_resources_count":len(DEFAULT.get("extra_resources") or []),
            "cache_size":len(CACHE),
            "flow":"1)faq 2)openai(gpt-4o-mini) 3)default(multi-bubble)"}

@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"[chat] json parse fail: {e}", file=sys.stderr)
        return JSONResponse(content=build_multi_bubble_reply(_join_default_bubbles()))

    utterance = (body.get("userRequest") or {}).get("utterance","").strip()
    utt_n     = norm(utterance)

    if not utterance:
        return JSONResponse(content=build_multi_bubble_reply(_join_default_bubbles()))

    cached = cache_get(utt_n)
    if cached and cached.startswith("MULTI_BUBBLE::"):
        try:
            return JSONResponse(content=json.loads(cached[len("MULTI_BUBBLE::"):]))
        except Exception:
            pass

    # (1) FAQ 매칭 (LLM 미사용, 즉시 응답)
    faq_hit = match_quick_faq(utterance)
    if faq_hit and faq_hit.get("answer"):
        bubbles = answers_bubbles_for(faq_hit["answer"])
        reply   = build_multi_bubble_reply(bubbles)
        cache_set(utt_n, "MULTI_BUBBLE::"+json.dumps(reply, ensure_ascii=False))
        print(f"[chat] faq-hit id={faq_hit.get('id')} utterance={utterance!r}", file=sys.stderr)
        return JSONResponse(content=reply)

    # (2) OpenAI 자유 답변 (env 키 + ALWAYS_TRY_LLM=1 일 때 발사)
    faq_hint = (faq_hit or {}).get("answer","") or ""
    if OPENAI_API_KEY and ALWAYS_TRY_LLM:
        try:
            ai_reply = await call_llm(utterance, faq_hint=faq_hint)
            bubbles  = answers_bubbles_for(ai_reply)
            reply    = build_multi_bubble_reply(bubbles)
            cache_set(utt_n, "MULTI_BUBBLE::"+json.dumps(reply, ensure_ascii=False))
            print(f"[chat] llm-ok model={OPENAI_MODEL} chars={len(ai_reply)}", file=sys.stderr)
            return JSONResponse(content=reply)
        except Exception as e:
            print(f"[chat] llm-fail {type(e).__name__}: {e} | utterance={utterance!r}",
                  file=sys.stderr)

    # (3) 키 없음 or 호출 실패 → default multi-bubble (긴 intro + 공식/지원단 + footer)
    bubbles = _join_default_bubbles()
    reply   = build_multi_bubble_reply(bubbles)
    cache_set(utt_n, "MULTI_BUBBLE::"+json.dumps(reply, ensure_ascii=False))
    return JSONResponse(content=reply)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
