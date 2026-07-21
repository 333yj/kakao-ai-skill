"""카카오 i 오픈빌더 AI 자동응답 스킬 서버 (FastAPI)
검색 기준일: 2026-07-21 / 공식 가이드 version 2.0 기준"""
import os, sys, json, re, time, urllib.request, urllib.error
from pathlib import Path
from fastapi import FastAPI, Request

# ───────── 환경 변수 ─────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# 무료 등급 quota(분당 15회)를 피하기 위해 가장 저렴한 모델 우선 호출.
# 429/실패 시 더 큰 모델로 폴백.
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]

# ───────── 상수 ─────────
KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.json"
MAX_OUT_CHARS = 990            # 카톡 simpleText 응답 안전 한도
MAX_CHUNK_CHARS = 900          # Gemini 컨텍스트 1청크 당 글자 상한
CACHE_TTL_SECONDS = 1800       # 동일 질문 답변 캐시 30분

# ───────── FastAPI 앱 & 부팅 로그 ─────────
app = FastAPI()


def log(msg: str) -> None:
    print(f"[kakao-ai-skill] {msg}", file=sys.stderr, flush=True)


# ───────── 부팅 시 knowledge.json 1회 로드 ─────────
try:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
    CHUNKS = KB.get("chunks", []) or []
    log(f"✅ Loaded {len(CHUNKS)} chunks from knowledge.json")
except Exception as e:
    log(f"❌ Failed to load knowledge.json: {e}")
    CHUNKS = []

# ───────── 동일 질문 캐시 (분당 호출 절감 → 429 회피) ─────────
answer_cache: dict = {}        # key: 정규화된 질문문자열, value: (timestamp, 답변)


def cache_get(q_key: str) -> str | None:
    item = answer_cache.get(q_key)
    if not item:
        return None
    ts, ans = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        answer_cache.pop(q_key, None)
        return None
    log(f"💾 cache hit: {q_key[:30]}")
    return ans


def cache_put(q_key: str, answer: str) -> None:
    answer_cache[q_key] = (time.time(), answer)
    # 캐시가 너무 커지지 않도록 200개 초과 시 오래된 것 절반 제거
    if len(answer_cache) > 200:
        sorted_items = sorted(answer_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in sorted_items[:100]:
            answer_cache.pop(k, None)


# ───────── 텍스트 정규화 & chunk 검색 ─────────
def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def search_chunks(query: str, k: int = 5):
    if not query:
        return []
    q = normalize(query)
    if len(q) < 2:
        return []
    q_grams = {q[i:i + 2] for i in range(len(q) - 1)}

    scored = []
    for c in CHUNKS:
        text = c.get("text", "") or ""
        if len(text) < 2:
            continue
        t = normalize(text)
        t_set = {t[i:i + 2] for i in range(len(t) - 1)}
        overlap = len(q_grams & t_set)

        # 보너스: 질문의 의미 있는 단어가 chunk 본문에 그대로 등장
        word_bonus = 0
        for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", query):
            if word in text:
                word_bonus += 3

        score = overlap + word_bonus
        if score > 0:
            snippet = text[:MAX_CHUNK_CHARS]
            scored.append((score, {
                "id": c.get("id", "?"),
                "chapter": c.get("chapter", "?"),
                "text": snippet,
            }))
    scored.sort(key=lambda x: -x[0])
    return [c for _s, c in scored[:k]]


# ───────── 프롬프트 빌더 ─────────
def build_prompt(user_msg: str, chunks: list) -> str:
    if not chunks:
        return (
            "당신은 일학습병행 상담도우미입니다.\n"
            f"사용자가 질문한 내용: \"{user_msg}\"\n"
            "참고할 수 있는 매뉴얼 본문이 없습니다.\n"
            "정확한 정보가 매뉴얼에서 확인되지 않음을 먼저 알리고, "
            "보다 자세한 상담은 1:1 상담 연결을 통해 문의해 주실 수 있도록 안내하세요.\n"
            "[답변]"
        )
    parts = [
        f"--- [{c['id']} / {c['chapter']}] ---\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n".join(parts)
    return (
        "당신은 일학습병행 상담도우미입니다. 아래 매뉴얼 본문 발췌 내용만 근거로 답변하세요.\n"
        "출처 chunk id는 답변 본문에 자연스럽게 녹여서 알려주세요.\n"
        "모르는 내용은 솔직히 답하세요.\n\n"
        f"[참고 매뉴얼 본문]\n{context}\n\n"
        f"[질문]\n{user_msg}\n\n[답변]"
    )


# ───────── Gemini 호출 (429 인지 + 1회 재시도 + 모델 폴백) ─────────
def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY env var not set on Render")

    last_err = "unknown"
    for model in GEMINI_MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 700,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )

        # 모델당 최대 2회 시도 (1차 + 429 시 1차 재시도)
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    cands = data.get("candidates", [])
                    if cands and cands[0].get("content", {}).get("parts"):
                        parts = cands[0]["content"]["parts"]
                        if parts and parts[0].get("text"):
                            text = parts[0]["text"].strip()
                            if text:
                                log(f"✅ Gemini answered via {model} ({len(text)} chars)")
                                return text
                    raise RuntimeError(f"empty response body from {model}")

            except urllib.error.HTTPError as e:
                log(f"⚠️ Gemini {model} attempt {attempt} → HTTP {e.code} {e.reason}")
                last_err = f"HTTP {e.code}"
                if e.code == 429:
                    # 분당 RPM 한도 초과. 12초 대기 후 같은 모델로 1회만 더.
                    log(f"⏳ 429: backing off 12s before retry")
                    time.sleep(12)
                    continue      # 같은 모델로 attempt=2 시도
                else:
                    break          # 다른 코드는 같은 모델로 재시도 안 함 → 다음 모델로

            except urllib.error.URLError as e:
                log(f"⚠️ Gemini {model} → URLError: {e}")
                last_err = "timeout/network"
                break              # 네트워크 문제는 다음 모델로

            except Exception as e:
                log(f"⚠️ Gemini {model} → {type(e).__name__}: {e}")
                last_err = type(e).__name__
                break

    raise RuntimeError(f"all Gemini models failed; last: {last_err}")


# ───────── 카카오 응답 포맷 ─────────
def build_simple_text_reply(text: str) -> dict:
    """카카오 공식 응답 포맷(version 2.0) — simpleText 만 (버튼 없음)"""
    safe = (text or "")[:MAX_OUT_CHARS]
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": safe}}],
        },
    }


# ───────── LLM 실패 시 안전한 폴백 답변 ─────────
FALLBACK_BY_REASON = {
    "HTTP 429": (
        "안녕하세요. 일학습병행 원스탑 상담 AI 챗봇입니다.\n"
        "현재 동시에 많은 문의가 들어와 잠시 응답이 지연되고 있어요.\n"
        "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."
    ),
    "empty": (
        "안녕하세요. 일학습병행 원스탑 상담 AI입니다.\n"
        "적절한 답변이 어렵습니다.\n"
        "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."
    ),
}
DEFAULT_FALLBACK = (
    "안녕하세요. 일학습병행 원스탑 상담 AI입니다.\n"
    "적절한 답변이 어렵습니다.\n"
    "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."
)


def make_fallback(reason_key: str) -> str:
    return FALLBACK_BY_REASON.get(reason_key, DEFAULT_FALLBACK)


# ───────── 헬스체크 ─────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "chunks_loaded": len(CHUNKS),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "cache_size": len(answer_cache),
    }


# ───────── 메인 엔드포인트 ─────────
@app.post("/api/chat")
async def chat(req: Request):
    # 요청 파싱 단계 try
    try:
        body = await req.json()
    except Exception as e:
        log(f"⚠️ request.json() 실패: {type(e).__name__}: {e}")
        return build_simple_text_reply(make_fallback("empty"))

    user_msg = ((body.get("userRequest") or {}).get("utterance") or "").strip()

    # 빈 메시지 안내
    if not user_msg:
        return build_simple_text_reply(
            "안녕하세요. 일학습병행 원스탑 상담 AI입니다. 무엇을 도와드릴까요?"
        )

    # 안전한 폴백 (어떤 단계에서든 항상 카톡 응답 보장)
    try:
        q_key = normalize(user_msg)
        log(f"Q: {user_msg}")

        # 1) 캐시 확인
        cached = cache_get(q_key)
        if cached:
            return build_simple_text_reply(cached)

        # 2) chunk 검색
        top = search_chunks(user_msg, k=5)
        log(f"matched: {[c['id'] for c in top]}")

        # 3) Gemini 호출
        prompt = build_prompt(user_msg, top)
        try:
            answer = call_gemini(prompt)
        except RuntimeError as e:
            err_msg = str(e)
            # 429는 별도 폴백 멘트
            if "429" in err_msg:
                log(f"❌ Gemini 응답 실패 (429) → 폴백")
                return build_simple_text_reply(make_fallback("HTTP 429"))
            log(f"❌ Gemini 응답 실패: {e} → 폴백")
            return build_simple_text_reply(make_fallback("empty"))

        # 4) 출처 footer + 응답 + 캐시 저장
        footer = ""
        if top:
            footer = "\n\n[출처 " + ", ".join(
                f"{c['id']}/{c['chapter']}" for c in top[:3]
            ) + "]"
        final_answer = (answer + footer)[:MAX_OUT_CHARS]

        cache_put(q_key, final_answer)
        return build_simple_text_reply(final_answer)

    except Exception as e:
        # 어떤 종류의 예외라도 카톡 응답 형식으로 돌려보냄 → 절대 500 안 남
        log(f"❌ TOP-LEVEL: {type(e).__name__}: {e}")
        return build_simple_text_reply(make_fallback("empty"))
