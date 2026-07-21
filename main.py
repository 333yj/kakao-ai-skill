"""카카오 i 오픈빌더 AI 자동응답 스킬 서버 (FastAPI)"""
import os, sys, json, re, urllib.request, urllib.error
from pathlib import Path
from fastapi import FastAPI, Request

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.json"
app = FastAPI()


def log(msg):
    print(f"[kakao-ai-skill] {msg}", file=sys.stderr, flush=True)


try:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
    CHUNKS = KB.get("chunks", [])
    log(f"✅ Loaded {len(CHUNKS)} chunks from knowledge.json")
except Exception as e:
    log(f"❌ Failed to load knowledge.json: {e}")
    CHUNKS = []


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def search_chunks(query: str, k: int = 5):
    q = normalize(query)
    if len(q) < 2:
        return []
    q_grams = set(q[i:i + 2] for i in range(len(q) - 1))
    scored = []
    for c in CHUNKS:
        t = normalize(c.get("text", ""))
        if len(t) < 2:
            continue
        t_set = set(t[i:i + 2] for i in range(len(t) - 1))
        overlap = len(q_grams & t_set)
        word_bonus = 0
        for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", query):
            if word and word in c.get("text", ""):
                word_bonus += 3
        score = overlap + word_bonus
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:k]]


def build_prompt(user_msg: str, chunks: list) -> str:
    blocks = [
        f"--- [{c.get('id', '?')} / {c.get('chapter', '?')}] ---\n{c.get('text', '')}"
        for c in chunks
    ]
    context = "\n\n".join(blocks)
    return (
        "당신은 일학습병행 상담도우미입니다. 아래 참고 원문 발췌만 근거로 답변하세요.\n"
        "답변 끝에 출처 chunk id를 표기하세요. 모르는 내용은 솔직히 답하세요.\n\n"
        f"[참고 원문]\n{context}\n\n[질문]\n{user_msg}\n\n[답변]\n"
    )


FALLBACK_MSG = (
    "안녕하세요. 일학습병행 원스탑 상담 AI입니다.\n"
    "적절한 답변이 어렵습니다. "
    "보다 자세한 상담은 1:1 상담 연결을 통해 문의 부탁드립니다."
)


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY env var not set on Render")

    last_err = "unknown"
    for model in GEMINI_MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        req_body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=req_body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        text = parts[0]["text"].strip()
                        if text:
                            log(f"✅ Gemini answered via {model} ({len(text)} chars)")
                            return text
                raise RuntimeError(f"empty response body from {model}")
        except urllib.error.HTTPError as e:
            log(f"⚠️ Gemini {model} → HTTP {e.code} {e.reason}")
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            log(f"⚠️ Gemini {model} → URLError: {e}")
            last_err = "timeout/network"
        except Exception as e:
            log(f"⚠️ Gemini {model} → {type(e).__name__}: {e}")
            last_err = type(e).__name__

    raise RuntimeError(f"all Gemini models failed; last: {last_err}")


def build_simple_text_reply(text: str):
    """카카오 공식 응답 포맷(version 2.0) — simpleText"""
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": trim_to_limit(text)
                    }
                }
            ],
        },
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "chunks_loaded": len(CHUNKS),
        "gemini_key_set": bool(GEMINI_API_KEY),
    }


@app.post("/api/chat")
async def chat(req: Request):
    try:
        body = await req.json()
    except Exception:
        return kakao_resp("요청을 해석할 수 없어요. 다시 시도 부탁드립니다.")

    user_msg = (body.get("userRequest", {}) or {}).get("utterance", "").strip()
    if not user_msg:
        return kakao_resp("안녕하세요. 일학습병행 상담 AI입니다. 무엇을 도와드릴까요?")
        # ← 이 줄의 따옴표 안 깨진 한글 깨짐(인코딩 문제) 넣지 않도록 유지; 그대로 OK

    try:
        log(f"Q: {user_msg}")
        top = search_chunks(user_msg, k=5)
        log(f"matched: {[c.get('id', '?') for c in top]}")
        prompt = build_prompt(user_msg, top)
        answer = call_gemini(prompt)
        footer = ""
        if top:
            footer = "\n\n[출처 " + ", ".join(
                f"{c.get('id', '?')}/{c.get('chapter', '?')}"
                for c in top[:3]
            ) + "]"
        return kakao_resp((answer + footer)[:990])
    except Exception as e:
        log(f"❌ TOP-LEVEL: {type(e).__name__}: {e}")
        return kakao_resp(FALLBACK_MSG)
