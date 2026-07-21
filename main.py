import os, json, re, urllib.request
from pathlib import Path
from fastapi import FastAPI, Request

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-2.0-flash"

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.json"

app = FastAPI()

# 부팅 시 1회만 로드
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)
CHUNKS = KB.get("chunks", [])


def normalize(s: str) -> str:
    """공백·대소문자 차이 제거"""
    return re.sub(r"\s+", "", s.lower())


def search_chunks(query: str, k: int = 5):
    """질문과 chunk 텍스트의 글자 2-gram 겹침 점수로 상위 k개 반환"""
    q = normalize(query)
    if len(q) < 2:
        return []
    q_grams = set(q[i:i + 2] for i in range(len(q) - 1))

    scored = []
    for c in CHUNKS:
        t = normalize(c["text"])
        if len(t) < 2:
            continue
        t_set = set(t[i:i + 2] for i in range(len(t) - 1))
        overlap = len(q_grams & t_set)
        # 단어 단위 매칭 보너스
        word_bonus = 0
        for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", query):
            if word and word in c["text"]:
                word_bonus += 3
        score = overlap + word_bonus
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:k]]


def build_prompt(user_msg: str, chunks: list) -> str:
    if not chunks:
        return (
            "당신은 일학습병행 상담도우미입니다.\n"
            f"사용자가 '{user_msg}' 라고 물었습니다.\n"
            "매뉴얼에 직접 답변할 단서가 없으므로 다음 답변을 그대로 출력하세요:\n"
            "\"챗봇으로 답변이 불가합니다. 자세한 상담은 1:1 상담 연결로 문의 부탁드립니다.\"\n"
        )

    blocks = []
    for c in chunks:
        blocks.append(
            f"--- [{c['id']} / {c['chapter']} / 본문 {c['char_offset_start']}~{c['char_offset_end']}자] ---\n"
            f"{c['text']}"
        )
    context = "\n\n".join(blocks)
    return (
        "당신은 일학습병행 상담도우미입니다. 아래 참고 원문 발췌의 내용만 근거로 답변하세요.\n"
        "답변 끝에 출처 chunk id를 (예: 출처: mc0011 / 제2장 학습기업) 형식으로 표기하세요.\n"
        "모르는 내용은 솔직히 '매뉴얼에 해당 내용이 명시되어 있지 않음'이라고 답하세요.\n\n"
        f"[참고 원문 발췌]\n{context}\n\n[질문]\n{user_msg}\n\n[답변]\n"
    )


def call_gemini(prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    req_body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=req_body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]


@app.get("/")
def root():
    return {"status": "ok", "chunks_loaded": len(CHUNKS)}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    user_msg = body.get("userRequest", {}).get("utterance", "").strip()

    if not user_msg:
        return {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {
                "text": "안녕하세요. 일학습병행 원스탑 상담 AI 챗봇입니다. 질문을 입력해 주세요."
            }}]}
        }

    top_chunks = search_chunks(user_msg, k=5)
    prompt = build_prompt(user_msg, top_chunks)
    answer = call_gemini(prompt)

    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": answer}}]}
    }
