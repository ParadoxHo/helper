import os
import glob
import asyncio
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import PyPDF2

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

# ==================== CONFIG ====================

ALLOWED_ORIGINS = [
    "https://assistics.vercel.app",
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "instructions")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "azs_instructions")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.05"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1200"))
GROQ_TOP_P = float(os.getenv("GROQ_TOP_P", "0.9"))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))

# Optional web fallback
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

MAX_FRAGMENT_LENGTH = int(os.getenv("MAX_FRAGMENT_LENGTH", "1000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))
TOP_K = int(os.getenv("TOP_K", "25"))
RAG_MIN_HITS = int(os.getenv("RAG_MIN_HITS", "3"))

DEVICE_KEYWORDS = {
    "rosslare": ["rosslare", "ac-b31", "ac-b32", "axtraxng"],
    "paradox": ["paradox", "evo", "evo640", "evo641", "evo192", "sp65", "sp4000", "dgp2-641bl", "dgp2-641rb", "dgp2-648bl"],
    "bosch": ["bosch", "bvms", "divar", "avenar"],
    "siemens": ["siemens", "vectis"],
    "3xlogic": ["3xlogic"],
    "provision": ["provision", "isr"],
    "satel": ["satel", "integra"],
    "babyware": ["babyware"],
    "hikvision": ["hikvision"],
}

# ==================== GLOBALS ====================

embedding_model = None
collection = None
chroma_client = None
is_rag_ready = False
history_store = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

# ==================== APP ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Start...")
    try:
        init_chroma()
        asyncio.create_task(load_instructions_async(force_reindex=False))
    except Exception as e:
        print(f"❌ Init error: {e}")
    yield
    print("🛑 Shutting down...")


app = FastAPI(title="Support AI (Instructions first, Web fallback)", lifespan=lifespan)

# CORS middleware only (важно для preflight OPTIONS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not set")
if not SERPER_API_KEY:
    print("ℹ️ SERPER_API_KEY not set (web fallback disabled)")

# ==================== MODELS ====================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    source: str = "instructions"  # instructions | web | none

# ==================== RAG ====================

def init_chroma():
    global chroma_client, collection
    if chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Collection found: {COLLECTION_NAME}, docs={collection.count()}")
        except Exception:
            print("ℹ️ Collection not found, creating new.")
            collection = chroma_client.create_collection(name=COLLECTION_NAME)


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Loading embedding model...")
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded")
    return embedding_model


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ PDF read error {pdf_path}: {e}")
    return text


def split_text_into_chunks(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    words = (text or "").split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        ch = " ".join(words[i:i + chunk_size]).strip()
        if ch:
            chunks.append(ch)
    return chunks


def detect_device_from_text(text: str) -> str:
    low = (text or "").lower()
    matched = []
    for dev, kws in DEVICE_KEYWORDS.items():
        if any(k in low for k in kws):
            matched.append(dev)
    return ",".join(matched) if matched else "unknown"


async def load_instructions_async(force_reindex: bool = False):
    global is_rag_ready
    try:
        await asyncio.to_thread(load_instructions_sync, force_reindex)
        is_rag_ready = True
        print("✅ RAG ready")
    except Exception as e:
        is_rag_ready = False
        print(f"❌ Instruction load error: {e}")


def load_instructions_sync(force_reindex: bool = False):
    global collection
    init_chroma()

    if force_reindex and collection is not None:
        try:
            collection.delete(where={})
            print("🧹 Collection cleared")
        except Exception as e:
            print(f"⚠️ Could not clear collection: {e}")

    if collection.count() > 0 and not force_reindex:
        print("ℹ️ Collection already populated, skip.")
        return

    if not os.path.exists(INSTRUCTIONS_DIR):
        os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)
        print(f"⚠️ Folder created but empty: {INSTRUCTIONS_DIR}")
        return

    files = glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.pdf")) + glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.txt"))
    if not files:
        print(f"⚠️ No PDF/TXT in {INSTRUCTIONS_DIR}")
        return

    print(f"📁 Found files: {[os.path.basename(f) for f in files]}")

    all_chunks, metas, ids = [], [], []
    idx = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"📄 Processing: {fname}")

        if fpath.lower().endswith(".pdf"):
            text = extract_text_from_pdf(fpath)
        else:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text or not text.strip():
            print(f"⚠️ Empty file: {fname}")
            continue

        chunks = split_text_into_chunks(text)
        if not chunks:
            continue

        by_name = detect_device_from_text(fname)
        by_text = detect_device_from_text(text[:25000])
        if by_name == "unknown":
            dev_tag = by_text
        elif by_text == "unknown":
            dev_tag = by_name
        else:
            dev_tag = ",".join(sorted(set((by_name + "," + by_text).split(","))))

        for ch in chunks:
            all_chunks.append(ch)
            metas.append({"source": fname, "device": dev_tag})
            ids.append(f"{fname}_{idx}")
            idx += 1

    if not all_chunks:
        print("⚠️ No chunks to index.")
        return

    model = get_embedding_model()
    embeddings = []
    total = len(all_chunks)
    print(f"🔄 Embedding {total} chunks...")

    for i, ch in enumerate(all_chunks):
        emb = list(model.embed([ch]))[0].tolist()
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            print(f"   embedded: {i+1}/{total}")

    batch = 64
    for i in range(0, total, batch):
        collection.add(
            embeddings=embeddings[i:i + batch],
            documents=all_chunks[i:i + batch],
            metadatas=metas[i:i + batch],
            ids=ids[i:i + batch],
        )
        print(f"   indexed: {min(i+batch, total)}/{total}")

    print(f"✅ Indexed chunks: {total}")


def extract_device_from_query(query: str) -> Optional[str]:
    q = (query or "").lower()
    for dev, kws in DEVICE_KEYWORDS.items():
        if any(k in q for k in kws):
            return dev
    return None


def search_instructions(query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict]]:
    if not is_rag_ready:
        return []

    init_chroma()
    if collection is None or collection.count() == 0:
        return []

    model = get_embedding_model()
    q_emb = list(model.embed([query]))[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k * 3)

    if not res or not res.get("documents") or not res["documents"][0]:
        return []

    frags: List[Tuple[str, Dict]] = []
    for i in range(len(res["documents"][0])):
        doc = res["documents"][0][i]
        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
        frags.append((doc, meta))

    target = extract_device_from_query(query)
    if target:
        filtered = [(d, m) for d, m in frags if target in m.get("device", "")]
        if filtered:
            return filtered[:top_k]
        return frags[:top_k]

    return frags[:top_k]


def build_context(frags: List[Tuple[str, Dict]]) -> str:
    if not frags:
        return "Brak dodatkowego kontekstu."

    parts = []
    total = 0
    for i, (doc, meta) in enumerate(frags, start=1):
        d = doc[:MAX_FRAGMENT_LENGTH] + ("..." if len(doc) > MAX_FRAGMENT_LENGTH else "")
        part = f"[Fragment {i} z pliku {meta.get('source', 'nieznany')}]\n{d}"
        if total + len(part) > MAX_CONTEXT_CHARS:
            parts.append("... (kontekst przycięty)")
            break
        parts.append(part)
        total += len(part)

    return "\n\n---\n".join(parts)

# ==================== WEB FALLBACK ====================

async def web_search_fallback(query: str, top_n: int = 5) -> List[Dict]:
    if not SERPER_API_KEY:
        return []

    payload = {"q": query, "num": top_n}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(SERPER_URL, headers=headers, json=payload)
            if r.status_code != 200:
                return []
            data = r.json()
            organic = data.get("organic", []) or []
            return [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "snippet": x.get("snippet", ""),
                }
                for x in organic[:top_n]
            ]
    except Exception:
        return []


def web_results_to_context(results: List[Dict]) -> str:
    if not results:
        return "Brak wyników web."
    out = []
    for i, r in enumerate(results, start=1):
        out.append(
            f"[Web {i}] {r.get('title', '')}\nURL: {r.get('link', '')}\nSnippet: {r.get('snippet', '')}"
        )
    return "\n\n---\n".join(out)

# ==================== LLM CALL ====================

async def call_groq(messages: List[Dict]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "Brak klucza Groq")

    max_retries = 5
    retry_delay = 1

    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_SECONDS) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": GROQ_TEMPERATURE,
                        "max_tokens": GROQ_MAX_TOKENS,
                        "top_p": GROQ_TOP_P,
                    },
                )

                if resp.status_code == 429:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue

                if resp.status_code != 200:
                    raise HTTPException(502, f"Groq: {resp.status_code}")

                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (2 ** attempt))

    raise HTTPException(503, "Serwis przeciążony")

# ==================== UTIL ====================

def cleanup():
    now = datetime.now()
    for sid in list(history_store.keys()):
        if now - history_store[sid]["last_updated"] > timedelta(minutes=60):
            del history_store[sid]

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {"status": "ok", "service": "assistics-backend"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(400, "Wiadomość nie może być pusta")

    sid = request.session_id or "default"
    cleanup()

    session = history_store[sid]
    session["last_updated"] = datetime.now()
    history = session["messages"]

    # 1) Always search instructions first
    frags = search_instructions(request.message, top_k=TOP_K)
    docs_found = len(frags)
    instr_context = build_context(frags) if frags else "Brak dodatkowego kontekstu."

    # 2) Fallback to web if weak/no instruction context
    source = "instructions"
    web_context = "Brak wyników web."
    if docs_found < RAG_MIN_HITS:
        web_results = await web_search_fallback(request.message, top_n=5)
        if web_results:
            web_context = web_results_to_context(web_results)
            source = "web"
        else:
            source = "none"

    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego.
Priorytet:
1) Najpierw korzystaj z KONTEKST_INSTRUKCJE.
2) Jeśli brak odpowiedzi w instrukcjach, użyj KONTEKST_WEB.
3) Jeśli nadal brak danych, odpowiedz dokładnie: "Nie znalazłem tej informacji. Skontaktuj się z serwisem."

ZASADY:
- Odpowiadaj TYLKO po polsku.
- Odpowiedź: krótka lista kroków z emoji 1️⃣ 2️⃣ 3️⃣ ...
- Bez wstępów i bez teorii.
- Nie wymyślaj kroków spoza kontekstu.
- Dla pytań EVO/Paradox zakładaj klawiaturę EVO640.

KONTEKST_INSTRUKCJE:
{instr_context}

KONTEKST_WEB:
{web_context}
""".strip()

    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}

    history.append({"role": "user", "content": request.message})

    if len(history) > 10:
        history = [history[0]] + history[-9:]

    try:
        reply = await call_groq(history)
        history.append({"role": "assistant", "content": reply})
        session["messages"] = history
        return ChatResponse(reply=reply, session_id=sid, source=source)
    except Exception:
        return ChatResponse(
            reply="Wystąpił błąd. Spróbuj później.",
            session_id=sid,
            source="none",
        )


@app.get("/health")
async def health():
    init_chroma()
    docs_count = collection.count() if collection else 0
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "serper_configured": bool(SERPER_API_KEY),
        "rag_ready": is_rag_ready,
        "instructions_loaded": docs_count,
        "instructions_dir": INSTRUCTIONS_DIR,
        "collection": COLLECTION_NAME,
    }


@app.get("/list_instructions")
async def list_instructions():
    if not os.path.exists(INSTRUCTIONS_DIR):
        return {"error": f"Brak folderu {INSTRUCTIONS_DIR}"}
    files = [f for f in os.listdir(INSTRUCTIONS_DIR) if os.path.isfile(os.path.join(INSTRUCTIONS_DIR, f))]
    return {"files": files, "count": len(files)}


@app.get("/reload")
async def reload_instructions():
    try:
        asyncio.create_task(load_instructions_async(force_reindex=True))
        return {"status": "reloading_full"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS powinien działać"}


@app.get("/favicon.ico")
async def favicon():
    # чтобы убрать 404 в браузере
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
