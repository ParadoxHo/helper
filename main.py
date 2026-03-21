import os
import glob
import asyncio
import logging
import uuid
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import httpx
import PyPDF2

# Отключаем телеметрию ChromaDB, чтобы не засорять логи
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistics")

# ================= CONFIG =================
CHROMA_DIR = "chroma_db"
DOCS_DIR = "instructions"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # мультиязычная
TOP_K = 10
MAX_CONTEXT_CHARS = 4000

# Groq (вы используете его)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TIMEOUT = 30

# ================= GLOBALS =================
embedder = None
chroma_client = None
collection = None
rag_ready = False

# История сессий (in‑memory, TTL)
sessions = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

# ================= FASTAPI =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up...")
    try:
        init_vector_db()
        await load_instructions()
        log.info("RAG ready")
    except Exception as e:
        log.error(f"Startup error: {e}")
    yield
    log.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# === CORS middleware (разрешаем все источники для упрощения, но можно сузить) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Явный обработчик OPTIONS для надёжности (необязательно, но пусть будет)
@app.options("/{rest_path:path}")
async def options_handler(rest_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

# ================= MODELS =================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    source: str = "rag"  # для отладки

# ================= VECTOR DB =================
def init_vector_db():
    global chroma_client, collection
    if chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            log.info(f"Loaded existing collection with {collection.count()} documents")
        except:
            collection = chroma_client.create_collection(COLLECTION_NAME)
            log.info("Created new collection")

def get_embedder():
    global embedder
    if embedder is None:
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        log.info("Embedding model loaded")
    return embedder

# ================= INDEXING =================
async def load_instructions():
    global rag_ready
    if collection.count() > 0:
        log.info("Collection already contains data, skipping indexing")
        rag_ready = True
        return

    files = glob.glob(f"{DOCS_DIR}/*.pdf") + glob.glob(f"{DOCS_DIR}/*.txt")
    if not files:
        log.warning("No files found in instructions folder")
        rag_ready = True
        return

    log.info(f"Found {len(files)} files, indexing...")
    all_chunks = []
    all_metas = []
    all_ids = []

    for file in files:
        if file.endswith(".pdf"):
            text = ""
            with open(file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t
        else:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        if not text.strip():
            continue

        # Разбивка на чанки
        words = text.split()
        chunk_size = 240
        overlap = 60
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                all_chunks.append(chunk)
                all_metas.append({"file": os.path.basename(file)})
                all_ids.append(f"{os.path.basename(file)}_{i}")

    if not all_chunks:
        log.warning("No text chunks extracted")
        rag_ready = True
        return

    embedder = get_embedder()
    embeddings = list(embedder.embed(all_chunks))

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            embeddings=[e.tolist() for e in embeddings[i:i+batch_size]],
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metas[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
        )
    log.info(f"Indexed {len(all_chunks)} chunks")
    rag_ready = True

# ================= RETRIEVAL =================
def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    if not rag_ready or collection is None or collection.count() == 0:
        return []
    embedder = get_embedder()
    q_emb = list(embedder.embed([query]))[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k * 2,
        include=["documents"]
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return []
    return docs[:top_k]

# ================= CONTEXT =================
def format_context(chunks: List[str]) -> str:
    total = 0
    out = []
    for i, chunk in enumerate(chunks, 1):
        chunk = chunk[:1000]  # обрезаем
        block = f"[{i}] {chunk}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n---\n".join(out)

# ================= LLM CALL (Groq) =================
async def call_groq(messages: List[Dict]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set")

    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
        for attempt in range(5):
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": 0.05,
                        "max_tokens": 800,
                    }
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    log.error(f"Groq error {resp.status_code}: {resp.text}")
                    raise HTTPException(502, "Groq API error")
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.RequestError as e:
                log.error(f"Groq request error: {e}")
                if attempt == 4:
                    raise HTTPException(503, "LLM service unavailable")
                await asyncio.sleep(2 ** attempt)
    raise HTTPException(503, "LLM service unavailable")

# ================= CHAT ENDPOINT =================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    sid = req.session_id or "default"

    # Обновляем время сессии
    sessions[sid]["last_updated"] = datetime.now()
    history = sessions[sid]["messages"]

    # 1. Поиск релевантных фрагментов
    retrieved = retrieve(req.message, TOP_K)
    context = format_context(retrieved)
    log.info(f"Retrieved {len(retrieved)} chunks, context length: {len(context)}")
    if context:
        log.info(f"Context preview: {context[:500]}...")
    else:
        log.info("Context is empty")

    # 2. Системный промпт (сильный, с примерами)
    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego systemów bezpieczeństwa na stacjach benzynowych.

**Zasady:**
- Odpowiadaj TYLKO po polsku.
- Używaj WYŁĄCZNIE informacji zawartych w poniższym kontekście.
- Jeśli kontekst nie zawiera odpowiedzi, napisz: "Nie znalazłem tej informacji w instrukcjach. Proszę skontaktować się z serwisem."
- Jeśli pytanie nie zawiera nazwy urządzenia (Rosslare, Paradox, Satel itp.), a kontekst jest pusty, zapytaj o nazwę urządzenia.
- Odpowiedź podawaj w formie numerowanej listy krok po kroku (1️⃣, 2️⃣, 3️⃣...). Między krokami zostaw pustą linię.
- Nie dodawaj wstępów ani podsumowań.

KONTEKST:
{context if context else "Brak kontekstu."}

**Przykład prawidłowej odpowiedzi (dla Rosslare):**
1️⃣ Wejdź w tryb programowania: naciśnij i przytrzymaj przycisk # przez 2 sekundy. Dioda Mode zgaśnie, dioda Door zapali się na czerwono. Wprowadź Kod Programowania (fabrycznie 1234). Po poprawnym kodzie dioda Door zaświeci się na zielono.
2️⃣ Naciśnij 8. Dioda Mode zaświeci się na czerwono, dioda Door na pomarańczowo. Wprowadź trzycyfrowy numer użytkownika (np. 001). Dioda Mode zacznie migać na czerwono – wprowadź Kod Programowania, aby potwierdzić usunięcie. Usłyszysz trzy krótkie sygnały.
3️⃣ Naciśnij 7. Dioda Door zaświeci się na pomarańczowo. Wprowadź ten sam numer użytkownika (001). Dioda Mode zacznie migać na zielono – wprowadź nowy czterocyfrowy kod (np. 9876). Po zaakceptowaniu dioda Mode przestanie migać.
4️⃣ Wyjdź z trybu programowania: naciśnij i przytrzymaj # przez 2 sekundy. Usłyszysz trzy sygnały, dioda Door zgaśnie, a dioda Mode zaświeci się na zielono.
"""

    # 3. Строим историю
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}
    history.append({"role": "user", "content": req.message})

    # Ограничиваем историю 12 сообщениями (system + 11)
    if len(history) > 12:
        history = [history[0]] + history[-11:]

    # 4. Вызываем Groq
    try:
        reply = await call_groq(history)
    except Exception as e:
        log.exception("LLM call failed")
        reply = "Wystąpił błąd. Spróbuj później."

    history.append({"role": "assistant", "content": reply})
    sessions[sid]["messages"] = history

    return ChatResponse(reply=reply, session_id=sid, source="rag")

# ================= HEALTH & UTILITIES =================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_ready": rag_ready,
        "documents": collection.count() if collection else 0,
        "session_count": len(sessions)
    }

@app.get("/reload")
async def reload():
    global rag_ready, collection
    if collection:
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.create_collection(COLLECTION_NAME)
    rag_ready = False
    asyncio.create_task(load_instructions())
    return {"status": "reloading"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
