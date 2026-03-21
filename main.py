import os
import glob
import asyncio
import logging
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

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistics")

# === CONFIG ===
CHROMA_DIR = "chroma_db"
DOCS_DIR = "instructions"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # мультиязычная
TOP_K = 10
MAX_CONTEXT_CHARS = 4000

# LLM: Google Gemini (бесплатно, 1500 запросов/день)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # или gemini-1.5-flash
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# Если Gemini недоступен, можно использовать Groq, но для простоты оставим Gemini
USE_GROQ = False  # переключите на True, если хотите Groq

# Groq (запасной вариант)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# === GLOBALS ===
embedder = None
chroma_client = None
collection = None
rag_ready = False

# История сессий (in‑memory, с TTL)
sessions = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

# === FASTAPI ===
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Явный обработчик OPTIONS для надёжности
@app.options("/chat")
async def options_chat():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })

# === MODELS ===
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# === VECTOR DB (ChromaDB) ===
def init_vector_db():
    global chroma_client, collection
    if chroma_client is None:
        import chromadb
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
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        log.info("Embedding model loaded")
    return embedder

# === INDEXING ===
async def load_instructions():
    global rag_ready
    if collection.count() > 0:
        log.info("Collection already contains data, skipping indexing")
        rag_ready = True
        return

    files = glob.glob(f"{DOCS_DIR}/*.pdf") + glob.glob(f"{DOCS_DIR}/*.txt")
    if not files:
        log.warning("No files found in instructions folder")
        rag_ready = True  # ничего не делаем, но не блокируем
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

        # Простое разбиение на чанки (без сложных алгоритмов)
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

    # Генерация эмбеддингов
    embedder = get_embedder()
    embeddings = list(embedder.embed(all_chunks))

    # Добавляем в ChromaDB
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

# === RETRIEVAL ===
def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    if not rag_ready:
        return []
    if collection is None or collection.count() == 0:
        return []
    embedder = get_embedder()
    q_emb = list(embedder.embed([query]))[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k * 2,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return []
    # Простая фильтрация по устройству (если нужно)
    # Здесь можно добавить, но пока вернём все
    return docs[:top_k]  # пока без MMR, просто top_k

# === CONTEXT FORMATTING ===
def format_context(chunks: List[str]) -> str:
    total = 0
    out = []
    for i, chunk in enumerate(chunks, 1):
        chunk = chunk[:1000]  # обрезаем слишком длинные
        block = f"[{i}] {chunk}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n---\n".join(out)

# === LLM CALL (Gemini) ===
async def call_gemini(messages: List[Dict]) -> str:
    # Формируем запрос для Gemini (он принимает другой формат)
    # Преобразуем список OpenAI-style сообщений в текст с системным промптом
    system = ""
    user = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user = m["content"]
    # Для Gemini system и user передаются в одном prompt
    full_prompt = f"{system}\n\nUser: {user}\n\nAsystent (po polsku):"

    async with httpx.AsyncClient(timeout=30) as client:
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 800,
                "topP": 0.9
            }
        }
        try:
            resp = await client.post(GEMINI_URL, json=payload)
            if resp.status_code != 200:
                log.error(f"Gemini error: {resp.status_code} {resp.text}")
                raise HTTPException(502, "Gemini API error")
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log.exception("Gemini call failed")
            raise HTTPException(503, "LLM service unavailable")

# === LLM CALL (Groq) — резерв ===
async def call_groq(messages: List[Dict]) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 800,
                    }
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    raise HTTPException(502, "Groq API error")
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.RequestError:
                await asyncio.sleep(2 ** attempt)
        raise HTTPException(503, "LLM service unavailable")

# === CHAT ENDPOINT ===
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

    # 2. Формируем системный промпт (сильный, с примерами)
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
1️⃣ Wejdź w tryb programowania: naciśnij i przytrzymaj przycisk # przez 2 sekundy. Wprowadź Kod Programowania (fabrycznie 1234).
2️⃣ Naciśnij 8, wprowadź numer użytkownika (np. 001), potwierdź Kodem Programowania.
3️⃣ Naciśnij 7, wprowadź ten sam numer użytkownika, wprowadź nowy kod (np. 9876).
4️⃣ Naciśnij i przytrzymaj # przez 2 sekundy, aby wyjść.
"""

    # 3. Строим историю диалога
    # Если история пуста, добавляем system prompt, иначе обновляем
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}
    history.append({"role": "user", "content": req.message})

    # Ограничиваем историю 12 сообщениями (system + 11)
    if len(history) > 12:
        history = [history[0]] + history[-11:]

    # 4. Вызываем LLM
    try:
        if GEMINI_API_KEY and not USE_GROQ:
            reply = await call_gemini(history)
        elif GROQ_API_KEY:
            reply = await call_groq(history)
        else:
            raise HTTPException(500, "No LLM API key configured")
    except Exception as e:
        log.exception("LLM call failed")
        reply = "Wystąpił błąd. Spróbuj później."

    history.append({"role": "assistant", "content": reply})
    sessions[sid]["messages"] = history

    return ChatResponse(reply=reply, session_id=sid)

# === HEALTH ===
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
        # Удаляем коллекцию и пересоздаём
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.create_collection(COLLECTION_NAME)
    rag_ready = False
    asyncio.create_task(load_instructions())
    return {"status": "reloading"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# === RUN ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
