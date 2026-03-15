import os
import glob
import sys
import asyncio
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import PyPDF2

# Отключаем телеметрию ChromaDB
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

# --- Более лёгкая модель (всего ~80 МБ) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Заменили на лёгкую

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Start...")
    try:
        init_chroma()
        # Запускаем индексацию в фоне, чтобы не блокировать старт
        asyncio.create_task(load_instructions_async())
    except Exception as e:
        print(f"❌ Błąd podczas inicjalizacji: {e}")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Wsparcie Techniczne AZS z RAG (lekka wersja)",
    lifespan=lifespan
)

# --- CORS ---
ALLOWED_ORIGINS = [
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app",
    "https://assistics.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY nie ustawiony!")

# ==================== RAG ====================
embedding_model = None
collection = None
chroma_client = None
COLLECTION_NAME = "azs_instructions"
is_rag_ready = False  # Флаг готовности RAG

DEVICE_KEYWORDS = {
    "rosslare": ["rosslare", "ac-b32", "axtraxng"],
    "paradox": ["paradox", "evo192", "sp65", "sp4000", "evo"],
    "bosch": ["bosch", "bvms", "divar", "avenar"],
    "siemens": ["siemens", "vectis"],
    "3xlogic": ["3xlogic"],
    "provision": ["provision", "isr"],
    "satel": ["satel", "integra"],
    "babyware": ["babyware"],
    "hikvision": ["hikvision"]
}

def init_chroma():
    global chroma_client, collection
    if chroma_client is None:
        CHROMA_DIR = "./chroma_db"
        os.makedirs(CHROMA_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            cnt = collection.count()
            print(f"✅ Znaleziono kolekcję {COLLECTION_NAME} z {cnt} dokumentami")
        except Exception:
            print("ℹ️ Kolekcja nie istnieje, tworzę nową.")
            collection = chroma_client.create_collection(name=COLLECTION_NAME)

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Ładowanie modelu fastembed (lekki)...")
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        print("✅ Model załadowany")
    return embedding_model

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Błąd PDF {pdf_path}: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Уменьшенный размер чанков для экономии памяти"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def load_instructions_async():
    """Асинхронная загрузка инструкций (чтобы не блокировать старт)"""
    global is_rag_ready
    try:
        load_instructions()
    except Exception as e:
        print(f"❌ Błąd podczas ładowania instrukcji: {e}")
    finally:
        is_rag_ready = True
        print("✅ RAG gotowy")

def load_instructions():
    global collection, is_rag_ready
    init_chroma()

    if collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        is_rag_ready = True
        return

    instr_dir = "instructions"
    if not os.path.exists(instr_dir):
        os.makedirs(instr_dir, exist_ok=True)
        print("⚠️ Folder instructions utworzony, ale pusty.")
        is_rag_ready = True
        return

    files = glob.glob(os.path.join(instr_dir, "*.pdf")) + glob.glob(os.path.join(instr_dir, "*.txt"))
    if not files:
        print("⚠️ Brak plików PDF/TXT w folderze instructions")
        is_rag_ready = True
        return

    print(f"📁 Znalezione pliki: {[os.path.basename(f) for f in files]}")

    all_chunks, metas, ids = [], [], []
    chunk_id = 0

    for fpath in files:
        print(f"📄 Przetwarzanie: {fpath}")
        fname = os.path.basename(fpath)
        if fpath.endswith('.pdf'):
            text = extract_text_from_pdf(fpath)
        else:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        if not text or not text.strip():
            print(f"⚠️ Plik {fname} pusty, pomijam.")
            continue

        chunks = split_text_into_chunks(text)
        print(f"   → {len(chunks)} fragmentów")
        matched = []
        low_fname = fname.lower()
        for dev, kw in DEVICE_KEYWORDS.items():
            if any(k in low_fname for k in kw):
                matched.append(dev)
        dev_tag = ",".join(matched) if matched else "unknown"

        for ch in chunks:
            all_chunks.append(ch)
            metas.append({"source": fname, "device": dev_tag})
            ids.append(f"{fname}_{chunk_id}")
            chunk_id += 1

        # Небольшая пауза для освобождения памяти
        time.sleep(0.1)  # синхронная пауза (можно заменить на asyncio.sleep, но load_instructions синхронная)

    if not all_chunks:
        print("⚠️ Nie znaleziono żadnych fragmentów tekstu")
        return

    model = get_embedding_model()
    print("🔄 Generowanie embeddignów (fastembed)...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        # Генерируем эмбеддинги по одному, чтобы не перегружать память
        emb = list(model.embed([chunk]))[0].tolist()
        embeddings.append(emb)
        if (i+1) % 10 == 0:
            print(f"   Przetworzono {i+1}/{len(all_chunks)} fragmentów")
            time.sleep(0.2)  # пауза после каждых 10

    print(f"   Wygenerowano {len(embeddings)} embeddingów")

    batch_size = 50  # уменьшенный размер батча
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            embeddings=embeddings[i:i+batch_size],
            documents=all_chunks[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"   Dodano {i+len(all_chunks[i:i+batch_size])} fragmentów")
        time.sleep(0.2)

    print(f"✅ Załadowano {len(all_chunks)} fragmentów instrukcji do bazy")
    is_rag_ready = True

def extract_device_from_query(query: str) -> Optional[str]:
    q = query.lower()
    for dev, kw in DEVICE_KEYWORDS.items():
        if any(k in q for k in kw):
            return dev
    return None

def search_instructions(query: str, top_k: int = 7) -> List[Tuple[str, Dict]]:
    if not is_rag_ready:
        print("⏳ RAG jeszcze nie gotowy, pomijam wyszukiwanie.")
        return []
    global collection
    init_chroma()
    if collection is None or collection.count() == 0:
        print("⚠️ Baza instrukcji pusta")
        return []
    model = get_embedding_model()
    q_emb = list(model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k*2)
    if not results or not results['documents']:
        return []

    frags = []
    for i in range(len(results['documents'][0])):
        frags.append((results['documents'][0][i], results['metadatas'][0][i] if results['metadatas'] else {}))

    target = extract_device_from_query(query)
    if target:
        filtered = [(d, m) for d, m in frags if target in m.get('device', '')]
        if filtered:
            print(f"🔍 Filtruję '{target}': {len(filtered)} fragmentów")
            return filtered[:top_k]
        print(f"⚠️ Brak '{target}', używam ogólnych")
    return frags[:top_k]

# ==================== Endpointy ====================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

history_store = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

def cleanup():
    now = datetime.now()
    for sid in list(history_store.keys()):
        if now - history_store[sid]["last_updated"] > timedelta(minutes=60):
            del history_store[sid]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(400, "Wiadomość nie może być pusta")
    if not GROQ_API_KEY:
        raise HTTPException(500, "Brak klucza Groq")

    sid = request.session_id or "default"
    cleanup()
    session = history_store[sid]
    session["last_updated"] = datetime.now()
    history = session["messages"]

    frags = search_instructions(request.message, 7)
    docs = [d for d,_ in frags]
    metas = [m for _,m in frags]

    context_parts = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        context_parts.append(f"[Źródło: {m.get('source')}]\n{d}")
    context = "\n\n".join(context_parts) if context_parts else "Brak dodatkowego kontekstu."

    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego. Twoim zadaniem jest pomóc operatorowi stacji benzynowej rozwiązać problem.

Dostępne systemy: CCTV (Bosch, Siemens, 3xLogic, Provision, Hikvision), alarmy (Paradox, Babyware, Bosch Fire), kontrola dostępu (Rosslare).

KONTEKST (jeśli dostępny):
{context}

ZASADY ODPOWIEDZI:
1. Odpowiadaj TYLKO po polsku.
2. Podawaj wyłącznie krótkie, konkretne instrukcje krok po kroku.
3. Każdy krok oznaczaj numerem i emotikoną (1️⃣, 2️⃣, 3️⃣, ...).
4. Nie dodawaj żadnych wstępów, podsumowań ani wyjaśnień technicznych.
5. Jeśli instrukcja wymaga sprawdzenia zasilania, napisz po prostu "Sprawdź zasilanie".
6. Jeśli problemu nie da się rozwiązać, napisz "Skontaktuj się z serwisem".
7. Nie używaj terminów takich jak "na podstawie instrukcji", "z dokumentacji" itp.
8. Odpowiedź powinna zawierać maksymalnie 5 kroków.
"""

    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}
    history.append({"role": "user", "content": request.message})
    if len(history) > 11:
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Zapytanie {sid}: {request.message[:50]}...")
        max_retries = 5
        retry_delay = 1
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": history,
                            "temperature": 0.2,
                            "max_tokens": 400,
                            "top_p": 0.9
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 429:
                        wait = retry_delay * (2 ** attempt)
                        print(f"⏳ Rate limit, próba {attempt+1}/{max_retries}, czekam {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code != 200:
                        raise HTTPException(502, f"Groq: {resp.status_code}")
                    data = resp.json()
                    reply = data["choices"][0]["message"]["content"]
                    break
                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise
                    wait = retry_delay * (2 ** attempt)
                    print(f"⏳ Timeout, próba {attempt+1}/{max_retries}, czekam {wait}s")
                    await asyncio.sleep(wait)
            else:
                return ChatResponse(reply="Serwis przeciążony. Spróbuj za chwilę.", session_id=sid)

        history.append({"role": "assistant", "content": reply})
        session["messages"] = history
        return ChatResponse(reply=reply, session_id=sid)

    except Exception as e:
        print(f"💥 Błąd: {e}")
        return ChatResponse(reply="Wystąpił błąd. Spróbuj później.", session_id=sid)

@app.get("/health")
async def health():
    init_chroma()
    docs_count = collection.count() if collection else 0
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "rag_ready": is_rag_ready,
        "instructions_loaded": docs_count
    }

@app.get("/list_instructions")
async def list_instructions():
    if not os.path.exists("instructions"):
        return {"error": "Brak folderu instructions"}
    files = [f for f in os.listdir("instructions") if os.path.isfile(os.path.join("instructions", f))]
    return {"files": files, "count": len(files)}

@app.get("/reload")
async def reload_instructions():
    try:
        if collection:
            collection.delete(where={})
        load_instructions()
        return {"loaded": collection.count()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import time  # добавлено для пауз
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
