import os
import glob
import sys
import asyncio
import re
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import PyPDF2

# Wyłączenie telemetrii ChromaDB
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

app = FastAPI(title="Wsparcie Techniczne AZS z RAG (fastembed)")

# TYLKO TE DWIE DOMENY – NIE ZMIENIAJ!
ALLOWED_ORIGINS = [
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app"
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

DEVICE_KEYWORDS = {
    "rosslare": ["rosslare", "ac-b32", "axtraxng"],
    "paradox": ["paradox", "evo192", "sp65", "sp4000"],
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
            print(f"✅ Znaleziono kolekcję {COLLECTION_NAME} z {collection.count()} dokumentami")
        except:
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            print(f"✅ Utworzono nową kolekcję {COLLECTION_NAME}")

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Ładowanie modelu fastembed...")
        from fastembed import TextEmbedding
        embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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

def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def load_instructions():
    global collection
    init_chroma()
    if collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane")
        return

    instr_dir = "instructions"
    if not os.path.exists(instr_dir):
        os.makedirs(instr_dir, exist_ok=True)
        return

    files = glob.glob(os.path.join(instr_dir, "*.pdf")) + glob.glob(os.path.join(instr_dir, "*.txt"))
    if not files:
        return

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
        if not text.strip():
            continue

        chunks = split_text_into_chunks(text)
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

    if not all_chunks:
        return

    model = get_embedding_model()
    embeddings = [emb.tolist() for emb in model.embed(all_chunks)]
    for i in range(0, len(all_chunks), 100):
        collection.add(
            embeddings=embeddings[i:i+100],
            documents=all_chunks[i:i+100],
            metadatas=metas[i:i+100],
            ids=ids[i:i+100]
        )
    print(f"✅ Załadowano {len(all_chunks)} fragmentów")

def extract_device_from_query(query: str) -> Optional[str]:
    q = query.lower()
    for dev, kw in DEVICE_KEYWORDS.items():
        if any(k in q for k in kw):
            return dev
    return None

def search_instructions(query: str, top_k: int = 7) -> List[Tuple[str, Dict]]:
    init_chroma()
    if collection is None or collection.count() == 0:
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

    if docs:
        print("📚 Znalezione:")
        for i, (d,m) in enumerate(zip(docs, metas)):
            print(f"   [{i}] Źródło: {m.get('source')}, urządzenie: {m.get('device')}")
            print(f"       {d[:150]}...")

    ctx_parts = []
    for i, (d,m) in enumerate(zip(docs, metas)):
        ctx_parts.append(f"[{i+1} z {m.get('source')} ({m.get('device')})]:\n{d}")
    context = "\n\n---\n".join(ctx_parts) or "Brak kontekstu."

    system = f"""
Jesteś inżynierem wsparcia (L2/L3) z 15-letnim doświadczeniem.
Systemy: CCTV (Bosch, Siemens, 3xLogic, Provision, Hikvision), Alarmy (Paradox, Babyware, Bosch Fire), KD (Rosslare).
Odpowiadaj TYLKO po polsku.
UŻYJ TEGO KONTEKSTU (jeśli pasuje do pytania):
{context}

Jeśli pytanie dotyczy innego urządzenia niż w kontekście – IGNORUJ kontekst.
Postępuj: diagnoza → szybkie przywrócenie → naprawa → prewencja.
Struktura: Problem, System, Krytyczność, Diagnoza, Szybkie przywrócenie, Naprawa trwała, Zapobieganie.
"""

    if not history:
        history.append({"role": "system", "content": system})
    else:
        history[0] = {"role": "system", "content": system}
    history.append({"role": "user", "content": request.message})
    if len(history) > 11:
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Zapytanie {sid}: {request.message[:50]}...")
        async with httpx.AsyncClient() as client:
            for attempt in range(5):
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": history,
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    timeout=30.0
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"⏳ Rate limit, próba {attempt+1}/5, czekam {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code != 200:
                    raise HTTPException(502, f"Groq: {resp.status_code}")
                data = resp.json()
                reply = data["choices"][0]["message"]["content"]
                break
            else:
                return ChatResponse(reply="Serwis przeciążony. Spróbuj za chwilę.", session_id=sid)

        history.append({"role": "assistant", "content": reply})
        session["messages"] = history
        return ChatResponse(reply=reply, session_id=sid)

    except Exception as e:
        print(f"💥 Błąd: {e}")
        return ChatResponse(reply="Wystąpił błąd. Spróbuj później.", session_id=sid)

# ==================== Pomocnicze ====================
@app.get("/health")
async def health():
    init_chroma()
    return {"status": "healthy", "groq_configured": bool(GROQ_API_KEY), "instructions_loaded": collection.count() if collection else 0}

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

@app.on_event("startup")
async def startup():
    print("🚀 Start...")
    init_chroma()
    load_instructions()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
