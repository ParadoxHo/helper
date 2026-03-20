import os
import glob
import sys
import asyncio
import time
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

# Разрешённые домены для CORS (фронтенд)
ALLOWED_ORIGINS = [
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app",
    "https://assistics.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Start...")
    try:
        init_chroma()
        asyncio.create_task(load_instructions_async())
    except Exception as e:
        print(f"❌ Błąd podczas inicjalizacji: {e}")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Wsparcie Techniczne AZS z RAG (poprawione)",
    lifespan=lifespan
)

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
is_rag_ready = False

# Словарь ключевых слов для определения устройства по имени файла
DEVICE_KEYWORDS = {
    "rosslare": ["rosslare", "ac-b31", "ac-b32", "axtraxng"],
    "paradox": ["paradox", "evo192", "sp65", "sp4000", "evo"],
    "bosch": ["bosch", "bvms", "divar", "avenar"],
    "siemens": ["siemens", "vectis"],
    "3xlogic": ["3xlogic"],
    "provision": ["provision", "isr"],
    "satel": ["satel", "integra"],
    "babyware": ["babyware"],
    "hikvision": ["hikvision"]
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
        print("⏳ Ładowanie modelu fastembed...")
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

def split_text_into_chunks(text: str, chunk_size: int = 150, overlap: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def load_instructions_async():
    global is_rag_ready
    try:
        await asyncio.to_thread(load_instructions_sync)
    except Exception as e:
        print(f"❌ Błąd podczas ładowania instrukcji: {e}")
        is_rag_ready = False
    else:
        is_rag_ready = True
        print("✅ RAG gotowy")

def load_instructions_sync():
    global collection
    init_chroma()

    if collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        return

    instr_dir = "instructions"
    if not os.path.exists(instr_dir):
        os.makedirs(instr_dir, exist_ok=True)
        print("⚠️ Folder instructions utworzony, ale pusty.")
        return

    files = glob.glob(os.path.join(instr_dir, "*.pdf")) + glob.glob(os.path.join(instr_dir, "*.txt"))
    if not files:
        print("⚠️ Brak plików PDF/TXT w folderze instructions")
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

        time.sleep(0.1)

    if not all_chunks:
        print("⚠️ Nie znaleziono żadnych fragmentów tekstu")
        return

    model = get_embedding_model()
    print("🔄 Generowanie embeddignów (fastembed)...")
    embeddings = []
    total = len(all_chunks)
    for i, chunk in enumerate(all_chunks):
        emb = list(model.embed([chunk]))[0].tolist()
        embeddings.append(emb)
        if (i+1) % 5 == 0:
            print(f"   Przetworzono {i+1}/{total} fragmentów")
            time.sleep(0.2)

    print(f"   Wygenerowano {len(embeddings)} embeddingów")

    batch_size = 30
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            embeddings=embeddings[i:i+batch_size],
            documents=all_chunks[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"   Dodano {i+len(all_chunks[i:i+batch_size])} fragmentów")
        time.sleep(0.3)

    print(f"✅ Załadowano {len(all_chunks)} fragmentów instrukcji do bazy")

def extract_device_from_query(query: str) -> Optional[str]:
    """Wykrywa nazwę urządzenia w zapytaniu użytkownika."""
    q = query.lower()
    for dev, kw in DEVICE_KEYWORDS.items():
        if any(k in q for k in kw):
            return dev
    return None

def search_instructions(query: str, top_k: int = 15) -> List[Tuple[str, Dict]]:
    """
    Zwraca fragmenty instrukcji dla wykrytego urządzenia.
    Jeśli urządzenie nie zostało wykryte, zwraca pustą listę.
    """
    if not is_rag_ready:
        print("⏳ RAG jeszcze nie gotowy, pomijam wyszukiwanie.")
        return []
    init_chroma()
    if collection is None or collection.count() == 0:
        print("⚠️ Baza instrukcji pusta")
        return []

    target = extract_device_from_query(query)
    if target is None:
        print("📌 Nie wykryto urządzenia – zwracam pustą listę.")
        return []

    model = get_embedding_model()
    q_emb = list(model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k*2)
    if not results or not results['documents']:
        return []

    frags = []
    for i in range(len(results['documents'][0])):
        frags.append((results['documents'][0][i], results['metadatas'][0][i] if results['metadatas'] else {}))

    filtered = [(d, m) for d, m in frags if target in m.get('device', '')]
    if filtered:
        print(f"🔍 Filtruję '{target}': znaleziono {len(filtered)} fragmentów")
        return filtered[:top_k]
    else:
        print(f"⚠️ Brak fragmentów dla urządzenia '{target}' – zwracam pustą listę.")
        return []

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

@app.options("/chat")
async def options_chat():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "https://assistics.vercel.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }
    )

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

    frags = search_instructions(request.message, top_k=15)
    docs = [d for d,_ in frags]
    metas = [m for _,m in frags]

    # Przygotowujemy kontekst (obcinamy długość fragmentów i sumaryczną długość)
    MAX_FRAGMENT_LENGTH = 500
    MAX_CONTEXT_CHARS = 3000
    context_parts = []
    total_chars = 0
    for i, (d, m) in enumerate(zip(docs, metas)):
        if len(d) > MAX_FRAGMENT_LENGTH:
            d = d[:MAX_FRAGMENT_LENGTH] + "..."
        part = f"[Fragment {i+1} z pliku {m.get('source', 'nieznany')}]\n{d}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            context_parts.append("... (kontekst przycięty)")
            break
        context_parts.append(part)
        total_chars += len(part)
    context = "\n\n---\n".join(context_parts) if context_parts else "Brak dodatkowego kontekstu."

    target_device = extract_device_from_query(request.message)

    # --- SYSTEM PROMPT (po polsku) ---
    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego z wieloletnim doświadczeniem w systemach bezpieczeństwa na stacjach benzynowych. Twoim zadaniem jest pomóc operatorowi stacji rozwiązać problem, podając **wyłącznie** informacje zawarte w poniższym kontekście. Nie korzystaj z własnej wiedzy.

KONTEKST (fragmenty instrukcji):
{context}

**ZASADY**:
1. Jeśli pytanie nie zawiera nazwy urządzenia (Rosslare, Paradox, Satel, Bosch, Hikvision itp.), a kontekst jest pusty – zapytaj: "Proszę podać nazwę urządzenia (np. Rosslare, Paradox, Satel)."
2. Jeśli kontekst jest pusty, ale urządzenie zostało podane – odpowiedz: "Nie znaleziono instrukcji dla urządzenia [nazwa]. Proszę skontaktować się z serwisem."
3. Jeśli kontekst zawiera fragmenty instrukcji, połącz je w jedną pełną, spójną procedurę. Opisz każdy krok od początku do końca – wejście w tryb programowania, kod, przyciski, potwierdzenia.
4. Nie pomijaj żadnego kroku, nawet jeśli w różnych fragmentach są rozproszone informacje – zbierz je wszystkie.
5. Nie mieszaj procedur z różnych urządzeń. Jeśli kontekst dotyczy innego urządzenia niż to, o które pyta użytkownik – zignoruj go (ale w tym przypadku kontekst powinien być już przefiltrowany).
6. Odpowiadaj **TYLKO po polsku**, w formie numerowanej listy z emoji (1️⃣, 2️⃣, 3️⃣...). Między krokami zostaw pustą linię (akapit).
7. Maksymalnie 10 kroków.
8. Nie dodawaj żadnych wstępów, podsumowań ani wyjaśnień technicznych – tylko same kroki.

**PRZYKŁAD POPRAWNEJ ODPOWIEDZI** (dla Rosslare AC-B31):
1️⃣ Wejdź w tryb programowania: naciśnij i przytrzymaj przycisk # przez 2 sekundy. Dioda Mode zgaśnie, dioda Door zapali się na czerwono. Wprowadź Kod Programowania (fabrycznie 1234). Po poprawnym kodzie dioda Door zaświeci się na zielono.

2️⃣ Usuń starego użytkownika: naciśnij 8. Dioda Mode zaświeci się na czerwono, dioda Door na pomarańczowo. Wprowadź trzycyfrowy numer użytkownika (np. 001). Dioda Mode zacznie migać na czerwono – wprowadź Kod Programowania, aby potwierdzić usunięcie. Usłyszysz trzy krótkie sygnały.

3️⃣ Dodaj nowego użytkownika: naciśnij 7. Dioda Door zaświeci się na pomarańczowo. Wprowadź ten sam numer użytkownika (001). Dioda Mode zacznie migać na zielono – wprowadź nowy czterocyfrowy kod (np. 9876). Po zaakceptowaniu dioda Mode przestanie migać.

4️⃣ Wyjdź z trybu programowania: naciśnij i przytrzymaj # przez 2 sekundy. Usłyszysz trzy sygnały, dioda Door zgaśnie, a dioda Mode zaświeci się na zielono.
"""

    # Jeśli brak kontekstu i brak urządzenia
    if not docs and target_device is None:
        return ChatResponse(
            reply="Proszę podać nazwę urządzenia (np. Rosslare, Paradox, Satel). Bez tej informacji nie mogę udzielić dokładnej instrukcji.",
            session_id=sid
        )

    # Jeśli brak kontekstu, ale urządzenie jest znane
    if not docs and target_device:
        return ChatResponse(
            reply=f"Nie znaleziono instrukcji dla urządzenia {target_device}. Proszę skontaktować się z serwisem.",
            session_id=sid
        )

    # Zarządzanie historią (system + ostatnie 5 wiadomości)
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}
    history.append({"role": "user", "content": request.message})
    if len(history) > 6:
        history = [history[0]] + history[-5:]

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
                            "temperature": 0.1,
                            "max_tokens": 800,
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

# ==================== Endpointy pomocnicze ====================

@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS powinien działać"}

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
        asyncio.create_task(load_instructions_async())
        return {"status": "reloading"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
