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

# Wyłączenie telemetrii ChromaDB (usuwa błąd "capture() takes 1 positional argument but 3 were given")
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

app = FastAPI(title="Wsparcie Techniczne AZS z RAG (fastembed)")

# CORS – zezwalamy tylko na frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assistics.netlify.app"],
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

# Słownik mapujący nazwy urządzeń na słowa kluczowe w nazwach plików
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
        print("⏳ Ładowanie modelu fastembed (może potrwać przy pierwszym uruchomieniu)...")
        # Sprawdzona wielojęzyczna model – dobrze radzi sobie z polskim
        embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Model fastembed załadowany")
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
        print(f"❌ Błąd podczas odczytu PDF {pdf_path}: {e}")
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
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        return

    instructions_dir = "instructions"
    if not os.path.exists(instructions_dir):
        print(f"⚠️ Folder '{instructions_dir}' nie istnieje. Tworzę.")
        os.makedirs(instructions_dir, exist_ok=True)
        return

    pdf_files = glob.glob(os.path.join(instructions_dir, "*.pdf"), recursive=False)
    txt_files = glob.glob(os.path.join(instructions_dir, "*.txt"), recursive=False)
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"⚠️ Brak plików PDF/TXT w folderze '{instructions_dir}'")
        return

    print(f"📁 Znalezione pliki: {[os.path.basename(f) for f in all_files]}")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0

    for file_path in all_files:
        print(f"📄 Przetwarzanie: {file_path}")
        filename = os.path.basename(file_path)

        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"❌ Błąd odczytu pliku {file_path}: {e}")
                continue

        if not text or not text.strip():
            print(f"⚠️ Plik {filename} jest pusty, pomijam.")
            continue

        chunks = split_text_into_chunks(text)
        print(f"   → {len(chunks)} fragmentów")

        # Określamy, jakie urządzenia pasują do nazwy pliku (dla metadanych)
        matched_devices = []
        lower_filename = filename.lower()
        for device, keywords in DEVICE_KEYWORDS.items():
            if any(kw in lower_filename for kw in keywords):
                matched_devices.append(device)
        device_tag = ",".join(matched_devices) if matched_devices else "unknown"

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename, "device": device_tag})
            all_ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1

    if not all_chunks:
        print("⚠️ Nie znaleziono żadnych fragmentów tekstu")
        return

    model = get_embedding_model()
    print("🔄 Generowanie embeddignów (fastembed)...")
    embeddings = list(model.embed(all_chunks))
    embeddings = [emb.tolist() for emb in embeddings]

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            embeddings=embeddings[i:i+batch_size],
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
        print(f"   Dodano {i+len(all_chunks[i:i+batch_size])} fragmentów")

    print(f"✅ Załadowano {len(all_chunks)} fragmentów instrukcji do bazy")

def extract_device_from_query(query: str) -> Optional[str]:
    """Próbuje wyciągnąć nazwę urządzenia z zapytania użytkownika."""
    lower_query = query.lower()
    for device, keywords in DEVICE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower_query:
                return device
    return None

def search_instructions(query: str, top_k: int = 10) -> List[Tuple[str, Dict]]:
    """
    Zwraca listę krotek (fragment, metadata) dla najbardziej pasujących fragmentów.
    """
    global collection
    init_chroma()
    if collection is None or collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = list(model.embed([query]))[0].tolist()

    # Najpierw szukamy ogólnie
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2  # weź więcej, potem przefiltrujemy
    )

    if not results or not results['documents']:
        return []

    # Próba wyciągnięcia urządzenia z zapytania
    target_device = extract_device_from_query(query)

    # Łączymy dokumenty z metadanymi
    fragments_with_meta = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        fragments_with_meta.append((doc, meta))

    # Jeśli w zapytaniu wykryto urządzenie, filtrujemy tylko te fragmenty,
    # które pochodzą z instrukcji tego urządzenia (jeśli w ogóle są)
    if target_device:
        filtered = [(doc, meta) for (doc, meta) in fragments_with_meta 
                    if target_device in meta.get('device', '')]
        if filtered:
            print(f"🔍 Filtruję po urządzeniu '{target_device}': znaleziono {len(filtered)} fragmentów")
            return filtered[:top_k]
        else:
            print(f"⚠️ Nie znaleziono fragmentów dla urządzenia '{target_device}', używam ogólnych")
            return fragments_with_meta[:top_k]
    else:
        return fragments_with_meta[:top_k]

# ==================== KONIEC RAG ====================

# ==================== Historia rozmów ====================
history_store: Dict[str, Dict] = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

def cleanup_old_sessions(max_age_minutes: int = 60):
    now = datetime.now()
    to_delete = []
    for sid, data in history_store.items():
        if now - data["last_updated"] > timedelta(minutes=max_age_minutes):
            to_delete.append(sid)
    for sid in to_delete:
        del history_store[sid]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# ==================== Endpointy ====================

@app.get("/")
async def root():
    return {"message": "Asystent techniczny AZS z RAG (fastembed) działa. Użyj /chat do wysyłania wiadomości."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API Groq")

    sid = request.session_id or "default"
    cleanup_old_sessions()
    session_data = history_store[sid]
    session_data["last_updated"] = datetime.now()
    history = session_data["messages"]

    # RAG – zaawansowane wyszukiwanie z filtracją
    fragments_with_meta = search_instructions(request.message, top_k=7)
    relevant_chunks = [doc for doc, _ in fragments_with_meta]
    metadatas = [meta for _, meta in fragments_with_meta]

    if relevant_chunks:
        print("📚 Znalezione fragmenty (z metadanymi):")
        for i, (chunk, meta) in enumerate(zip(relevant_chunks, metadatas)):
            print(f"   [{i}] Źródło: {meta.get('source', 'nieznane')}, urządzenie: {meta.get('device', 'nieznane')}")
            print(f"       {chunk[:150]}...")
    else:
        print("📭 Nie znaleziono żadnych fragmentów")

    # Przygotowujemy kontekst z podaniem źródła
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(relevant_chunks, metadatas)):
        source = meta.get('source', 'nieznane')
        device = meta.get('device', 'nieznane')
        context_parts.append(f"[Fragment {i+1} z pliku {source} (dotyczy: {device})]:\n{chunk}")
    context = "\n\n---\n".join(context_parts) if context_parts else "Brak dodatkowego kontekstu."

    # System prompt – w pełni po polsku, zgodny z Twoimi wymaganiami
    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego (L2/L3) z ponad 15-letnim doświadczeniem w integracji systemów bezpieczeństwa w UE.
Twoim zadaniem jest rozwiązywanie problemów technicznych zgłaszanych przez operatorów stacji benzynowych.

**Dostępne systemy (poziom użytkownika):**
- CCTV: Bosch (BVMS, DIVAR IP, Avenar), Siemens Vectis, 3xLogic, Provision ISR
- Alarmy: Paradox (EVO192, SP65, SP4000), Babyware, Bosch Avenar Fire
- Kontrola dostępu: Rosslare (AC-B32, AxTraxNG)

**DODATKOWY KONTEKST Z INSTRUKCJI (UŻYJ GO JAKO PODSTAWY ODPOWIEDZI):**
{context}

**ZASADY PRACY:**
1.  Odpowiadaj TYLKO po polsku.
2.  Jeśli w kontekście znajduje się informacja o konkretnym urządzeniu, a pytanie użytkownika dotyczy innego urządzenia – NIE UŻYWAJ tej informacji.
3.  Jeśli kontekst nie zawiera odpowiedzi na pytanie, poinformuj o tym i zaproponuj kontakt z serwisem.
4.  Postępuj zgodnie z inżynierskim procesem: diagnoza → szybkie przywrócenie → naprawa trwała → prewencja.
5.  Bądź konkretny, zadawaj pytania, prowadź użytkownika.

**STRUKTURA ODPOWIEDZI:**
- **Problem:** (krótki opis)
- **System:** (jaki system/urządzenie)
- **Krytyczność:** (CRITICAL/HIGH/MEDIUM/LOW)
- **Diagnoza:** (kroki, które podjąłeś/zadajesz)
- **Szybkie przywrócenie:** (tymczasowe rozwiązanie, jeśli możliwe)
- **Naprawa trwała:** (root cause i właściwe rozwiązanie)
- **Zapobieganie:** (rekomendacja na przyszłość)
"""

    # Zarządzanie historią
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}

    history.append({"role": "user", "content": request.message})

    if len(history) > 11:
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Zapytanie (sesja {sid}): {request.message[:50]}...")
        if relevant_chunks:
            print(f"   Znaleziono {len(relevant_chunks)} fragmentów w instrukcjach")

        max_retries = 3
        retry_delay = 1

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",  # stabilna, szybka, wysokie limity
                            "messages": history,
                            "temperature": 0.3,
                            "max_tokens": 500,
                            "top_p": 0.9
                        },
                        timeout=30.0
                    )

                    if response.status_code == 429:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⏳ Rate limit, próba {attempt+1}/{max_retries}, czekam {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        print(f"❌ Błąd Groq: {response.status_code} - {response.text}")
                        raise HTTPException(status_code=502, detail="Błąd komunikacji z Groq")

                    data = response.json()
                    reply = data["choices"][0]["message"]["content"]
                    break

                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⏳ Timeout, próba {attempt+1}/{max_retries}, czekam {wait_time}s...")
                    await asyncio.sleep(wait_time)
            else:
                return ChatResponse(
                    reply="Przepraszam, serwis tymczasowo niedostępny. Spróbuj później.",
                    session_id=sid
                )

        history.append({"role": "assistant", "content": reply})
        session_data["messages"] = history

        print(f"✅ Odpowiedź wysłana (sesja {sid})")
        return ChatResponse(reply=reply, session_id=sid)

    except httpx.TimeoutException:
        return ChatResponse(reply="Przepraszam, serwis nie odpowiada. Spróbuj za chwilę.", session_id=sid)
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        return ChatResponse(reply="Wystąpił błąd. Proszę spróbować później.", session_id=sid)

# ==================== Dodatkowe endpointy pomocnicze ====================

@app.options("/chat")
async def options_chat():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "https://assistics.netlify.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.options("/{rest_path:path}")
async def preflight_handler(rest_path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "https://assistics.netlify.app",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.get("/health")
async def health():
    init_chroma()
    docs_count = collection.count() if collection else 0
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "rag_ready": docs_count > 0,
        "instructions_loaded": docs_count
    }

@app.get("/list_instructions")
async def list_instructions():
    if not os.path.exists("instructions"):
        return {"exists": False, "error": "Folder 'instructions' nie istnieje"}
    files = os.listdir("instructions")
    files = [f for f in files if os.path.isfile(os.path.join("instructions", f))]
    return {
        "exists": True,
        "files": files,
        "count": len(files)
    }

@app.get("/reload")
async def reload_instructions():
    try:
        # Opcjonalnie: wyczyszczenie kolekcji przed ponownym załadowaniem
        # if collection:
        #     collection.delete(where={})
        load_instructions()
        count = collection.count() if collection else 0
        return {"status": "ok", "loaded": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/list_models")
async def list_models():
    """Endpoint do sprawdzania dostępnych modeli Groq."""
    if not GROQ_API_KEY:
        return {"error": "Brak klucza API"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
            )
            if response.status_code == 200:
                models = response.json()
                model_ids = [m["id"] for m in models.get("data", [])]
                return {"available_models": model_ids}
            else:
                return {"error": f"Błąd {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug_env")
async def debug_env():
    return {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "instructions_exists": os.path.exists("instructions"),
        "instructions_dir_list": os.listdir(".") if os.path.exists(".") else []
    }

@app.on_event("startup")
async def startup_event():
    print("🚀 Uruchamianie aplikacji...")
    init_chroma()
    load_instructions()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
