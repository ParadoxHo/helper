import os
import glob
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import PyPDF2

# RAG
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

app = FastAPI(title="Wsparcie Techniczne AZS z RAG")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assistics.netlify.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY nie ustawiony!")

# ==================== RAG ====================
# Многоязычная модель – хорошо понимает польский
print("Ładowanie modelu embeddignów (multilingual)...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ChromaDB с сохранением на диск
CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection_name = "azs_instructions"

# Пытаемся получить существующую коллекцию
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"✅ Znaleziono kolekcję {collection_name} z {collection.count()} dokumentami")
except:
    collection = None
    print("ℹ️ Kolekcja nie istnieje – zostanie utworzona przy pierwszym ładowaniu")

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"❌ Błąd podczas odczytu PDF {pdf_path}: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def load_instructions():
    global collection
    
    if collection and collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        return
    
    if collection is None:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"✅ Utworzono nową kolekcję {collection_name}")
    
    # Ищем файлы в папке instructions
    instructions_dir = "instructions"
    if not os.path.exists(instructions_dir):
        print(f"⚠️ Folder '{instructions_dir}' nie istnieje. Tworzę.")
        os.makedirs(instructions_dir, exist_ok=True)
        return
    
    pdf_files = glob.glob(f"{instructions_dir}/*.pdf")
    txt_files = glob.glob(f"{instructions_dir}/*.txt")
    all_files = pdf_files + txt_files
    
    if not all_files:
        print(f"⚠️ Brak plików w folderze '{instructions_dir}'")
        return
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0
    
    for file_path in all_files:
        print(f"📄 Przetwarzanie: {file_path}")
        filename = os.path.basename(file_path)
        
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        if not text.strip():
            print(f"⚠️ Plik {filename} jest pusty, pomijam.")
            continue
        
        chunks = split_text_into_chunks(text)
        print(f"   → {len(chunks)} fragmentów")
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename})
            all_ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1
    
    if not all_chunks:
        print("⚠️ Nie znaleziono żadnych fragmentów tekstu")
        return
    
    print("🔄 Generowanie embeddignów...")
    embeddings = embedding_model.encode(all_chunks).tolist()
    
    # Добавляем пачками, чтобы избежать переполнения памяти
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

@app.on_event("startup")
async def startup_event():
    load_instructions()

def search_instructions(query: str, top_k: int = 3) -> List[str]:
    if collection is None or collection.count() == 0:
        return []
    
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if results and results['documents']:
        return results['documents'][0]
    return []
# ==================== RAG END ====================

# Хранилище истории диалогов
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

@app.get("/")
async def root():
    return {"message": "Asystent techniczny AZS z RAG działa. Użyj /chat do wysyłania wiadomości."}

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

    # RAG: ищем релевантные фрагменты
    relevant_chunks = search_instructions(request.message, top_k=3)
    context = "\n\n---\n".join(relevant_chunks) if relevant_chunks else "Brak dodatkowego kontekstu."

    # Системный промпт
    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego pierwszej linii dla systemów bezpieczeństwa na stacjach benzynowych.
Twoi rozmówcy to pracownicy stacji (operatorzy), którzy nie są specjalistami. Mówią po polsku.

ZNASZ NASTĘPUJĄCE SYSTEMY (POZIOM UŻYTKOWNIKA):
- Monitoring: Bosch DIVAR, Bosch DIP, 3xLogic, Provision, Hikvision
- Alarmy: Paradox EVO192, SP65, SP4000, Satel Integra
- Kontrola dostępu: Rosslare B32 (zmiana kodu użytkownika)

MASZ DOSTĘP DO NASTĘPUJĄCYCH INSTRUKCJI (KONTEKST):
{context}

ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i rzeczowo.
2. Jeśli kontekst zawiera instrukcję – użyj jej jako podstawy odpowiedzi.
3. Jeśli nie ma kontekstu lub nie zawiera odpowiedzi, postępuj według ogólnych zasad diagnostyki:
   - sprawdź zasilanie (kable, korki)
   - sprawdź połączenia sieciowe (diody)
   - zresetuj urządzenie (odłącz zasilanie na 10 sekund)
   - sprawdź czystość czujek (kurz, pajęczyny)
   - wymień baterię w czujce bezprzewodowej
4. Jeśli problem jest poważny, zasugeruj wezwanie serwisu.
5. Bądź uprzejmy i cierpliwy.
6. Pamiętaj kontekst rozmowy – odpowiadaj na pytania użytkownika w sposób ciągły.
"""

    # Формируем историю
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        # Обновляем system prompt (первое сообщение)
        history[0] = {"role": "system", "content": system_prompt}

    history.append({"role": "user", "content": request.message})

    # Ограничиваем длину истории
    if len(history) > 11:
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Zapytanie (sesja {sid}): {request.message[:50]}...")
        if relevant_chunks:
            print(f"   Znaleziono {len(relevant_chunks)} fragmentów w instrukcjach")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": history,
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "top_p": 0.9
                },
                timeout=30.0
            )

            if response.status_code != 200:
                print(f"❌ Błąd Groq: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="Błąd komunikacji z Groq")

            data = response.json()
            reply = data["choices"][0]["message"]["content"]

        history.append({"role": "assistant", "content": reply})
        session_data["messages"] = history

        print(f"✅ Odpowiedź wysłana (sesja {sid})")
        return ChatResponse(reply=reply, session_id=sid)

    except httpx.TimeoutException:
        return ChatResponse(reply="Przepraszam, serwis nie odpowiada. Spróbuj ponownie za chwilę.", session_id=sid)
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        return ChatResponse(reply="Wystąpił błąd. Proszę spróbować później.", session_id=sid)

@app.options("/chat")
async def options_chat():
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.get("/health")
async def health():
    docs_count = collection.count() if collection else 0
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "rag_ready": bool(collection and collection.count() > 0),
        "instructions_loaded": docs_count
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
