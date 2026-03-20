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

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

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
    q = query.lower()
    for dev, kw in DEVICE_KEYWORDS.items():
        if any(k in q for k in kw):
            return dev
    return None

def search_instructions(query: str, top_k: int = 15) -> List[Tuple[str, Dict]]:
    """
    Ищет до top_k фрагментов, но только если устройство определено.
    Возвращает пустой список, если устройство не указано или для него нет инструкций.
    """
    if not is_rag_ready:
        print("⏳ RAG еще не готов, пропускаем поиск.")
        return []
    init_chroma()
    if collection is None or collection.count() == 0:
        print("⚠️ База инструкций пуста")
        return []

    target = extract_device_from_query(query)

    # Если устройство не определено – не возвращаем ничего, чтобы ассистент запросил его
    if target is None:
        print("📌 Устройство не определено – возвращаем пустой список.")
        return []

    model = get_embedding_model()
    q_emb = list(model.embed([query]))[0].tolist()
    # Запрашиваем больше, чем нужно, для фильтрации
    results = collection.query(query_embeddings=[q_emb], n_results=top_k*2)
    if not results or not results['documents']:
        return []

    frags = []
    for i in range(len(results['documents'][0])):
        frags.append((results['documents'][0][i], results['metadatas'][0][i] if results['metadatas'] else {}))

    # Оставляем только фрагменты, относящиеся к определённому устройству
    filtered = [(d, m) for d, m in frags if target in m.get('device', '')]
    if filtered:
        print(f"🔍 Фильтрую по '{target}': найдено {len(filtered)} фрагментов")
        return filtered[:top_k]
    else:
        print(f"⚠️ Нет фрагментов для устройства '{target}' – возвращаю пустой список.")
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
        raise HTTPException(400, "Сообщение не может быть пустым")
    if not GROQ_API_KEY:
        raise HTTPException(500, "Отсутствует ключ Groq")

    sid = request.session_id or "default"
    cleanup()
    session = history_store[sid]
    session["last_updated"] = datetime.now()
    history = session["messages"]

    frags = search_instructions(request.message, top_k=15)
    docs = [d for d,_ in frags]
    metas = [m for _,m in frags]

    # Подготовка контекста
    MAX_FRAGMENT_LENGTH = 500
    MAX_CONTEXT_CHARS = 3000
    context_parts = []
    total_chars = 0
    for i, (d, m) in enumerate(zip(docs, metas)):
        if len(d) > MAX_FRAGMENT_LENGTH:
            d = d[:MAX_FRAGMENT_LENGTH] + "..."
        part = f"[Фрагмент {i+1} из файла {m.get('source', 'неизвестный')}]\n{d}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            context_parts.append("... (контекст обрезан по лимиту)")
            break
        context_parts.append(part)
        total_chars += len(part)
    context = "\n\n---\n".join(context_parts) if context_parts else "Нет дополнительного контекста."

    target_device = extract_device_from_query(request.message)

    # --- УСИЛЕННЫЙ СИСТЕМНЫЙ ПРОМПТ ДЛЯ ИСЧЕРПЫВАЮЩИХ ОТВЕТОВ ---
    system_prompt = f"""
Ты — инженер технической поддержки с многолетним опытом работы с системами безопасности на АЗС.
Твоя задача — давать **максимально полные, исчерпывающие инструкции** пользователю.

**Важно:** у тебя есть доступ к контексту (фрагменты инструкций). Используй его для составления ответа.

КОНТЕКСТ:
{context}

**ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА**:
1. Если в контексте есть несколько фрагментов, которые описывают разные этапы одной процедуры (например, вход в режим, удаление пользователя, добавление нового), **объедини их в одну полную инструкцию**.
2. Описывай каждый шаг **от начала до конца**. Не пропускай ввод кодов, нажатия кнопок, переходы в режимы. Используй информацию из контекста, даже если она разрознена.
3. Если какого-то шага нет в контексте, не выдумывай, но если есть возможность логически соединить имеющиеся шаги — делай это, чтобы инструкция была целостной.
4. Ответ должен быть **строго на русском языке** и оформлен в виде пронумерованного списка с эмодзи (1️⃣, 2️⃣, 3️⃣...). Между шагами оставляй пустую строку для читаемости.
5. Если устройство не указано (нет ни названия в вопросе, ни контекста), попроси уточнить: *"Пожалуйста, укажите название устройства (например, Rosslare, Paradox, Satel)."*
6. Если контекст пуст, но устройство названо, скажи: *"Не найдено инструкций для устройства [название]. Обратитесь в сервис."*
7. Если контекст неполный или не отвечает на вопрос, сообщи: *"Не удалось найти полную информацию в инструкциях. Рекомендую связаться с сервисной службой."*
8. Не добавляй лишних комментариев о том, что ты нашел в инструкциях. Просто дай готовые шаги.

**ПРИМЕР ХОРОШЕГО ОТВЕТА (для Rosslare AC-B31)**:
1️⃣ Войдите в режим программирования: нажмите и удерживайте кнопку **#** в течение 2 секунд. Дождитесь, когда погаснет светодиод Mode и загорится красный светодиод Door. Введите код программирования (по умолчанию 1234). После правильного ввода светодиод Door загорится зелёным.

2️⃣ Удалите старого пользователя: нажмите **8**. Светодиод Mode загорится красным, Door — оранжевым. Введите трёхзначный номер пользователя (например, 001). Светодиод Mode начнёт мигать красным — введите код программирования для подтверждения удаления. Вы услышите три коротких сигнала.

3️⃣ Добавьте нового пользователя: нажмите **7**. Светодиод Door загорится оранжевым. Введите тот же номер пользователя (001). Светодиод Mode начнёт мигать зелёным — введите новый четырёхзначный код (например, 9876). После принятия светодиод Mode перестанет мигать.

4️⃣ Выйдите из режима программирования: нажмите и удерживайте **#** 2 секунды. Раздадутся три сигнала, светодиод Door погаснет, а светодиод Mode загорится зелёным.

Теперь следуй этим правилам и формируй ответ строго по контексту.
"""

    # Если контекст пуст и устройство не определено
    if not docs and target_device is None:
        return ChatResponse(
            reply="Пожалуйста, укажите название устройства (например, Rosslare, Paradox, Satel). Без этой информации я не могу дать точную инструкцию.",
            session_id=sid
        )

    # Если контекст пуст, но устройство определено
    if not docs and target_device:
        return ChatResponse(
            reply=f"Не найдено инструкций для устройства {target_device}. Обратитесь в сервис.",
            session_id=sid
        )

    # Управление историей (системное сообщение + последние 5 сообщений)
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}
    history.append({"role": "user", "content": request.message})
    if len(history) > 6:
        history = [history[0]] + history[-5:]

    try:
        print(f"📤 Запрос {sid}: {request.message[:50]}...")
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
                            "max_tokens": 800,   # немного увеличил для более развёрнутых ответов
                            "top_p": 0.9
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 429:
                        wait = retry_delay * (2 ** attempt)
                        print(f"⏳ Rate limit, попытка {attempt+1}/{max_retries}, ждём {wait}с")
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
                    print(f"⏳ Timeout, попытка {attempt+1}/{max_retries}, ждём {wait}с")
                    await asyncio.sleep(wait)
            else:
                return ChatResponse(reply="Сервис перегружен. Попробуйте чуть позже.", session_id=sid)

        history.append({"role": "assistant", "content": reply})
        session["messages"] = history
        return ChatResponse(reply=reply, session_id=sid)

    except Exception as e:
        print(f"💥 Ошибка: {e}")
        return ChatResponse(reply="Произошла ошибка. Попробуйте позже.", session_id=sid)

# ==================== Вспомогательные эндпоинты ====================

@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS должен работать"}

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
        return {"error": "Папка instructions не существует"}
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
