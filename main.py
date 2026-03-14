import os
import glob
import sys
import asyncio
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import PyPDF2

# RAG – лёгкие компоненты
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

app = FastAPI(title="Wsparcie Techniczne AZS z RAG (fastembed)")

# CORS – разрешаем только фронтенд
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assistics.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY не установлен!")

# ==================== RAG ====================
embedding_model = None
collection = None
chroma_client = None
COLLECTION_NAME = "azs_instructions"

def init_chroma():
    global chroma_client, collection
    if chroma_client is None:
        CHROMA_DIR = "./chroma_db"
        os.makedirs(CHROMA_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Найдена коллекция {COLLECTION_NAME} с {collection.count()} документами")
        except:
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            print(f"✅ Создана новая коллекция {COLLECTION_NAME}")

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Загрузка модели fastembed (может занять время при первом запуске)...")
        # Проверенная мультиязычная модель – хорошо работает с польским
        embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Модель fastembed загружена")
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
        print(f"❌ Ошибка при чтении PDF {pdf_path}: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    """Разбивает текст на фрагменты по 300 слов (для лучшего соответствия)."""
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
        print("ℹ️ Инструкции уже загружены, пропускаем.")
        return

    instructions_dir = "instructions"
    if not os.path.exists(instructions_dir):
        print(f"⚠️ Папка '{instructions_dir}' не существует. Создаю.")
        os.makedirs(instructions_dir, exist_ok=True)
        return

    pdf_files = glob.glob(os.path.join(instructions_dir, "*.pdf"), recursive=False)
    txt_files = glob.glob(os.path.join(instructions_dir, "*.txt"), recursive=False)
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"⚠️ В папке '{instructions_dir}' нет PDF/TXT файлов")
        return

    print(f"📁 Найденные файлы: {[os.path.basename(f) for f in all_files]}")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0

    for file_path in all_files:
        print(f"📄 Обработка: {file_path}")
        filename = os.path.basename(file_path)

        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"❌ Ошибка чтения файла {file_path}: {e}")
                continue

        if not text or not text.strip():
            print(f"⚠️ Файл {filename} пуст, пропускаю.")
            continue

        chunks = split_text_into_chunks(text)
        print(f"   → {len(chunks)} фрагментов")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename})
            all_ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1

    if not all_chunks:
        print("⚠️ Не найдено ни одного текстового фрагмента")
        return

    # Создаём эмбеддинги
    model = get_embedding_model()
    print("🔄 Генерация эмбеддингов (fastembed)...")
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
        print(f"   Добавлено {i+len(all_chunks[i:i+batch_size])} фрагментов")

    print(f"✅ Загружено {len(all_chunks)} фрагментов инструкций в базу")

def search_instructions(query: str, top_k: int = 7) -> List[str]:
    """Возвращает список наиболее релевантных фрагментов (top_k = 7)."""
    global collection
    init_chroma()
    if collection is None or collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = list(model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if results and results['documents']:
        return results['documents'][0]
    return []
# ==================== КОНЕЦ RAG ====================

# ==================== История разговоров ====================
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

# ==================== Эндпоинты ====================

@app.get("/")
async def root():
    return {"message": "Ассистент технической поддержки АЗС с RAG (fastembed) работает. Используйте /chat для отправки сообщений."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Отсутствует ключ API Groq")

    sid = request.session_id or "default"
    cleanup_old_sessions()
    session_data = history_store[sid]
    session_data["last_updated"] = datetime.now()
    history = session_data["messages"]

    # RAG – поиск фрагментов
    relevant_chunks = search_instructions(request.message, top_k=7)
    if relevant_chunks:
        print("📚 Найденные фрагменты:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"   [{i}] {chunk[:200]}...")
    else:
        print("📭 Не найдено ни одного фрагмента")

    context = "\n\n---\n".join(relevant_chunks) if relevant_chunks else "Нет дополнительного контекста."

    # Улучшенный системный промпт – отвечать только из контекста
    system_prompt = f"""
Ты – инженер первой линии технической поддержки для систем безопасности на автозаправочных станциях.
Твои собеседники – сотрудники станции (операторы), которые не являются специалистами. Они говорят по‑польски.

ТЫ ЗНАЕШЬ СЛЕДУЮЩИЕ СИСТЕМЫ (НА УРОВНЕ ПОЛЬЗОВАТЕЛЯ):
- Видеонаблюдение: Bosch DIVAR, Bosch DIP, 3xLogic, Provision, Hikvision
- Сигнализация: Paradox EVO192, SP65, SP4000, Satel Integra
- Контроль доступа: Rosslare B32 (смена кода пользователя)

У ТЕБЯ ЕСТЬ ДОСТУП К СЛЕДУЮЩИМ ИНСТРУКЦИЯМ (КОНТЕКСТ). 
**ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО НА ОСНОВЕ ЭТОГО КОНТЕКСТА.**
Если контекст не содержит ответа на вопрос, скажи: "Извините, я не нашёл этой информации в инструкциях. Пожалуйста, свяжитесь с сервисной службой."

КОНТЕКСТ:
{context}

ПРАВИЛА:
1. Отвечай ТОЛЬКО по‑польски, кратко и по существу.
2. Используй исключительно информацию из контекста. Не добавляй собственных знаний.
3. Если контекст содержит инструкцию – приведи её пошагово.
4. Если нет контекста или он не содержит ответа, сообщи об этом и предложи обратиться в сервис.
5. Будь вежлив и терпелив.
6. Помни контекст разговора – отвечай на вопросы пользователя последовательно.
"""

    # Управление историей
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}

    history.append({"role": "user", "content": request.message})

    if len(history) > 11:
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Запрос (сессия {sid}): {request.message[:50]}...")
        if relevant_chunks:
            print(f"   Найдено {len(relevant_chunks)} фрагментов в инструкциях")

        # Вызов Groq с повторными попытками при rate limit
        max_retries = 3
        retry_delay = 1  # начальная задержка в секундах

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
                            "model": "llama-3.3-70b-versatile",  # актуальная модель
                            "messages": history,
                            "temperature": 0.3,
                            "max_tokens": 600,  # уменьшено для экономии токенов
                            "top_p": 0.9
                        },
                        timeout=30.0
                    )

                    if response.status_code == 429:
                        # Превышен лимит, ждём и повторяем
                        wait_time = retry_delay * (2 ** attempt)  # экспоненциальная задержка
                        print(f"⏳ Rate limit, попытка {attempt+1}/{max_retries}, ждём {wait_time}с...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        print(f"❌ Ошибка Groq: {response.status_code} - {response.text}")
                        raise HTTPException(status_code=502, detail="Ошибка связи с Groq")

                    data = response.json()
                    reply = data["choices"][0]["message"]["content"]
                    break  # успешно получили ответ

                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⏳ Тайм-аут, попытка {attempt+1}/{max_retries}, ждём {wait_time}с...")
                    await asyncio.sleep(wait_time)

            else:
                # Все попытки исчерпаны
                return ChatResponse(
                    reply="Извините, сервис временно недоступен. Попробуйте позже.",
                    session_id=sid
                )

        history.append({"role": "assistant", "content": reply})
        session_data["messages"] = history

        print(f"✅ Ответ отправлен (сессия {sid})")
        return ChatResponse(reply=reply, session_id=sid)

    except httpx.TimeoutException:
        return ChatResponse(reply="Извините, сервис не отвечает. Попробуйте ещё раз через минуту.", session_id=sid)
    except Exception as e:
        print(f"💥 Ошибка: {str(e)}")
        return ChatResponse(reply="Произошла ошибка. Пожалуйста, попробуйте позже.", session_id=sid)

# Явная обработка OPTIONS для preflight
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
        return {"exists": False, "error": "Папка 'instructions' не существует"}
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
        # При необходимости можно очистить коллекцию перед перезагрузкой:
        # if collection:
        #     collection.delete(where={})
        load_instructions()
        count = collection.count() if collection else 0
        return {"status": "ok", "loaded": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
    print("🚀 Запуск приложения...")
    init_chroma()
    load_instructions()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
