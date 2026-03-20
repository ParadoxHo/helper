import os
import glob
import sys
import asyncio
import time
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import httpx
import PyPDF2

# Отключаем телеметрию ChromaDB
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from fastembed import TextEmbedding

# Разрешённые домены для CORS (можно сузить)
ALLOWED_ORIGINS = [
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app",
    "https://assistics.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "instructions")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "azs_instructions")

# ==================== LLM ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1200"))
GROQ_TOP_P = float(os.getenv("GROQ_TOP_P", "0.9"))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30.0"))

# ==================== RAG ====================
embedding_model = None
collection = None
chroma_client = None
is_rag_ready = False
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Контекст (увеличено, чтобы entry/мысли не “выпали”)
MAX_FRAGMENT_LENGTH = int(os.getenv("MAX_FRAGMENT_LENGTH", "1000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))

# Кол-во фрагментов
TOP_K_MAIN = int(os.getenv("TOP_K_MAIN", "25"))
TOP_K_ENTRY = int(os.getenv("TOP_K_ENTRY", "12"))
TOP_K_MENU = int(os.getenv("TOP_K_MENU", "15"))

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
    "hikvision": ["hikvision"],
}

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
    title="Wsparcie Techniczne AZS z RAG (poprawione - Rosslare entry + CORS +完整步骤)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY nie ustawiony!")


# ==================== RAG ====================

def init_chroma():
    global chroma_client, collection
    if chroma_client is None:
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
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Błąd PDF {pdf_path}: {e}")
    return text


def split_text_into_chunks(text: str, chunk_size: int = 150, overlap: int = 20) -> List[str]:
    # chunking po словах стабильнее, чем по символам
    words = (text or "").split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
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

    # Если уже загружено, не перезаливаем
    if collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        return

    if not os.path.exists(INSTRUCTIONS_DIR):
        os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)
        print(f"⚠️ Folder {INSTRUCTIONS_DIR} utworzony, ale pusty.")
        return

    files = glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.pdf")) + glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.txt"))
    if not files:
        print(f"⚠️ Brak plików PDF/TXT w folderze {INSTRUCTIONS_DIR}")
        return

    print(f"📁 Znalezione pliki: {[os.path.basename(f) for f in files]}")

    all_chunks, metas, ids = [], [], []
    chunk_id = 0

    for fpath in files:
        print(f"📄 Przetwarzanie: {fpath}")
        fname = os.path.basename(fpath)

        if fpath.endswith(".pdf"):
            text = extract_text_from_pdf(fpath)
        else:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
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
        if (i + 1) % 5 == 0:
            print(f"   Przetworzono {i + 1}/{total} fragmentów")
            time.sleep(0.2)

    print(f"   Wygenerowano {len(embeddings)} embeddingów")

    batch_size = 30
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            embeddings=embeddings[i:i + batch_size],
            documents=all_chunks[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
        print(f"   Dodano {i + len(all_chunks[i:i + batch_size])} fragmentów")
        time.sleep(0.3)

    print(f"✅ Załadowano {len(all_chunks)} fragmentów instrukcji do bazy")


def extract_device_from_query(query: str) -> Optional[str]:
    q = (query or "").lower()
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
        return []

    model = get_embedding_model()
    q_emb = list(model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k * 2)

    if not results or not results.get("documents"):
        return []

    frags = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        frags.append((doc, meta))

    filtered = [(d, m) for d, m in frags if target in m.get("device", "")]
    return filtered[:top_k]


def context_has_entry_markers(context: str) -> bool:
    c = (context or "").lower()

    # PL + EN markers for entry into programming mode
    has_time = any(x in c for x in ["2 sek", "2 sekundy", "2 seconds", "2 second"])
    has_hash = "#" in c
    has_programming = any(x in c for x in ["tryb programowania", "programowania", "programming mode", "program code", "programming code"])
    has_led = any(x in c for x in ["dioda", "led", "door led", "mode led", "dioda door", "dioda mode"])

    # Code markers vary by manual language
    has_code = any(x in c for x in ["kod programowania", "program code", "programing code", "4-digit", "czterocyfrow"])
    return (has_time and has_hash and has_programming and (has_led or has_code))


def is_rosslare_user_code_change_intent(message: str) -> bool:
    """
    Приближённая детекция: если спрашивают про "zmienić kod" и это НЕ явные меню коды (open/normal/programming),
    предполагаем, что речь про код пользователя.
    """
    m = (message or "").lower()
    if "rosslare" not in m:
        return False
    if not any(x in m for x in ["zmieni", "zmiana", "wymień", "replace"]):
        return False
    if "programowani" in m or "program code" in m:
        return False
    if "open code" in m or "open" in m:
        return False
    if "normal" in m or "secure" in m or "bypass" in m:
        return False
    if "door release" in m or "fail safe" in m or "fail secure" in m:
        return False
    return "kod" in m


def build_context_with_entry_and_menu(query: str, target_device: str) -> Tuple[str, List[Tuple[str, Dict]]]:
    """
    Сбор контекста:
    1) фрагменты входа в tryb/programming mode
    2) если Rosslare и похоже на изменение кода пользователя — фрагменты menu 8 (delete) и menu 7 (add)
    3) фрагменты основной релевантности
    """
    # Entry-focused query (важно: search_instructions фильтрует по target_device из текста запроса)
    entry_query = f"{target_device} wejście do trybu programowania tryb programowania kod programowania 2 sekundy # door led mode led"
    frags_entry = search_instructions(entry_query, top_k=TOP_K_ENTRY)

    # Rosslare change user code: include menu 8 and menu 7 explicitly
    frags_menu = []
    if target_device == "rosslare" and is_rosslare_user_code_change_intent(query):
        menu8_query = f"{target_device} menu 8 usuwanie użytkowników usuń użytkownika Kod Programowania potwierdź"
        menu7_query = f"{target_device} menu 7 dodanie kodu użytkownika Enroll Primary Secondary numer użytkownika dioda Mode Door"
        frags_menu8 = search_instructions(menu8_query, top_k=TOP_K_MENU)
        frags_menu7 = search_instructions(menu7_query, top_k=TOP_K_MENU)
        frags_menu = frags_menu8 + frags_menu7

    frags_main = search_instructions(query, top_k=TOP_K_MAIN)

    # Дедупликация по тексту
    seen = set()
    ordered: List[Tuple[str, Dict]] = []
    for lst in [frags_entry, frags_menu, frags_main]:
        for d, m in lst:
            key = hash(d)
            if key in seen:
                continue
            seen.add(key)
            ordered.append((d, m))

    # Собираем context с лимитами
    context_parts: List[str] = []
    total_chars = 0
    for i, (d, m) in enumerate(ordered):
        dd = d
        if len(dd) > MAX_FRAGMENT_LENGTH:
            dd = dd[:MAX_FRAGMENT_LENGTH] + "..."

        part = f"[Fragment {i + 1} z pliku {m.get('source', 'nieznany')}]\n{dd}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            context_parts.append("... (kontekst przycięty)")
            break
        context_parts.append(part)
        total_chars += len(part)

    context = "\n\n---\n".join(context_parts) if context_parts else "Brak dodatkowego kontekstu."
    return context, ordered


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
async def options_chat(request: Request):
    """
    Исправление CORS preflight:
    обязательно отдаём Access-Control-Allow-Origin на OPTIONS.
    """
    origin = request.headers.get("origin") or ""
    if origin and origin.rstrip("/") in ALLOWED_ORIGINS:
        allow_headers = request.headers.get("access-control-request-headers")
        if not allow_headers:
            allow_headers = "Content-Type"

        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin.rstrip("/"),
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": allow_headers,
                "Vary": "Origin",
            },
        )
    return Response(status_code=400)


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

    target_device = extract_device_from_query(request.message)
    if target_device is None:
        return ChatResponse(
            reply="Proszę podać nazwę urządzenia (np. Rosslare, Paradox, Satel). Bez tej informacji nie mogę udzielić dokładnej instrukcji.",
            session_id=sid,
        )

    # 1) context: entry-first + Rosslare menu8/menu7 if needed
    context, _ = build_context_with_entry_and_menu(request.message, target_device)

    # 2) guard: ensure entry exists in context to prevent skipping it
    if not context or context.strip() == "Brak dodatkowego kontekstu." or not context_has_entry_markers(context):
        return ChatResponse(
            reply="Nie znalazłem pełnego opisu wejścia do trybu programowania w instrukcjach. Proszę skontaktować się z serwisem.",
            session_id=sid,
        )

    system_prompt = f"""
Jesteś inżynierem wsparcia technicznego z wieloletnim doświadczeniem w systemach bezpieczeństwa na stacjach benzynowych.
Twoim zadaniem jest pomóc operatorowi stacji rozwiązać problem, podając wyłącznie informacje zawarte w poniższym kontekście.
Nie korzystaj z własnej wiedzy.

KONTEKST (fragmenty instrukcji):
{context}

ZASADY:
1️⃣ Odpowiadaj TYLKO po polsku.
2️⃣ Odpowiedź ma być w formie numerowanej listy z emoji: 1️⃣, 2️⃣, 3️⃣...
3️⃣ Między krokami zostaw pustą linię (akapit).
4️⃣ Nie dodawaj wstępów, podsumowań ani wyjaśnień technicznych — tylko kroki.
5️⃣ Maksymalnie 16 kroków.
6️⃣ Nie mieszaj procedur różnych urządzeń.
7️⃣ Nie pomijaj kroków. Jeśli w kontekście wejście/wyjście jest rozproszone po fragmentach — złącz i przepisz pełną sekwencję.
8️⃣ Wyjście z Trybu Programowania: ZAWSZE jest przyciskiem „#” wciśniętym i przytrzymanym przez 2 sekundy (3 krótkie sygnały). NIE używaj menu (np. 7 lub 8) jako wyjścia.

ROSSLARE (AC-B31 / AC-B32) — ZMIANA KODU UŻYTKOWNIKA:
Jeśli pytanie dotyczy „zmienić kod użytkownika”, „zmiana kodu użytkownika”, „kod wprowadzony dla użytkownika” itp.:
   - zastosuj procedurę: najpierw USUNIĘCIE użytkownika (menu „8”), a następnie DODANIE użytkownika z nowym kodem (menu „7”).
   - Opisz DWIE pełne procedury (USUNIĘCIE i DODANIE) oraz obie części ZACZNIJ od „Wejście do Trybu Programowania”:
       Wejście (# przez 2 sekundy) -> Kod Programowania (fabrycznie 1234) -> gotowość
       ... -> wyjście (# przez 2 sekundy) jeśli wynika z kontekstu
       ponownie Wejście (# przez 2 sekundy) -> Kod Programowania -> dalsze kroki
   - Po kroku „7” wprowadź kod użytkownika zgodnie z miganiem diody „Mode”:
       miganie na zielono = Kod Podstawowy
       miganie na czerwono = Kod Dodatkowy

TRYB PROGRAMOWANIA — WYMÓG STARTU:
Odpowiedź ZAWSZE zaczynaj od pełnego wejścia do Trybu Programowania:
1️⃣ „#” przez 2 sekundy -> Mode/ Door LED zgodnie z kontekstem -> wprowadź Kod Programowania (fabrycznie 1234) -> Door LED zielona = jesteś w trybie.

Jeśli w KONTEKŚCIE nie ma odpowiedzi na wymaganą część procedury — napisz dokładnie:
„Nie znalazłem tej informacji w instrukcjach. Proszę skontaktować się z serwisem.”
""".strip()

    # 3) history: держим только user/assistant, system формируем заново в groq_messages
    if not history:
        history.append({"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}

    history.append({"role": "user", "content": request.message})
    # ограничим историю, чтобы не “перенастраивался стиль”
    if len(history) > 10:
        history = [history[0]] + history[-9:]

    try:
        print(f"📤 Zapytanie {sid}: {request.message[:80]}...")

        groq_messages = history  # здесь уже есть system в history[0]
        max_retries = 5
        retry_delay = 1

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": GROQ_MODEL,
                            "messages": groq_messages,
                            "temperature": GROQ_TEMPERATURE,
                            "max_tokens": GROQ_MAX_TOKENS,
                            "top_p": GROQ_TOP_P,
                        },
                        timeout=GROQ_TIMEOUT_SECONDS,
                    )

                    if resp.status_code == 429:
                        wait = retry_delay * (2 ** attempt)
                        print(f"⏳ Rate limit, próba {attempt + 1}/{max_retries}, czekam {wait}s")
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
                    print(f"⏳ Timeout, próba {attempt + 1}/{max_retries}, czekam {wait}s")
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
        "instructions_loaded": docs_count,
        "instructions_dir": INSTRUCTIONS_DIR,
        "chroma_dir": CHROMA_DIR,
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
