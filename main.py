import os
import glob
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

# Разрешённые домены для CORS (можешь сузить позже)
ALLOWED_ORIGINS = [
    "https://assistics.netlify.app",
    "https://pagggge.vercel.app",
    "https://assistics.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Пути (можно переопределить переменными окружения)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "instructions")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "azs_instructions")

# RAG настройки
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# LLM настройки
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1200"))
GROQ_TOP_P = float(os.getenv("GROQ_TOP_P", "0.9"))

# Настройки контекста
MAX_FRAGMENT_LENGTH = int(os.getenv("MAX_FRAGMENT_LENGTH", "900"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))
TOP_K_MAIN = int(os.getenv("TOP_K_MAIN", "25"))
TOP_K_ENTRY = int(os.getenv("TOP_K_ENTRY", "12"))

# Ограничение истории для LLM
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))  # роли user/assistant

# Timeout для Groq
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30.0"))

# ==================== FastAPI ====================

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
    title="Wsparcie Techniczne AZS z RAG (poprawione - enhanced)",
    lifespan=lifespan,
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
    "hikvision": ["hikvision"],
}


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
    # Разбиение по словам: в PDF часто "крошится" на шум, поэтому так стабильнее чем по символам
    words = text.split()
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

    # Не загружаем повторно, если база уже не пустая
    if collection.count() > 0:
        print("ℹ️ Instrukcje już załadowane, pomijam.")
        return

    if not os.path.exists(INSTRUCTIONS_DIR):
        os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)
        print(f"⚠️ Folder {INSTRUCTIONS_DIR} utworzony, ale pusty.")
        return

    files = (
        glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.pdf")) +
        glob.glob(os.path.join(INSTRUCTIONS_DIR, "*.txt"))
    )
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
    """Вykrywa nazwę urządzenia w zapytaniu użytkownika (по ключевым словам)."""
    q = (query or "").lower()
    for dev, kw in DEVICE_KEYWORDS.items():
       
