import os
import glob
import sys
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import PyPDF2

# RAG компоненты
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

app = FastAPI(title="Wsparcie Techniczne AZS z RAG (fastembed)")

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

# ==================== RAG z fastembed ====================
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
            print(f"✅ Znaleziono kolekcję {COLLECTION_NAME} z {collection.count()} dokumentami")
        except:
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            print(f"✅ Utworzono nową kolekcję {COLLECTION_NAME}")

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Ładowanie modelu fastembed (może potrwać przy pierwszym uruchomieniu)...")
        # Model e5-small – lekki, wielojęzyczny, dobry do polskiego
        embedding_model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
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

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename})
            all_ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1

    if not all_chunks:
        print("⚠️ Nie znaleziono żadnych fragmentów tekstu")
        return

    # Tworzymy embeddingi za pomocą fastembed
    model = get_embedding_model()
    print("🔄 Generowanie embeddignów (fastembed)...")
    embeddings = list(model.embed(all_chunks))  # zwraca generator, konwertujemy na listę
    # fastembed zwraca numpy array, trzeba zrzutować na listę float
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

def search_instructions(query: str, top_k: int = 5) -> List[str]:
    global collection
    init_chroma()
    if collection is None or collection.count() == 0:
        return []

    model = get_embedding_model()
    # Wygeneruj embedding dla zapytania
    query_embedding = list(model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if results and results['documents']:
        return results['documents'][0]
    return []
# ==================== KONIEC RAG ====================

# Reszta kodu (historia, endpointy) zostaje bez zmian
# (poniżej tylko skrót, ale w rzeczywistym pliku trzeba wkleić całą dalszą część z poprzedniej wersji)

# ... (tu wklej całą dalszą część z poprzedniego main.py, od historii po endpointy i startup) ...
