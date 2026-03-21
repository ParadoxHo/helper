# FINAL ELITE RAG BACKEND (Production hardened: Railway + Vercel + CORS safe)

import os

# ===== TELEMETRY FIX (must be before chroma import)
os.environ["ANONYMIZED_TELEMETRY"]="False"
os.environ["CHROMA_TELEMETRY_ENABLED"]="false"

import glob
import asyncio
import uuid
import logging
import time
import math
import hashlib
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import httpx
import PyPDF2

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

# ================= CONFIG =================

class Config:

    CHROMA_DIR="chroma_db"

    DOC_DIR="instructions"

    COLLECTION="rag_collection"

    TOP_K=15

    FETCH_K=60

    SIM_THRESHOLD=0.42

    MAX_CONTEXT=9000

    MAX_CHUNK=900

    LLM_CONCURRENCY=8

    RATE_LIMIT=120

    CACHE_TTL=600

    SESSION_TTL_HOURS=2

    RETRIES=5

    TIMEOUT=45

    MODEL="llama-3.1-8b-instant"

    ALLOWED_ORIGINS=[
        "https://assistics.vercel.app",
        "http://localhost:3000"
    ]

CONFIG=Config()

# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log=logging.getLogger()

# ================= GLOBALS =================

embedder=None

chroma=None

collection=None

rag_ready=False

sessions=defaultdict(lambda:{"m":deque(maxlen=12),"t":datetime.now()})

rate=defaultdict(list)

cache={}

llm_sem=asyncio.Semaphore(CONFIG.LLM_CONCURRENCY)

# ================= SECURITY =================

BLOCK_PATTERNS=[
"ignore previous instructions",
"reveal system prompt",
"print hidden",
"show config",
"developer message"
]


def sanitize(text):

    low=text.lower()

    for p in BLOCK_PATTERNS:

        if p in low:

            return ""

    return text

# ================= UTILS =================

def hash_text(t):

    return hashlib.sha1(t.encode()).hexdigest()

# ================= RATE =================

def check_rate(k):

    now=time.time()

    b=rate[k]

    b[:]=[x for x in b if now-x<60]

    if len(b)>CONFIG.RATE_LIMIT:

        raise HTTPException(429)

    b.append(now)

# ================= CLEANER =================

async def cleaner():

    while True:

        now=datetime.now()

        for k in list(sessions.keys()):

            if now-sessions[k]["t"]>timedelta(hours=CONFIG.SESSION_TTL_HOURS):

                del sessions[k]

        for k in list(cache.keys()):

            if time.time()-cache[k][1]>CONFIG.CACHE_TTL:

                del cache[k]

        await asyncio.sleep(900)

# ================= FASTAPI =================

app=FastAPI()

# ===== PRODUCTION CORS CONFIG
app.add_middleware(

    CORSMiddleware,

    allow_origins=CONFIG.ALLOWED_ORIGINS,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

    expose_headers=["*"],

    max_age=600
)

# ===== EXTRA CORS SAFETY FOR RAILWAY
@app.middleware("http")
async def cors_headers(request:Request,call_next):

    response=await call_next(request)

    origin=request.headers.get("origin")

    if origin in CONFIG.ALLOWED_ORIGINS:

        response.headers["Access-Control-Allow-Origin"]=origin

    response.headers["Access-Control-Allow-Methods"]="*"

    response.headers["Access-Control-Allow-Headers"]="*"

    response.headers["Access-Control-Allow-Credentials"]="true"

    return response

# ===== PREFLIGHT FIX
@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path:str):

    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin":"https://assistics.vercel.app",
            "Access-Control-Allow-Methods":"*",
            "Access-Control-Allow-Headers":"*",
            "Access-Control-Allow-Credentials":"true"
        }
    )

# ================= LIFESPAN =================

@asynccontextmanager
async def lifespan(app:FastAPI):

    init_vector()

    asyncio.create_task(index_docs(False))

    asyncio.create_task(cleaner())

    yield

app.router.lifespan_context=lifespan

# ================= MODELS =================

class ChatRequest(BaseModel):

    message:str

    session_id:Optional[str]=None

class ChatResponse(BaseModel):

    reply:str

    source:str

    session_id:str

# ================= EMBEDDINGS =================

def get_embedder():

    global embedder

    if embedder:

        return embedder

    embedder=TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return embedder

# ================= VECTOR =================

def init_vector():

    global chroma,collection

    if chroma:

        return

    os.makedirs(CONFIG.CHROMA_DIR,exist_ok=True)

    chroma=chromadb.PersistentClient(

        path=CONFIG.CHROMA_DIR,

        settings=Settings(
            anonymized_telemetry=False
        )
    )

    try:

        collection=chroma.get_collection(CONFIG.COLLECTION)

    except:

        collection=chroma.create_collection(CONFIG.COLLECTION)

# ================= CHUNK =================

def chunk(text,size=240,overlap=60):

    words=text.split()

    step=size-overlap

    out=[]

    for i in range(0,len(words),step):

        c=" ".join(words[i:i+size])

        if c:

            out.append(c)

    return out

# ================= INDEX =================

async def index_docs(force):

    global rag_ready

    await asyncio.to_thread(index_sync,force)

    rag_ready=True


def index_sync(force):

    if collection.count()>0 and not force:

        return

    files=(
        glob.glob(CONFIG.DOC_DIR+"/*.pdf")+
        glob.glob(CONFIG.DOC_DIR+"/*.txt")
    )

    if not files:

        log.warning("no docs")

        return

    texts=[]

    metas=[]

    ids=[]

    for f in files:

        if f.endswith("pdf"):

            txt=""

            reader=PyPDF2.PdfReader(open(f,'rb'))

            for p in reader.pages:

                t=p.extract_text()

                if t:

                    txt+=t

        else:

            txt=open(f,encoding="utf8",errors="ignore").read()

        parts=chunk(txt)

        for p in parts:

            texts.append(p)

            metas.append({
                "file":os.path.basename(f)
            })

            ids.append(str(uuid.uuid4()))

    model=get_embedder()

    emb=list(model.embed(texts))

    b=64

    for i in range(0,len(texts),b):

        collection.add(
            embeddings=[x.tolist() for x in emb[i:i+b]],
            documents=texts[i:i+b],
            metadatas=metas[i:i+b],
            ids=ids[i:i+b]
        )

    log.info(f"indexed {len(texts)} chunks")

# ================= COSINE =================

def cosine(a,b):

    s=sum(x*y for x,y in zip(a,b))

    na=math.sqrt(sum(x*x for x in a))

    nb=math.sqrt(sum(x*x for x in b))

    return s/(na*nb+1e-9)

# ================= MMR =================

def mmr(qemb,docs,embs):

    selected=[]

    while len(selected)<CONFIG.TOP_K and docs:

        best=None

        best_score=-1

        for i in range(len(docs)):

            rel=cosine(qemb,embs[i])

            div=0

            if selected:

                div=max(
                    cosine(embs[i],embs[j])
                    for j in selected
                )

            score=0.7*rel-0.3*div

            if score>best_score:

                best_score=score

                best=i

        selected.append(best)

        docs.pop(best)

        embs.pop(best)

    return selected

# ================= RETRIEVE =================

def retrieve(q):

    if not rag_ready:

        return []

    model=get_embedder()

    qemb=list(model.embed([q]))[0].tolist()

    res=collection.query(
        query_embeddings=[qemb],
        n_results=CONFIG.FETCH_K,
        include=[
            "documents",
            "metadatas",
            "embeddings",
            "distances"
        ]
    )

    docs=res["documents"][0]

    metas=res["metadatas"][0]

    embs=res["embeddings"][0]

    dists=res["distances"][0]

    filtered=[]

    femb=[]

    for d,m,e,dist in zip(docs,metas,embs,dists):

        score=1-dist

        if score<CONFIG.SIM_THRESHOLD:

            continue

        filtered.append((d,m))

        femb.append(e)

    if not filtered:

        return []

    idxs=mmr(qemb,filtered.copy(),femb.copy())

    return [filtered[i] for i in idxs]

# ================= CONTEXT =================

def build_context(frags):

    total=0

    out=[]

    for i,(d,m) in enumerate(frags,1):

        d=sanitize(d)

        t=d[:CONFIG.MAX_CHUNK]

        block=f"[{i} {m.get('file')}]\n{t}"

        if total+len(block)>CONFIG.MAX_CONTEXT:

            break

        total+=len(block)

        out.append(block)

    return "\n---\n".join(out)

# ================= LLM =================

async def llm(messages):

    key=hash_text(str(messages))

    if key in cache:

        return cache[key][0]

    async with llm_sem:

        async with httpx.AsyncClient(timeout=CONFIG.TIMEOUT) as c:

            for i in range(CONFIG.RETRIES):

                try:

                    r=await c.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization":f"Bearer {os.getenv('GROQ_API_KEY')}"
                        },
                        json={
                            "model":CONFIG.MODEL,
                            "messages":messages,
                            "temperature":0.05,
                            "max_tokens":1200
                        }
                    )

                    if r.status_code==200:

                        txt=r.json()["choices"][0]["message"]["content"]

                        cache[key]=(txt,time.time())

                        return txt

                    await asyncio.sleep(2**i)

                except httpx.RequestError:

                    await asyncio.sleep(2**i)

    raise HTTPException(503)

# ================= CHAT =================

@app.post("/chat")
async def chat(req:ChatRequest):

    sid=req.session_id or "default"

    check_rate(sid)

    q=req.message.strip()

    if not q:

        raise HTTPException(400)

    frags=retrieve(q)

    ctx=build_context(frags)

    source="rag"

    sys=f"""
Jesteś inżynierem wsparcia technicznego.
Odpowiadaj tylko z kontekstu.
Lista kroków.
Bez zgadywania.

KONTEKST:
{ctx}
"""

    h=sessions[sid]["m"]

    if not h:

        h.append({"role":"system","content":sys})

    else:

        h[0]={"role":"system","content":sys}

    h.append({"role":"user","content":q})

    reply=await llm(list(h))

    h.append({"role":"assistant","content":reply})

    sessions[sid]["t"]=datetime.now()

    return ChatResponse(
        reply=reply,
        source=source,
        session_id=sid
    )

# ================= HEALTH =================

@app.get("/health")
async def health():

    return {
        "rag_ready":rag_ready,
        "documents":collection.count() if collection else 0,
        "cache":len(cache)
    }

@app.get("/reload")
async def reload():

    asyncio.create_task(index_docs(True))

    return {"status":"reindexing"}

@app.get("/favicon.ico")
async def favicon():

    return Response(status_code=204)

# ================= RUN =================

if __name__=="__main__":

    import uvicorn

    port=int(os.getenv("PORT",8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
