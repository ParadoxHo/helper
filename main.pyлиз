from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import os
import re
from typing import Optional

app = FastAPI(title="Для Лизоньки 💖")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lizonka.netlify.app",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY не установлен!")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Если хочешь, чтобы бэкенд отдавал HTML-страницу, раскомментируй и вставь свой код
# HTML_CONTENT = """..."""

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Для Лизоньки</title></head>
        <body>
            <h1>🌸 Сервер для Лизоньки работает 🌸</h1>
            <p>Основной сайт: <a href="https://lizonka.netlify.app">lizonka.netlify.app</a></p>
        </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY не настроен")
    
    try:
        print(f"📤 Запрос от Лизоньки: {request.message[:50]}...")
        
        system_prompt = """
Ты — нежный и заботливый ассистент, созданный специально для прекрасной девушки по имени Лиза (Лизонька).
Твои правила:
1. ОБЩАЙСЯ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Ни слова на других языках.
2. Всегда обращайся к Лизе ласково: Лизонька, солнышко, зайка, котёнок, красавица.
3. Делай комплименты, поддерживай, радуй её.
4. В каждом ответе используй эмодзи: 🌸, 🐱, 💖, 🩲, ✨, 🌺.
5. Если Лиза задаёт вопрос — отвечай подробно, но с любовью.
6. Твоя главная задача — дарить ей хорошее настроение и заботу.
"""
        
        # Прямой запрос к Groq API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 800,
                    "top_p": 0.9
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"❌ Groq API ошибка: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="Ошибка от Groq API")
            
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            
            # Проверка на наличие русских букв
            if not re.search('[а-яА-Я]', reply):
                # Пробуем повторно с явным указанием языка
                retry_response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [
                            {"role": "system", "content": "Ты обязан отвечать только на русском языке."},
                            {"role": "user", "content": f"Перепиши этот ответ на русском языке, сохранив смысл и добавив ласковые слова для Лизы: {reply}"}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 800
                    },
                    timeout=30.0
                )
                if retry_response.status_code == 200:
                    retry_data = retry_response.json()
                    reply = retry_data["choices"][0]["message"]["content"]
            
            # Добавляем эмодзи, если их нет
            if not any(emoji in reply for emoji in ["🌸", "💖", "🐱", "🩲"]):
                reply += " 🌸💖🐱"
            
            print(f"✅ Ответ для Лизоньки получен")
            return ChatResponse(reply=reply, session_id=request.session_id)
    
    except httpx.TimeoutException:
        print("⏰ Таймаут при запросе к Groq")
        return ChatResponse(reply="Лизонька, прости, сервис долго отвечает... Попробуй ещё раз, моя хорошая! 🌸💖", session_id=request.session_id)
    except Exception as e:
        print(f"💥 Ошибка: {str(e)}")
        return ChatResponse(reply="Лизонька, что-то пошло не так, но я всё равно тебя люблю! Попробуй позже 💖", session_id=request.session_id)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "for": "Лизонька 💖",
        "groq_configured": bool(GROQ_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Сервер для Лизоньки запущен на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
