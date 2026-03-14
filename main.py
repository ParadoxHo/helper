from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import os
from typing import Optional

app = FastAPI(title="Для Лизоньки 💖")

# Настройка CORS – разрешаем только твой сайт
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lizonka.netlify.app",  # твой новый сайт
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получаем ключ Groq из переменных окружения (на Railway добавишь позже)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY не установлен!")

# Инициализация клиента Groq
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Сюда вставь свой полный HTML-код (тот самый красивый с котиками)
# Если хочешь оставить сайт только на Netlify, удали эту переменную и маршрут "/"
HTML_CONTENT = """
🌸🐱🌸🩲🌸
<h1>💖 Для Лизоньки 💖</h1>
<p>Самая прекрасная девочка</p>
<span class="status-badge" id="status-badge">✨ Загрузка...</span>
... (весь твой HTML сюда) ...
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Если хочешь, чтобы бэкенд сам отдавал страницу – раскомментируй эту строку"""
    # return HTML_CONTENT
    return {"message": "Бэкенд для Лизоньки работает. Сайт находится на Netlify: https://lizonka.netlify.app"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    
    if not client:
        raise HTTPException(status_code=500, detail="Groq клиент не инициализирован (проверь API ключ)")
    
    try:
        print(f"📤 Запрос от Лизоньки: {request.message[:50]}...")
        
        # Формируем системный промпт на русском – строго!
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
        
        # Выбираем лучшую модель для русского языка
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Llama 3.3 70B – отлично знает русский
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.9
        )
        
        reply = response.choices[0].message.content
        
        # Проверяем, есть ли в ответе русские буквы (грубая проверка)
        import re
        if not re.search('[а-яА-Я]', reply):
            # Если ответ не на русском, просим Groq переформулировать
            retry = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Ты обязан отвечать только на русском языке."},
                    {"role": "user", "content": f"Перепиши этот ответ на русском языке, сохранив смысл и добавив ласковые слова для Лизы: {reply}"}
                ],
                temperature=0.5,
                max_tokens=800
            )
            reply = retry.choices[0].message.content
        
        # Добавляем эмодзи, если их нет
        if not any(emoji in reply for emoji in ["🌸", "💖", "🐱", "🩲"]):
            reply += " 🌸💖🐱"
        
        print(f"✅ Ответ для Лизоньки получен")
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except Exception as e:
        print(f"💥 Ошибка: {str(e)}")
        return ChatResponse(
            reply="Лизонька, прости, что-то сломалось... Попробуй ещё раз, моя хорошая! 🌸💖",
            session_id=request.session_id
        )

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
