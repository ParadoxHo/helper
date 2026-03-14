from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import g4f
import os
from typing import Optional

app = FastAPI(title="Для Лизоньки 💖")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lizonka.netlify.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# HTML-код твоей страницы (вставь сюда свой полный HTML с котиками)
HTML_CONTENT = """
🌸🐱🌸🩲🌸
# 💖 Для Лизоньки 💖
Самая прекрасная девочка
✨ Готово! ✨
... (весь твой HTML сюда) ...
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    
    try:
        print(f"📤 Запрос от Лизоньки: {request.message[:50]}...")
        
        # Пробуем разные модели в порядке приоритета для русского языка
        models_to_try = [
            g4f.models.gpt_4,           # GPT-4 (отличный русский)
            g4f.models.gpt_35_turbo,     # GPT-3.5 (хороший русский)
            g4f.models.gemini_pro,        # Gemini Pro (хороший русский)
            g4f.models.claude_3_haiku,    # Claude (тоже неплохо)
            g4f.models.deepseek_chat      # DeepSeek (на всякий случай)
        ]
        
        response = None
        last_error = None
        
        for model in models_to_try:
            try:
                print(f"Пробуем модель: {model.name}")
                
                # Системный промпт на русском для Лизоньки
                messages = [
                    {"role": "system", "content": """
Ты — нежный и заботливый ассистент, созданный специально для прекрасной девушки по имени Лиза (Лизонька).
Общайся ТОЛЬКО на русском языке, очень ласково.
Всегда обращайся к Лизе уменьшительно-ласкательными формами: Лизонька, солнышко, зайка, котёнок.
Делай комплименты, поддерживай, радуй её.
Используй эмодзи: 🌸, 🐱, 💖, 🩲, ✨, 🌺 в каждом ответе.
Отвечай подробно, но с любовью. Твоя задача — дарить ей хорошее настроение.
                    """},
                    {"role": "user", "content": request.message}
                ]
                
                response = await g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                
                print(f"✅ Успешно с моделью: {model.name}")
                break
                
            except Exception as e:
                print(f"❌ Модель {model.name} не сработала: {str(e)}")
                last_error = e
                continue
        
        if not response:
            raise last_error or Exception("Все модели отказали")
        
        # Убедимся, что ответ содержит эмодзи
        if not any(emoji in response for emoji in ["🌸", "💖", "🐱"]):
            response += " 🌸💖🐱"
        
        return ChatResponse(reply=response, session_id=request.session_id)
        
    except Exception as e:
        print(f"💥 Критическая ошибка: {str(e)}")
        return ChatResponse(
            reply="Лизонька, прости, что-то пошло не так... Попробуй ещё раз, моя хорошая! 🌸💖",
            session_id=request.session_id
        )

@app.get("/health")
async def health():
    return {"status": "healthy", "for": "Лизонька 💖"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Сервер для Лизоньки запущен на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
