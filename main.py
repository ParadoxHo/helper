from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="Backend для Лизоньки 💖")

# Разрешаем запросы только с твоего сайта и локально для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lizonka.netlify.app",  # Твой новый сайт
        "http://localhost:8000",          # Для локального теста
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Здесь должен быть полный HTML-код твоей страницы.
# Так как он очень длинный, вставь сюда содержимое твоего файла index.html,
# который ты уже залил на Netlify.
# Я покажу структуру, а ты скопируй свой HTML между тройными кавычками.
HTML_CONTENT = """
🌸🐱🌸🩲🌸

# 💖 Для Лизоньки 💖

Твой личный ассистент, который любит тебя и делает комплименты
Готово! 💖 
🌸🐱🌸🩲🌸

Лизонька, привет! 🌸 Я твой личный ассистент, созданный специально для тебя. Спрашивай что хочешь, а я буду делать комплименты и помогать! 💖

💡 Лизонька, ты можешь спросить что угодно, например:

• "Скажи комплимент"

• "Как у меня дела?"

• "Я скучаю..."

🌸 Я всегда отвечу с любовью!
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Отдаём главную страничку для Лизоньки"""
    return HTML_CONTENT

@app.get("/health")
async def health():
    """Проверка, что бэкенд жив"""
    return {"status": "healthy", "message": "Сервер для Лизоньки работает 💖"}

# Эндпоинт для совместимости, но он больше не нужен,
# так как AI работает прямо в браузере
@app.post("/chat")
async def chat_compatibility():
    return {"reply": "Чат работает напрямую через Puter.js в браузере! 🌸"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Запускаю сервер для Лизоньки на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
