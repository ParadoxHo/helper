from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Google Gemini (FREE)")

# Настройка CORS с твоим доменом
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://asistics.netlify.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://assistics.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получаем ключ API из переменных окружения
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY не установлен!")

# Настройка Gemini
genai.configure(api_key=GEMINI_API_KEY)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Системный промпт на польском языке
SYSTEM_PROMPT = """
Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce.
Twoja specjalizacja:
- Systemy kontroli dostępu (szlabany, czytniki kart, zamki elektroniczne)
- Monitoring wizyjny (kamery IP, rejestratory)
- Systemy alarmowe (czujki, sygnalizacja)

WAŻNE ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i konkretnie.
2. Zawsze zaczynaj od najprostszych rzeczy (sprawdź zasilanie, kable).
3. Jeśli problem jest prosty – podaj instrukcję krok po kroku.
4. Jeśli problem wymaga interwencji technika – zaproponuj wezwanie serwisu.
"""

@app.get("/")
def root():
    return {
        "message": "AZS Support Bot API (Google Gemini FREE) działa! 🇵🇱",
        "status": "online",
        "provider": "Google AI Studio",
        "model": "gemini-2.0-flash (1500 requests/day FREE)",
        "endpoints": {
            "chat": "/chat - wyślij wiadomość POST z {'message': 'tekst'}"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API Gemini")
    
    try:
        print(f"📤 Wysyłam zapytanie do Gemini: {request.message[:50]}...")
        
        # Используем модель Gemini 2.0 Flash (1500 запросов/день бесплатно)
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 800,
                "top_p": 0.9,
            }
        )
        
        # Формируем запрос с системным промптом
        full_prompt = f"{SYSTEM_PROMPT}\n\nUżytkownik: {request.message}\nAsystent:"
        
        response = model.generate_content(full_prompt)
        
        reply = response.text
        
        print(f"✅ Otrzymano odpowiedź od Gemini ({len(reply)} znaków)")
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except Exception as e:
        print(f"💥 Błąd Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_key_configured": bool(GEMINI_API_KEY),
        "model": "gemini-2.0-flash (FREE)",
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera Gemini FREE na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
