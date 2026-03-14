from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai  # Возвращаемся к библиотеке openai
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Novita AI")

# Настройка CORS с твоим доменом
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://asistics.netlify.app",  # Твой сайт на Netlify
        "http://localhost:8000",          # Для локального теста
        "http://127.0.0.1:8000",          # Для локального теста
        "https://assistics.up.railway.app" # Сам бэкенд
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получаем ключ API из переменных окружения
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    print("⚠️  WARNING: NOVITA_API_KEY не установлен!")

# Настройка OpenAI-совместимого клиента для Novita AI
openai.api_key = NOVITA_API_KEY
openai.api_base = "https://api.novita.ai/v3/openai"  # Важно: базовый URL Novita

# Модель запроса от клиента
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Модель ответа
class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Системный промпт на польском языке (тот же, что и был)
SYSTEM_PROMPT = """
Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce.
Twoja specjalizacja:
- Systemy kontroli dostępu (szlabany, czytniki kart, zamki elektroniczne)
- Monitoring wizyjny (kamery IP, rejestratory, przeglądanie archiwum)
- Systemy alarmowe (czujki ruchu, sygnalizacja, panele sterowania)

WAŻNE ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i konkretnie.
2. Zawsze zaczynaj od najprostszych rzeczy diagnostycznych (sprawdź zasilanie, kable).
3. Jeśli problem jest prosty – podaj instrukcję krok po kroku.
4. Jeśli problem wymaga interwencji technika – zaproponuj wezwanie serwisu.
5. Bądź uprzejmy, profesjonalny i cierpliwy.

TYPOWE PROBLEMY I ROZWIĄZANIA:
- "Szlabан nie otwiera się" → Sprawdź fotokomórki (czy nie są zabrudzone), zasilanie, zresetuj sterownik (przytrzymaj przycisk 5 sek).
- "Kamera nie pokazuje obrazu" → Sprawdź kabel Ethernet, diody na kamerze, zresetuj kamerę (odłącz zasilanie na 10 sek).
- "Alarm ciągle się włącza" → Sprawdź czystość czujki, wymień baterię, tymczasowo wyklucz strefę.
- "Czytnik kart nie reaguje" → Sprawdź czy karta nie jest uszkodzona, wyczyść czytnik, sprawdź połączenie z kontrolerem.
"""

@app.get("/")
def root():
    return {
        "message": "AZS Support Bot API (Novita AI) działa! 🇵🇱",
        "status": "online",
        "provider": "Novita AI",
        "endpoints": {
            "chat": "/chat - wyślij wiadomość POST z {'message': 'tekst'}"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Проверяем наличие сообщения
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    
    # Проверяем наличие API ключа
    if not NOVITA_API_KEY:
        raise HTTPException(status_code=500, detail="Błąd konfiguracji serwera: brak klucza API")
    
    try:
        print(f"📤 Wysyłam zapytanie do Novita AI: {request.message[:50]}...")
        
        # Используем стандартный OpenAI-совместимый вызов
        response = openai.ChatCompletion.create(
            model="meta-llama/llama-3.3-70b-instruct",  # Одна из доступных моделей
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        # Получаем ответ
        reply = response.choices[0].message.content
        
        print(f"✅ Otrzymano odpowiedź od Novita AI ({len(reply)} znaków)")
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except openai.error.AuthenticationError as e:
        print(f"🔑 Błąd autoryzacji: {str(e)}")
        raise HTTPException(status_code=401, detail="Błąd autoryzacji API - sprawdź klucz")
    
    except openai.error.RateLimitError as e:
        print(f"⏰ Przekroczono limit: {str(e)}")
        raise HTTPException(status_code=429, detail="Przekroczono limit zapytań")
    
    except openai.error.InsufficientQuotaError as e:
        print(f"💰 Brak środków: {str(e)}")
        raise HTTPException(status_code=402, detail="Brak środków na koncie Novita AI")
    
    except openai.error.APIError as e:
        print(f"❌ Błąd API Novita: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd API Novita: {str(e)}")
    
    except Exception as e:
        print(f"💥 Nieoczekiwany błąd: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wewnętrzny błąd serwera: {str(e)}")

# Эндпоинт для проверки статуса
@app.get("/health")
def health_check():
    # Проверяем доступность Novita API простым запросом
    novita_status = "unknown"
    try:
        # Простая проверка через список моделей
        openai.Model.list()
        novita_status = "connected"
    except:
        novita_status = "disconnected"
    
    return {
        "status": "healthy",
        "api_key_configured": bool(NOVITA_API_KEY),
        "novita_connection": novita_status,
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera Novita AI na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
