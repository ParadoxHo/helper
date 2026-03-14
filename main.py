from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - DeepSeek")

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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("⚠️  WARNING: DEEPSEEK_API_KEY не установлен!")

# Модель запроса от клиента
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Модель ответа
class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Системный промпт на польском языке (специально для техподдержки АЗС)
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
        "message": "AZS Support Bot API (DeepSeek) działa! 🇵🇱",
        "status": "online",
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
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="Błąd konfiguracji serwera: brak klucza API")
    
    # Подготовка запроса к DeepSeek API
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",  # Можно использовать "deepseek-reasoner" для сложных случаев
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ],
        "temperature": 0.3,        # Низкая температура = более предсказуемые ответы
        "max_tokens": 800,          # Максимальная длина ответа
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stream": False
    }
    
    try:
        print(f"📤 Wysyłam zapytanie do DeepSeek: {request.message[:50]}...")
        
        # Отправляем запрос к DeepSeek
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        # Проверяем статус ответа
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ Błąd DeepSeek API: {response.status_code} - {error_text}")
            
            # Пытаемся распарсить ошибку
            try:
                error_json = response.json()
                error_message = error_json.get('error', {}).get('message', 'Nieznany błąd')
            except:
                error_message = error_text
            
            raise HTTPException(
                status_code=500, 
                detail=f"Błąd DeepSeek API ({response.status_code}): {error_message}"
            )
        
        # Получаем ответ
        result = response.json()
        reply = result['choices'][0]['message']['content']
        
        # Логируем успех
        print(f"✅ Otrzymano odpowiedź od DeepSeek ({len(reply)} znaków)")
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except requests.exceptions.Timeout:
        print("⏰ Timeout przy połączeniu z DeepSeek")
        raise HTTPException(status_code=504, detail="Przekroczono czas oczekiwania na odpowiedź")
    
    except requests.exceptions.ConnectionError:
        print("🔌 Błąd połączenia z DeepSeek")
        raise HTTPException(status_code=503, detail="Nie można połączyć się z serwerem DeepSeek")
    
    except Exception as e:
        print(f"💥 Nieoczekiwany błąd: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wewnętrzny błąd serwera: {str(e)}")

# Добавим эндпоинт для проверки статуса
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_key_configured": bool(DEEPSEEK_API_KEY),
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
