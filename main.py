from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Novita AI (Free)")

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
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    print("⚠️  WARNING: NOVITA_API_KEY не установлен!")

# Настройка OpenAI-совместимого клиента для Novita AI
openai.api_key = NOVITA_API_KEY
openai.api_base = "https://api.novita.ai/v3/openai"

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
- Monitoring wizyjny (kamery IP, rejestratory, przeglądanie archiwum)
- Systemy alarmowe (czujki ruchu, sygnalizacja, panele sterowania)

WAŻNE ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i konkretnie.
2. Zawsze zaczynaj od najprostszych rzeczy diagnostycznych (sprawdź zasilanie, kable).
3. Jeśli problem jest prosty – podaj instrukcję krok po kroku.
4. Jeśli problem wymaga interwencji technika – zaproponuj wezwanie serwisu.
5. Bądź uprzejmy, profesjonalny i cierpliwy.
"""

@app.get("/")
def root():
    return {
        "message": "AZS Support Bot API (Novita AI - FREE model) działa! 🇵🇱",
        "status": "online",
        "provider": "Novita AI",
        "model": "meta-llama/llama-3.3-3b-instruct (FREE)",
        "endpoints": {
            "chat": "/chat - wyślij wiadomość POST z {'message': 'tekst'}"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    
    if not NOVITA_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API")
    
    try:
        print(f"📤 Wysyłam zapytanie do Novita AI (FREE): {request.message[:50]}...")
        
        # 🔥 ИЗМЕНЕНИЕ ТОЛЬКО ЗДЕСЬ: используем бесплатную модель
        response = openai.ChatCompletion.create(
            model="meta-llama/llama-3.3-3b-instruct",  # Бесплатная модель!
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        reply = response.choices[0].message.content
        print(f"✅ Otrzymano odpowiedź ({len(reply)} znaków)")
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except openai.error.APIError as e:
        error_str = str(e)
        if "NOT_ENOUGH_BALANCE" in error_str or "not enough balance" in error_str.lower():
            print(f"💰 Brak środków - ale to dziwne, bo model FREE powinien działać!")
            raise HTTPException(
                status_code=402, 
                detail="Problem z darmowym modelem. Spróbuj później lub dodaj środki na konto."
            )
        else:
            print(f"❌ Błąd API Novita: {error_str}")
            raise HTTPException(status_code=500, detail=f"Błąd API: {error_str}")
    
    except openai.error.AuthenticationError as e:
        print(f"🔑 Błąd autoryzacji: {str(e)}")
        raise HTTPException(status_code=401, detail="Błąd autoryzacji - sprawdź klucz API")
    
    except openai.error.RateLimitError as e:
        print(f"⏰ Przekroczono limit: {str(e)}")
        raise HTTPException(status_code=429, detail="Przekroczono limit zapytań")
    
    except Exception as e:
        print(f"💥 Nieoczekiwany błąd: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wewnętrzny błąd serwera: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_key_configured": bool(NOVITA_API_KEY),
        "model": "meta-llama/llama-3.3-3b-instruct (FREE)",
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera Novita AI (FREE model) na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
