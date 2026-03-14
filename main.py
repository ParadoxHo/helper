from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import asyncio
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Puter.js (Free, No Key)")

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
        "message": "AZS Support Bot API (Puter.js FREE) działa! 🇵🇱",
        "status": "online",
        "provider": "Puter.js + Cohere",
        "note": "Działa bez klucza API!",
        "endpoints": {
            "chat": "/chat - wyślij wiadomość POST z {'message': 'tekst'}"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    
    try:
        print(f"📤 Wysyłam zapytanie do Puter.js: {request.message[:50]}...")
        
        # Puter.js API endpoint - darmowy, nie wymaga klucza
        url = "https://api.puter.com/v1/ai/chat"
        
        # Przygotuj wiadomości z system promptem
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ]
        
        # Użyj modelu Cohere (darmowy przez Puter.js)
        payload = {
            "model": "cohere/command-r-plus-08-2024",  # Najlepszy model Cohere
            "messages": messages,
            "stream": False,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Błąd Puter.js: {response.status} - {error_text}")
                    
                    # Spróbuj z fallbackowym modelem
                    print("🔄 Próbuję z modelem fallback...")
                    payload["model"] = "cohere/command-r7b-12-2024"  # Lżejszy model
                    
                    async with session.post(url, json=payload) as retry_response:
                        if retry_response.status != 200:
                            raise HTTPException(
                                status_code=500, 
                                detail="Nie można połączyć się z Puter.js"
                            )
                        
                        result = await retry_response.json()
                else:
                    result = await response.json()
        
        # Wyciągnij odpowiedź z wyniku
        reply = result.get('message', {}).get('content', '')
        if not reply:
            reply = result.get('text', 'Brak odpowiedzi')
        
        print(f"✅ Otrzymano odpowiedź od Puter.js ({len(reply)} znaków)")
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except asyncio.TimeoutError:
        print("⏰ Timeout przy połączeniu z Puter.js")
        raise HTTPException(status_code=504, detail="Przekroczono czas oczekiwania")
    
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "provider": "Puter.js (no API key needed)",
        "model": "cohere/command-r-plus-08-2024",
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera Puter.js na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
