from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import re
from typing import Optional

app = FastAPI(title="Wsparcie Techniczne AZS")

# Разрешаем CORS для твоего фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assistics.netlify.app", "http://localhost:8000"],  # без слеша в конце!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY nie ustawiony!")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Asystent techniczny AZS działa. Użyj /chat do wysyłania wiadomości."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API Groq")

    try:
        print(f"📤 Zapytanie: {request.message[:50]}...")

        system_prompt = """
Jesteś inżynierem wsparcia technicznego pierwszej linii dla systemów bezpieczeństwa na stacjach benzynowych.
Twoi rozmówcy to pracownicy stacji (operatorzy), którzy nie są specjalistami. Mówią po polsku.

ZNASZ NASTĘPUJĄCE SYSTEMY (POZIOM UŻYTKOWNIKA):
- Monitoring: Bosch DIVAR, Bosch DIP, 3xLogic, Provision, Hikvision
- Alarmy: Paradox EVO192, SP65, SP4000, Satel Integra
- Kontrola dostępu: Rosslare B32 (zmiana kodu użytkownika)

ZADANIA UŻYTKOWNIKA:
- zmiana kodu użytkownika w Paradox, Satel, Rosslare
- raportowanie awarii: kamera nie działa, brak obrazu, czujka fałszywie alarmuje, czytnik nie reaguje

ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i rzeczowo.
2. Nie podawaj instrukcji programowania – jeśli użytkownik o to pyta, powiedz, że to może zrobić tylko serwisant.
3. Dla typowych problemów sugeruj proste czynności:
   - sprawdź zasilanie (kable, korki)
   - sprawdź połączenia sieciowe (diody na kamerze)
   - zresetuj urządzenie (odłącz zasilanie na 10 sekund)
   - sprawdź czystość czujek (pajęczyny, kurz)
   - wymień baterię w czujce bezprzewodowej
4. Jeśli problem jest poważny, zasugeruj wezwanie serwisu.
5. Bądź uprzejmy i cierpliwy.
"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",  # актуальная модель
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "top_p": 0.9
                },
                timeout=30.0
            )

            if response.status_code != 200:
                print(f"❌ Błąd Groq: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="Błąd komunikacji z Groq")

            data = response.json()
            reply = data["choices"][0]["message"]["content"]

            print(f"✅ Odpowiedź wysłana")
            return ChatResponse(reply=reply, session_id=request.session_id)

    except httpx.TimeoutException:
        return ChatResponse(reply="Przepraszam, serwis nie odpowiada. Spróbuj ponownie za chwilę.", session_id=request.session_id)
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        return ChatResponse(reply="Wystąpił błąd. Proszę spróbować później.", session_id=request.session_id)

# Явная обработка OPTIONS для preflight (на всякий случай)
@app.options("/chat")
async def options_chat():
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.get("/health")
async def health():
    return {"status": "healthy", "groq_configured": bool(GROQ_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
