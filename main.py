from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import os
import re
from typing import Optional

app = FastAPI(title="Wsparcie Techniczne AZS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lizonka.netlify.app",  # или твой будущий сайт поддержки
        "http://localhost:8000"
    ],
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

        # === NOWY SYSTEM PROMPT DLA UŻYTKOWNIKÓW ===
        system_prompt = """
Jesteś inżynierem wsparcia technicznego pierwszej linii dla systemów bezpieczeństwa na stacjach benzynowych.
Twoi rozmówcy to pracownicy stacji (operatorzy), którzy nie są specjalistami. Mówią po polsku.

ZNASZ NASTĘPUJĄCE SYSTEMY (POZIOM UŻYTKOWNIKA):
- Monitoring: Bosch DIVAR, Bosch DIP, 3xLogic, Provision, Hikvision
- Alarmy: Paradox EVO192, SP65, SP4000, Satel Integra
- Kontrola dostępu: Rosslare B32

ZADANIA UŻYTKOWNIKA:
- zmiana kodu użytkownika w Paradox, Satel, Rosslare
- zmiana kodu otwierania drzwi w Rosslare
- raportowanie awarii: kamera nie działa, brak obrazu, szlaban się nie otwiera, czujka fałszywie alarmuje

ZASADY:
1. Odpowiadaj TYLKO po polsku, krótko i rzeczowo.
2. Nie podawaj instrukcji programowania ani zaawansowanej konfiguracji – jeśli użytkownik o to pyta, powiedz, że to może zrobić tylko serwisant.
3. Dla typowych problemów sugeruj proste czynności:
   - sprawdź zasilanie (czy kabel nie jest odłączony, czy nie wywaliło korków)
   - sprawdź połączenia sieciowe (kabel Ethernet, diody na kamerze)
   - zresetuj urządzenie (odłącz zasilanie na 10 sekund)
   - sprawdź czystość czujek (pajęczyny, kurz)
   - wymień baterię w czujce bezprzewodowej
4. Jeśli problem jest poważny (brak reakcji na reset, uszkodzenie mechaniczne), zasugeruj wezwanie serwisu.
5. Bądź uprzejmy i cierpliwy. Używaj prostego języka.

PRZYKŁADY:
- Pytanie: "Szlaban się nie otwiera po przyłożeniu karty"
  → Odpowiedź: "Proszę sprawdzić, czy dioda na czytniku świeci. Jeśli nie, sprawdź zasilanie. Jeśli świeci, ale brama nie reaguje, zresetuj sterownik – odłącz zasilanie na 10 sekund."
- Pytanie: "Jak zmienić kod w alarmie Paradox?"
  → Odpowiedź: "Aby zmienić kod użytkownika, należy wejść w tryb programowania użytkownika: na klawiaturze wpisz [kod serwisowy] + [0] + [nowy kod]. Dokładną instrukcję znajdziesz w dokumentacji."
- Pytanie: "Kamera nie nagrywa"
  → Odpowiedź: "Sprawdź, czy na rejestratorze jest wolne miejsce na dysku. Jeśli nie, nagrania starsze niż X dni są automatycznie kasowane. Możesz też sprawdzić, czy kamera jest widoczna w menu – jeśli nie, sprawdź połączenie sieciowe."

PAMIĘTAJ: Twoim celem jest pomóc użytkownikowi rozwiązać prosty problem lub skierować go do serwisu.
"""

        # Wywołanie Groq API (tak samo jak wcześniej)
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
                    "temperature": 0.3,  # niższa temperatura = bardziej przewidywalne odpowiedzi
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

            # Opcjonalnie: sprawdzenie czy odpowiedź jest po polsku (zawiera polskie znaki)
            if not re.search('[ąćęłńóśźżĄĆĘŁŃÓŚŹŻa-zA-Z]', reply):
                # Jeśli odpowiedź nie zawiera polskich liter, może być problem – ale rzadko
                print("⚠️ Odpowiedź może nie być po polsku, ale zwracamy jak jest.")

            print(f"✅ Odpowiedź wysłana")
            return ChatResponse(reply=reply, session_id=request.session_id)

    except httpx.TimeoutException:
        return ChatResponse(reply="Przepraszam, serwis nie odpowiada. Spróbuj ponownie za chwilę.", session_id=request.session_id)
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        return ChatResponse(reply="Wystąpił błąd. Proszę spróbować później.", session_id=request.session_id)

@app.get("/health")
async def health():
    return {"status": "healthy", "groq_configured": bool(GROQ_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
