from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import re
from typing import Optional, List, Dict
from collections import defaultdict
from datetime import datetime, timedelta

app = FastAPI(title="Wsparcie Techniczne AZS")

# CORS – разрешаем твой фронтенд
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assistics.netlify.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY nie ustawiony!")

# Хранилище историй диалогов (в памяти, для демо)
# Структура: { session_id: {"messages": list, "last_updated": datetime} }
history_store: Dict[str, Dict] = defaultdict(lambda: {"messages": [], "last_updated": datetime.now()})

def cleanup_old_sessions(max_age_minutes: int = 60):
    """Удаляет сессии, которые не обновлялись больше max_age_minutes."""
    now = datetime.now()
    to_delete = []
    for sid, data in history_store.items():
        if now - data["last_updated"] > timedelta(minutes=max_age_minutes):
            to_delete.append(sid)
    for sid in to_delete:
        del history_store[sid]

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

    # Определяем session_id (если не передан, используем "default")
    sid = request.session_id or "default"
    cleanup_old_sessions()  # очищаем старые сессии

    session_data = history_store[sid]
    session_data["last_updated"] = datetime.now()
    history = session_data["messages"]

    # Системный промпт (на польском)
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
6. Pamiętaj kontekst rozmowy – odpowiadaj na pytania użytkownika w sposób ciągły.
"""

    # Если история пуста, добавляем system prompt
    if not history:
        history.append({"role": "system", "content": system_prompt})

    # Добавляем текущее сообщение пользователя
    history.append({"role": "user", "content": request.message})

    # Ограничиваем длину истории (оставляем system + последние 10 сообщений)
    if len(history) > 11:  # system + 10 сообщений (user/assistant)
        history = [history[0]] + history[-10:]

    try:
        print(f"📤 Zapytanie (sesja {sid}): {request.message[:50]}...")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",  # актуальная модель
                    "messages": history,
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

        # Добавляем ответ ассистента в историю
        history.append({"role": "assistant", "content": reply})

        # Сохраняем обновленную историю
        session_data["messages"] = history

        print(f"✅ Odpowiedź wysłana (sesja {sid})")
        return ChatResponse(reply=reply, session_id=sid)

    except httpx.TimeoutException:
        return ChatResponse(reply="Przepraszam, serwis nie odpowiada. Spróbuj ponownie za chwilę.", session_id=sid)
    except Exception as e:
        print(f"💥 Błąd: {str(e)}")
        return ChatResponse(reply="Wystąpił błąd. Proszę spróbować później.", session_id=sid)

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
