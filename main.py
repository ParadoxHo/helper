from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot")

# Разрешаем запросы с любых доменов (для тестирования)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получаем ключ API из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("WARNING: OPENAI_API_KEY не установлен!")

# Модель запроса от клиента
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # для логирования, опционально

# Модель ответа
class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None

# Системный промпт на польском языке
SYSTEM_PROMPT = """
Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych.
Twoja specjalizacja:
- Systemy kontroli dostępu (szlabany, czytniki kart, zamki)
- Monitoring wizyjny (kamery, rejestratory)
- Systemy alarmowe (czujki, sygnalizacja)

Zasady:
1. Odpowiadaj zawsze po polsku, krótko i rzeczowo.
2. Jeśli klient opisuje problem, zadaj pytania diagnostyczne (sprawdź zasilanie, połączenia sieciowe).
3. Jeśli problem jest prosty – podaj instrukcję rozwiązania.
4. Jeśli problem wymaga interwencji technika – zaproponuj wezwanie serwisu.
5. Bądź uprzejmy i profesjonalny.

Przykładowe problemy:
- "Szlabан nie otwiera się" → sprawdź fotokomórki, zasilanie, zresetuj sterownik.
- "Kamera nie pokazuje obrazu" → sprawdź kabel Ethernet, zresetuj kamerę.
- "Alarm fałszywy" → sprawdź czystość czujki, baterię, wyklucz strefę tymczasowo.
"""

@app.get("/")
def root():
    return {"message": "AZS Support Bot API działa! Użyj /chat do wysyłania wiadomości."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")
    
    try:
        # Вызываем OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # можно заменить на gpt-4
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        
        # Здесь можно добавить сохранение диалога в базу данных
        
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except Exception as e:
        # Логируем ошибку
        print(f"Błąd OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)