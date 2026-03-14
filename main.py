from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Puter.js Client")

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

@app.get("/")
def root():
    return {
        "message": "AZS Support Bot API - Puter.js Gateway",
        "status": "online",
        "instructions": "Ten serwer służy jako proxy dla Puter.js AI. Użyj /chat z przeglądarki.",
    }

@app.get("/client.js")
def get_client_script():
    """Zwraca skrypt JavaScript do osadzenia na stronie"""
    js_content = """
    // Puter.js Client for AZS Support Bot
    // Ten skrypt łączy się z Puter.js bezpośrednio z przeglądarki
    
    (function() {
        console.log('AZS Puter.js client loading...');
        
        // Funkcja inicjalizująca Puter.js
        window.initPuterChat = function(apiUrl, sessionId) {
            console.log('Initializing Puter.js chat...');
            
            // Dodaj skrypt Puter.js jeśli nie istnieje
            if (!document.querySelector('script[src="https://js.puter.com/v2/"]')) {
                const script = document.createElement('script');
                script.src = 'https://js.puter.com/v2/';
                script.async = true;
                script.onload = function() {
                    console.log('Puter.js loaded successfully');
                    window.puterReady = true;
                };
                script.onerror = function() {
                    console.error('Failed to load Puter.js');
                    window.puterReady = false;
                };
                document.head.appendChild(script);
            }
            
            // Funkcja do wysyłania wiadomości do Puter.js AI
            window.sendToPuterAI = async function(message) {
                console.log('Sending to Puter AI:', message);
                
                if (!window.puter || !window.puter.ai) {
                    throw new Error('Puter.js not loaded yet');
                }
                
                try {
                    // Używamy modelu Solar Pro 3 (działa z polskim)
                    const response = await window.puter.ai.chat(message, {
                        model: 'upstage/solar-pro-3',  // Dobry model dla polskiego
                        system_prompt: `Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce. Odpowiadaj TYLKO po polsku, krótko i konkretnie.`
                    });
                    
                    console.log('Puter AI response received');
                    
                    // Puter.ai.chat zwraca różne formaty
                    if (typeof response === 'string') {
                        return response;
                    } else if (response && response.message) {
                        return response.message.content || response.message;
                    } else if (response && response.text) {
                        return response.text;
                    } else if (response && response.content) {
                        return response.content;
                    } else {
                        return JSON.stringify(response);
                    }
                } catch (error) {
                    console.error('Puter AI error:', error);
                    throw error;
                }
            };
            
            console.log('Puter.js chat initialized');
        };
        
        // Automatyczna inicjalizacja po załadowaniu strony
        window.addEventListener('load', function() {
            if (window.initialApiUrl) {
                window.initPuterChat(window.initialApiUrl, window.initialSessionId);
            }
        });
    })();
    """
    return HTMLResponse(content=js_content, media_type="application/javascript")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ten endpoint jest teraz tylko pośrednikiem - właściwa odpowiedź będzie z Puter.js w przeglądarce"""
    return ChatResponse(
        reply="Klient Puter.js został załadowany. Odpowiedź zostanie wygenerowana bezpośrednio w Twojej przeglądarce.",
        session_id=request.session_id
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "puter.js client proxy",
        "cors_origins": ["https://asistics.netlify.app", "localhost"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Uruchamianie serwera Puter.js proxy na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
