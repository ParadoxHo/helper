from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import datetime
from typing import Optional

app = FastAPI(title="AZS Support Bot - Puter.js")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wsparcie AZS - Asystent AI</title>
    <style>
        /* style pozostają bez zmian */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { font-size: 1.8em; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 0.9em; }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f7fb;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 15px;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .message.bot {
            align-self: flex-start;
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-bottom-left-radius: 5px;
        }
        .message.typing {
            background: #e0e0e0;
            color: #666;
            font-style: italic;
        }
        .message.error {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ef9a9a;
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e4ff;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e4ff;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input input:disabled { background: #f5f5f5; cursor: not-allowed; }
        .chat-input button {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }
        .chat-input button:hover:not(:disabled) { transform: scale(1.1); }
        .chat-input button:disabled { opacity: 0.5; cursor: not-allowed; }
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            background: #4CAF50;
            color: white;
            border-radius: 12px;
            font-size: 0.7em;
            margin-left: 10px;
            vertical-align: middle;
        }
        .status-badge.error { background: #f44336; }
        .status-badge.warning { background: #ff9800; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🚀 Wsparcie Techniczne AZS</h1>
            <p>Asystent AI • Polski • Puter.js</p>
            <span class="status-badge" id="status-badge">Inicjalizacja...</span>
        </div>
        
        <div class="chat-messages" id="chat-messages"></div>
        
        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Wpisz pytanie..." autocomplete="off" disabled>
            <button id="send-button" disabled>➤</button>
        </div>
    </div>

    <script>
        (function() {
            'use strict';
            
            console.log('Skrypt startuje, czekam na elementy...');
            
            // Funkcja, która uruchomi się, gdy elementy będą dostępne
            function whenElementsReady(callback) {
                const checkInterval = setInterval(function() {
                    const messagesDiv = document.getElementById('chat-messages');
                    const input = document.getElementById('message-input');
                    const sendButton = document.getElementById('send-button');
                    const statusBadge = document.getElementById('status-badge');
                    
                    if (messagesDiv && input && sendButton && statusBadge) {
                        clearInterval(checkInterval);
                        console.log('Elementy znalezione, inicjalizacja...');
                        callback(messagesDiv, input, sendButton, statusBadge);
                    }
                }, 10); // sprawdzaj co 10ms
            }

            whenElementsReady(function(messagesDiv, input, sendButton, statusBadge) {
                // Reszta kodu z poprzedniej wersji, ale z użyciem przekazanych elementów
                // (poniżej wklej całą logikę, tylko zamień zmienne na te z parametrów)
                
                // Wiadomość powitalna
                const welcomeMsg = document.createElement('div');
                welcomeMsg.className = 'message bot';
                welcomeMsg.textContent = 'Cześć! 👋 Jestem asystentem technicznym. Opisz swój problem związany ze szlabanami, kamerami lub alarmami.';
                messagesDiv.appendChild(welcomeMsg);

                let puterReady = false;
                let currentModel = '';

                function addMessage(text, sender, isError = false) {
                    const msgDiv = document.createElement('div');
                    msgDiv.className = `message ${sender}${isError ? ' error' : ''}`;
                    msgDiv.textContent = text;
                    messagesDiv.appendChild(msgDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    return msgDiv;
                }

                function addTempMessage(text) {
                    const tempId = 'temp_' + Date.now();
                    const msgDiv = document.createElement('div');
                    msgDiv.id = tempId;
                    msgDiv.className = 'message bot typing';
                    msgDiv.textContent = text;
                    messagesDiv.appendChild(msgDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    return tempId;
                }

                function removeTempMessage(id) {
                    const el = document.getElementById(id);
                    if (el) el.remove();
                }

                function updateStatus(text, type = 'normal') {
                    statusBadge.textContent = text;
                    statusBadge.className = 'status-badge';
                    if (type === 'error') statusBadge.classList.add('error');
                    else if (type === 'warning') statusBadge.classList.add('warning');
                }

                async function loadPuterJS() {
                    return new Promise((resolve, reject) => {
                        if (window.puter) {
                            console.log('Puter.js już załadowany');
                            resolve();
                            return;
                        }
                        
                        console.log('Ładowanie Puter.js...');
                        updateStatus('Ładowanie Puter.js...');
                        
                        const script = document.createElement('script');
                        script.src = 'https://js.puter.com/v2/';
                        script.async = true;
                        
                        script.onload = () => {
                            console.log('✅ Puter.js załadowany');
                            updateStatus('Puter.js gotowy');
                            puterReady = true;
                            input.disabled = false;
                            sendButton.disabled = false;
                            input.focus();
                            resolve();
                        };
                        
                        script.onerror = (error) => {
                            console.error('❌ Błąd ładowania Puter.js:', error);
                            updateStatus('Błąd ładowania', 'error');
                            reject(new Error('Failed to load Puter.js'));
                        };
                        
                        document.head.appendChild(script);
                    });
                }

                const models = [
                    'upstage/solar-pro-3',
                    'cohere/command-r-plus-08-2024',
                    'claude-3-haiku'
                ];

                async function sendMessage() {
                    const text = input.value.trim();
                    if (!text || !puterReady) return;

                    input.value = '';
                    sendButton.disabled = true;
                    input.disabled = true;
                    
                    addMessage(text, 'user');
                    
                    const tempId = addTempMessage('✎ Asystent pisze...');

                    try {
                        let response = null;
                        let lastError = null;

                        for (const model of models) {
                            try {
                                console.log(`Próba modelu: ${model}`);
                                updateStatus(`Próba: ${model.split('/').pop()}`, 'warning');
                                
                                response = await window.puter.ai.chat(text, {
                                    model: model,
                                    system_prompt: `Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce. Specjalizacja: szlabany, kamery, alarmy, kontrola dostępu. Odpowiadaj TYLKO po polsku, krótko i konkretnie. Zawsze zaczynaj od diagnostyki: sprawdź zasilanie, kable, połączenia.`
                                });
                                
                                currentModel = model;
                                updateStatus(`Model: ${model.split('/').pop()}`);
                                console.log(`✅ Sukces z modelem: ${model}`);
                                break;
                                
                            } catch (e) {
                                console.log(`Model ${model} nie działa:`, e.message);
                                lastError = e;
                                continue;
                            }
                        }

                        removeTempMessage(tempId);

                        if (!response) throw lastError || new Error('All models failed');

                        let reply = '';
                        if (typeof response === 'string') reply = response;
                        else if (response?.message?.content) reply = response.message.content;
                        else if (response?.text) reply = response.text;
                        else if (response?.content) reply = response.content;
                        else reply = JSON.stringify(response);

                        addMessage(reply, 'bot');

                    } catch (error) {
                        console.error('Błąd Puter.js:', error);
                        removeTempMessage(tempId);
                        
                        let errorMsg = '❌ Błąd: ';
                        if (error.message?.includes('auth')) {
                            errorMsg += 'Wymagane logowanie do Puter. Odśwież stronę.';
                            updateStatus('Wymaga logowania', 'error');
                        } else if (error.message?.includes('quota')) {
                            errorMsg += 'Przekroczono limit. Spróbuj później.';
                            updateStatus('Limit exceeded', 'error');
                        } else {
                            errorMsg += 'Problem z połączeniem.';
                            updateStatus('Błąd', 'error');
                        }
                        
                        addMessage(errorMsg, 'bot', true);
                    } finally {
                        sendButton.disabled = false;
                        input.disabled = false;
                        input.focus();
                    }
                }

                async function initialize() {
                    try {
                        await loadPuterJS();
                        
                        const examples = ['Szlaban nie otwiera się', 'Kamera nie nagrywa', 'Alarm ciągle się włącza'];
                        
                        const helpMsg = document.createElement('div');
                        helpMsg.className = 'message bot';
                        helpMsg.innerHTML = '💡 Przykładowe pytania:<br>' + examples.map(ex => `• "${ex}"`).join('<br>');
                        messagesDiv.appendChild(helpMsg);
                        
                        console.log('Inicjalizacja zakończona');
                        
                    } catch (error) {
                        console.error('Inicjalizacja nieudana:', error);
                        updateStatus('Błąd inicjalizacji', 'error');
                        addMessage('❌ Nie można załadować asystenta. Odśwież stronę.', 'bot', true);
                    }
                }

                // Dodajemy nasłuchiwacze
                sendButton.addEventListener('click', sendMessage);
                
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                input.addEventListener('input', () => {
                    sendButton.disabled = !input.value.trim() || !puterReady;
                });

                // Start
                initialize();
            });
        })();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(
        reply="Ten endpoint nie jest używany. Puter.js działa bezpośrednio w przeglądarce.",
        session_id=request.session_id
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "puter.js direct browser",
        "timestamp": str(datetime.datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Serwer Puter.js uruchomiony na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
