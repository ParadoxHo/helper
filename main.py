from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import datetime
from typing import Optional

# Создаём приложение
app = FastAPI(title="AZS Support Bot - Puter.js")

# Настройка CORS
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

# HTML шаблон с чатом
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wsparcie AZS - Asystent AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
        
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
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
        
        .chat-input input:focus {
            border-color: #667eea;
        }
        
        .chat-input input:disabled {
            background: #f5f5f5;
            cursor: not-allowed;
        }
        
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
        
        .chat-input button:hover:not(:disabled) {
            transform: scale(1.1);
        }
        
        .chat-input button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
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
        
        .status-badge.error {
            background: #f44336;
        }
        
        .status-badge.warning {
            background: #ff9800;
        }
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
            
            console.log('Script started');
            
            // Funkcja bezpiecznego pobierania elementów
            function getElement(id) {
                const el = document.getElementById(id);
                if (!el) {
                    console.error('Element not found:', id);
                }
                return el;
            }

            // Inicjalizacja po załadowaniu DOM
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', init);
            } else {
                init();
            }

            function init() {
                console.log('Initializing chat...');
                
                // Pobierz wszystkie elementy
                const messagesDiv = getElement('chat-messages');
                const input = getElement('message-input');
                const sendButton = getElement('send-button');
                const statusBadge = getElement('status-badge');

                if (!messagesDiv || !input || !sendButton || !statusBadge) {
                    console.error('Critical error: DOM elements not found');
                    return;
                }

                // Dodaj wiadomość powitalną
                const welcomeMsg = document.createElement('div');
                welcomeMsg.className = 'message bot';
                welcomeMsg.textContent = 'Cześć! 👋 Jestem asystentem technicznym. Opisz swój problem związany ze szlabanami, kamerami lub alarmami.';
                messagesDiv.appendChild(welcomeMsg);

                const sessionId = 'sesja_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                let puterReady = false;
                let currentModel = '';

                // Funkcja dodawania wiadomości
                function addMessage(text, sender, isError = false) {
                    const msgDiv = document.createElement('div');
                    msgDiv.className = `message ${sender}${isError ? ' error' : ''}`;
                    msgDiv.textContent = text;
                    messagesDiv.appendChild(msgDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    return msgDiv;
                }

                // Funkcja dodawania wiadomości tymczasowej (np. "pisze...")
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

                // Funkcja usuwania wiadomości tymczasowej
                function removeTempMessage(id) {
                    const el = document.getElementById(id);
                    if (el) el.remove();
                }

                // Aktualizacja statusu
                function updateStatus(text, type = 'normal') {
                    statusBadge.textContent = text;
                    statusBadge.className = 'status-badge';
                    if (type === 'error') {
                        statusBadge.classList.add('error');
                    } else if (type === 'warning') {
                        statusBadge.classList.add('warning');
                    }
                }

                // Ładowanie Puter.js
                async function loadPuterJS() {
                    return new Promise((resolve, reject) => {
                        if (window.puter) {
                            console.log('Puter.js already loaded');
                            resolve();
                            return;
                        }
                        
                        console.log('Loading Puter.js...');
                        updateStatus('Ładowanie Puter.js...');
                        
                        const script = document.createElement('script');
                        script.src = 'https://js.puter.com/v2/';
                        script.async = true;
                        
                        script.onload = () => {
                            console.log('✅ Puter.js loaded successfully');
                            updateStatus('Puter.js gotowy');
                            puterReady = true;
                            input.disabled = false;
                            sendButton.disabled = false;
                            input.focus();
                            resolve();
                        };
                        
                        script.onerror = (error) => {
                            console.error('❌ Failed to load Puter.js:', error);
                            updateStatus('Błąd ładowania', 'error');
                            reject(new Error('Failed to load Puter.js'));
                        };
                        
                        document.head.appendChild(script);
                    });
                }

                // Lista modeli do wypróbowania
                const models = [
                    'upstage/solar-pro-3',
                    'cohere/command-r-plus-08-2024',
                    'claude-3-haiku',
                    'gpt-3.5-turbo'
                ];

                // Wysyłanie wiadomości
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

                        // Próbuj różne modele
                        for (const model of models) {
                            try {
                                console.log(`Trying model: ${model}`);
                                updateStatus(`Próba: ${model.split('/').pop()}`, 'warning');
                                
                                response = await window.puter.ai.chat(text, {
                                    model: model,
                                    system_prompt: `Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce.
Twoja specjalizacja: szlabany, kamery, alarmy, kontrola dostępu.
Odpowiadaj TYLKO po polsku, krótko i konkretnie.
Zawsze zaczynaj od diagnostyki: sprawdź zasilanie, kable, połączenia.
Jeśli nie znasz odpowiedzi, zasugeruj wezwanie serwisu.`
                                });
                                
                                currentModel = model;
                                updateStatus(`Model: ${model.split('/').pop()}`);
                                console.log(`✅ Success with model: ${model}`);
                                break; // Sukces, wyjdź z pętli
                                
                            } catch (e) {
                                console.log(`Model ${model} failed:`, e.message);
                                lastError = e;
                                continue; // Spróbuj następny model
                            }
                        }

                        removeTempMessage(tempId);

                        if (!response) {
                            throw lastError || new Error('All models failed');
                        }

                        // Wyciągnij odpowiedź z różnych formatów
                        let reply = '';
                        if (typeof response === 'string') {
                            reply = response;
                        } else if (response?.message?.content) {
                            reply = response.message.content;
                        } else if (response?.text) {
                            reply = response.text;
                        } else if (response?.content) {
                            reply = response.content;
                        } else {
                            reply = JSON.stringify(response);
                        }

                        addMessage(reply, 'bot');

                    } catch (error) {
                        console.error('Puter.js error:', error);
                        removeTempMessage(tempId);
                        
                        let errorMsg = '❌ Błąd: ';
                        if (error.message?.includes('authentication') || error.message?.includes('auth')) {
                            errorMsg += 'Wymagane logowanie do Puter. Odśwież stronę i zaloguj się.';
                            updateStatus('Wymaga logowania', 'error');
                        } else if (error.message?.includes('quota') || error.message?.includes('limit')) {
                            errorMsg += 'Przekroczono limit Puter. Spróbuj później.';
                            updateStatus('Limit exceeded', 'error');
                        } else if (error.message?.includes('network') || error.message?.includes('fetch')) {
                            errorMsg += 'Problem z połączeniem. Sprawdź internet.';
                            updateStatus('Błąd sieci', 'error');
                        } else {
                            errorMsg += 'Nieznany błąd. Spróbuj ponownie.';
                            updateStatus('Błąd', 'error');
                        }
                        
                        addMessage(errorMsg, 'bot', true);
                    } finally {
                        sendButton.disabled = false;
                        input.disabled = false;
                        input.focus();
                    }
                }

                // Inicjalizacja
                async function initialize() {
                    try {
                        await loadPuterJS();
                        
                        // Dodaj przykładowe pytania
                        const examples = [
                            'Szlaban nie otwiera się',
                            'Kamera nie nagrywa',
                            'Alarm ciągle się włącza'
                        ];
                        
                        const helpMsg = document.createElement('div');
                        helpMsg.className = 'message bot';
                        helpMsg.innerHTML = '💡 Przykładowe pytania:<br>' + 
                            examples.map(ex => `• "${ex}"`).join('<br>');
                        messagesDiv.appendChild(helpMsg);
                        
                        console.log('Initialization complete');
                        
                    } catch (error) {
                        console.error('Initialization failed:', error);
                        updateStatus('Błąd inicjalizacji', 'error');
                        addMessage('❌ Nie można załadować asystenta. Odśwież stronę.', 'bot', true);
                    }
                }

                // Event listeners
                sendButton.addEventListener('click', sendMessage);
                
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                input.addEventListener('input', () => {
                    const hasText = input.value.trim().length > 0;
                    sendButton.disabled = !hasText || !puterReady;
                });

                // Start
                initialize();
            }
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
    """Ten endpoint jest zachowany dla kompatybilności, ale nie jest używany"""
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
    print(f"🌐 Otwórz http://localhost:{port} w przeglądarce")
    uvicorn.run(app, host="0.0.0.0", port=port)
