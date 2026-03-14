HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wsparcie AZS - Asystent AI</title>
    <style>
        /* STYLE - zostawiamy bez zmian */
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
        
        .chat-input button:hover {
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
        
        .error-badge {
            background: #f44336;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🚀 Wsparcie Techniczne AZS</h1>
            <p>Asystent AI • Polski</p>
            <span class="status-badge" id="status-badge">Inicjalizacja...</span>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message bot">
                Cześć! 👋 Jestem asystentem technicznym. Opisz swój problem związany ze szlabanami, kamerami lub alarmami.
            </div>
        </div>
        
        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Wpisz pytanie..." autocomplete="off">
            <button id="send-button" disabled>➤</button>
        </div>
    </div>

    <script>
        // Funkcja inicjalizacji - uruchamiana po załadowaniu DOM
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM załadowany, inicjalizacja...');
            
            // Pobieramy elementy
            const messagesDiv = document.getElementById('chat-messages');
            const input = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const statusBadge = document.getElementById('status-badge');
            
            // Sprawdzamy czy elementy istnieją
            if (!messagesDiv || !input || !sendButton || !statusBadge) {
                console.error('Nie znaleziono elementów DOM!');
                return;
            }
            
            console.log('Elementy DOM znalezione:', {messagesDiv, input, sendButton, statusBadge});
            
            const sessionId = 'sesja_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            let puterReady = false;
            let currentModel = 'upstage/solar-pro-3';

            // Funkcja dodawania wiadomości
            function addMessage(text, sender, isTyping = false) {
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${sender}${isTyping ? ' typing' : ''}`;
                msgDiv.textContent = text;
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                return msgDiv;
            }

            // Funkcja aktualizacji statusu
            function updateStatus(text, isError = false) {
                statusBadge.textContent = text;
                statusBadge.className = 'status-badge' + (isError ? ' error-badge' : '');
            }

            // Ładowanie Puter.js
            async function loadPuterJS() {
                return new Promise((resolve, reject) => {
                    if (window.puter) {
                        resolve();
                        return;
                    }
                    
                    const script = document.createElement('script');
                    script.src = 'https://js.puter.com/v2/';
                    script.async = true;
                    
                    script.onload = () => {
                        console.log('✅ Puter.js loaded');
                        updateStatus('Puter.js gotowy');
                        puterReady = true;
                        sendButton.disabled = false;
                        input.disabled = false;
                        input.focus();
                        resolve();
                    };
                    
                    script.onerror = () => {
                        console.error('❌ Failed to load Puter.js');
                        updateStatus('Błąd ładowania', true);
                        reject(new Error('Failed to load Puter.js'));
                    };
                    
                    document.head.appendChild(script);
                });
            }

            // Lista modeli do prób
            const models = [
                'upstage/solar-pro-3',
                'cohere/command-r-plus-08-2024',
                'claude-3-haiku'
            ];

            // Wysyłanie wiadomości
            async function sendMessage() {
                const text = input.value.trim();
                if (!text || !puterReady) return;

                input.value = '';
                sendButton.disabled = true;
                
                addMessage(text, 'user');
                
                const typingMsg = addMessage('✎ Asystent pisze...', 'bot', true);

                try {
                    let response;
                    let lastError;

                    // Próba z różnymi modelami
                    for (const model of models) {
                        try {
                            console.log(`Próba modelu: ${model}`);
                            response = await window.puter.ai.chat(text, {
                                model: model,
                                system_prompt: `Jesteś inżynierem wsparcia technicznego dla systemów bezpieczeństwa na stacjach benzynowych w Polsce.
Specjalizacja: szlabany, kamery, alarmy, kontrola dostępu.
Odpowiadaj TYLKO po polsku, krótko i konkretnie.
Zawsze zaczynaj od diagnostyki: sprawdź zasilanie, kable, połączenia.`
                            });
                            
                            currentModel = model;
                            updateStatus(`Model: ${model.split('/').pop()}`);
                            break;
                            
                        } catch (e) {
                            console.log(`Model ${model} nie działa:`, e);
                            lastError = e;
                            continue;
                        }
                    }

                    typingMsg.remove();

                    if (!response) {
                        throw lastError || new Error('Żaden model nie odpowiedział');
                    }

                    // Wyciągnij odpowiedź
                    let reply = '';
                    if (typeof response === 'string') reply = response;
                    else if (response?.message?.content) reply = response.message.content;
                    else if (response?.text) reply = response.text;
                    else if (response?.content) reply = response.content;
                    else reply = JSON.stringify(response);

                    addMessage(reply, 'bot');

                } catch (error) {
                    console.error('Puter.js error:', error);
                    typingMsg.remove();
                    
                    let errorMsg = '❌ Błąd: ';
                    if (error.message?.includes('authentication')) {
                        errorMsg += 'Wymagane logowanie do Puter. Odśwież stronę i zaloguj się.';
                        updateStatus('Wymaga logowania', true);
                    } else if (error.message?.includes('quota')) {
                        errorMsg += 'Przekroczono limit Puter. Spróbuj później.';
                        updateStatus('Limit exceeded', true);
                    } else {
                        errorMsg += 'Problem z połączeniem. Spróbuj ponownie.';
                        updateStatus('Błąd połączenia', true);
                    }
                    
                    addMessage(errorMsg, 'bot');
                } finally {
                    sendButton.disabled = false;
                    input.focus();
                }
            }

            // Inicjalizacja
            async function initialize() {
                updateStatus('Ładowanie Puter.js...');
                
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
                    
                } catch (error) {
                    updateStatus('Błąd inicjalizacji', true);
                    addMessage('❌ Nie można załadować asystenta. Odśwież stronę.', 'bot');
                }
            }

            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            input.addEventListener('input', () => {
                sendButton.disabled = !input.value.trim() || !puterReady;
            });

            // Uruchom inicjalizację
            initialize();
        });
    </script>
</body>
</html>
"""
