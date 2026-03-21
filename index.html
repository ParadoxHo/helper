<!DOCTYPE html>
<html lang="pl">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Wsparcie Techniczne</title>

<link rel="preconnect"
href="https://fonts.googleapis.com">

<link rel="preconnect"
href="https://fonts.gstatic.com"
crossorigin>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
rel="stylesheet">

<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

body{

font-family:Inter;

background:
linear-gradient(145deg,#eef2f6,#d9e0eb);

height:100vh;

display:flex;

align-items:center;

justify-content:center;

padding:15px;

}

.chat{

width:100%;
max-width:1200px;

height:95vh;

background:
rgba(255,255,255,0.75);

backdrop-filter:blur(20px);

border-radius:40px;

display:flex;

flex-direction:column;

overflow:hidden;

box-shadow:
0 30px 60px -20px rgba(0,20,40,0.3);

}

.header{

padding:20px 30px;

background:rgba(255,255,255,0.4);

border-bottom:1px solid rgba(255,255,255,0.7);

}

.top{

display:flex;

justify-content:space-between;

align-items:center;

}

.brand{

display:flex;

gap:12px;

align-items:center;

}

.brand i{

font-size:28px;

background:white;

padding:10px;

border-radius:15px;

}

.status{

background:#e6f0ff;

padding:7px 16px;

border-radius:30px;

font-size:.85rem;

display:flex;

gap:8px;

align-items:center;

}

.messages{

flex:1;

overflow:auto;

padding:25px;

display:flex;

flex-direction:column;

gap:14px;

}

.msg{

max-width:75%;

padding:14px 18px;

border-radius:25px;

position:relative;

animation:fade .2s;

word-wrap:break-word;

}

@keyframes fade{

from{
opacity:0;
transform:translateY(10px);
}

}

.user{

background:#0a2647;

color:white;

align-self:flex-end;

border-bottom-right-radius:8px;

}

.bot{

background:white;

align-self:flex-start;

border-bottom-left-radius:8px;

}

.time{

font-size:.7rem;

opacity:.6;

margin-top:6px;

}

.actions{

position:absolute;

top:-10px;

right:-10px;

display:none;

}

.msg:hover .actions{

display:block;

}

.copy{

border:none;

background:white;

border-radius:8px;

padding:4px 7px;

cursor:pointer;

box-shadow:0 3px 10px rgba(0,0,0,.1);

}

.footer{

padding:20px;

background:rgba(255,255,255,0.4);

border-top:1px solid rgba(255,255,255,0.6);

}

.inputWrap{

display:flex;

gap:12px;

}

textarea{

flex:1;

border:none;

resize:none;

padding:15px;

border-radius:25px;

font-family:Inter;

outline:none;

max-height:140px;

}

button{

border:none;

background:#0a2647;

color:white;

width:60px;

height:60px;

border-radius:50%;

cursor:pointer;

font-size:1.3rem;

transition:.2s;

}

button:hover{

background:#1e3a5f;

}

button:disabled{

opacity:.4;

cursor:not-allowed;

}

.typing{

background:#eef2f6;

font-style:italic;

}

.dot{

animation:blink 1.4s infinite;

}

.dot:nth-child(2){

animation-delay:.2s;

}

.dot:nth-child(3){

animation-delay:.4s;

}

@keyframes blink{

0%{opacity:.2}

20%{opacity:1}

100%{opacity:.2}

}

code{

background:#0a2647;

color:white;

padding:3px 6px;

border-radius:6px;

}

pre{

background:#0a2647;

color:white;

padding:14px;

border-radius:15px;

overflow:auto;

margin-top:10px;

}

.retry{

margin-top:10px;

background:#e74c3c;

padding:6px 12px;

border-radius:10px;

font-size:.8rem;

cursor:pointer;

color:white;

border:none;

}

</style>

</head>

<body>

<div class="chat">

<div class="header">

<div class="top">

<div class="brand">

<i class="fas fa-shield-halved"></i>

<h2>Wsparcie</h2>

</div>

<div class="status"
id="status">

<i class="fas fa-circle"></i>

<span>Connecting</span>

</div>

</div>

</div>

<div class="messages"
id="messages">

</div>

<div class="footer">

<div class="inputWrap">

<textarea
id="input"
placeholder="Wpisz pytanie..."
rows="1">

</textarea>

<button
id="send"
disabled>

<i class="fas fa-paper-plane">

</i>

</button>

</div>

</div>

</div>

<script>

const API =
"https://assistics.up.railway.app/chat";

const session =
crypto.randomUUID();

const box =
document.getElementById("messages");

const input =
document.getElementById("input");

const send =
document.getElementById("send");

let sending=false;

function now(){

return new Date()
.toLocaleTimeString([],{
hour:'2-digit',
minute:'2-digit'
});

}

function save(){

localStorage.setItem(
"chat_history",
box.innerHTML
);

}

function load(){

const data=
localStorage.getItem(
"chat_history"
);

if(data)
box.innerHTML=data;

}

function status(text,error=false){

document.getElementById(
"status"
).innerHTML=

`<i class="fas fa-circle"
style="color:${
error?'#e74c3c':'#27ae60'
}"></i>
<span>${text}</span>`;

}

function message(text,type){

const div=
document.createElement("div");

div.className=
"msg "+type;

div.innerHTML=

marked.parse(text)+

`<div class="time">
${now()}
</div>

<div class="actions">

<button class="copy">
📋
</button>

</div>`;

div.querySelector(".copy")
.onclick=()=>

navigator.clipboard
.writeText(
div.innerText
);

box.appendChild(div);

box.scrollTop=
box.scrollHeight;

save();

return div;

}

function typing(){

const t=
document.createElement("div");

t.id="typing";

t.className=
"msg bot typing";

t.innerHTML=

`Asystent pisze

<span class="dot">•</span>

<span class="dot">•</span>

<span class="dot">•</span>`;

box.appendChild(t);

box.scrollTop=
box.scrollHeight;

}

function removeTyping(){

const t=
document.getElementById(
"typing"
);

if(t)
t.remove();

}

async function sendMessage(text){

if(sending)
return;

if(!text.trim())
return;

sending=true;

send.disabled=true;

input.disabled=true;

message(text,"user");

input.value="";

typing();

try{

const controller=
new AbortController();

const timeout=
setTimeout(
()=>controller.abort(),
25000
);

const response=
await fetch(API,{

method:"POST",

headers:{
"Content-Type":
"application/json"
},

body:JSON.stringify({

message:text,

session_id:session

}),

signal:controller.signal

});

clearTimeout(timeout);

removeTyping();

if(!response.ok)
throw Error(response.status);

const data=
await response.json();

streamReply(
data.reply ||
"Brak odpowiedzi"
);

status("Online");

}

catch(e){

removeTyping();

status("Offline",true);

const m=
message(
"❌ Błąd połączenia",
"bot"
);

const retry=
document.createElement(
"button"
);

retry.className=
"retry";

retry.textContent=
"Spróbuj ponownie";

retry.onclick=
()=>sendMessage(text);

m.appendChild(retry);

}

sending=false;

send.disabled=false;

input.disabled=false;

input.focus();

}

function streamReply(text){

const m=
message("","bot");

let i=0;

function type(){

if(i<text.length){

m.childNodes[0]
.textContent+=
text.charAt(i);

i++;

box.scrollTop=
box.scrollHeight;

setTimeout(
type,
10
);

}

else save();

}

type();

}

input.addEventListener(
"input",
()=>{

send.disabled=
!input.value.trim();

input.style.height=
"auto";

input.style.height=
input.scrollHeight+"px";

}

);

input.addEventListener(
"keydown",
e=>{

if(
e.key==="Enter"
&&!e.shiftKey
){

e.preventDefault();

sendMessage(
input.value
);

}

}

);

send.onclick=
()=>sendMessage(
input.value
);

load();

message(
"Cześć 👋 Opisz problem.",
"bot"
);

status("Gotowy");

</script>

</body>

</html>
