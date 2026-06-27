from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import os

app = FastAPI(title="Patraratorn Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://patraratorn.web.app"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """คุณคือ AI Assistant ส่วนตัวของ "ภัทรธร สุภาพ"..."""

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=1000)
    provider: str = "deepseek"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if req.provider == "deepseek" and DEEPSEEK_API_KEY:
        return _call_deepseek(req.message)
    elif GEMINI_API_KEY:
        return _call_gemini(req.message)
    raise HTTPException(403, "No API key configured")

def _call_gemini(message: str) -> dict:
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={GEMINI_API_KEY}",
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": message}]}]
        }, timeout=15)
    data = resp.json()
    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return {"reply": text}

def _call_deepseek(message: str) -> dict:
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ]
        }, timeout=15)
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"reply": text}
