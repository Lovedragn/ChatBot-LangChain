from fastapi import FastAPI
from pydantic import BaseModel
from Services.chatbot import ask_bot, handle_action
from Services.Database import init_db

app = FastAPI(title="LangChain Gemini Chatbot")

# Initialize DB
init_db()

class ChatRequest(BaseModel):
    user_input: str

@app.get("/")
def root():
    return {"message": "LangChain Gemini Chatbot Running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    ai_response = ask_bot(request.user_input)
    print(ai_response)  # For debugging
    result = handle_action(ai_response)
    return {"reply": result}
