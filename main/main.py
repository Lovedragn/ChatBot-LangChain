import fastapi

app = fastapi.FastAPI()

@app.post("/")
def prompt_request():
    RAG()
