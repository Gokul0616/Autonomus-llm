from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
async def generate_text(request: dict):
    return {"text": model.generate(request["prompt"])}