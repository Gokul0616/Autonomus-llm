from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
async def generate_text(request: dict):
    return {"text": model.generate(request["prompt"])}

# deepseek/api.py
from fastapi import FastAPI
from brain.model_manager import ModelManager
from brain.router import TaskRouter

app = FastAPI()
manager = ModelManager()
router = TaskRouter()

@app.post("/generate")
async def generate(text: str):
    model_type = router.classify_task(text)
    model = manager.load_model(model_type)
    with torch.inference_mode():
        return model.generate(text)