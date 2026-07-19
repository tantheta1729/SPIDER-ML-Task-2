import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from assistant import orchestrate_healthcare_assistant

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat_endpoint(request: QueryRequest):
    try:
        ai_reply, confidence_level = orchestrate_healthcare_assistant(request.question)
        return {
            "reply": ai_reply,
            "confidence": confidence_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🛠️ FIXED FOLDER MAPPING ENGINE:
# This calculates the exact path where main.py lives, and targets the 'static' folder right next to it.
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir_path = os.path.join(current_dir, "static")

print(f"📁 System is explicitly serving frontend assets from: {static_dir_path}")

app.mount("/", StaticFiles(directory=static_dir_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)