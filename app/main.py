from typing import Dict

from fastapi import FastAPI

from app.routers import tts_llm_response, tts_response

app = FastAPI()

# For a simpler task with audio-to-text transcription
app.include_router(tts_response.router)
app.include_router(tts_llm_response.router)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Health Check"}
