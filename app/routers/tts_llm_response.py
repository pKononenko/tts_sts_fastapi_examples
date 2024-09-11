import os

from fastapi import APIRouter, File, UploadFile

from app.models import llm_responder, whisper_transcriber
from app.schemas.schemas import TranscriptionOutput

router = APIRouter(tags=["TTS with LLM"])


@router.post("/tts-llm-processor")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    audio_file_path = f"temp_{file.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Use the transcriber class to get the transcription
    transcription, lang = whisper_transcriber.transcribe(audio_file_path)

    # Remove temporary audio file after transcription
    os.remove(audio_file_path)

    # Get answer from LLM
    response = llm_responder.get_llm_response(transcription)

    return response  #TranscriptionOutput(transcription=transcription, lang=lang)
