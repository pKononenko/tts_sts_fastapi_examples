import os

from fastapi import APIRouter, File, UploadFile

from app.models import whisper_transcriber
from app.schemas.schemas import TranscriptionOutput

router = APIRouter(tags=["Basic TTS"])


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(
    ...)) -> TranscriptionOutput:
    # Save the uploaded file temporarily
    audio_file_path = f"temp_{file.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Use the transcriber class to get the transcription
    transcription, lang = whisper_transcriber.transcribe(audio_file_path)

    # Remove temporary audio file after transcription
    os.remove(audio_file_path)

    return TranscriptionOutput(transcription=transcription, lang=lang)
