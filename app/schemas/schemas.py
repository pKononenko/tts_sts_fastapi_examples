from pydantic import BaseModel


class TranscriptionOutput(BaseModel):
    transcription: str
    lang: str
