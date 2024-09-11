import os

from dotenv import load_dotenv

from app.models.llm import LLMResponder
from app.models.tts import WhisperTranscriber

load_dotenv()


whisper_transcriber = WhisperTranscriber(model_size="large-v3",
                                         device="cuda",
                                         compute_type="float32")

llm_responder = LLMResponder(api_key=os.environ.get("OPENAI_API_KEY"))
