
from faster_whisper import WhisperModel


class WhisperTranscriber:

    def __init__(self,
                 model_size: str = "medium",
                 device: str = "cpu",
                 compute_type: str = "int8"):
        """
        Initializes the Whisper model with the specified parameters.

        Parameters:
        - model_size: Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large',
            'large-v2', 'large-v3')
        - device: Device to run the model on ('cpu' or 'cuda')
        - compute_type: Type of computation ('int8', 'int16', 'float32')
        """
        self.model = WhisperModel(model_size,
                                  device=device,
                                  compute_type=compute_type)

    def transcribe(self, audio_file_path: str) -> tuple[str, str]:
        """
        Transcribes the given audio file and returns the transcription as a string.

        Parameters:
        - audio_file_path (str): Path to the audio file to be transcribed.

        Returns:
        - transcription (str): The transcription of the audio file as a string.
        """
        segments, info = self.model.transcribe(audio_file_path)
        transcription = " ".join([segment.text for segment in segments])
        return transcription, info.language
