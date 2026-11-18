
import whisper
import os

# Load the Whisper model once
MODEL = whisper.load_model("base")


def transcribe_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found.")

    result = MODEL.transcribe(
        audio_path,
        fp16=False
    )

    transcript = result["text"].strip()

    if not transcript:
        raise ValueError("No speech detected in the audio.")

    return transcript
