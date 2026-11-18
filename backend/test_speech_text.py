
from app.services.speech_to_text import transcribe_audio
from app.services.speech_text_analysis import analyze_speech_text

audio_path = "test_audio.wav"

# Step 1: Convert audio into text
transcript = transcribe_audio(audio_path)

print("\nTranscribed Speech:")
print(transcript)

# Step 2: Analyse the meaning of the spoken words
linguistic_result = analyze_speech_text(transcript)

print("\nLinguistic Emotion Result:")
print(linguistic_result)
