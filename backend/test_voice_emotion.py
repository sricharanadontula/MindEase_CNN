
from app.services.voice_emotion import analyze_voice_emotion

audio_path = "test_audio.wav"

print("\nTesting voice-tone emotion analysis...")

result = analyze_voice_emotion(audio_path)

print("\nVoice Emotion Result:")
print(result)
