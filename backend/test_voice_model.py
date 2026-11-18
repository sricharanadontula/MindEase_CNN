
from speechbrain.inference.interfaces import foreign_class

print("Loading pretrained voice emotion model...")

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

print("Model loaded successfully!")

print("Testing sample audio...")

audio_path = "test_audio.wav"

result = classifier.classify_file(audio_path)

print("\nVoice Emotion Result:")
print("Probabilities:", result[0])
print("Confidence:", result[1])
print("Emotion:", result[3])
