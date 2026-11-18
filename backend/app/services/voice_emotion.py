
from speechbrain.inference.interfaces import foreign_class

print("Loading voice emotion recognition model...")

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

print("Voice emotion model loaded successfully!")


EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral"
}



def analyze_voice_emotion(audio_path):

    import os

    audio_path = os.path.abspath(audio_path)

    result = classifier.classify_file(
        audio_path.replace("\\", "/")
    )

    probabilities = result[0]
    confidence = result[1]
    predicted_emotion = result[3]

    if isinstance(predicted_emotion, (list, tuple)):
        predicted_emotion = predicted_emotion[0]

    emotion_code = str(predicted_emotion).lower()

    emotion = EMOTION_MAP.get(
        emotion_code,
        emotion_code
    )

    confidence_value = float(
        confidence.squeeze().item()
    ) * 100

    return {
        "emotion": emotion,
        "confidence": round(confidence_value, 2)
    }
