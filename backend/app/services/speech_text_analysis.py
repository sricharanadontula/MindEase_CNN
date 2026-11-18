
import os
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables from backend/.env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)


def analyze_speech_text(transcript):

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    prompt = f"""
    Analyse the emotional meaning of the following spoken text.

    Identify the primary emotion expressed through the words,
    not the speaker's voice or facial expression.

    Return only valid JSON with this structure:

    {{
        "emotion": "sad",
        "confidence": 85,
        "explanation": "The speaker expresses feeling low."
    }}

    Possible emotions:
    happy, sad, angry, neutral, fear, surprise, frustrated.

    Spoken text:
    {transcript}
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    response_text = response.text.strip()

    # Remove Markdown code fences
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "", 1)
        response_text = response_text.replace("```", "").strip()

    # Convert JSON text into a Python dictionary
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemini returned invalid JSON: {response_text}"
        ) from error

    return result
