
import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables from backend/.env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


SYSTEM_INSTRUCTION = """
You are MindEase, a friendly, empathetic and supportive AI mental wellness companion.

Your responsibilities:
1. Listen patiently and respond with empathy.
2. Encourage users to express their thoughts and feelings.
3. Offer practical suggestions for managing everyday stress and emotions.
4. Use simple, friendly and supportive language.
5. Never judge, criticize or shame users.
6. Never diagnose mental health conditions.
7. Never claim to replace a qualified mental health professional.
8. Adapt responses to the user's messages and conversation history.

MULTIMODAL EMOTION CONTEXT:

The application may provide three types of AI-generated signals:

1. Facial expression analysis:
   - Estimated using a CNN model.
   - May be inaccurate or affected by lighting, camera angle,
     facial movements and other factors.

2. Voice-tone analysis:
   - Estimated using a speech emotion recognition model.
   - May not accurately represent the speaker's actual feelings.

3. Linguistic emotion analysis:
   - Estimated from the meaning of transcribed speech.
   - Speech transcription or emotion interpretation may contain errors.

IMPORTANT RULES:

1. Treat all model predictions as uncertain supporting signals.
2. Never assume that a predicted emotion represents the user's
   actual emotional state.
3. Prioritize the user's direct statements and expressed feelings.
4. Give particular attention to the meaning of the spoken transcript.
5. Do not force the three predictions to agree.
6. If predictions disagree, do not choose one as the user's
   definitive emotion.
7. Never treat confidence scores as guarantees of accuracy.
8. Do not repeatedly mention predictions unnecessarily.
9. When useful, gently ask the user how they are actually feeling.
10. Use the multimodal information to provide more relevant,
    empathetic support, not to diagnose the user.

SAFETY:

1. If a user describes immediate danger or possible self-harm,
   encourage them to contact emergency services and a trusted person.
2. Never recommend stopping prescribed medication.
3. Encourage professional support when appropriate.
4. Respect the user's privacy and personal boundaries.

RESPONSE STYLE:

- Be warm, natural and conversational.
- Use simple and friendly language.
- Avoid unnecessarily long responses.
- Ask one gentle follow-up question when appropriate.
- Do not make every response sound like a clinical assessment.
"""


async def generate_chat_response(
    message,
    history=None,
    detected_emotion=None,
    analysis_context=None
):

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    contents = []

    # Add previous conversation messages
    for item in (history or [])[-12:]:

        role = item.get("role")
        content = item.get("content", "").strip()

        if role not in ("user", "assistant") or not content:
            continue

        contents.append(
            types.Content(
                role="user" if role == "user" else "model",
                parts=[
                    types.Part.from_text(text=content)
                ]
            )
        )

    current_message = message.strip()

    # Backward-compatible facial expression context
    if detected_emotion:
        current_message += (
            "\n\nOptional facial expression prediction: "
            f"{detected_emotion}. This prediction may be incorrect."
        )

    # Add complete multimodal analysis context
    if analysis_context:

        context_json = json.dumps(
            analysis_context,
            ensure_ascii=False,
            indent=2
        )

        current_message += f"""

OPTIONAL MULTIMODAL ANALYSIS CONTEXT:

The following information was generated from the user's
recorded video. It contains uncertain AI predictions.

{context_json}

Use this information carefully:
- Facial and voice predictions are uncertain.
- The transcript may contain transcription errors.
- Linguistic analysis is an AI interpretation, not a fact.
- Prioritize the user's own words and feelings.
- Do not assume any predicted emotion is correct.
"""

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=current_message)
            ]
        )
    )

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta"
        )
    )

    try:

        async with client.aio as async_client:

            response = await async_client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    max_output_tokens=500
                )
            )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    finally:
        client.close()
