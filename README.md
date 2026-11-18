# MindEase 2.0 – AI-Powered Mental Wellness Companion

## 1. Project Overview

**MindEase 2.0** is an AI-powered mental wellness companion designed to provide supportive and empathetic interactions using facial expression recognition, voice emotion analysis, speech transcription, and a Generative AI chatbot.

The application analyses facial expressions from captured images and combines facial, voice-tone, and linguistic signals from recorded videos to provide contextual support through a Gemini-powered chatbot.

MindEase aims to encourage users to express their feelings and receive supportive guidance through a simple conversational interface.

## 2. Key Features

- **Facial Expression Recognition:** Uses a custom-trained CNN V3 model to classify facial expressions from captured images.
- **Video-Based Multimodal Analysis:** Analyses facial expressions, voice tone, and spoken words from a single video recording.
- **Voice Emotion Recognition:** Uses SpeechBrain to estimate emotion from voice audio.
- **Speech-to-Text:** Uses Whisper to convert spoken words into text.
- **Linguistic Emotion Analysis:** Uses Gemini to interpret emotional signals from the transcribed speech.
- **AI Mental Wellness Chatbot:** Uses Google Gemini to generate supportive and empathetic responses.
- **Automatic Video Response:** Generates a chatbot response automatically after successful video analysis.
- **Conversation History:** Maintains conversational context for more relevant responses.

## 3. Technologies Used

### Frontend
- React.js
- Vite
- JavaScript
- HTML
- CSS

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning and AI
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- OpenCV
- NumPy
- SpeechBrain
- Whisper
- Google Gemini API

### Tools
- Visual Studio Code
- FFmpeg
- Git and GitHub

## 4. System Architecture

MindEase 2.0 follows a frontend-backend architecture.

1. The React frontend allows users to capture photos, record videos, and interact with the chatbot.
2. The FastAPI backend receives uploaded media.
3. The CNN V3 model analyses facial expressions.
4. For recorded videos, audio is extracted using FFmpeg.
5. SpeechBrain performs voice-tone emotion analysis.
6. Whisper generates a speech transcript.
7. Gemini performs linguistic emotion analysis.
8. The multimodal analysis results are passed to the Gemini chatbot.
9. Gemini generates a supportive response that is displayed in the chat interface.

## 5. Multimodal Emotion Analysis Workflow

The video analysis pipeline combines three sources of information:

| Analysis | Technology | Purpose |
|---|---|---|
| Facial expression | CNN V3 | Estimates facial expression |
| Voice emotion | SpeechBrain | Estimates emotion from voice tone |
| Speech transcription | Whisper | Converts speech into text |
| Linguistic emotion | Gemini | Interprets emotional signals in spoken language |
| Supportive response | Gemini | Generates contextual chatbot responses |

The analysis results are processed in the backend and passed to the chatbot as context. Detailed technical predictions are hidden from the normal chat interface.

**Note:** These AI-generated predictions are estimates and may be affected by recording quality, lighting, background noise, and model limitations. They are not a diagnosis or a definitive measurement of a person's emotional state.

## 6. Machine Learning Model

### CNN V3 – Facial Expression Recognition

The project uses a custom-trained CNN V3 model for facial expression classification.

**Model details:**
- Model type: Convolutional Neural Network (CNN)
- Input image size: 224 × 224 × 3
- Number of emotion classes: 5
- Framework: TensorFlow / Keras
- Model file: `mindease_emotion_model_v3.keras`

**Supported classes:**
1. Angry
2. Happy
3. Neutral
4. Sad
5. Surprise

**Model evaluation:**
- Test accuracy: 64.92%
- Test samples: 6,043

The model is intended for facial expression estimation and should not be interpreted as a reliable assessment of a person's actual emotional or mental health condition.

## 7. Project Structure

```text
MindEase_2.0/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── ExpressionResult.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   └── emotion.py
│   │   ├── services/
│   │   │   ├── chat_service.py
│   │   │   ├── face_detection.py
│   │   │   ├── image_processing.py
│   │   │   ├── video_processing.py
│   │   │   ├── speech_to_text.py
│   │   │   ├── speech_text_analysis.py
│   │   │   └── voice_emotion.py
│   │   └── main.py
│   │
│   ├── ml/
│   │   └── saved_models/
│   │       └── mindease_emotion_model_v3.keras
│   │
│   ├── requirements.txt
│   └── .env.example
│
├── README.md
└── .gitignore
```

*The structure above highlights the main project files. Include any additional modules, assets, or configuration files required by your actual implementation.*

## 8. Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.11 or a compatible version supported by the dependencies.
- Node.js and npm.
- FFmpeg.
- Google Gemini API key.

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
cd MindEase_2.0
```

### Step 2: Backend Setup

Navigate to the backend folder:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file inside the backend folder.

Add your Gemini API configuration:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Replace the placeholder with your own API key.

Never upload your actual `.env` file or API key to GitHub.

### Step 4: Install FFmpeg

Install FFmpeg and ensure that its executable is accessible through your system PATH.

Verify installation:

```bash
ffmpeg -version
```

### Step 5: Run the Backend

From the backend directory:

```bash
uvicorn app.main:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 6: Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm run dev
```

Open the local URL displayed by Vite, usually:

```text
http://localhost:5173
```

## 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/emotion/predict` | Facial expression prediction from an image |
| POST | `/emotion/video` | Multimodal analysis of a recorded video |
| POST | `/chat/message` | Generates a Gemini chatbot response |

## 10. Application Workflow

### Photo Analysis
1. Open the camera.
2. Capture a photo.
3. The backend processes the image using CNN V3.
4. The result can be used as context for the AI companion.

### Video Analysis and Automatic Chatbot Response
1. Open the camera and microphone.
2. Record a video containing facial expressions and speech.
3. The video is uploaded to the FastAPI backend.
4. Facial, voice-tone, and linguistic analyses are performed.
5. The analysis context is passed to Gemini.
6. Gemini automatically generates a supportive response.
7. The response appears in the existing chat interface without displaying the detailed technical analysis.

### Text Chat
1. Enter a message in the chat interface.
2. The message is sent to the backend.
3. Gemini generates a response using the conversation history and available analysis context.
4. The response is displayed in the chat.

## 11. Privacy and Responsible Use

- Camera and microphone access require user permission.
- AI predictions are uncertain and should not be treated as facts.
- MindEase is not a medical or psychological diagnostic system.
- The chatbot is intended for general emotional support and does not replace qualified mental health professionals.
- API keys and other credentials should be stored securely in environment variables.

## 12. Future Enhancements

- Improve facial expression classification accuracy.
- Enhance voice emotion recognition.
- Improve multimodal context integration.
- Add multilingual speech support.
- Improve chatbot personalization.
- Introduce secure user authentication and privacy controls.
- Explore additional mental wellness support features.

## 13. Conclusion

MindEase 2.0 demonstrates the integration of machine learning, speech processing, computer vision, and Generative AI into a multimodal mental wellness companion.

By combining facial expression estimation, voice-tone analysis, speech transcription, and conversational AI, the project aims to provide a more contextual and supportive user interaction experience.

---

**Project:** MindEase 2.0  
**Category:** Artificial Intelligence | Machine Learning | Generative AI | Mental Wellness  
**Frontend:** React.js  
**Backend:** FastAPI  
**AI Model:** CNN V3 + SpeechBrain + Whisper + Google Gemini
