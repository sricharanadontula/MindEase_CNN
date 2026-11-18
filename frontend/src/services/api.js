
const API_BASE_URL = "http://127.0.0.1:8000";

// CNN V3 facial expression detection (photo)
export const analyzeEmotion = async (imageBlob) => {
  const formData = new FormData();
  formData.append("file", imageBlob, "captured_image.jpg");

  const response = await fetch(`${API_BASE_URL}/emotion/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Emotion detection failed.");
  }

  return await response.json();
};

// Complete recorded-video analysis
export const analyzeRecordedVideo = async (videoBlob) => {
  const formData = new FormData();
  formData.append("file", videoBlob, "recording.webm");

  const response = await fetch(`${API_BASE_URL}/emotion/video`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Video analysis failed.");
  }

  return await response.json();
};

// Gemini AI chatbot with multimodal context
export const sendChatMessage = async (
  message,
  history = [],
  detectedEmotion = null,
  analysisContext = null
) => {
  const response = await fetch(`${API_BASE_URL}/chat/message`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
      detected_emotion: detectedEmotion,
      analysis_context: analysisContext,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Unable to generate AI response.");
  }

  return await response.json();
};
