
import React, { useState, useRef, useEffect } from "react";

import {
  analyzeEmotion,
  analyzeRecordedVideo,
  sendChatMessage,
} from "./services/api";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [isSending, setIsSending] = useState(false);

  const [cameraOpen, setCameraOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const [detectedEmotion, setDetectedEmotion] = useState(null);
  const [analysisContext, setAnalysisContext] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const messagesEndRef = useRef(null);

  // Automatically scroll to the latest chat message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isSending]);

  // Release camera and microphone
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOpen(false);
  };

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }

      if (
        mediaRecorderRef.current?.state === "recording"
      ) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  // Open live camera and microphone
  const openCamera = async () => {
    try {
      setCameraError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      streamRef.current = stream;
      setCameraOpen(true);
    } catch (error) {
      console.error("Camera error:", error);

      setCameraError(
        "Unable to access camera or microphone. Please allow permissions."
      );
    }
  };

  // Attach camera stream
  useEffect(() => {
    if (
      cameraOpen &&
      videoRef.current &&
      streamRef.current
    ) {
      videoRef.current.srcObject = streamRef.current;

      videoRef.current.play().catch(() => {});
    }
  }, [cameraOpen]);

  // Capture photo and analyse using CNN V3
  const capturePhoto = async () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      setCameraError("Camera is not ready yet.");
      return;
    }

    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const photoBlob = await new Promise((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.9);
    });

    if (!photoBlob) {
      setCameraError("Unable to capture photo.");
      return;
    }

    const photoUrl = URL.createObjectURL(photoBlob);
    const photoId = Date.now();

    setCameraError("");

    setMessages((previous) => [
      ...previous,
      {
        id: photoId,
        text: "",
        preview: photoUrl,
        fileType: "image",
        sender: "user",
        emotionLoading: true,
        emotionResult: null,
        emotionError: null,
      },
    ]);

    stopCamera();

    try {
      const response = await analyzeEmotion(photoBlob);

      if (!response?.success || !response?.result) {
        throw new Error(response?.message || "Facial expression analysis failed.");
      }

      const result = response.result;

      // Keep the photo analysis as context for Gemini, without displaying
      // the technical prediction in the chat.
      setDetectedEmotion(null);

      const context = {
        facial_analysis: result,
        voice_analysis: null,
        transcript: null,
        linguistic_analysis: null,
      };

      setAnalysisContext(context);

      // Generate a supportive response automatically after photo analysis.
      await generateMediaSupportResponse(context, "photo");
    } catch (error) {
      console.error("Emotion detection error:", error);
      setCameraError(error.message || "Photo analysis failed.");
    }
  };

  // Generate a supportive Gemini response automatically after photo/video analysis
  const generateMediaSupportResponse = async (context, mediaType) => {
    const mediaPrompt = `
The user has just shared a ${mediaType} for emotional support.

Respond warmly and naturally using the available analysis context and,
for a video, the spoken transcript if one is available.

Acknowledge the user's words when a transcript is available. Offer one or
two practical suggestions if appropriate, and ask one gentle follow-up
question. If there is no transcript, gently ask how the user is feeling.

Treat facial-expression, voice-tone, and linguistic predictions as uncertain
AI signals. Do not assume they prove how the user feels. Do not diagnose.
Keep the response concise, simple, and supportive.
`;

    setIsSending(true);

    try {
      const response = await sendChatMessage(
        mediaPrompt,
        chatHistory,
        null,
        context
      );

      if (!response?.success || !response?.reply?.trim()) {
        throw new Error(
          "Gemini did not return a valid response for the uploaded media."
        );
      }

      const aiReply = response.reply.trim();

      // Show only the supportive response in the existing chat UI.
      setMessages((previous) => [
        ...previous,
        {
          id: Date.now(),
          text: aiReply,
          sender: "assistant",
        },
      ]);

      const historyEntry =
        `The user shared a ${mediaType} for emotional support. ` +
        (context.transcript?.trim()
          ? `Spoken transcript: ${context.transcript.trim()}`
          : "No speech transcript was available.");

      setChatHistory((previous) => [
        ...previous,
        { role: "user", content: historyEntry },
        { role: "assistant", content: aiReply },
      ]);
    } catch (error) {
      console.error("Automatic media chatbot response error:", error);

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now(),
          text:
            "I've reviewed the information from your upload. I'm here to listen. " +
            "Would you like to tell me a little more about how you're feeling?",
          sender: "assistant",
          isError: true,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  // Start video recording
  const startRecording = () => {
    if (!streamRef.current) return;

    chunksRef.current = [];

    const formats = [
      "video/webm;codecs=vp8,opus",
      "video/webm;codecs=vp9,opus",
      "video/webm",
    ];

    const supportedFormat = formats.find((format) =>
      MediaRecorder.isTypeSupported(format)
    );

    if (!supportedFormat) {
      setCameraError(
        "Your browser does not support video recording."
      );
      return;
    }

    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: supportedFormat,
    });

    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onerror = (event) => {
      setCameraError(
        "Video recording failed: " +
          (event.error?.message || "Unknown error")
      );

      setRecording(false);
    };

    recorder.onstop = async () => {
      const videoBlob = new Blob(chunksRef.current, {
        type: recorder.mimeType,
      });

      if (videoBlob.size === 0) {
        setCameraError(
          "The recorded video is empty. Please try again."
        );

        stopCamera();
        setRecording(false);
        return;
      }

      const videoUrl = URL.createObjectURL(videoBlob);
      const videoId = Date.now();

      setMessages((previous) => [
        ...previous,
        {
          id: videoId,
          text: "",
          preview: videoUrl,
          fileType: "video",
          sender: "user",
          videoAnalysisLoading: true,
          videoAnalysis: null,
          videoAnalysisError: null,
        },
      ]);

      stopCamera();
      setRecording(false);

      try {
        // Send complete recording to FastAPI
        const response = await analyzeRecordedVideo(videoBlob);

        if (!response.success || !response.result) {
          throw new Error(
            response.message || "Video analysis failed."
          );
        }

        const result = response.result;

        // Keep all multimodal results as context for Gemini.
        // They are intentionally not displayed in the chat UI.
        const context = {
          facial_analysis: result.facial_analysis,
          voice_analysis: result.voice_analysis,
          transcript: result.transcript,
          linguistic_analysis: result.linguistic_analysis,
        };

        setAnalysisContext(context);
        setDetectedEmotion(null);

        // Automatically generate a supportive response after video analysis.
        await generateMediaSupportResponse(context, "video");
      } catch (error) {
        console.error("Video analysis error:", error);
        setCameraError(error.message || "Video analysis failed.");
      }
    };

    recorder.start(1000);
    setRecording(true);
    setCameraError("");
  };

  // Stop video recording
  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;

    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
  };

  // Send message to Gemini
  const handleSend = async () => {
    const userText = message.trim();

    if (!userText || isSending) return;

    const userMessage = {
      id: Date.now(),
      text: userText,
      sender: "user",
    };

    // Display user message immediately
    setMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    setMessage("");
    setIsSending(true);

    try {
      const response = await sendChatMessage(
        userText,
        chatHistory,
        detectedEmotion,
        analysisContext
      );

      console.log("Gemini response:", response);

      if (
        !response?.success ||
        !response?.reply ||
        !response.reply.trim()
      ) {
        throw new Error(
          "Gemini did not return a valid response."
        );
      }

      const aiReply = response.reply.trim();

      const assistantMessage = {
        id: Date.now() + 1,
        text: aiReply,
        sender: "assistant",
      };

      // Display Gemini response
      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);

      // Update conversation history
      setChatHistory((previous) => [
        ...previous,
        {
          role: "user",
          content: userText,
        },
        {
          role: "assistant",
          content: aiReply,
        },
      ]);
    } catch (error) {
      console.error("Gemini chat error:", error);

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          text:
            "I'm sorry, I couldn't generate a response right now. Please try again.",
          sender: "assistant",
          isError: true,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  // Send message using Enter
  const handleKeyDown = (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app">
      <h1>MindEase 2.0</h1>

      <p>AI Facial Expression & Multimodal Chatbot</p>

      <div className="chat-container">
        <div className="chat-messages">

          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Hello! 👋</h2>

              <p>
                Welcome to MindEase. I'm here to listen.
                How are you feeling today?
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.sender}`}
            >
              {msg.text && (
                <p>{msg.text}</p>
              )}

              {/* Show the uploaded photo only; keep analysis hidden. */}
              {msg.fileType === "image" && (
                <img
                  src={msg.preview}
                  alt="Uploaded photo"
                  className="uploaded-image"
                />
              )}

              {/* Show the recorded video only; keep analysis hidden. */}
              {msg.fileType === "video" && (
                <video
                  src={msg.preview}
                  controls
                  className="uploaded-video"
                />
              )}
            </div>
          ))}

          {isSending && (
            <div className="message assistant">
              <p>MindEase is thinking...</p>
            </div>
          )}

          {/* Reference for automatic scrolling */}
          <div ref={messagesEndRef} />
        </div>

        {cameraError && (
          <p className="camera-error">
            {cameraError}
          </p>
        )}

        {cameraOpen && (
          <div className="camera-preview">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="camera-video"
            />

            <div className="camera-controls">
              {!recording && (
                <>
                  <button onClick={capturePhoto}>
                    📸 Capture Photo
                  </button>

                  <button onClick={startRecording}>
                    🔴 Start Recording
                  </button>
                </>
              )}

              {recording && (
                <button onClick={stopRecording}>
                  ⏹ Stop Recording
                </button>
              )}

              {!recording && (
                <button onClick={stopCamera}>
                  ✕ Close Camera
                </button>
              )}
            </div>
          </div>
        )}

        <div className="chat-input">
          <button
            className="upload-button"
            onClick={openCamera}
            disabled={cameraOpen || isSending}
            title="Open camera"
          >
            📷
          </button>

          <input
            type="text"
            placeholder="Type your message..."
            value={message}
            onChange={(event) =>
              setMessage(event.target.value)
            }
            onKeyDown={handleKeyDown}
            disabled={isSending}
          />

          <button
            onClick={handleSend}
            disabled={isSending || !message.trim()}
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
