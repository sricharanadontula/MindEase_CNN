
import cv2
import tempfile
import os
import subprocess
import numpy as np

from pathlib import Path

from app.services.image_processing import predict_emotion
from app.services.speech_to_text import transcribe_audio
from app.services.speech_text_analysis import analyze_speech_text
from app.services.voice_emotion import analyze_voice_emotion


FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"


# Extract audio from recorded video
def extract_audio(video_path, audio_path):

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_path
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True
    )


def get_video_duration(video_path):

    import json

    command = [
        FFPROBE_PATH,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=duration,nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    # First try the container duration.
    format_duration = data.get("format", {}).get("duration")

    if format_duration and format_duration != "N/A":
        duration = float(format_duration)

        if duration > 0:
            return duration

    # If container duration is unavailable, calculate
    # duration from the video frame count and frame rate.
    streams = data.get("streams", [])

    if not streams:
        raise ValueError("No readable video stream found.")

    stream = streams[0]

    frame_count = stream.get("nb_read_frames")

    if not frame_count or frame_count == "N/A":
        frame_count = stream.get("nb_frames")

    frame_rate = (
        stream.get("avg_frame_rate")
        or stream.get("r_frame_rate")
    )

    if frame_count and frame_rate and frame_rate != "0/0":
        numerator, denominator = map(int, frame_rate.split("/"))

        if denominator > 0 and numerator > 0:
            duration = int(frame_count) * denominator / numerator

            if duration > 0:
                return duration

    raise ValueError("Unable to determine video duration.")




# Extract and analyse video frames using FFmpeg

def analyze_video_frames(video_path, frame_count=5):

    predictions = []

    with tempfile.TemporaryDirectory() as frame_dir:

        output_pattern = os.path.join(
            frame_dir,
            "frame_%04d.jpg"
        )

        # Extract frames directly without needing video duration.
        command = [
            FFMPEG_PATH,
            "-y",
            "-i", video_path,
            "-vf", "fps=2",
            "-q:v", "2",
            output_pattern
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise ValueError(
                "FFmpeg could not extract video frames: "
                + result.stderr[-1000:]
            )

        # Collect extracted frames.
        frame_files = sorted([
            os.path.join(frame_dir, filename)
            for filename in os.listdir(frame_dir)
            if filename.lower().endswith(".jpg")
        ])

        if not frame_files:
            raise ValueError(
                "No readable frames could be extracted from the video."
            )

        # Select up to five frames evenly.
        selected_indices = np.linspace(
            0,
            len(frame_files) - 1,
            min(frame_count, len(frame_files)),
            dtype=int
        )

        for index in selected_indices:

            with open(frame_files[index], "rb") as image_file:
                image_bytes = image_file.read()

            result = predict_emotion(image_bytes)
            predictions.append(result)

    if not predictions:
        raise ValueError("Unable to analyse video frames.")

    # Count predicted expressions.
    emotion_counts = {}

    for result in predictions:
        emotion = result["emotion"]

        emotion_counts[emotion] = (
            emotion_counts.get(emotion, 0) + 1
        )

    dominant_emotion = max(
        emotion_counts,
        key=emotion_counts.get
    )

    average_confidence = np.mean([
        result["confidence"]
        for result in predictions
    ])

    return {
        "dominant_emotion": dominant_emotion,
        "average_confidence": round(
            float(average_confidence), 2
        ),
        "frames_analysed": len(predictions),
        "frame_results": predictions
    }



# Complete multimodal video analysis
def analyze_recorded_video(video_bytes):

    with tempfile.TemporaryDirectory() as temp_dir:

        video_path = os.path.join(
            temp_dir,
            "recording.webm"
        )

        audio_path = os.path.join(
            temp_dir,
            "recording.wav"
        )

        # Save uploaded video
        with open(video_path, "wb") as video_file:
            video_file.write(video_bytes)

        # 1. Facial expression analysis
        facial_result = analyze_video_frames(video_path)

        # 2. Extract audio
        extract_audio(video_path, audio_path)

        # 3. Voice-tone emotion analysis
        voice_result = analyze_voice_emotion(audio_path)

        # 4. Speech transcription
        transcript = transcribe_audio(audio_path)

        # 5. Linguistic emotion analysis
        linguistic_result = analyze_speech_text(transcript)

        return {
            "facial_analysis": facial_result,
            "voice_analysis": voice_result,
            "transcript": transcript,
            "linguistic_analysis": linguistic_result
        }
