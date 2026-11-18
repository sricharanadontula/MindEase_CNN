
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.image_processing import predict_emotion
from app.services.video_processing import analyze_recorded_video

router = APIRouter(
    prefix="/emotion",
    tags=["Emotion Detection"]
)


# 1. Photo-based facial expression detection
@router.post("/predict")
async def detect_emotion(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        result = predict_emotion(image_bytes)

        return {
            "success": True,
            "message": "Facial expression prediction completed.",
            "result": result
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Emotion detection failed."
        )


# 2. Multimodal emotion analysis from recorded video
@router.post("/video")
async def analyze_video(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid video."
        )

    try:
        video_bytes = await file.read()

        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        result = analyze_recorded_video(video_bytes)

        return {
            "success": True,
            "message": "Video emotion analysis completed.",
            "result": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Video emotion analysis failed."
        )
