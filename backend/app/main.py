
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import emotion, chat

app = FastAPI(
    title="MindEase API",
    description="AI-powered mental health companion backend",
    version="2.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register emotion detection route
app.include_router(emotion.router)
app.include_router(chat.router)

@app.get("/")
def home():
    return {
        "message": "Welcome to MindEase API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
