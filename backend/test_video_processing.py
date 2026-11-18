from app.services.video_processing import analyze_recorded_video

print("Starting video analysis...")

with open("video.webm.mp4", "rb") as video_file:
    video_bytes = video_file.read()

result = analyze_recorded_video(video_bytes)

print("\nVIDEO ANALYSIS RESULT:")
print(result)