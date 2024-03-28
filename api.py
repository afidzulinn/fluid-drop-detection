from fastapi import FastAPI, BackgroundTasks
import cv2
import time
# import numpy as np
import torch
from pathlib import Path
import platform

app = FastAPI()

if platform.system() == "Windows":
    path = Path('yolov5/runs/train/exp2/weights/best.pt').resolve()  # Use resolve() to get the absolute path
else:
    path = Path('yolov5/runs/train/exp2/weights/best.pt')

model = torch.hub.load('Ultralytics/yolov5', 'custom', path=str(path), device='cpu', force_reload=True)

#
last_drop_time = None
total_drops = 0
total_duration = 0

def detect_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.xyxy[0]  # Extract detected objects
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def duration_between_drops(frame, last_drop_time):
    detections = detect_drops(frame)
    if detections is not None and len(detections) > 0:
        current_time = time.time()
        if last_drop_time:
            time_diff = current_time - last_drop_time
            return time_diff, current_time
    return None, last_drop_time

def process_frame(frame):
    global total_drops, last_drop_time, total_duration

    # Count total drops
    drop_count = count_total_drops(frame)
    total_drops += drop_count

    # Calculate duration between drops
    duration, last_drop_time = duration_between_drops(frame, last_drop_time)
    if duration is not None:
        total_duration += duration
        average_duration = total_duration / total_drops
        print("Total drops:", total_drops)
        print("Average duration between drops:", average_duration)

@app.post("/start_detection")
async def start_detection(background_tasks: BackgroundTasks):
    def detect_objects():
        video_capture = cv2.VideoCapture(0)  # Open default camera (change to your camera index if needed)
        if not video_capture.isOpened():
            print("Error: Unable to access the camera.")
            return

        global total_drops, last_drop_time, total_duration
        total_drops = 0
        last_drop_time = None
        total_duration = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            process_frame(frame)
            time.sleep(0.1)  # Adjust sleep time if needed for performance

        video_capture.release()
        cv2.destroyAllWindows()

    background_tasks.add_task(detect_objects)

    return {"message": "Object detection started."}

@app.post("/stop_detection")
async def stop_detection():
    # Implement stopping object detection if needed
    return {"message": "Object detection stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    return {"total_drops": total_drops, "average_duration": total_duration / total_drops if total_drops > 0 else 0}


# if __name__ == "__main__":