import cv2
import pandas as pd
import numpy as np
import torch
import time

path = '/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')

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

# Main program
if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)  # Open default camera (change to your camera index if needed)
    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        exit()

    total_drops = 0
    last_drop_time = None
    drop_count = 0
    total_duration = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

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

        # Display the frame with detected drops
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()