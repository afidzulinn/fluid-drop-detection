import cv2
import time
import torch
import pandas as pd

path = 'D:/infusion/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='cpu')


def detect_infusion_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Extract detected objects
    return detections


def count_drops_in_one_minute():
    start_time = time.time()
    drop_count = 0

    while time.time() - start_time < 60:
        frame = capture_frame()  # Capture a frame from the camera
        detections = detect_infusion_drops(frame)
        drop_count += len(detections)

    return drop_count

def time_between_consecutive_drops():
    last_drop_time = None

    while True:
        frame = capture_frame()  # Capture a frame from the camera
        detections = detect_infusion_drops(frame)

        if detections:
            current_time = time.time()
            if last_drop_time:
                time_diff = current_time - last_drop_time
                return time_diff
            last_drop_time = current_time

# def capture_frame():
#    cap = cv2.VideoCapture(0)
#    ret, frame = cap.read()
#    cap.release()
#    return frame

def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    ret, frame = cap.read()
    cap.release()
    return frame


# Main program
if __name__ == "__main__":
    # Example usage:
    drops_in_one_minute = count_drops_in_one_minute()
    print("Drops in one minute:", drops_in_one_minute)

    time_between_drops = time_between_consecutive_drops()
    print("Time between consecutive drops:", time_between_drops)
