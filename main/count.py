import cv2
import time
import torch

path = '/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')
video_path = '../video/infusion_drop.mp4'

def detect_drops(frame):

    results = model(frame)
    detections = results.xyxy[0]
    return detections

def count_total_drops(video_path):
    cap = cv2.VideoCapture(video_path)
    total_drops = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_drops(frame)
        total_drops += len(detections)

    cap.release()
    return total_drops

def duration_between_drops(video_path):
    cap = cv2.VideoCapture(video_path)
    last_drop_time = None
    drop_durations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_drops(frame)

        if detections is not None and len(detections) > 0:
            current_time = time.time()
            if last_drop_time:
                time_diff = current_time - last_drop_time
                drop_durations.append(time_diff)
            last_drop_time = current_time

    cap.release()
    return drop_durations

# Main program
if __name__ == "__main__":

    total_drops = count_total_drops(video_path)
    print("Total drops detected:", total_drops)

    drop_durations = duration_between_drops(video_path)
    if drop_durations:
        average_duration = sum(drop_durations) / len(drop_durations)
        print("Average duration between drops:", average_duration)
    else:
        print("No drops detected in the video.")
