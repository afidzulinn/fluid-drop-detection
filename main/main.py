import cv2
import time
import torch

path = '/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')
video_path = '../video/infusion_drop.mp4'

def detect_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.xyxy[0]
    return detections

def count_drops_in_one_minute(video_path):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    drop_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_drops(frame)
        drop_count += len(detections)

        # Check if one minute has elapsed
        if time.time() - start_time >= 60:
            break

    cap.release()
    return drop_count

def time_between_consecutive_drops(video_path):
    cap = cv2.VideoCapture(video_path)
    last_drop_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_drops(frame)

        if detections is not None and len(detections) > 0:
            current_time = time.time()
            if last_drop_time:
                time_diff = current_time - last_drop_time
                return time_diff
            last_drop_time = current_time

    cap.release()
    return None

# Main program
if __name__ == "__main__":

    drops_in_one_minute = count_drops_in_one_minute(video_path)
    print("Drops in one minute:", drops_in_one_minute)

    time_between_drops = time_between_consecutive_drops(video_path)
    print("Time between consecutive drops:", time_between_drops)
