import cv2
import time
import torch

path = '/home/van/Detta/infusion/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')

def detect_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.xyxy[0]  # Extract detected objects
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

def draw_stats_on_frame(frame, total_drops, average_duration):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)
    font_color1 = (255, 255, 255)
    line_type = 2

    text_total_drops = f"Total drops: {total_drops}"
    text_average_duration = f"Average duration: {average_duration:.2f} seconds"

    cv2.putText(frame, text_total_drops, (20, 40), font, font_scale, font_color, line_type)
    cv2.putText(frame, text_average_duration, (20, 80), font, font_scale, font_color1, line_type)

    return frame

# Main program
if __name__ == "__main__":
    video_path = 'video/infus.mp4'

    total_drops = count_total_drops(video_path)
    drop_durations = duration_between_drops(video_path)

    if drop_durations:
        average_duration = sum(drop_durations) / len(drop_durations)
    else:
        average_duration = 0

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = 'Results/output.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_stats = draw_stats_on_frame(frame, total_drops, average_duration)
        out.write(frame_with_stats)

    cap.release()
    out.release()

    print(f"Video with stats saved to: {output_path}")
