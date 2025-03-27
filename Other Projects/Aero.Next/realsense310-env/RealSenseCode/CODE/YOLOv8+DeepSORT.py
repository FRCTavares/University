import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os
import torch
import json
from datetime import datetime

# ------------- Configuration -------------
DETECTION_CONFIDENCE = 0.5  # Minimum confidence threshold for detections
PERSON_CLASS_ID = 0         # YOLO class ID for "person"

# Model options (adjust as needed)
MODELS = {
    'nano': 'yolov8n.pt',    # Fastest model (nano)
    'small': 'yolov8s.pt',   # Balanced model
    'medium': 'yolov8m.pt',  # Higher accuracy
}
current_model = 'nano'

# Determine device (GPU if available; on Raspberry Pi you'll likely use CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODELS[current_model]).to(device)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# ------------- RealSense Setup -------------
pipeline = rs.pipeline()
config = rs.config()

# Use lower resolution and framerate for better performance on embedded devices
width, height = 640, 480
fps = 15
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

pipeline.start(config)

# Align depth frame to color frame
align = rs.align(rs.stream.color)

# ------------- Miscellaneous Setup -------------
frame_count = 0
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

# Recording variables
recording = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# For smoothing and history
depth_history = {}  # Depth values per track
box_history = {}    # Bounding box history per track

# Target tracking variables
target_id = None     # Single target to follow
target_ids = set()   # Multi-target mode (tracking up to max_targets)
max_targets = 3

# Debug and performance monitoring
debug_mode = False
process_times = []

# For trajectory recording (if needed)
trajectory_data = {}
trajectory_recording = False

# ------------- Main Loop -------------
try:
    while True:
        start_time = time.time()
        frame_count += 1
        
        # Process every 2nd frame to reduce load
        if frame_count % 2 != 0:
            continue

        # Get frames from RealSense and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Optional: Measure detection time in debug mode
        if debug_mode:
            detection_start = time.time()

        # YOLOv8 detection on the RGB frame
        results = model.predict(frame, verbose=False, device=device)[0]

        if debug_mode:
            detection_time = time.time() - detection_start
            tracking_start = time.time()

        # Prepare detections for DeepSORT (format: ([x, y, w, h], confidence, class))
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == PERSON_CLASS_ID and conf > DETECTION_CONFIDENCE:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Update DeepSORT tracker with current detections
        tracks = tracker.update_tracks(detections, frame=frame)

        if debug_mode:
            tracking_time = time.time() - tracking_start
            draw_start = time.time()

        # Clean depth and box history (remove old track entries)
        current_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        depth_history = {k: v for k, v in depth_history.items() if k in current_track_ids}
        box_history = {k: v for k, v in box_history.items() if k in current_track_ids}

        # Process each confirmed track
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Smooth bounding box coordinates using previous values
            if track_id in box_history:
                smooth_factor = 0.7  # Adjust smoothing as desired
                prev_l, prev_t, prev_r, prev_b = box_history[track_id]
                l = int(smooth_factor * prev_l + (1 - smooth_factor) * l)
                t = int(smooth_factor * prev_t + (1 - smooth_factor) * t)
                r = int(smooth_factor * prev_r + (1 - smooth_factor) * r)
                b = int(smooth_factor * prev_b + (1 - smooth_factor) * b)

            box_history[track_id] = (l, t, r, b)

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            
            # Determine center of bounding box
            center_x = (l + r) // 2
            center_y = (t + b) // 2

            # Get and smooth depth at the center (in meters)
            if depth_frame and track_id in depth_history:
                current_depth = depth_frame.get_distance(center_x, center_y)
                if current_depth > 0:
                    alpha = 0.3  # Smoothing factor
                    smoothed_depth = alpha * current_depth + (1 - alpha) * depth_history[track_id]
                    depth_history[track_id] = smoothed_depth
                else:
                    smoothed_depth = depth_history[track_id]
            elif depth_frame:
                smoothed_depth = depth_frame.get_distance(center_x, center_y)
                depth_history[track_id] = smoothed_depth

            # Display depth info
            if depth_frame:
                try:
                    depth = depth_frame.get_distance(center_x, center_y)
                    if 0 < depth < 10:  # Only valid depths within 10m
                        cv2.putText(frame, f'ID: {track_id} {depth:.2f}m', (l, t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f'ID: {track_id} (invalid depth)', (l, t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frame, f'ID: {track_id}', (l, t - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'ID: {track_id}', (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Highlight target(s) differently if applicable
            if track_id == target_id:
                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 3)
            elif track_id in target_ids:
                cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 3)

        if debug_mode:
            draw_time = time.time() - draw_start
            process_times.append({
                'detection': detection_time,
                'tracking': tracking_time,
                'drawing': draw_time,
                'total': time.time() - start_time
            })
            if len(process_times) > 10:
                avg_times = {k: sum(p[k] for p in process_times[-10:]) / 10 for k in process_times[0]}
                cv2.putText(frame, f"Det: {avg_times['detection']*1000:.1f}ms", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Track: {avg_times['tracking']*1000:.1f}ms", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Recording video if toggled
        if recording and out is not None:
            out.write(frame)

        # Calculate and display FPS
        fps_val = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps_val:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + DeepSORT + RealSense", frame)

        # Key event handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                frame_height, frame_width = frame.shape[:2]
                out = cv2.VideoWriter(f"{output_dir}/recording_{timestamp}.avi", fourcc, fps, (frame_width, frame_height))
                recording = True
            else:
                out.release()
                recording = False
        elif key == ord('t'):
            if tracks:
                closest_track = min(
                    [t for t in tracks if t.is_confirmed()],
                    key=lambda t: depth_history.get(t.track_id, float('inf'))
                )
                target_id = closest_track.track_id
                print(f"Now tracking target ID: {target_id}")
        elif key == ord('c'):
            target_id = None
            print("Cleared target tracking")
        elif key == ord('m'):
            if tracks and len(target_ids) < max_targets:
                untracked = [t for t in tracks if t.is_confirmed() and t.track_id not in target_ids]
                if untracked:
                    closest = min(untracked, key=lambda t: depth_history.get(t.track_id, float('inf')))
                    target_ids.add(closest.track_id)
                    print(f"Added target ID: {closest.track_id}, now tracking {len(target_ids)} targets")
        elif key == ord('1'):
            current_model = 'nano'
            model = YOLO(MODELS[current_model]).to(device)
            print(f"Switched to {current_model} model")
        elif key == ord('2'):
            current_model = 'small'
            model = YOLO(MODELS[current_model]).to(device)
            print(f"Switched to {current_model} model")
        elif key == ord('d'):
            debug_mode = not debug_mode
            process_times = []
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('j'):
            trajectory_recording = not trajectory_recording
            if not trajectory_recording and trajectory_data:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(f"{output_dir}/trajectory_{timestamp}.json", 'w') as f:
                    json.dump(trajectory_data, f)
                trajectory_data = {}
                print("Saved trajectory data")
            else:
                print("Started trajectory recording")

        # Record trajectory data if enabled
        if trajectory_recording:
            frame_timestamp = time.time()
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    center_x, center_y = (l + r) // 2, (t + b) // 2
                    depth = depth_history.get(track_id, None)
                    if track_id not in trajectory_data:
                        trajectory_data[track_id] = []
                    trajectory_data[track_id].append({
                        'timestamp': frame_timestamp,
                        'x': center_x,
                        'y': center_y,
                        'depth': depth,
                        'box': [l, t, r, b]
                    })
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
