import numpy as np
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.optimize import linear_sum_assignment

def lerp_box(box_start: list, box_end: list, t: float) -> list:
    """
    Linearly interpolate box coordinates (x1, y1, x2, y2).
    """

    x1 = int(box_start[0] + (box_end[0] - box_start[0]) * t)
    y1 = int(box_start[1] + (box_end[1] - box_start[1]) * t)
    x2 = int(box_start[2] + (box_end[2] - box_start[2]) * t)
    y2 = int(box_start[3] + (box_end[3] - box_start[3]) * t)
    return [x1, y1, x2, y2]

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute the IoU matrix between two sets of boxes.
    """

    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((len(boxes_a), len(boxes_b)))

    inter_ul = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    inter_lr = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter_wh = np.clip(inter_lr - inter_ul, 0, None)
    
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = inter_area / np.clip(union_area, 1e-6, None)
    return iou


def process_video(video_path: Path, output_path: Path, model_name: str, 
                  conf_threshold: float, frame_skip: int, iou_threshold: float,
                  max_age_frames: int):
    """
    Two-pass detection and tracking with LERP and persistent IDs with "memory" for occlusions
    """
    
    # Pass 1: Keyframe collection
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: couldn't open the file {video_path}")
        return

    print("Pass 1: Keyframe detection...")
    keyframe_detections = {} # {frame_num: (boxes_np_array, confs_np_array)}
    total_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if total_frames % frame_skip == 0:
            results = model(frame, classes=[0], conf=conf_threshold, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            keyframe_detections[total_frames] = (boxes, confs)
        
        total_frames += 1

    print(f"{len(keyframe_detections)} keyframes detected.")
    cap.release()

    # Pass 2: Interpolation and video writing
    print("Pass 2: Interpolation and video writing...")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Настройка VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    next_track_id = 0
    
    # {track_id: {"box": [], "conf": float, "last_seen_frame": int}}
    active_tracks = {}

    # {track_id: {"box_start": [], "box_end": [], "conf": float, 
    #             "frame_start_lerp": int, "frame_end_lerp": int}}
    interpolated_tracks = {}
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= total_frames:
            break

        # Tracker updates (on keyframes)
        if frame_count in keyframe_detections:
            current_keyframe = frame_count
            current_boxes, current_confs = keyframe_detections[current_keyframe]
            
            # 1. CLeaning interpolated tracks
            interpolated_tracks.clear()

            # 2. Removing "dead" tracks that are "too old"
            dead_tracks = []
            for track_id, data in active_tracks.items():
                if (current_keyframe - data["last_seen_frame"]) > max_age_frames:
                    dead_tracks.append(track_id)
            for track_id in dead_tracks:
                del active_tracks[track_id]

            # 3. Preparing for matching
            prev_tracks_list = list(active_tracks.items())
            prev_boxes = np.array([data["box"] for _, data in prev_tracks_list])

            # 4. Matching "live" tracks (prev_boxes) with new detections (current_boxes)
            cost_matrix = 1.0 - iou_matrix(prev_boxes, current_boxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_prev_indices = set()
            matched_current_indices = set()
            new_active_tracks = active_tracks.copy()
            
            # 5. Processing matched trackings (Re-ID)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > (1.0 - iou_threshold):
                    continue
                    
                track_id, start_data = prev_tracks_list[r]
                
                interpolated_tracks[track_id] = {
                    "box_start": start_data["box"],
                    "box_end": current_boxes[c],
                    "conf": current_confs[c],
                    "frame_start_lerp": start_data["last_seen_frame"],
                    "frame_end_lerp": current_keyframe
                }
                
                new_active_tracks[track_id] = {
                    "box": current_boxes[c],
                    "conf": current_confs[c],
                    "last_seen_frame": current_keyframe
                }
                
                matched_prev_indices.add(r)
                matched_current_indices.add(c)

            for c in range(len(current_boxes)):
                if c not in matched_current_indices:
                    track_id = next_track_id
                    next_track_id += 1
                    box_new = current_boxes[c]
                    conf_new = current_confs[c]
                    
                    interpolated_tracks[track_id] = {
                        "box_start": box_new, "box_end": box_new, "conf": conf_new,
                        "frame_start_lerp": current_keyframe, 
                        "frame_end_lerp": current_keyframe
                    }
                    new_active_tracks[track_id] = {
                        "box": box_new, "conf": conf_new, "last_seen_frame": current_keyframe
                    }

            active_tracks = new_active_tracks
            
        # Draw interpolated boxes for the current frame
        current_frame_draw = frame.copy()
        
        for track_id, data in interpolated_tracks.items():
            
            t_denominator = data["frame_end_lerp"] - data["frame_start_lerp"]
            if t_denominator == 0:
                t = 1.0
            else:
                t = (frame_count - frame_skip - data["frame_start_lerp"]) / t_denominator
            t = max(0.0, min(1.0, t)) # Clamp t to [0.0, 1.0]
            
            interp_box = lerp_box(data["box_start"], data["box_end"], t)
            
            x1, y1, x2, y2 = interp_box
            label = f"ID: {track_id} | {data['conf']:.2f}"
            cv2.rectangle(current_frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(current_frame_draw, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(current_frame_draw)
        frame_count += 1
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video saved to {output_path}")

