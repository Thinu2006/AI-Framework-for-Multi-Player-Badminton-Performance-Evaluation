import os
import sys
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
sys.path.append(_PARENT_DIR)
sys.path.append(os.path.join(_PARENT_DIR, "court_detection"))
sys.path.append(os.path.join(_PARENT_DIR, "player_detection"))
sys.path.append(os.path.join(_PARENT_DIR, "pose_estimate"))
sys.path.append(os.path.join(_PARENT_DIR, "shuttle_detection"))

from court_mapper import CourtMapper
from auto_detector import AutoCourtDetector
from detect_player import get_valid_players
from pose_estimation import MoveNetPoseEstimator, draw_pose_skeleton
from TrackNet import ResNet_Track
from shot_detection.shot_classification import BadmintonShotDetector, get_zone_at_pixel

from main_pipeline import get_player_feet_position

def load_tracknet_model(weights_path: str):
    """Loads pre-trained TrackNet shuttle detection model."""
    model = ResNet_Track(input_shape=(3, 288, 512))
    if os.path.exists(weights_path + ".index"):
        try:
            model.load_weights(weights_path).expect_partial()
        except Exception as e:
            pass
    return model

def main():
    input_video = os.path.join(_PARENT_DIR, "Input", "testing3.mp4")
    output_video = os.path.join(_CUR_DIR, "shot_detection_output.mp4")
    
    if not os.path.exists(input_video):
        print(f"Error: Could not find input video {input_video}")
        return

    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame from video.")
        cap.release()
        return

    print("1. Auto-detecting Court Corners & Initializing Court Mapper...")
    try:
        auto_detector = AutoCourtDetector(top_mask_ratio=0.25)
        corners_pixel = auto_detector.detect(frame)
        print(f"Detected corners: {corners_pixel}")
    except Exception as e:
        print(f"Auto court detection notice: {e}. Using fallback corner points.")
        corners_pixel = {
            'far_left': (260.07, 375.29),
            'far_right': (795.27, 385.66),
            'near_left': (130.92, 690.77), 
            'near_right': (972.10, 702.64)
        }

    court_mapper = CourtMapper(corners_pixel, alpha=0.30)
    court_poly = court_mapper.geometry.pts_pixel.astype(np.int32)
    
    # Initialize the shot detector with the court geometry
    shot_detector = BadmintonShotDetector(
        geometry=court_mapper.geometry, 
        max_missing_gap=4, 
        min_direction_consistency=0.70
    )
    
    print("2. Loading YOLOv8 Model for Player Detection...")
    yolo_model_path = os.path.join(_PARENT_DIR, "player_detection", "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt" 
    yolo_model = YOLO(yolo_model_path)

    print("3. Loading MoveNet Model for Pose Estimation...")
    movenet_model_path = os.path.join(_PARENT_DIR, "pose_estimate", "movenet_thunder.tflite")
    if not os.path.exists(movenet_model_path):
        root_movenet = os.path.join(_PARENT_DIR, "movenet_thunder.tflite")
        if os.path.exists(root_movenet): movenet_model_path = root_movenet
        else: movenet_model_path = "movenet_thunder.tflite"
    pose_estimator = MoveNetPoseEstimator(movenet_model_path)

    print("4. Loading TrackNet Model for Shuttle Detection...")
    weights_path = os.path.join(_PARENT_DIR, "shuttle_detection", "weights", "TrackNet")
    tracknet_model = load_tracknet_model(weights_path)
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    gray_imgs = deque(maxlen=3)
    frame_count = 0
    last_shot_info = None
    
    print("Processing video frames to find a shot event...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_imgs.append(np.expand_dims(gray_curr, axis=2))
        
        if len(gray_imgs) < 3:
            continue
        
        # --- 1. Court Mapping ---
        output_frame = court_mapper.process_frame(
            frame.copy(), draw_zones=True, draw_lines=True, draw_corners=True, draw_hud=False
        )
        
        # --- 2. Player Detection & Pose Estimation ---
        players = get_valid_players(
            frame=frame, yolo_model=yolo_model, court_polygon=court_poly, track=False, conf_threshold=0.35
        )
        
        player_zones = {}
        for p in players:
            bbox = p['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            
            keypoints = pose_estimator.estimate_player_pose(frame, bbox, margin=0.15)
            output_frame = draw_pose_skeleton(output_frame, keypoints, min_conf=0.20)
            
            # Map feet to a court zone and assign Far/Near role
            feet_pos = get_player_feet_position(keypoints, bbox)
            role = 'Far Player' if feet_pos[1] < height / 2 else 'Near Player'
            zone = get_zone_at_pixel(court_mapper.geometry, feet_pos)
            player_zones[role] = zone

        # --- 3. Shuttle Detection ---
        shuttle_pos = None
        img_input = np.concatenate(list(gray_imgs), axis=2)
        img_input = cv2.resize(img_input, (512, 288))
        img_input = np.moveaxis(img_input, -1, 0)
        img_input = np.expand_dims(img_input, axis=0).astype('float32') / 255.0

        y_pred = tracknet_model.predict(img_input, batch_size=1, verbose=0)
        y_pred = (y_pred > 0.5).astype('float32')
        h_pred = (y_pred[0] * 255).astype('uint8')

        if np.amax(h_pred) > 0:
            cnts, _ = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                target = max(rects, key=lambda r: r[2] * r[3])
                scale_x = width / 512.0
                scale_y = height / 288.0
                cx_pred = int(scale_x * (target[0] + target[2] / 2.0))
                cy_pred = int(scale_y * (target[1] + target[3] / 2.0))
                shuttle_pos = (cx_pred, cy_pred)

        if shuttle_pos:
            cv2.circle(output_frame, shuttle_pos, 5, (0, 0, 255), -1, cv2.LINE_AA)
            
        # --- 4. Shot Detection ---
        # Feed the coordinates and zones into the BadmintonShotDetector
        shot_event = shot_detector.process_frame(
            frame_idx=frame_count,
            shuttle_pos=shuttle_pos,
            player_zones=player_zones
        )
        
        if shot_event:
            print(f"\n>>> SHOT DETECTED at frame {frame_count}!")
            print(f"Shot Type: {shot_event['shot_type']} (Conf: {shot_event['confidence']})")
            print(f"Hitter: {shot_event['hitter']} ({shot_event['hitter_zone']})")
            print(f"Target Zone: {shot_event['shuttle_target_zone']}")
            last_shot_info = shot_event
            
        # Draw the active shot status on the frame
        if last_shot_info:
            cv2.putText(output_frame, f"LAST SHOT: {last_shot_info['shot_type'].upper()}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(output_frame, f"Hitter: {last_shot_info['hitter_zone']}", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output_frame, f"Target: {last_shot_info['shuttle_target_zone']}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(output_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            

    cap.release()
    if out is not None:
        out.release()
    print(f"\nSuccess! Full pipeline test video saved to {output_video}")

if __name__ == '__main__':
    main()
