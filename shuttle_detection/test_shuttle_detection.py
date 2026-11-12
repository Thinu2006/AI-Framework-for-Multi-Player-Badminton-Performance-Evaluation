import os
import sys
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Setup paths to import the necessary modules from other folders
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
sys.path.append(_PARENT_DIR)
sys.path.append(os.path.join(_PARENT_DIR, "court_detection"))
sys.path.append(os.path.join(_PARENT_DIR, "player_detection"))
sys.path.append(os.path.join(_PARENT_DIR, "pose_estimate"))

from court_mapper import CourtMapper
from detect_player import get_valid_players
from pose_estimation import MoveNetPoseEstimator, draw_pose_skeleton
from TrackNet import ResNet_Track

def load_tracknet_model(weights_path: str):
    """Loads pre-trained TrackNet shuttle detection model."""
    model = ResNet_Track(input_shape=(3, 288, 512))
    if os.path.exists(weights_path + ".index"):
        try:
            model.load_weights(weights_path).expect_partial()
            print("TrackNet weights loaded successfully.")
        except Exception as e:
            print(f"Warning loading TrackNet weights: {e}")
    else:
        print(f"Warning: TrackNet weights not found at {weights_path}")
    return model

def main():
    input_video = os.path.join(_PARENT_DIR, "Input", "test1.mp4")
    output_image = os.path.join(_CUR_DIR, "shuttle_detection_frame.jpg")
    
    if not os.path.exists(input_video):
        print(f"Error: Could not find input video {input_video}")
        return

    # Use the default corner coordinates from main_pipeline.py
    corners_pixel = {
        'far_left': (260.07, 375.29),
        'far_right': (795.27, 385.66),
        'near_left': (130.92, 690.77), 
        'near_right': (972.10, 702.64)
    }
    
    print("1. Initializing Court Mapper...")
    court_mapper = CourtMapper(corners_pixel, alpha=0.30)
    court_poly = court_mapper.geometry.pts_pixel.astype(np.int32)
    
    print("2. Loading YOLOv8 Model for Player Detection...")
    yolo_model_path = os.path.join(_PARENT_DIR, "player_detection", "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt" 
    yolo_model = YOLO(yolo_model_path)

    print("3. Loading MoveNet Model for Pose Estimation...")
    movenet_model_path = os.path.join(_PARENT_DIR, "pose_estimate", "movenet_thunder.tflite")
    if not os.path.exists(movenet_model_path):
        root_movenet = os.path.join(_PARENT_DIR, "movenet_thunder.tflite")
        if os.path.exists(root_movenet):
            movenet_model_path = root_movenet
        else:
            movenet_model_path = "movenet_thunder.tflite"
    pose_estimator = MoveNetPoseEstimator(movenet_model_path)

    print("4. Loading TrackNet Model for Shuttle Detection...")
    weights_path = os.path.join(_CUR_DIR, "weights", "TrackNet")
    tracknet_model = load_tracknet_model(weights_path)
    
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # TrackNet uses 3 consecutive grayscale frames
    gray_imgs = deque(maxlen=3)
    
    print("Processing frames to find shuttle...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Reached end of video without finding shuttle.")
            break
            
        frame_count += 1
        
        # Add frame to TrackNet deque
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_imgs.append(np.expand_dims(gray_curr, axis=2))
        
        # We need at least 3 frames for TrackNet to make a prediction
        if len(gray_imgs) < 3:
            continue
            
        if frame_count < 30:
            continue
            
        print(f"Running detection on frame {frame_count}...")
        
        # --- 1. Court Mapping ---
        output_frame = court_mapper.process_frame(
            frame.copy(), 
            draw_zones=True, 
            draw_lines=True, 
            draw_corners=True, 
            draw_hud=False
        )
        
        # --- 2. Player Detection & Pose Estimation ---
        players = get_valid_players(
            frame=frame,
            yolo_model=yolo_model,
            court_polygon=court_poly,
            track=False,
            conf_threshold=0.35
        )
        
        for p in players:
            bbox = p['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            
            keypoints = pose_estimator.estimate_player_pose(frame, bbox, margin=0.15)
            output_frame = draw_pose_skeleton(output_frame, keypoints, min_conf=0.20)

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
            print(f"Shuttle detected at {shuttle_pos} on frame {frame_count}")
            # Draw red dot for shuttle
            cv2.circle(output_frame, shuttle_pos, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(output_frame, shuttle_pos, 5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            print(f"Shuttle not visible in frame {frame_count}.")
            
        # Save the output image and exit
        cv2.imwrite(output_image, output_frame)
        print(f"Success! Saved full pipeline frame {frame_count} to {output_image}")
        break
        
    cap.release()

if __name__ == '__main__':
    main()
