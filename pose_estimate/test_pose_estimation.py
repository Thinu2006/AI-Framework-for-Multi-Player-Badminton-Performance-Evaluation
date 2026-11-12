import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Setup paths to import the necessary modules from other folders
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
sys.path.append(_PARENT_DIR)
sys.path.append(os.path.join(_PARENT_DIR, "court_detection"))
sys.path.append(os.path.join(_PARENT_DIR, "player_detection"))

from court_mapper import CourtMapper
from auto_detector import AutoCourtDetector
from detect_player import get_valid_players
from pose_estimation import MoveNetPoseEstimator, draw_pose_skeleton

def main():
    input_video = os.path.join(_PARENT_DIR, "Input", "NewSession.mp4")
    output_image = os.path.join(_CUR_DIR, "pose_estimation_frame.jpg")
    
    if not os.path.exists(input_video):
        print(f"Error: Could not find input video {input_video}")
        return

    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame from video.")
        cap.release()
        return

    print("Auto-detecting court corners...")
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

    print("Initializing Court Mapper...")
    court_mapper = CourtMapper(corners_pixel, alpha=0.30)
    court_poly = court_mapper.geometry.pts_pixel.astype(np.int32)
    
    print("Loading YOLOv8 Model for Player Detection...")
    yolo_model_path = os.path.join(_PARENT_DIR, "player_detection", "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt" 
    yolo_model = YOLO(yolo_model_path)

    print("Loading MoveNet Model for Pose Estimation...")
    movenet_model_path = os.path.join(_CUR_DIR, "movenet_thunder.tflite")
    if not os.path.exists(movenet_model_path):
        # The movenet model is typically located in the root folder according to main_pipeline.py
        root_movenet = os.path.join(_PARENT_DIR, "movenet_thunder.tflite")
        if os.path.exists(root_movenet):
            movenet_model_path = root_movenet
        else:
            movenet_model_path = "movenet_thunder.tflite"

    pose_estimator = MoveNetPoseEstimator(movenet_model_path)
    
    print("Detecting players within the court area...")
    players = get_valid_players(
        frame=frame,
        yolo_model=yolo_model,
        court_polygon=court_poly,
        track=False,
        conf_threshold=0.35
    )
    
    # Draw court overlay
    output_frame = court_mapper.process_frame(
        frame.copy(), 
        draw_zones=True, 
        draw_lines=True, 
        draw_corners=True, 
        draw_hud=False
    )
    
    # Process each detected player for pose estimation
    print(f"Estimating poses for {len(players)} players...")
    for p in players:
        bbox = p['bbox']
        
        # 1. Draw Player Bounding Box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        
        # 2. Run Pose Estimation on the ROI
        keypoints = pose_estimator.estimate_player_pose(frame, bbox, margin=0.15)
        
        # 3. Draw Skeleton over the output frame
        output_frame = draw_pose_skeleton(output_frame, keypoints, min_conf=0.20)
        
    # Save the output image
    cv2.imwrite(output_image, output_frame)
    print(f"Success! Saved processed frame with pose skeletons to {output_image}")
    
    cap.release()

if __name__ == '__main__':
    main()
