import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path to import court detection and player detection
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
sys.path.append(_PARENT_DIR)
sys.path.append(os.path.join(_PARENT_DIR, "court_detection"))

from court_mapper import CourtMapper
from auto_detector import AutoCourtDetector
from detect_player import get_valid_players

def main():
    # Use test1.mp4 from the Input directory
    input_video = os.path.join(_PARENT_DIR, "Input", "test1.mp4")
    output_image = os.path.join(_CUR_DIR, "player_detection_frame.jpg")
    
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
    
    print("Loading YOLOv8 Model...")
    yolo_model_path = os.path.join(_CUR_DIR, "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt" # Fallback to default download behavior
    yolo_model = YOLO(yolo_model_path)
    
    print("Detecting players within the court area...")
    
    # Run player detection
    # get_valid_players filters out any bounding box whose feet aren't strictly inside the polygon
    players = get_valid_players(
        frame=frame,
        yolo_model=yolo_model,
        court_polygon=court_poly,
        track=False,
        conf_threshold=0.35,
        polygon_margin=0
    )
    
    # 1. Overlay court lines, corners, and zones on the original frame
    output_frame = court_mapper.process_frame(
        frame.copy(), 
        draw_zones=True, 
        draw_lines=True, 
        draw_corners=True, 
        draw_hud=False
    )
    
    # 2. Draw bounding boxes for detected players
    for p in players:
        x1, y1, x2, y2 = p['bbox']
        conf = p['conf']
        # Draw bounding box (Green)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw label
        label = f"Player {conf:.2f}"
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(18, y1 - 8)
        cv2.rectangle(output_frame, (x1, ty - t_h - 4), (x1 + t_w + 6, ty + 4), (20, 20, 25), -1)
        cv2.rectangle(output_frame, (x1, ty - t_h - 4), (x1 + t_w + 6, ty + 4), (0, 255, 0), 1)
        cv2.putText(output_frame, label, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
        # Draw a red dot at feet to visualize the filter anchor point
        feet_x = int(x1 + (x2 - x1) / 2.0)
        feet_y = int(y2)
        cv2.circle(output_frame, (feet_x, feet_y), 4, (0, 0, 255), -1) 
    
    # Save the output image
    cv2.imwrite(output_image, output_frame)
    print(f"Success! Detected {len(players)} players inside/near the court boundaries.")
    print(f"Saved processed frame to {output_image}")
    
    cap.release()

if __name__ == '__main__':
    main()
