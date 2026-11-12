import os
import sys
import cv2

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_CUR_DIR))

from court_mapper import CourtMapper

def main():
    input_video = os.path.join(os.path.dirname(_CUR_DIR), "Input", "NewSession.mp4")
    output_image = os.path.join(_CUR_DIR, "court_detection_frame.jpg")
    
    if not os.path.exists(input_video):
        print(f"Error: Could not find input video {input_video}")
        return

    print("Initializing Court Mapper...")
    from court_mapper import CourtMapper
    from auto_detector import AutoCourtDetector
    
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
        return
        
    print("Auto-detecting court corners...")
    detector = AutoCourtDetector(top_mask_ratio=0.25)
    try:
        corners_pixel = detector.detect(frame)
        print(f"Detected corners: {corners_pixel}")
    except Exception as e:
        print(f"Detection failed: {e}")
        cap.release()
        return
    court_mapper = CourtMapper(corners_pixel, alpha=0.30)
        
    print("Processing first frame...")
    # 1. Overlay court lines, corners, and zones on the original frame
    output_frame = court_mapper.process_frame(
        frame.copy(), 
        draw_zones=True, 
        draw_lines=True, 
        draw_corners=True, 
        draw_hud=False
    )
    
    # Save the output image
    cv2.imwrite(output_image, output_frame)
    print(f"Success! Saved processed frame to {output_image}")
    
    cap.release()

if __name__ == '__main__':
    main()
