import cv2

def get_valid_players(frame, yolo_model, court_polygon=None, track=False, conf_threshold=0.4, polygon_margin=0):
    """Runs YOLOv8 on the frame to detect persons (class 0).
    Filters the results to only include bounding boxes whose bottom-center (feet)
    is strictly inside the given mapped court_polygon.
    
    Returns a list of dictionaries containing:
        - bbox: (x1, y1, x2, y2)
        - conf: float
        - track_id: int (if track=True, else -1)
    """
    valid_players = []
    
    if track:
        results = yolo_model.track(frame, classes=[0], verbose=False, persist=True)
    else:
        results = yolo_model(frame, classes=[0], verbose=False)
        
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract track_id if available
            track_id = -1
            if track and box.id is not None:
                track_id = int(box.id[0])
                
            # Filter by court polygon if provided
            if court_polygon is not None and len(court_polygon) > 0:
                feet_x = x1 + (x2 - x1) / 2.0
                feet_y = float(y2)
                dist = cv2.pointPolygonTest(court_polygon, (feet_x, feet_y), measureDist=True)
                
                # Exclude detections outside the mapped court area (dist < polygon_margin)
                if dist < polygon_margin:
                    continue
            
            valid_players.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "track_id": track_id
            })
            
    return valid_players
