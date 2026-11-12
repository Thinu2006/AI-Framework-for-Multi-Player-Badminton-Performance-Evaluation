import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO

# Add subdirectories to module search path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_BASE_DIR, "court_detection"))
sys.path.append(os.path.join(_BASE_DIR, "player_detection"))
sys.path.append(os.path.join(_BASE_DIR, "pose_estimate"))
sys.path.append(os.path.join(_BASE_DIR, "shuttle_detection"))
sys.path.append(os.path.join(_BASE_DIR, "shot_detection"))
sys.path.append(_BASE_DIR)  

from court_detection.court_mapper import CourtMapper
from court_detection.court_geometry import CourtGeometry
from player_detection.detect_player import get_valid_players
from pose_estimate.pose_estimation import MoveNetPoseEstimator, draw_pose_skeleton
from shuttle_detection.TrackNet import ResNet_Track
from shot_detection import BadmintonShotDetector
import json
from csv_exporter import export_player_csvs


def get_player_feet_position(keypoints: dict, bbox: tuple) -> tuple:
    """
    Computes the floor position of a player's feet using MoveNet pose keypoints (ankles 15 & 16).
    Falls back to bottom-center of bounding box if ankle keypoints are not confident.
    """
    x1, y1, x2, y2 = bbox
    ankle_pts = []
    
    if keypoints:
        for idx in (15, 16):
            kp = keypoints.get(idx) or keypoints.get(str(idx))
            if kp and isinstance(kp, dict):
                score = kp.get('score', 0.0)
                pt = kp.get('pt')
                if score >= 0.20 and pt:
                    ankle_pts.append(pt)
                    
    if ankle_pts:
        feet_x = sum(p[0] for p in ankle_pts) / float(len(ankle_pts))
        feet_y = sum(p[1] for p in ankle_pts) / float(len(ankle_pts))
        return (feet_x, feet_y)
        
    return ((x1 + x2) / 2.0, float(y2))


def get_player_zone_name(geometry: CourtGeometry, feet_pixel: tuple) -> str:
    """
    Given a player's feet pixel coordinate (x, y), uses inverse Homography
    to project into 2D court metric space and returns the active zone name,
    or the nearest zone if standing slightly outside court boundaries.
    """
    if geometry is None or feet_pixel is None:
        return "Court"
    try:
        court_pts = geometry.pixel_to_court([feet_pixel])
        cx, cy = float(court_pts[0][0]), float(court_pts[0][1])
        
        best_zone = None
        max_dist = -float('inf')
        
        for zone in geometry.get_zones():
            poly_2d = np.array(zone['poly_2d'], dtype=np.float32)
            dist = cv2.pointPolygonTest(poly_2d, (cx, cy), measureDist=True)
            if dist >= 0:
                return zone['name']
            if dist > max_dist:
                max_dist = dist
                best_zone = zone['name']
                
        # If outside but within 300cm of court boundaries, return nearest zone
        if max_dist > -300.0 and best_zone:
            return best_zone
            
        return "Outside Court"
    except Exception:
        return "Court"




def load_tracknet_model(weights_path: str = "shuttle_detection/weights/TrackNet"):
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


def draw_segment_annotation(frame: np.ndarray, shot_dict: dict) -> np.ndarray:
    """
    Renders the segment annotation badge displaying shot type, hitter, direction,
    shuttle travelling zones (e.g., near_left_front -> far_right_rear), frame range, and confidence.
    """
    if shot_dict is None:
        return frame

    shot_type_str = shot_dict.get('shot_type', 'unknown').upper()
    hitter_str = "Far Player" if shot_dict.get('hitter') == 'far_player' else "Near Player"
    dir_raw = shot_dict.get('travel_direction', '')
    dir_str = "Far -> Near" if dir_raw == 'far_to_near' else "Near -> Far"

    zone_flow = shot_dict.get('zone_transition')
    if not zone_flow:
        hz = shot_dict.get('hitter_zone', 'unknown')
        rz = shot_dict.get('receiver_zone', 'unknown')
        stz = shot_dict.get('shuttle_target_zone', 'unknown')
        eff_target = rz if rz != 'unknown' else stz
        zone_flow = f"{hz} -> {eff_target}"

    frames_str = f"{shot_dict.get('start_frame')}-{shot_dict.get('end_frame')}"
    conf_val = shot_dict.get('confidence', 0.0)

    lines = [
        f"Shot: {shot_type_str}",
        f"Hitter: {hitter_str}",
        f"Direction: {dir_str}",
        f"Travel Zones: {zone_flow}",
        f"Frames: {frames_str}",
        f"Confidence: {conf_val:.2f}"
    ]

    # Position annotation card top-left below HUD
    x, y = 20, 140
    w, h = 330, 150

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 255), 2, cv2.LINE_AA)

    # Title header
    cv2.putText(frame, f"SHOT #{shot_dict.get('shot_id', 1)} ANNOTATION", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    cur_y = y + 40
    for line in lines:
        cv2.putText(frame, line, (x + 12, cur_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        cur_y += 18

    return frame


def process_video_pipeline(
    input_path: str = "Input/NewSession.mp4",
    output_dir: str = "NewSession",
    output_name: str = "NewSession_output.mp4",
    target_roles: list = None,
    report_enabled_roles: list = None,
    progress_callback = None
):
    """
    Executes combined court, player role assignment, shuttle tracking, pose estimation,
    and 2-pass frame-by-frame shot annotation writing to annotated_shots.mp4 and shot_predictions.json.
    Supports filtering by target_roles (e.g. ['Far Player'], ['Near Player'], or both)
    and live 7-stage progress_callback updates.
    """
    def _notify(stage_idx, stage_name, pct, detail="", log_msg=None):
        if progress_callback:
            try:
                progress_callback(stage_idx, stage_name, pct, detail, log_msg or detail)
            except RuntimeError as re:
                if "cancelled" in str(re).lower():
                    raise re
                print(f"Progress callback error: {re}")
            except Exception as e:
                print(f"Progress callback error: {e}")

    _notify(1, "Stage 1/7: Court & Zone Calibration", 5, "Initializing court mapping & tactical zones...", "Calibrated court boundaries and zone layout")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # Derive a safe session name from the actual uploaded video filename
    _raw_name = os.path.splitext(os.path.basename(input_path))[0]
    import re as _re
    session_name = _re.sub(r'[^a-zA-Z0-9_-]', '_', _raw_name)[:40] or "session"

    annotated_video_path = os.path.join(output_dir, f"{session_name}_annotated.mp4")
    json_output_path     = os.path.join(output_dir, f"{session_name}_shot_predictions.json")

    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found!")
        return

    # 1. Initialize Models & Utilities
    _notify(1, "Stage 1/7: Court & Zone Calibration", 5, "Setting up court boundaries and grid zones...", "Calibrating court boundaries automatically")
    from court_detection.auto_detector import AutoCourtDetector
    
    temp_cap = cv2.VideoCapture(input_path)
    ret, first_frame = temp_cap.read()
    temp_cap.release()
    
    if not ret or first_frame is None:
        raise RuntimeError("Failed to read video to detect court.")
        
    try:
        auto_detector = AutoCourtDetector(top_mask_ratio=0.25)
        corners_pixel = auto_detector.detect(first_frame)
    except Exception as e:
        raise RuntimeError(f"Automatic court calibration failed: {e}")
        
    court_mapper = CourtMapper(corners_pixel, alpha=0.30)
    geometry = court_mapper.geometry
    court_poly = geometry.pts_pixel.astype(np.int32)

    _notify(1, "Stage 1/7: Court & Zone Calibration", 8, "Loading player motion and posture trackers...", "Loaded player detection system")
    yolo_model_path = os.path.join("player_detection", "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt"
    yolo_model = YOLO(yolo_model_path)

    _notify(1, "Stage 1/7: Court & Zone Calibration", 10, "Loading posture and shuttle tracking models...", "Loaded posture estimator")
    movenet_model_path = "movenet_thunder.tflite"
    pose_estimator = MoveNetPoseEstimator(movenet_model_path)

    tracknet_model = load_tracknet_model()
    _notify(1, "Stage 1/7: Court & Zone Calibration", 12, "Shuttle flight tracker ready", "Loaded shuttle tracking system")

    # Shot Detector
    shot_detector = BadmintonShotDetector(geometry=geometry, max_missing_gap=4, min_direction_consistency=0.70)

    # 2. Open Video Stream (PASS 1: Detection & Frame Caching)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n--- Pass 1: Analyzing Video Stream: {input_path} ---")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total Frames: {total_frames}")

    gray_imgs = deque(maxlen=3)
    shuttle_trajectory = deque(maxlen=20)
    shuttle_ratio = height / 288.0

    role_colors = {
        'Far Player': (255, 140, 0),
        'Near Player': (0, 100, 255)
    }

    frames_cache    = []
    frame_idx     = 0
    # Dict to store per-frame keypoints: {frame_idx: {'Near Player': {...}, 'Far Player': {...}}}
    frame_keypoints: dict = {}
    shuttle_keypoints_dict: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        output_frame = frame.copy()

        # STEP 1: Detect Court & Zones
        output_frame = court_mapper.process_frame(
            output_frame, draw_zones=True, draw_lines=True, draw_corners=True, draw_hud=False
        )

        # STEP 2: Detect Players & Assign Near / Far Roles (strictly inside mapped court area)
        raw_players = get_valid_players(
            frame, 
            yolo_model, 
            court_polygon=court_poly, 
            track=True, 
            conf_threshold=0.35, 
            polygon_margin=0
        )
        raw_players.sort(key=lambda p: p['bbox'][3])
        
        assigned_players = {}
        if len(raw_players) >= 2:
            assigned_players['Far Player'] = raw_players[0]
            assigned_players['Near Player'] = raw_players[-1]
        elif len(raw_players) == 1:
            p = raw_players[0]
            feet_y = p['bbox'][3]
            if feet_y < 535:
                assigned_players['Far Player'] = p
            else:
                assigned_players['Near Player'] = p

        if target_roles:
            assigned_players = {r: p for r, p in assigned_players.items() if r in target_roles}

        player_zones = {}
        for role, pdata in assigned_players.items():
            x1, y1, x2, y2 = pdata['bbox']
            color = role_colors.get(role, (0, 255, 0))

            # STEP 2a: Estimate Pose Keypoints FIRST for accurate feet positioning
            keypoints = pose_estimator.estimate_player_pose(frame, (x1, y1, x2, y2), margin=0.15)
            output_frame = draw_pose_skeleton(output_frame, keypoints, min_conf=0.20)

            # STEP 2b: Use Ankle keypoints if detected with confidence, else fall back to bbox bottom-center
            feet_pos = get_player_feet_position(keypoints, (x1, y1, x2, y2))
            zone_name = get_player_zone_name(geometry, feet_pos)
            player_zones[role] = zone_name

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            badge_text = f"{role} | Zone: {zone_name}"
            (t_w, t_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            ty = max(18, y1 - 8)
            cv2.rectangle(output_frame, (x1, ty - t_h - 4), (x1 + t_w + 6, ty + 4), (20, 20, 25), -1)
            cv2.rectangle(output_frame, (x1, ty - t_h - 4), (x1 + t_w + 6, ty + 4), color, 1)
            cv2.putText(output_frame, badge_text, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Cache keypoints for this frame + role (serialise pt tuples → lists)
            if frame_idx not in frame_keypoints:
                frame_keypoints[frame_idx] = {}
            serialisable_kps = {
                str(k): {
                    "name":  v["name"],
                    "pt":    [round(v["pt"][0], 2), round(v["pt"][1], 2)],
                    "score": round(v["score"], 4),
                }
                for k, v in keypoints.items()
            }
            frame_keypoints[frame_idx][role] = serialisable_kps


        # STEP 3: Detect Shuttle & Draw Trajectory
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_imgs.append(np.expand_dims(gray_curr, axis=2))

        shuttle_pos = None
        if len(gray_imgs) == 3:
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
                    shuttle_trajectory.append(shuttle_pos)
                    shuttle_keypoints_dict[str(frame_idx)] = {
                        "x": float(cx_pred),
                        "y": float(cy_pred),
                        "confidence": 0.90
                    }

        if shuttle_pos:
            cv2.circle(output_frame, shuttle_pos, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(output_frame, shuttle_pos, 5, (0, 255, 255), 1, cv2.LINE_AA)

        # Convert shuttle position to 2D court metric coordinates for top-view shot detection
        shuttle_court_pos = None
        if shuttle_pos:
            try:
                c_pts = geometry.pixel_to_court([shuttle_pos])
                shuttle_court_pos = (float(c_pts[0][0]), float(c_pts[0][1]))
            except Exception:
                shuttle_court_pos = None

        # STEP 4: Process Frame in Shot Detector using 2D Top View Court Metric Coordinates
        shot_detector.process_frame(frame_idx, shuttle_pos, player_zones, shuttle_court_pos=shuttle_court_pos)

        # HUD Header Summary Overlay
        hud = output_frame.copy()
        cv2.rectangle(hud, (20, 20), (550, 115), (15, 15, 25), -1)
        cv2.addWeighted(hud, 0.75, output_frame, 0.25, 0, output_frame)
        cv2.rectangle(output_frame, (20, 20), (550, 115), (0, 255, 200), 1, cv2.LINE_AA)

        cv2.putText(output_frame, "BADMINTON MULTI-ANALYSIS SYSTEM", (32, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2, cv2.LINE_AA)
        
        far_str = player_zones.get('Far Player', 'N/A')
        near_str = player_zones.get('Near Player', 'N/A')
        shuttle_str = f"({shuttle_pos[0]}, {shuttle_pos[1]})" if shuttle_pos else "Tracking..."

        cv2.putText(output_frame, f"Far Player: {far_str}", (32, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 100), 1, cv2.LINE_AA)
        cv2.putText(output_frame, f"Near Player: {near_str}", (32, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(output_frame, f"Shuttle: {shuttle_str} | Frame: {frame_idx}/{total_frames}", (32, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        frames_cache.append(output_frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = 10 + int((frame_idx / max(1, total_frames)) * 55)
            _notify(2, f"Stage 2-4/7: Tracking Video ({frame_idx}/{total_frames})", pct, f"Tracking player positioning, body posture & shuttle flight ({frame_idx}/{total_frames} frames)")

    cap.release()

    # Finalize shot detection
    _notify(5, "Stage 5/7: Shot Identification & Rally Export", 75, "Finalizing stroke recognition and exporting rally clips...")
    completed_shots = shot_detector.finish_stream()
    print(f"\n--- Stage 5: Shot Analysis Complete: {len(completed_shots)} shots detected ---")
    for s in completed_shots:
        print(json.dumps(s, indent=2))

    # Save shot_predictions.json  (uses session_name so each job gets its own file)
    with open(json_output_path, "w") as f:
        json.dump(completed_shots, f, indent=2)

    output_dir_json = os.path.join(output_dir, f"{session_name}_shot_predictions.json")
    with open(output_dir_json, "w") as f:
        json.dump(completed_shots, f, indent=2)

    # Save frame_keypoints.json
    kp_output_path = os.path.join(output_dir, f"{session_name}_frame_keypoints.json")
    frame_keypoints_str_keys = {str(k): v for k, v in frame_keypoints.items()}
    with open(kp_output_path, "w") as f:
        json.dump(frame_keypoints_str_keys, f, indent=2)

    # Save shuttle_keypoints.json
    shuttle_output_path = os.path.join(output_dir, f"{session_name}_shuttle_keypoints.json")
    with open(shuttle_output_path, "w") as f:
        json.dump(shuttle_keypoints_dict, f, indent=2)

    # Export player-wise CSVs from shot predictions + frame keypoints
    print("\n--- Exporting Player-wise Shot CSVs & Court Area Mapping ---")
    csv_paths = export_player_csvs(
        shots_json_path   = output_dir_json,
        keypoints_json_path = kp_output_path,
        output_dir        = output_dir,
    )

    # Export court mapping area CSV
    from csv_exporter import export_court_csv
    export_court_csv(geometry, corners_pixel, os.path.join(output_dir, "mapped_court_area.csv"))
    export_court_csv(geometry, corners_pixel, os.path.join(output_dir, "court_mapping.csv"))

    # Map frame indices to shot predictions for Pass 2 rendering
    frame_to_shot = {}
    for shot in completed_shots:
        for f in range(shot['start_frame'], shot['end_frame'] + 1):
            frame_to_shot[f] = shot

    # PASS 2: Render exact segment annotation badges, write annotated_shots.mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_h, out_w = frames_cache[0].shape[:2]
    out_annotated = cv2.VideoWriter(annotated_video_path, fourcc, fps, (out_w, out_h))
    out_pipeline = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frames_out_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    for idx, f_frame in enumerate(frames_cache, start=1):
        shot_info = frame_to_shot.get(idx)
        annotated_frame = draw_segment_annotation(f_frame.copy(), shot_info)

        out_annotated.write(annotated_frame)
        out_pipeline.write(annotated_frame)

        frame_file = f"frame_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_dir, frame_file), annotated_frame)
        cv2.imwrite(os.path.join(frames_out_dir, frame_file), annotated_frame)

    out_annotated.release()
    out_pipeline.release()

    # --- Stage 6: Technique & Flaw Analysis ---
    _notify(6, "Stage 6/7: Technique & Flaw Analysis", 88, "Evaluating swing form, posture & joint angles...")
    from error_detection.stage6_error_pipeline import run_stage6_pipeline
    abs_output_dir = os.path.abspath(output_dir)
    run_stage6_pipeline(
        shots_json_path    = os.path.join(abs_output_dir, f"{session_name}_shot_predictions.json"),
        keypoints_json_path= os.path.join(abs_output_dir, f"{session_name}_frame_keypoints.json"),
        shuttle_json_path  = os.path.join(abs_output_dir, f"{session_name}_shuttle_keypoints.json"),
        output_dir         = abs_output_dir,
        output_json_name   = f"{session_name}_biomechanical_errors.json",
        report_enabled_roles=report_enabled_roles
    )

    # --- Stage 7: Grounded AI Coach Advice ---
    _notify(7, "Stage 7/7: AI Coach Advice Synthesis", 95, "Generating personalized coaching feedback & custom drills...")
    from RAG.stage7_feedback_pipeline import run_stage7_pipeline
    run_stage7_pipeline(
        errors_json_path = os.path.join(abs_output_dir, f"{session_name}_biomechanical_errors.json"),
        output_dir       = abs_output_dir,
        output_json_name = f"{session_name}_coaching_feedback.json"
    )

    _notify(7, "Completed", 100, "Analysis complete! Report generated.")


if __name__ == "__main__":
    process_video_pipeline()

