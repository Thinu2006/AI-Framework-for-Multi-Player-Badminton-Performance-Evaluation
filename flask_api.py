import os
import sys
import jwt
import json
import uuid
import datetime
import threading
import traceback
import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from functools import wraps

import database
import main_pipeline
import pipeline_logger as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Ensure logs directory exists on startup
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

@app.errorhandler(Exception)
def handle_global_exception(e):
    response = jsonify({"message": f"Server error: {str(e)}"})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response, 500

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key-badminton-court-detection')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'Input')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'Match')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max video upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory store for active analysis jobs
JOBS = {}

# Initialize database
try:
    database.init_db()
except Exception as e:
    print(f"[Backend Warning] Database initialization notice: {e}")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(None, None, *args, **kwargs)

        token = None
        if 'Authorization' in request.headers:
            parts = request.headers['Authorization'].split()
            if len(parts) == 2 and parts[0] == 'Bearer':
                token = parts[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_coach_id = data['coach_id']
            current_coach_name = data['name']
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(current_coach_id, current_coach_name, *args, **kwargs)

    return decorated

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    name = data.get('name')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([name, username, email, password]):
        return jsonify({'message': 'Missing required registration fields'}), 400

    if database.coach_exists(username, email):
        return jsonify({'message': 'A coach with this username or email already exists.'}), 409

    success = database.create_coach(name, username, email, password)
    if success:
        return jsonify({'message': 'Registration successful!'}), 201
    else:
        return jsonify({'message': 'Registration failed.'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({'message': 'Missing required login fields'}), 400

    coach = database.verify_coach_login(username, email, password)
    
    if coach:
        token = jwt.encode({
            'coach_id': coach['coach_id'],
            'name': coach['name'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'message': 'Login successful!',
            'token': token,
            'coach': coach
        }), 200

    return jsonify({'message': 'Invalid credentials. Please verify Username, Email, and Password.'}), 401

@app.route('/api/dashboard', methods=['GET'])
@token_required
def dashboard(current_coach_id, current_coach_name):
    return jsonify({
        'message': f'Welcome to Badminton Vision Dashboard, {current_coach_name}!',
        'coach_id': current_coach_id,
        'coach_name': current_coach_name
    }), 200

@app.route('/api/players', methods=['GET'])
@token_required
def get_players(current_coach_id, current_coach_name):
    players = database.get_players_by_coach(current_coach_id)
    serialized_players = []
    for p in players:
        p_copy = p.copy()
        if p_copy.get('created_at'):
            p_copy['created_at'] = str(p_copy['created_at'])
        serialized_players.append(p_copy)
    return jsonify(serialized_players), 200

@app.route('/api/players', methods=['POST'])
@token_required
def add_player(current_coach_id, current_coach_name):
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    full_name = data.get('full_name')
    contact_number = data.get('contact_number')
    age = data.get('age')
    age_group = data.get('age_group')
    
    if not full_name:
        return jsonify({'message': 'Full name is required'}), 400
        
    try:
        age_val = int(age) if age else None
    except ValueError:
        return jsonify({'message': 'Age must be a valid integer'}), 400

    valid_groups = ['U11', 'U13', 'U15', 'U17', 'U19']
    if age_group and age_group not in valid_groups:
        return jsonify({'message': 'Invalid age group'}), 400
        
    success = database.create_player(current_coach_id, full_name, contact_number, age_val, age_group)
    if success:
        return jsonify({'message': 'Player added successfully!'}), 201
    else:
        return jsonify({'message': 'Failed to add player.'}), 500

@app.route('/api/players/<int:player_id>', methods=['DELETE'])
@token_required
def remove_player(current_coach_id, current_coach_name, player_id):
    success = database.delete_player(player_id, current_coach_id)
    if success:
        return jsonify({'message': 'Player deleted successfully!'}), 200
    else:
        return jsonify({'message': 'Failed to delete player.'}), 500

@app.route('/api/sessions', methods=['GET'])
@token_required
def get_sessions(current_coach_id, current_coach_name):
    sessions = database.get_sessions_by_coach(current_coach_id)
    serialized_sessions = []
    
    match_dir = os.path.join(BASE_DIR, "Match")
    
    for s in sessions:
        s_copy = s.copy()
        if s_copy.get('created_at'):
            s_copy['created_at'] = str(s_copy['created_at'])
            
        sid = s_copy.get('session_id')
        target_dir = resolve_session_output_dir(sid)
        
        try:
            res = load_session_results(target_dir, session_data=s_copy)
            
            players_list = res.get('players', [])
            target_p = {}
            for pl in players_list:
                if pl.get('shot_statistics', {}).get('total_shots', 0) > 0:
                    target_p = pl
                    break
            if not target_p and len(players_list) > 0:
                target_p = players_list[0]
            
            valid_preds = [
                s for s in (res.get('shot_predictions') or []) 
                if isinstance(s, dict) and str(s.get('shot_type', '')).lower() != 'unknown'
            ]
            
            total_shots = target_p.get('shot_statistics', {}).get('total_shots', 0)
            if total_shots == 0 and len(valid_preds) > 0:
                total_shots = len(valid_preds)

            errors_list = target_p.get('detected_errors', [])
            error_count = len(errors_list)
            
            court_side_raw = target_p.get('court_side')
            if not court_side_raw and s_copy.get('participants'):
                parts = s_copy.get('participants')
                if isinstance(parts, list) and len(parts) > 0:
                    court_side_raw = parts[0].get('court_side')
            side_lower = str(court_side_raw or 'near_player').lower()
            court_side = "Near Side" if "near" in side_lower else ("Far Side" if "far" in side_lower else "Near Side")
            
            sess_status = s_copy.get('status', 'Completed')
            
            if sess_status != 'Completed':
                badge_cat = 'Analysis Incomplete'
            elif error_count == 0:
                badge_cat = 'Excellent Technique'
            elif error_count <= 2:
                badge_cat = 'Minor Improvements Needed'
            else:
                badge_cat = 'Needs Improvement'
                
            top_shots = list(target_p.get('shot_statistics', {}).get('shot_distribution', {}).keys())
            shot_clause = f"in {top_shots[0].lower()}s" if top_shots else "during stroke execution"
            
            if error_count == 0:
                ai_sum = f"{total_shots} shots analysed. No major biomechanical issues detected. Technique was consistent throughout."
            else:
                ai_sum = f"{total_shots} shots analysed. {error_count} biomechanical issues detected {shot_clause}. Improvement recommended."
                
            s_copy['total_shots'] = total_shots
            s_copy['error_count'] = error_count
            s_copy['court_side'] = court_side
            s_copy['technique_badge'] = badge_cat
            s_copy['summary'] = ai_sum
            s_copy['pipeline_version'] = "v1.0"
            s_copy['report_version'] = "v1.0"
            s_copy['processing_date'] = s_copy.get('created_at')
            s_copy['analysis_duration'] = "Not available."
            
        except Exception as err:
            print(f"[Warning] Failed to enrich session {sid}: {err}")
            s_copy['total_shots'] = s_copy.get('total_shots', 0)
            s_copy['error_count'] = s_copy.get('error_count', 0)
            s_copy['court_side'] = "Near Side"
            s_copy['pipeline_version'] = "v1.0"
            s_copy['report_version'] = "v1.0"
            s_copy['analysis_duration'] = "Not available."
            
        serialized_sessions.append(s_copy)
        
    return jsonify(serialized_sessions), 200

@app.route('/api/sessions', methods=['POST'])
@token_required
def add_session(current_coach_id, current_coach_name):
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    video_name = data.get('video_name')
    players = data.get('players')
    status = data.get('status', 'Completed')
    technique_badge = data.get('technique_badge', 'Good Technique')
    summary = data.get('summary', '')
    participants = data.get('participants', [])
    
    if not video_name or not players:
        return jsonify({'message': 'Video name and players are required'}), 400
        
    session_id = database.create_session(current_coach_id, video_name, players, status, technique_badge, summary, participants)
    if session_id:
        return jsonify({'message': 'Session added successfully!', 'session_id': session_id}), 201
    else:
        return jsonify({'message': 'Failed to add session.'}), 500

@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
@token_required
def remove_session(current_coach_id, current_coach_name, session_id):
    success = database.delete_session(session_id, current_coach_id)
    if success:
        return jsonify({'message': 'Session deleted successfully!'}), 200
    else:
        return jsonify({'message': 'Failed to delete session.'}), 500

def run_pipeline_job(job_id, video_path, output_dir, target_roles, coach_id, video_name, player_names_str, participants, report_enabled_roles=None, test_mode=False, ground_truth_path=None):
    # ── Per-job log file ───────────────────────────────────────────────────
    logger = pl.create_logger(job_id)
    logger.log_event("System", f"Pipeline job started for video: {video_name}")
    logger.log_event("System", f"Output directory: {output_dir}")
    logger.log_event("System", f"Target roles: {target_roles}")
    logger.log_event("System", f"Report roles: {report_enabled_roles}")

    def progress_cb(stage_idx, stage_name, pct, detail="", log_msg=""):
        if job_id in JOBS:
            if JOBS[job_id].get("status") == "Cancelled":
                raise RuntimeError("Job cancelled by user")
            JOBS[job_id]["stage_idx"] = stage_idx
            JOBS[job_id]["stage_name"] = stage_name
            JOBS[job_id]["percent"] = pct
            JOBS[job_id]["detail"] = detail
        db_job = database.get_job(job_id)
        if db_job and db_job.get("status") == "Cancelled":
            if job_id in JOBS:
                JOBS[job_id]["status"] = "Cancelled"
            raise RuntimeError("Job cancelled by user")
        database.update_job_progress(job_id, stage_idx, stage_name, pct, detail, log_msg)
        # Mirror every stage progress update to the job log file
        logger.log_event(stage_name, log_msg or detail, level="STAGE")

    try:
        # Save a clean copy of the raw unannotated input video in output_dir
        if video_path and os.path.exists(video_path):
            try:
                import shutil
                shutil.copy2(video_path, os.path.join(output_dir, "raw_input.mp4"))
            except Exception:
                pass

        progress_cb(1, "Stage 1/7: Court & Zone Calibration", 5, "Initializing court mapping & tactical zones...", "Calibrating court boundaries and zone layout")
        main_pipeline.process_video_pipeline(
            input_path=video_path,
            output_dir=output_dir,
            output_name="output.mp4",
            target_roles=target_roles,
            report_enabled_roles=report_enabled_roles,
            progress_callback=progress_cb
        )
        
        summary_text = "Biomechanical analysis completed. Footwork & stroke mechanics evaluated."
        badge = "Good Technique"

        # Search for the session's biomechanical errors file (name is derived from uploaded video)
        import glob as _glob
        err_candidates = (
            _glob.glob(os.path.join(output_dir, "*_biomechanical_errors.json")) +
            [os.path.join(output_dir, "biomechanical_errors.json")]
        )
        err_file = next((p for p in err_candidates if os.path.exists(p)), None)
        if err_file:
            try:
                with open(err_file, 'r') as f:
                    err_data = json.load(f)
                errs = err_data.get("shots", []) if isinstance(err_data, dict) else err_data
                active_errs = [s for s in errs if s.get("errors") and len(s.get("errors")) > 0]
                if len(active_errs) > 0:
                    badge = "Needs Improvement"
                    summary_text = f"Biomechanical errors flagged in {len(active_errs)} shot segment(s)."
            except Exception:
                pass

        if test_mode and ground_truth_path:
            progress_cb(7, "Test Evaluation", 98, "Running test metrics evaluation against ground truth...", "Running test evaluation module")
            import testing.evaluation
            try:
                testing.evaluation.run_test_evaluation(job_id, output_dir, ground_truth_path)
                badge = "Test Complete"
                summary_text = "Test mode evaluation run against ground truth."
                video_name = f"(TEST RUN) {video_name}"
            except Exception as eval_err:
                logger.log_event("System", f"Test evaluation failed: {eval_err}", level="ERROR")
                summary_text = f"Test evaluation failed: {eval_err}"

        session_id = database.create_session(
            coach_id=coach_id,
            video_name=video_name,
            players=player_names_str,
            status="Completed",
            technique_badge=badge,
            summary=summary_text,
            participants=participants
        )
        
        logger.log_event("System", f"Pipeline completed successfully. Session #{session_id} created.", level="SUCCESS")
        progress_cb(7, "Analysis Complete", 100, "Report generated", f"Session #{session_id} saved to database. Report ready.")
        if job_id in JOBS:
            JOBS[job_id]["status"] = "Completed"
            JOBS[job_id]["session_id"] = session_id
        database.complete_job(job_id, session_id)

    except Exception as e:
        tb_str = traceback.format_exc()
        error_detail = str(e)

        # ── Classify common failure causes ────────────────────────────────
        error_category = "Pipeline Error"
        if any(kw in error_detail.lower() for kw in ["api key", "apikey", "api_key", "quota", "credit", "billing", "limit", "429", "rate limit"]):
            error_category = "AI/LLM API Error — API key may be invalid, quota exceeded, or credits exhausted"
        elif any(kw in error_detail.lower() for kw in ["connection", "timeout", "network", "resolve", "socket"]):
            error_category = "Network Error — Could not reach external service"
        elif any(kw in error_detail.lower() for kw in ["no video", "not found", "no such file", "corrupt", "codec"]):
            error_category = "Video File Error — File missing, corrupt, or unsupported format"
        elif any(kw in error_detail.lower() for kw in ["memory", "memoryerror", "oom"]):
            error_category = "Memory Error — Insufficient RAM to process video"

        if (job_id in JOBS and JOBS[job_id].get("status") == "Cancelled") or "cancelled" in error_detail.lower():
            logger.log_event("System", "Analysis cancelled by user.", level="CANCELLED")
            logger.close()
            pl.close_logger(job_id)
            print(f"Pipeline job {job_id} cancelled: {e}")
            if job_id in JOBS:
                JOBS[job_id]["status"] = "Cancelled"
                JOBS[job_id]["detail"] = "Analysis cancelled by user."
            database.cancel_job(job_id, "Analysis cancelled by user.")
            return

        # Log the classified error with full traceback to file
        logger.log_event("System", f"FAILED — {error_category}: {error_detail}", level="ERROR")
        logger.log_error(f"{error_category}: {error_detail}")
        try:
            logger._log_file.write(f"  Full traceback:\n{tb_str}\n")
            logger._log_file.flush()
        except Exception:
            pass

        print(f"Pipeline job failed [{error_category}]: {e}")
        print(tb_str)
        if job_id in JOBS:
            JOBS[job_id]["status"] = "Failed"
            JOBS[job_id]["detail"] = f"{error_category}: {error_detail}"
        database.fail_job(job_id, f"{error_category}: {error_detail}")

    finally:
        pl.close_logger(job_id)

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
@token_required
def analyze_video(current_coach_id, current_coach_name):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    if 'video' not in request.files:
        return jsonify({'message': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'message': 'No video file selected'}), 400

    player_names_str = request.form.get('players', 'Unknown Players')
    target_roles_raw = request.form.get('target_roles', '["Far Player", "Near Player"]')
    participants_raw = request.form.get('participants', '[]')
    
    test_mode = request.form.get('test_mode') == 'true'
    ground_truth_path = None
    if test_mode and 'ground_truth' in request.files:
        gt_file = request.files['ground_truth']
        if gt_file and gt_file.filename != '':
            gt_filename = f"{uuid.uuid4().hex[:8]}_{gt_file.filename}"
            ground_truth_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
            gt_file.save(ground_truth_path)
    
    try:
        participants = json.loads(participants_raw)
    except Exception:
        participants = []

    # Standardize court_side values: 'near_player' / 'far_player'
    # Role values: 'target_player' / 'opponent'
    # report_enabled: True / False
    formatted_participants = []
    if isinstance(participants, list):
        for p in participants:
            side_raw = str(p.get('court_side', 'near_player')).lower()
            norm_side = 'near_player' if ('near' in side_raw or 'bottom' in side_raw) else 'far_player'
            formatted_participants.append({
                'player_id': p.get('player_id'),
                'court_side': norm_side,
                'role': p.get('role', 'target_player'),
                'report_enabled': p.get('report_enabled', True)
            })

    # Case 1: 1 selected player -> Target player (report enabled) + Anonymous Opponent (tracking only)
    if len(formatted_participants) == 1:
        p0 = formatted_participants[0]
        p0['role'] = 'target_player'
        p0['report_enabled'] = True
        
        opp_side = 'far_player' if p0['court_side'] == 'near_player' else 'near_player'
        p1 = {
            'player_id': None,
            'court_side': opp_side,
            'role': 'opponent',
            'report_enabled': False
        }
        formatted_participants.append(p1)
    elif len(formatted_participants) == 2:
        # Case 2: 2 selected players -> Both target players (reports enabled)
        for p in formatted_participants:
            p['role'] = 'target_player'
            p['report_enabled'] = True
    elif len(formatted_participants) == 0:
        formatted_participants = [
            {'player_id': None, 'court_side': 'near_player', 'role': 'target_player', 'report_enabled': True},
            {'player_id': None, 'court_side': 'far_player', 'role': 'target_player', 'report_enabled': True}
        ]

    # Calculate report_enabled_roles for Stage 6 filtering
    report_enabled_roles = [p['court_side'] for p in formatted_participants if p.get('report_enabled')]

    # Always track both players in video stream for complete shot and shuttle context
    target_roles = ["Far Player", "Near Player"]

    import re
    raw_name = os.path.splitext(file.filename)[0]
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', raw_name)[:20]
    if not sanitized_name:
        sanitized_name = "video"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:4]
    job_id = f"job_{sanitized_name}_{now_str}_{short_uuid}"
    output_dir = os.path.join(BASE_DIR, "Match", f"session_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{file.filename}")
    file.save(save_path)
    
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    init_log = f"[{ts}] [System] Video '{file.filename}' uploaded. Initializing background pipeline worker..."
    database.create_job(job_id, "Saving video file...", init_log)

    JOBS[job_id] = {
        "status": "Processing",
        "stage_idx": 1,
        "stage_name": "Stage 1/7: Court Setup",
        "percent": 5,
        "detail": "Saving video file...",
        "logs": [init_log],
        "output_dir": output_dir,
        "session_id": None,
        "is_test_mode": test_mode
    }
    
    t = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, save_path, output_dir, target_roles, current_coach_id, file.filename, player_names_str, formatted_participants, report_enabled_roles, test_mode, ground_truth_path)
    )
    t.daemon = True
    t.start()

    return jsonify({"message": "Analysis job started", "job_id": job_id}), 202

@app.route('/api/analyze/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        job = database.get_job(job_id)
    if not job:
        return jsonify({"message": "Job not found"}), 404
    return jsonify(job), 200

@app.route('/api/analyze/cancel/<job_id>', methods=['POST', 'OPTIONS'])
@token_required
def cancel_analysis(current_coach_id, current_coach_name, job_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cancel_log = f"[{ts}] [Cancel] Analysis job cancelled by user."

    if job_id in JOBS:
        JOBS[job_id]["status"] = "Cancelled"
        JOBS[job_id]["detail"] = "Analysis cancelled by user."
        if "logs" in JOBS[job_id] and isinstance(JOBS[job_id]["logs"], list):
            JOBS[job_id]["logs"].append(cancel_log)

    database.cancel_job(job_id, "Analysis cancelled by user.")
    return jsonify({"message": "Job cancelled successfully", "job_id": job_id}), 200

import report_view_builder

def resolve_session_output_dir(session_id=None):
    match_dir = os.path.join(BASE_DIR, "Match")
    if session_id:
        try:
            job = database.get_job_by_session_id(session_id)
            if job and job.get("job_id"):
                j_dir = os.path.join(match_dir, f"session_{job['job_id']}")
                if os.path.exists(j_dir):
                    return j_dir
        except Exception as e:
            print(f"[Warning] Error resolving session dir for session {session_id}: {e}")

        subdirs = [
            os.path.join(match_dir, d) for d in os.listdir(match_dir) 
            if os.path.isdir(os.path.join(match_dir, d)) and d.startswith("session_")
        ]
        for sd in subdirs:
            if str(session_id) in sd:
                return sd

    # Fallback to latest subdirectory or root Match
    subdirs = [
        os.path.join(match_dir, d) for d in os.listdir(match_dir) 
        if os.path.isdir(os.path.join(match_dir, d)) and d.startswith("session_")
    ]
    if subdirs:
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return subdirs[0]

    return match_dir

def load_session_results(output_dir, session_data=None):
    import glob as _glob

    def _find_first(patterns):
        """Return first existing file matching any glob pattern."""
        for pat in patterns:
            matches = sorted(_glob.glob(pat), key=os.path.getmtime, reverse=True)
            if matches:
                return matches[0]
        return None

    # Shot predictions: match any *_shot_predictions.json or shot_predictions.json
    shots_path = _find_first([
        os.path.join(output_dir, "*_shot_predictions.json"),
        os.path.join(output_dir, "shot_predictions.json"),
    ])
    shots_data = []
    if shots_path:
        try:
            with open(shots_path, 'r') as f:
                shots_data = json.load(f)
        except Exception:
            pass

    # Biomechanical errors
    err_path = _find_first([
        os.path.join(output_dir, "*_biomechanical_errors.json"),
        os.path.join(output_dir, "biomechanical_errors.json"),
    ])
    err_data = []
    if err_path:
        try:
            with open(err_path, 'r') as f:
                err_data = json.load(f)
        except Exception:
            pass

    # Coaching feedback
    fb_path = _find_first([
        os.path.join(output_dir, "*_coaching_feedback.json"),
        os.path.join(output_dir, "coaching_feedback.json"),
    ])
    fb_data = []
    if fb_path:
        try:
            with open(fb_path, 'r') as f:
                fb_data = json.load(f)
        except Exception:
            pass

    # Frame keypoints
    kp_path = _find_first([
        os.path.join(output_dir, "*_frame_keypoints.json"),
        os.path.join(output_dir, "frame_keypoints.json"),
    ])
    frame_kp = None
    if kp_path:
        try:
            with open(kp_path, 'r') as f:
                frame_kp = json.load(f)
        except Exception:
            pass

    # Shuttle keypoints
    shuttle_path = _find_first([
        os.path.join(output_dir, "*_shuttle_keypoints.json"),
        os.path.join(output_dir, "shuttle_keypoints.json"),
    ])
    shuttle_kp = None
    if shuttle_path:
        try:
            with open(shuttle_path, 'r') as f:
                shuttle_kp = json.load(f)
        except Exception:
            pass

    participants = session_data.get("participants") if isinstance(session_data, dict) else []

    report_view = report_view_builder.build_report_view_model(
        session=session_data,
        participants=participants,
        shot_predictions=shots_data,
        biomechanical_errors=err_data,
        coaching_feedback=fb_data,
        frame_keypoints=frame_kp,
        shuttle_keypoints=shuttle_kp,
        output_dir=output_dir
    )

    # Attach backward compatible fields for seamless frontend transition
    report_view["shot_predictions"] = [
        s for s in (shots_data if isinstance(shots_data, list) else [])
        if isinstance(s, dict) and s.get("shot_type", "").lower() != "unknown"
    ]
    report_view["biomechanical_errors"] = report_view.get("players", [{}])[0].get("detected_errors", []) if report_view.get("players") else []
    report_view["coaching_feedback"] = report_view.get("players", [{}])[0].get("coaching_feedback", []) if report_view.get("players") else []
    report_view["group_diagnostics"] = fb_data.get("group_diagnostics", {}) if isinstance(fb_data, dict) else {}

    return report_view

@app.route('/api/report/latest', methods=['GET'])
@token_required
def get_latest_report(current_coach_id, current_coach_name):
    target_dir = resolve_session_output_dir(None)
    res = load_session_results(target_dir)
    res["output_dir"] = target_dir
    return jsonify(res), 200

@app.route('/api/report/<int:session_id>', methods=['GET'])
@token_required
def get_session_report(current_coach_id, current_coach_name, session_id):
    session = database.get_session_by_id(session_id)
    target_dir = resolve_session_output_dir(session_id)
    res = load_session_results(target_dir, session_data=session)
    res["output_dir"] = target_dir
    return jsonify(res), 200

@app.route('/api/test_report/<int:session_id>', methods=['GET'])
@token_required
def get_test_report(current_coach_id, current_coach_name, session_id):
    session = database.get_session_by_id(session_id)
    target_dir = resolve_session_output_dir(session_id)
    
    test_results = {}
    test_results_path = os.path.join(target_dir, "test_results.json")
    if os.path.exists(test_results_path):
        try:
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load test_results.json: {e}")
            
    res = load_session_results(target_dir, session_data=session)
    res["test_results"] = test_results
    return jsonify(res), 200

def find_first_matching(patterns):
    """Return first existing file matching any glob pattern."""
    import glob as _glob
    for pat in patterns:
        matches = sorted(_glob.glob(pat), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    return None

def find_raw_input_video(output_dir):
    """Locate unannotated raw input video file for clean clip extraction."""
    import shutil
    import glob as _glob

    # 1. Check for raw_input copy inside output_dir
    direct_candidates = [
        os.path.join(output_dir, "raw_input.mp4"),
        os.path.join(output_dir, "input.mp4"),
        os.path.join(output_dir, "input_video.mp4"),
        os.path.join(output_dir, "original.mp4"),
    ]
    for cand in direct_candidates:
        if os.path.exists(cand) and os.path.getsize(cand) > 1000:
            return cand

    # 2. Check Input/ directory matching job_id or session folder key
    input_dir = os.path.join(BASE_DIR, "Input")
    if os.path.exists(input_dir):
        folder_name = os.path.basename(output_dir)
        job_key = folder_name.replace("session_", "")
        matches = sorted(_glob.glob(os.path.join(input_dir, f"*{job_key}*.mp4")), key=os.path.getmtime, reverse=True)
        if matches and os.path.exists(matches[0]):
            try:
                shutil.copy2(matches[0], os.path.join(output_dir, "raw_input.mp4"))
            except Exception:
                pass
            return matches[0]

        # Check for unannotated input files in Input/
        all_inputs = sorted(_glob.glob(os.path.join(input_dir, "*.mp4")), key=os.path.getmtime, reverse=True)
        for input_cand in all_inputs:
            bn = os.path.basename(input_cand)
            if not bn.endswith("_annotated.mp4") and not bn.endswith("_predict.mp4") and not bn.startswith("shot_clip_"):
                try:
                    shutil.copy2(input_cand, os.path.join(output_dir, "raw_input.mp4"))
                except Exception:
                    pass
                return input_cand

    # 3. Any non-annotated mp4 in output_dir
    all_mp4s = _glob.glob(os.path.join(output_dir, "*.mp4"))
    for mp4 in all_mp4s:
        bn = os.path.basename(mp4)
        if not bn.endswith("_annotated.mp4") and not bn.startswith("shot_clip_") and bn != "output.mp4":
            return mp4

    # 4. Fallback to annotated video if raw video cannot be found
    return find_first_matching([
        os.path.join(output_dir, "*_annotated.mp4"),
        os.path.join(output_dir, "output.mp4"),
        os.path.join(output_dir, "*.mp4")
    ])

def generate_shot_clip(output_dir, shot_id):
    """
    Extracts exact frame segment [start_frame .. end_frame] for shot_id
    from the RAW UNANNOTATED video and builds a shot_clip_{shot_id}.mp4 file inside output_dir.
    """
    clip_filename = f"shot_clip_{shot_id}.mp4"
    clip_path = os.path.join(output_dir, clip_filename)
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
        return clip_path

    pred_path = find_first_matching([
        os.path.join(output_dir, "*_shot_predictions.json"),
        os.path.join(output_dir, "shot_predictions.json")
    ])
    if not pred_path or not os.path.exists(pred_path):
        return None

    try:
        with open(pred_path, encoding="utf-8") as f:
            preds = json.load(f)

        shot_info = next((s for s in preds if str(s.get("shot_id")) == str(shot_id)), None)
        if not shot_info:
            return None

        start_frame = shot_info.get("start_frame", 1)
        end_frame = shot_info.get("end_frame", 30)

        # Buffer frames slightly (+/- 5 frames for context)
        start_frame = max(1, start_frame - 5)
        end_frame = end_frame + 5

        video_src = find_raw_input_video(output_dir)
        if not video_src or not os.path.exists(video_src):
            return None

        cap = cv2.VideoCapture(video_src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 4x Slow Motion: divide FPS by 4.0 so output video plays 4x slower for coach error identification
        clip_fps = max(4.0, round(fps / 4.0, 2))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            return None

        out = None
        # Try native MSMF H.264 on Windows first for guaranteed HTML5 browser compatibility
        test_msmf = cv2.VideoWriter(clip_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), clip_fps, (width, height))
        if test_msmf.isOpened():
            out = test_msmf
        else:
            # Fallback to default API with H.264 / avc1 / mp4v
            for fourcc_str in ['avc1', 'H264', 'mp4v']:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                test_out = cv2.VideoWriter(clip_path, fourcc, clip_fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    break

        if not out or not out.isOpened():
            cap.release()
            return None

        curr_frame = 1
        written_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= curr_frame <= end_frame:
                out.write(frame)
                written_count += 1
            if curr_frame > end_frame:
                break
            curr_frame += 1

        cap.release()
        out.release()

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000 and written_count > 0:
            return clip_path
        return None
    except Exception as e:
        print(f"[Error] Failed to generate shot clip for shot {shot_id}: {e}")
        return None

@app.route('/api/shot_clip/<session_id>/<int:shot_id>', methods=['GET'])
def get_shot_clip(session_id, shot_id):
    try:
        sid = None if str(session_id) == 'latest' else session_id
        target_dir = resolve_session_output_dir(sid)
        if not os.path.exists(target_dir):
            return jsonify({"message": "Session directory not found"}), 404

        clip_path = generate_shot_clip(target_dir, shot_id)
        if clip_path and os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            return send_file(clip_path, mimetype='video/mp4')

        # Fallback to serving the main raw video (not annotated)
        full_video = find_raw_input_video(target_dir)
        if not full_video or not os.path.exists(full_video):
            full_video = find_first_matching([
                os.path.join(target_dir, "*.mp4"),
            ])
        if full_video and os.path.exists(full_video):
            return send_file(full_video, mimetype='video/mp4')
        return jsonify({"message": "Shot clip video not found"}), 404
    except Exception as e:
        print(f"[Error] get_shot_clip error: {e}")
        return jsonify({"message": str(e)}), 500

@app.route('/api/frame/<session_id>/<int:frame_num>', methods=['GET'])
def get_frame_image(session_id, frame_num):
    """
    Serves exact evidence image frame for a given session and frame number.
    Extracts frame on-the-fly using OpenCV if PNG does not already exist on disk.
    """
    try:
        sid = None if str(session_id) == 'latest' else session_id
        target_dir = resolve_session_output_dir(sid)
        if not target_dir or not os.path.exists(target_dir):
            return jsonify({"message": "Session directory not found"}), 404

        # Check all possible frame filename patterns (raw frames only — skip annotated)
        candidates = [
            f"frame_{frame_num:04d}.png",
            f"frame_{frame_num}.png",
            f"frame_{frame_num:03d}.png",
            f"frame_{frame_num:05d}.png",
            f"shot_frame_{frame_num}.png"
        ]

        for c_name in candidates:
            # Direct in output_dir (raw frames only)
            c_path = os.path.join(target_dir, c_name)
            if os.path.exists(c_path) and os.path.getsize(c_path) > 1000:
                return send_file(c_path, mimetype='image/png')

        # Dynamically extract frame using OpenCV from RAW input video only (no annotated overlays)
        video_src = find_raw_input_video(target_dir)
        if not video_src or not os.path.exists(video_src):
            # Last resort: any mp4 that is NOT annotated
            all_mp4s = find_first_matching([
                os.path.join(target_dir, "*.mp4"),
            ])
            if all_mp4s and "_annotated" not in os.path.basename(all_mp4s).lower():
                video_src = all_mp4s

        if video_src and os.path.exists(video_src):
            cap = cv2.VideoCapture(video_src)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
            ret, frame = cap.read()
            cap.release()
            if ret:
                save_name = f"frame_{frame_num:04d}.png"
                save_path = os.path.join(target_dir, save_name)
                cv2.imwrite(save_path, frame)
                return send_file(save_path, mimetype='image/png')

        return jsonify({"message": "Frame image not found"}), 404
    except Exception as e:
        print(f"[Error] get_frame_image error: {e}")
        return jsonify({"message": str(e)}), 500

@app.route('/api/media/<path:filename>', methods=['GET'])
def serve_media(filename):
    """Serves media files like annotated MP4s or output frames."""
    match_dir = os.path.join(BASE_DIR, "Match")
    subdirs = [os.path.join(match_dir, d) for d in os.listdir(match_dir) if os.path.isdir(os.path.join(match_dir, d)) and d.startswith("session_")]
    if subdirs:
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    possible_dirs = []
    for sd in subdirs:
        possible_dirs.extend([sd, os.path.join(sd, "annotated_frames")])
    possible_dirs.extend([
        BASE_DIR,
        match_dir,
        os.path.join(match_dir, "annotated_frames"),
        os.path.join(BASE_DIR, "Input")
    ])

    for d in possible_dirs:
        file_path = os.path.join(d, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return send_file(file_path)
    return jsonify({"message": "File not found"}), 404





# ─────────────────────────────────────────────────────────────────────────────
# Log File Endpoints  (no auth required — coaches debug from UI)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/logs', methods=['GET'])
def list_logs():
    """
    Returns a list of all available job log files with metadata.
    GET /api/logs
    """
    try:
        files = pl.list_log_files()
        return jsonify({"logs": files, "logs_dir": os.path.join(BASE_DIR, "logs")}), 200
    except Exception as e:
        return jsonify({"message": f"Error listing logs: {str(e)}"}), 500


@app.route('/api/logs/<job_id>', methods=['GET'])
def get_job_log(job_id):
    """
    Returns the full log for a given job_id as plain text.
    GET /api/logs/<job_id>
    Optional query param: ?format=json  → wraps in a JSON envelope
    """
    try:
        log_path = pl.get_log_path(job_id)
        if not os.path.exists(log_path):
            return jsonify({"message": f"Log file not found for job: {job_id}"}), 404

        fmt = request.args.get("format", "text")
        if fmt == "json":
            content = pl.PipelineLogger(job_id, redirect_stdout=False, redirect_stderr=False).read_all()
            return jsonify({"job_id": job_id, "log": content}), 200
        else:
            return send_file(log_path, mimetype="text/plain", as_attachment=False)
    except Exception as e:
        return jsonify({"message": f"Error reading log: {str(e)}"}), 500


@app.route('/api/logs/<job_id>/tail', methods=['GET'])
def get_job_log_tail(job_id):
    """
    Returns the last N lines of the log for quick error inspection.
    GET /api/logs/<job_id>/tail?lines=100
    """
    try:
        log_path = pl.get_log_path(job_id)
        if not os.path.exists(log_path):
            return jsonify({"message": f"Log file not found for job: {job_id}"}), 404

        n = int(request.args.get("lines", 100))
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        tail = "".join(all_lines[-n:])

        return jsonify({"job_id": job_id, "lines_returned": min(n, len(all_lines)), "tail": tail}), 200
    except Exception as e:
        return jsonify({"message": f"Error reading log tail: {str(e)}"}), 500


@app.route('/api/logs/<job_id>/errors', methods=['GET'])
def get_job_log_errors(job_id):
    """
    Returns only ERROR and FAILED lines from the log file.
    Useful for the frontend to surface API failures without showing full raw log.
    GET /api/logs/<job_id>/errors
    """
    try:
        log_path = pl.get_log_path(job_id)
        if not os.path.exists(log_path):
            return jsonify({"job_id": job_id, "errors": []}), 200

        error_lines = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if any(tag in stripped for tag in ["[ERROR]", "[STDERR]", "FAILED", "Error", "Exception", "Traceback", "API Error", "quota", "credit", "api key"]):
                    error_lines.append(stripped)

        return jsonify({"job_id": job_id, "error_count": len(error_lines), "errors": error_lines}), 200
    except Exception as e:
        return jsonify({"message": f"Error extracting errors from log: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Badminton Analysis Flask API on http://localhost:{port}")
    app.run(debug=True, use_reloader=False, port=port, host='0.0.0.0')
