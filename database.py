import os
import sys
import json
import datetime
import sqlite3
import hashlib

# PostgreSQL configuration from environment
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "badminton_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "20201220")

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "badminton.db")

USE_SQLITE = False

def get_connection():
    global USE_SQLITE
    if not USE_SQLITE:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=2
            )
            return conn, "pg"
        except Exception as e:
            print(f"[Database Notice] Could not connect to PostgreSQL ({e}). Falling back to local SQLite database at {SQLITE_DB_PATH}")
            USE_SQLITE = True

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def init_db():
    conn, mode = get_connection()
    cursor = conn.cursor()
    
    if mode == "pg":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coaches (
                coach_id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id SERIAL PRIMARY KEY,
                coach_id INT REFERENCES coaches(coach_id) ON DELETE CASCADE,
                full_name VARCHAR(255) NOT NULL,
                contact_number VARCHAR(50),
                age INT,
                age_group VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id SERIAL PRIMARY KEY,
                coach_id INT REFERENCES coaches(coach_id) ON DELETE CASCADE,
                video_name VARCHAR(255) NOT NULL,
                players VARCHAR(255) NOT NULL,
                status VARCHAR(50) DEFAULT 'Completed',
                technique_badge VARCHAR(50) DEFAULT 'Good Technique',
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_players (
                session_player_id SERIAL PRIMARY KEY,
                session_id INT REFERENCES sessions(session_id) ON DELETE CASCADE,
                player_id INT REFERENCES players(player_id) ON DELETE CASCADE,
                court_side VARCHAR(50) NOT NULL,
                role VARCHAR(50) DEFAULT 'target_player',
                report_enabled BOOLEAN DEFAULT TRUE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id VARCHAR(100) PRIMARY KEY,
                status VARCHAR(50) DEFAULT 'Processing',
                stage_idx INT DEFAULT 1,
                stage_name VARCHAR(255) DEFAULT 'Stage 1/7: Initializing',
                percent INT DEFAULT 5,
                detail TEXT DEFAULT 'Starting pipeline worker...',
                logs TEXT DEFAULT '[]',
                session_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coaches (
                coach_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                coach_id INTEGER REFERENCES coaches(coach_id) ON DELETE CASCADE,
                full_name TEXT NOT NULL,
                contact_number TEXT,
                age INTEGER,
                age_group TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                coach_id INTEGER REFERENCES coaches(coach_id) ON DELETE CASCADE,
                video_name TEXT NOT NULL,
                players TEXT NOT NULL,
                status TEXT DEFAULT 'Completed',
                technique_badge TEXT DEFAULT 'Good Technique',
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_players (
                session_player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
                player_id INTEGER REFERENCES players(player_id) ON DELETE CASCADE,
                court_side TEXT NOT NULL,
                role TEXT DEFAULT 'target_player',
                report_enabled INTEGER DEFAULT 1
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'Processing',
                stage_idx INTEGER DEFAULT 1,
                stage_name TEXT DEFAULT 'Stage 1/7: Initializing',
                percent INTEGER DEFAULT 5,
                detail TEXT DEFAULT 'Starting pipeline worker...',
                logs TEXT DEFAULT '[]',
                session_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
    # Check and add missing columns to session_players for existing databases
    try:
        if mode == "pg":
            cursor.execute("ALTER TABLE session_players ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'target_player';")
            cursor.execute("ALTER TABLE session_players ADD COLUMN IF NOT EXISTS report_enabled BOOLEAN DEFAULT TRUE;")
        else:
            cursor.execute("PRAGMA table_info(session_players);")
            columns = [row[1] for row in cursor.fetchall()]
            if 'role' not in columns:
                cursor.execute("ALTER TABLE session_players ADD COLUMN role TEXT DEFAULT 'target_player';")
            if 'report_enabled' not in columns:
                cursor.execute("ALTER TABLE session_players ADD COLUMN report_enabled INTEGER DEFAULT 1;")
    except Exception as e:
        print(f"[Database Notice] Migration notice for session_players: {e}")
        
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def create_coach(name: str, username: str, email: str, password: str) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    password_hash = hash_password(password)
    ph = "%s" if mode == "pg" else "?"
    
    try:
        cursor.execute(
            f"INSERT INTO coaches (name, username, email, password_hash) VALUES ({ph}, {ph}, {ph}, {ph})",
            (name, username, email, password_hash)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to create coach: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def coach_exists(username: str, email: str) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    
    try:
        cursor.execute(f"SELECT 1 FROM coaches WHERE username = {ph} OR email = {ph}", (username, email))
        row = cursor.fetchone()
        return row is not None
    except Exception as e:
        print(f"[Database Error] Failed to check coach existence: {e}")
        return False
    finally:
        conn.close()

def get_coach(username: str, email: str) -> dict:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
            
        cursor.execute(
            f"SELECT coach_id, name, username, email, password_hash FROM coaches WHERE username = {ph} AND email = {ph}",
            (username, email)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"[Database Error] Failed to retrieve coach: {e}")
        return None
    finally:
        conn.close()

def verify_coach_login(username: str, email: str, password: str) -> dict:
    coach = get_coach(username, email)
    if not coach:
        return None
    
    hashed_input = hash_password(password)
    if coach["password_hash"] == hashed_input:
        return {
            "coach_id": coach["coach_id"],
            "name": coach["name"],
            "username": coach["username"],
            "email": coach["email"]
        }
    return None

def create_player(coach_id: int, full_name: str, contact_number: str, age: int, age_group: str) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(
            f"INSERT INTO players (coach_id, full_name, contact_number, age, age_group) VALUES ({ph}, {ph}, {ph}, {ph}, {ph})",
            (coach_id, full_name, contact_number, age, age_group)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to create player: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_players_by_coach(coach_id: int) -> list:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(
            f"SELECT player_id, full_name, contact_number, age, age_group, created_at FROM players WHERE coach_id = {ph} ORDER BY created_at DESC",
            (coach_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[Database Error] Failed to retrieve players: {e}")
        return []
    finally:
        conn.close()

def delete_player(player_id: int, coach_id: int) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(
            f"DELETE FROM players WHERE player_id = {ph} AND coach_id = {ph}",
            (player_id, coach_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to delete player: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def create_session(coach_id: int, video_name: str, players: str, status: str, technique_badge: str, summary: str, participants: list = None) -> int:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            cursor.execute(
                f"INSERT INTO sessions (coach_id, video_name, players, status, technique_badge, summary) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}) RETURNING session_id",
                (coach_id, video_name, players, status, technique_badge, summary)
            )
            session_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"INSERT INTO sessions (coach_id, video_name, players, status, technique_badge, summary) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
                (coach_id, video_name, players, status, technique_badge, summary)
            )
            session_id = cursor.lastrowid
        
        if participants:
            for p in participants:
                raw_pid = p.get('player_id')
                p_id = int(raw_pid) if raw_pid is not None and str(raw_pid).isdigit() else None
                side = p.get('court_side', 'near_player')
                role = p.get('role', 'target_player')
                is_rep = bool(p.get('report_enabled', True))
                report_enabled = is_rep if mode == "pg" else (1 if is_rep else 0)

                cursor.execute(
                    f"INSERT INTO session_players (session_id, player_id, court_side, role, report_enabled) VALUES ({ph}, {ph}, {ph}, {ph}, {ph})",
                    (session_id, p_id, side, role, report_enabled)
                )
        conn.commit()
        return session_id
    except Exception as e:
        print(f"[Database Error] Failed to create session: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_sessions_by_coach(coach_id: int) -> list:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()

        # Fetch all sessions for this coach
        cursor.execute(
            f"SELECT session_id, video_name, players, status, technique_badge, summary, created_at FROM sessions WHERE coach_id = {ph} ORDER BY created_at DESC",
            (coach_id,)
        )
        rows = cursor.fetchall()
        sessions = [dict(row) for row in rows]

        # For each session, also fetch its participants from session_players
        for sess in sessions:
            sid = sess["session_id"]
            try:
                cursor.execute(
                    f"""SELECT sp.session_player_id, sp.player_id, sp.court_side, sp.role, sp.report_enabled,
                               COALESCE(p.full_name, 'Unknown Player') AS full_name
                        FROM session_players sp
                        LEFT JOIN players p ON sp.player_id = p.player_id
                        WHERE sp.session_id = {ph}""",
                    (sid,)
                )
                p_rows = cursor.fetchall()
                sess["participants"] = [dict(r) for r in p_rows]
            except Exception as pe:
                print(f"[Database Warning] Could not fetch participants for session {sid}: {pe}")
                sess["participants"] = []

        return sessions
    except Exception as e:
        print(f"[Database Error] Failed to retrieve sessions: {e}")
        return []
    finally:
        conn.close()

def get_session_by_id(session_id: int) -> dict:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(
            f"SELECT session_id, coach_id, video_name, players, status, technique_badge, summary, created_at FROM sessions WHERE session_id = {ph}",
            (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        sess_dict = dict(row)

        # Fetch associated participant records
        cursor.execute(
            f"""SELECT sp.session_player_id, sp.player_id, sp.court_side, sp.role, sp.report_enabled, p.full_name 
                FROM session_players sp 
                LEFT JOIN players p ON sp.player_id = p.player_id 
                WHERE sp.session_id = {ph}""",
            (session_id,)
        )
        p_rows = cursor.fetchall()
        sess_dict["participants"] = [dict(r) for r in p_rows]
        return sess_dict
    except Exception as e:
        print(f"[Database Error] Failed to retrieve session {session_id}: {e}")
        return None
    finally:
        conn.close()

def delete_session(session_id: int, coach_id: int) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(f"DELETE FROM sessions WHERE session_id = {ph} AND coach_id = {ph}", (session_id, coach_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to delete session: {e}")
        return False
    finally:
        conn.close()

def create_job(job_id: str, detail: str = "Starting pipeline worker...", initial_log: str = None) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    logs_json = json.dumps([initial_log]) if initial_log else "[]"
    try:
        cursor.execute(
            f"INSERT INTO jobs (job_id, status, stage_idx, stage_name, percent, detail, logs) VALUES ({ph}, 'Processing', 1, 'Stage 1/7: Initializing', 5, {ph}, {ph})",
            (job_id, detail, logs_json)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to create job {job_id}: {e}")
        return False
    finally:
        conn.close()

def update_job_progress(job_id: str, stage_idx: int, stage_name: str, percent: int, detail: str, log_msg: str = None) -> bool:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        
        cursor.execute(f"SELECT logs FROM jobs WHERE job_id = {ph}", (job_id,))
        row = cursor.fetchone()
        logs_list = []
        if row:
            raw = row['logs'] if mode == "pg" else row[0]
            if raw:
                try:
                    logs_list = json.loads(raw)
                except Exception:
                    logs_list = []

        if log_msg or detail:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] [{stage_name}] {log_msg or detail}"
            if len(logs_list) == 0 or logs_list[-1] != line:
                logs_list.append(line)

        cursor.execute(
            f"UPDATE jobs SET stage_idx = {ph}, stage_name = {ph}, percent = {ph}, detail = {ph}, logs = {ph} WHERE job_id = {ph}",
            (stage_idx, stage_name, percent, detail, json.dumps(logs_list), job_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to update job {job_id}: {e}")
        return False
    finally:
        conn.close()

def complete_job(job_id: str, session_id: int) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(
            f"UPDATE jobs SET status = 'Completed', percent = 100, stage_name = 'Analysis Complete', session_id = {ph} WHERE job_id = {ph}",
            (session_id, job_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to complete job {job_id}: {e}")
        return False
    finally:
        conn.close()

def fail_job(job_id: str, error_detail: str) -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(
            f"UPDATE jobs SET status = 'Failed', detail = {ph} WHERE job_id = {ph}",
            (error_detail, job_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to set fail job {job_id}: {e}")
        return False
    finally:
        conn.close()

def cancel_job(job_id: str, detail: str = "Analysis cancelled by user") -> bool:
    conn, mode = get_connection()
    cursor = conn.cursor()
    ph = "%s" if mode == "pg" else "?"
    try:
        cursor.execute(
            f"UPDATE jobs SET status = 'Cancelled', detail = {ph} WHERE job_id = {ph}",
            (detail, job_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to cancel job {job_id}: {e}")
        return False
    finally:
        conn.close()

def get_job(job_id: str) -> dict:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(f"SELECT job_id, status, stage_idx, stage_name, percent, detail, logs, session_id, created_at FROM jobs WHERE job_id = {ph}", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        job_dict = dict(row)
        if isinstance(job_dict.get('logs'), str):
            try:
                job_dict['logs'] = json.loads(job_dict['logs'])
            except Exception:
                job_dict['logs'] = []
        return job_dict
    except Exception as e:
        print(f"[Database Error] Failed to get job {job_id}: {e}")
        return None
    finally:
        conn.close()

def get_job_by_session_id(session_id: int) -> dict:
    conn, mode = get_connection()
    ph = "%s" if mode == "pg" else "?"
    try:
        if mode == "pg":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(f"SELECT job_id, status, stage_idx, stage_name, percent, detail, session_id, created_at FROM jobs WHERE session_id = {ph} ORDER BY created_at DESC", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    except Exception as e:
        print(f"[Database Error] Failed to get job by session_id {session_id}: {e}")
        return None
    finally:
        conn.close()
