"""
pipeline_logger.py
==================
Per-job file logger for the Coach Vision analysis pipeline.

Creates a dedicated log file under  logs/<job_id>.log  for every analysis job
and redirects all print(), Python warnings, and explicit log_event() calls into
that file. The frontend can retrieve the log via:

    GET /api/logs/<job_id>          → returns full log text
    GET /api/logs/<job_id>/tail     → returns last 100 lines

Usage inside flask_api.py / run_pipeline_job:
    from pipeline_logger import PipelineLogger
    logger = PipelineLogger(job_id)
    with logger:           # redirects sys.stdout and sys.stderr
        run_pipeline()
    logger.log_event(stage_name, detail, level="INFO")
    logger.log_error("LLM API key quota exhausted", exc=e)
"""

import os
import sys
import datetime
import traceback
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe tee-writer: writes to BOTH the original stream AND the log file
# ─────────────────────────────────────────────────────────────────────────────
class _TeeWriter:
    """Writes simultaneously to *original* stream and *log_file*."""

    def __init__(self, original, log_file, prefix=""):
        self._orig = original
        self._file = log_file
        self._prefix = prefix
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return
        with self._lock:
            # Write raw to original console
            try:
                self._orig.write(text)
                self._orig.flush()
            except Exception:
                pass
            # Write timestamped to log file
            try:
                if text.strip():
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    tagged = f"[{ts}]{self._prefix} {text.rstrip()}\n"
                else:
                    tagged = text
                self._file.write(tagged)
                self._file.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Logger class
# ─────────────────────────────────────────────────────────────────────────────
class PipelineLogger:
    """
    Manages a per-job log file and optionally redirects sys.stdout / sys.stderr.

    Parameters
    ----------
    job_id : str
        Unique job identifier (used as the log filename).
    redirect_stdout : bool
        If True (default), replaces sys.stdout with a tee that mirrors to file.
    redirect_stderr : bool
        If True (default), replaces sys.stderr with a tee that mirrors to file.
    """

    def __init__(self, job_id: str, redirect_stdout: bool = True, redirect_stderr: bool = True):
        self.job_id = job_id
        self.log_path = os.path.join(LOGS_DIR, f"{job_id}.log")
        self._redirect_stdout = redirect_stdout
        self._redirect_stderr = redirect_stderr

        self._orig_stdout = None
        self._orig_stderr = None
        self._log_file = None
        self._lock = threading.Lock()

        # Open the log file immediately
        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._write_header()

    def _write_header(self):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"\n{'=' * 70}\n"
            f"  Coach Vision Analysis Log\n"
            f"  Job ID : {self.job_id}\n"
            f"  Started: {ts}\n"
            f"{'=' * 70}\n"
        )
        self._log_file.write(header)
        self._log_file.flush()

    # ── Context-manager support ──────────────────────────────────────────────
    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        if self._redirect_stdout:
            sys.stdout = _TeeWriter(self._orig_stdout, self._log_file, "[STDOUT]")
        if self._redirect_stderr:
            sys.stderr = _TeeWriter(self._orig_stderr, self._log_file, "[STDERR]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log any uncaught exception that escaped the pipeline
        if exc_type is not None:
            self.log_error(f"Uncaught pipeline exception: {exc_val}", exc=exc_val)
        self._restore_streams()
        return False  # do not suppress exception

    def _restore_streams(self):
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

    # ── Explicit event logging ───────────────────────────────────────────────
    def log_event(self, stage_name: str, detail: str = "", level: str = "INFO"):
        """Log a pipeline stage progress event."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] [{stage_name}] {detail}\n"
        with self._lock:
            try:
                self._log_file.write(line)
                self._log_file.flush()
            except Exception:
                pass

    def log_error(self, message: str, exc: Exception = None):
        """Log an error with optional full traceback."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with self._lock:
            try:
                self._log_file.write(f"\n[{ts}] [ERROR] {message}\n")
                if exc is not None:
                    tb = traceback.format_exc()
                    self._log_file.write(f"  Traceback:\n{tb}\n")
                self._log_file.flush()
            except Exception:
                pass

    def log_api_error(self, service_name: str, exc: Exception = None):
        """Convenience wrapper to flag API / external service failures."""
        detail = str(exc) if exc else "Unknown API error"
        self.log_error(
            f"External service failure [{service_name}]: {detail}"
            f"\n  >>> This may be caused by: expired API key, quota exceeded, or network timeout.",
            exc=exc,
        )

    def close(self):
        """Flush and close the log file. Restores stdout/stderr if hijacked."""
        self._restore_streams()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._log_file.write(f"\n[{ts}] [END] Log closed.\n{'=' * 70}\n")
            self._log_file.flush()
            self._log_file.close()
        except Exception:
            pass

    # ── Read helpers (for API endpoint) ─────────────────────────────────────
    def read_all(self) -> str:
        """Return the full log text."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def read_tail(self, n: int = 100) -> str:
        """Return the last *n* lines of the log."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except FileNotFoundError:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Registry: one logger per active job (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_registry: dict[str, PipelineLogger] = {}
_registry_lock = threading.Lock()


def create_logger(job_id: str) -> PipelineLogger:
    """Create, register and return a logger for *job_id*."""
    logger = PipelineLogger(job_id, redirect_stdout=False, redirect_stderr=False)
    with _registry_lock:
        _registry[job_id] = logger
    return logger


def get_logger(job_id: str) -> PipelineLogger | None:
    """Return the active logger for *job_id*, or None."""
    return _registry.get(job_id)


def close_logger(job_id: str):
    """Close and deregister the logger for *job_id*."""
    with _registry_lock:
        logger = _registry.pop(job_id, None)
    if logger:
        logger.close()


def get_log_path(job_id: str) -> str:
    """Return the filesystem path for *job_id*.log."""
    return os.path.join(LOGS_DIR, f"{job_id}.log")


def list_log_files() -> list[dict]:
    """Return a list of all log file metadata (job_id, size, modified)."""
    results = []
    for fname in os.listdir(LOGS_DIR):
        if fname.endswith(".log"):
            fpath = os.path.join(LOGS_DIR, fname)
            stat = os.stat(fpath)
            results.append({
                "job_id": fname[:-4],
                "filename": fname,
                "size_bytes": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    results.sort(key=lambda x: x["modified"], reverse=True)
    return results
