"""
Smart Workplace Occupancy & Employee Flow Analytics Platform
Backend – Flask + OpenCV-contrib LBPH face recognition
"""
import base64
import io
import os
import sqlite3
import threading
import time
import base64
import json
import logging
from datetime import datetime, date, timedelta
from functools import wraps
from pathlib import Path
import math

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Optional CV imports ────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    cv2 = None
    np = None
    CV2_OK = False

SFACE_OK = False
YUNET_OK = False
if CV2_OK:
    try:
        import onnxruntime
        SFACE_OK = True
        YUNET_OK = True
    except ImportError:
        pass

from flask import (
    Flask, Response, flash, jsonify, redirect,
    render_template, request, send_file, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "employees"
DB_PATH    = DATA_DIR / "smartwork.db"
MODEL_PATH = DATA_DIR / "face_model.yml"
SFACE_MODEL_PATH = DATA_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_MODEL_PATH = DATA_DIR / "face_detection_yunet_2023mar.onnx"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

# ── Business config ────────────────────────────────────────────────────────────
WORKDAY_START  = "08:30"
WORKDAY_END    = "18:00"
LATE_AFTER     = "09:30"
COOLDOWN_SECS  = 90
SFACE_THRESHOLD = 0.363  # Cosine distance threshold for SFace

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SMARTWORK_SECRET", "sw-demo-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024 * 1024
app.config["MAX_FORM_PARTS_SIZE"] = 64 * 1024 * 1024

# ── Shared mutable state (protected by locks) ──────────────────────────────────
_cam_lock   = threading.Lock()
_model_lock = threading.Lock()
_face_detector   = None
_face_recognizer = None
_model_trained   = False
_is_training     = False
_known_embeddings = [] # List of tuples (employee_id, embedding)
_id_to_name: dict[int, str] = {}
_last_seen: dict[int, datetime] = {}
_last_status: dict[int, str] = {}
_recent_detections: list[dict] = []   # ring-buffer, newest first


# ══════════════════════════════════════════════════════════════════════════════
#  DB helpers
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    ensure_dirs()
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'admin',
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS employees (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name  TEXT NOT NULL,
                department TEXT NOT NULL,
                position   TEXT NOT NULL,
                email      TEXT UNIQUE NOT NULL,
                photo_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS employee_photos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                photo_path  TEXT    NOT NULL,
                label       TEXT    DEFAULT 'reference',
                created_at  TEXT    NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                event_type  TEXT    NOT NULL CHECK(event_type IN ('entry','exit')),
                event_time  TEXT    NOT NULL,
                confidence  REAL    DEFAULT 0,
                source      TEXT    NOT NULL DEFAULT 'webcam',
                FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER,
                alert_type  TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                severity    TEXT    NOT NULL DEFAULT 'medium',
                created_at  TEXT    NOT NULL,
                resolved    INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE SET NULL
            );
        """)
        if not db.execute("SELECT id FROM users WHERE email=?", ("admin@smartwork.local",)).fetchone():
            db.execute(
                "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                ("Administrateur", "admin@smartwork.local",
                 generate_password_hash("admin123"), "admin", datetime.now().isoformat()),
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Auth decorators
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def login_required(fn):
    @wraps(fn)
    def _w(*a, **kw):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return _w


def admin_required(fn):
    @wraps(fn)
    def _w(*a, **kw):
        if session.get("role") != "admin":
            flash("Accès réservé aux administrateurs.", "warning")
            return redirect(url_for("dashboard"))
        return fn(*a, **kw)
    return _w


# ══════════════════════════════════════════════════════════════════════════════
#  Face-recognition helpers (SFace)
# ══════════════════════════════════════════════════════════════════════════════

def init_face_recognition():
    global _face_detector, _face_recognizer
    if not CV2_OK or not SFACE_OK or not YUNET_OK:
        return
    if YUNET_MODEL_PATH.exists():
        _face_detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000
        )
    if SFACE_MODEL_PATH.exists():
        _face_recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config=""
        )

def _get_face_embedding(img):
    """Extract face embedding using YuNet and SFace."""
    if _face_detector is None or _face_recognizer is None:
        return None
    
    height, width, _ = img.shape
    _face_detector.setInputSize((width, height))
    _, faces = _face_detector.detect(img)
    
    if faces is None or len(faces) == 0:
        return None
    
    # Get the largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    face = faces[0]
    
    aligned_face = _face_recognizer.alignCrop(img, face)
    embedding = _face_recognizer.feature(aligned_face)
    return embedding, face

def process_and_save_photos(employee_id, original_photo_path):
    """Save the initial uploaded photo path, then auto-retrain."""
    # The original_photo_path is already saved by the route if it's an upload.
    threading.Thread(target=train_model, daemon=True).start()

def process_burst_photos(employee_id, base64_list):
    """Process a list of 16 base64 photos from the webcam burst capture."""
    now = datetime.now()
    with get_db() as db:
        for i, b64_str in enumerate(base64_list):
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            try:
                img_data = base64.b64decode(b64_str)
                filename = f"{int(now.timestamp())}_{employee_id}_burst_{i}.jpg"
                filepath = UPLOAD_DIR / filename
                with open(filepath, "wb") as f:
                    f.write(img_data)
                
                db.execute(
                    "INSERT INTO employee_photos(employee_id,photo_path,label,created_at) VALUES(?,?,?,?)",
                    (employee_id, f"uploads/employees/{filename}", f"burst_{i}", now.isoformat())
                )
            except Exception as e:
                logger.error(f"Error saving burst photo {i}: {e}")
                
    threading.Thread(target=train_model, daemon=True).start()


def train_model() -> bool:
    """Load embeddings for all stored employee photos. Thread-safe."""
    global _model_trained, _id_to_name, _known_embeddings, _is_training
    if not CV2_OK or not SFACE_OK or _face_detector is None or _face_recognizer is None:
        return False

    with _model_lock:
        _is_training = True

    try:
        new_embeddings = []
        name_map = {}
    
        with get_db() as db:
            ref_rows = db.execute("""
                SELECT ep.employee_id, ep.photo_path, e.first_name, e.last_name
                FROM employee_photos ep
                JOIN employees e ON e.id = ep.employee_id
            """).fetchall()
            emp_rows = db.execute("""
                SELECT id as employee_id, photo_path, first_name, last_name
                FROM employees WHERE photo_path IS NOT NULL
            """).fetchall()
    
        all_photos = (
            [(r["employee_id"], r["photo_path"], f"{r['first_name']} {r['last_name']}") for r in ref_rows]
            + [(r["employee_id"], r["photo_path"], f"{r['first_name']} {r['last_name']}") for r in emp_rows]
        )
    
        for emp_id, photo_path, name in all_photos:
            if not photo_path:
                continue
                
            full = BASE_DIR / "static" / photo_path if not Path(photo_path).is_absolute() else Path(photo_path)
            if not full.exists():
                continue
                
            img = cv2.imread(str(full))
            if img is None:
                continue
                
            res = _get_face_embedding(img)
            if res is not None:
                embedding, _ = res
                new_embeddings.append((emp_id, embedding))
                name_map[emp_id] = name
    
        with _model_lock:
            if new_embeddings:
                _known_embeddings = new_embeddings
                _id_to_name = name_map
                _model_trained = True
                return True
            _model_trained = False
            return False
    finally:
        with _model_lock:
            _is_training = False


# ══════════════════════════════════════════════════════════════════════════════
#  Business logic
# ══════════════════════════════════════════════════════════════════════════════

def today_bounds():
    start = datetime.combine(date.today(), datetime.min.time())
    return start.isoformat(), (start + timedelta(days=1)).isoformat()


def event_counts_today() -> dict:
    s, e = today_bounds()
    with get_db() as db:
        rows = db.execute(
            "SELECT event_type, COUNT(*) c FROM attendance_events "
            "WHERE event_time>=? AND event_time<? GROUP BY event_type", (s, e)
        ).fetchall()
    counts = {"entry": 0, "exit": 0}
    for r in rows:
        counts[r["event_type"]] = r["c"]
    return counts


def current_presence() -> dict:
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) c FROM employees").fetchone()["c"]
        rows  = db.execute("""
            SELECT ae.employee_id, ae.event_type
            FROM attendance_events ae
            JOIN (SELECT employee_id, MAX(event_time) t FROM attendance_events GROUP BY employee_id) lx
              ON lx.employee_id=ae.employee_id AND lx.t=ae.event_time
        """).fetchall()
    present  = sum(1 for r in rows if r["event_type"] == "entry")
    absent   = max(total - present, 0)
    occupancy = round(present / total * 100, 1) if total else 0
    return {"total": total, "present": present, "absent": absent, "occupancy": occupancy}


def add_alert(employee_id, alert_type: str, message: str, severity: str = "medium"):
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    with get_db() as db:
        exists = db.execute(
            "SELECT id FROM alerts WHERE employee_id IS ? AND alert_type=? AND resolved=0 AND created_at>?",
            (employee_id, alert_type, one_hour_ago)
        ).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO alerts(employee_id,alert_type,message,severity,created_at) VALUES(?,?,?,?,?)",
                (employee_id, alert_type, message, severity, datetime.now().isoformat())
            )


def register_detection(employee_id: int, confidence: float = 1.0):
    now = datetime.now()
    last = _last_seen.get(employee_id)
    if last and (now - last).total_seconds() < COOLDOWN_SECS:
        return None

    with get_db() as db:
        prev = db.execute(
            "SELECT event_type FROM attendance_events WHERE employee_id=? ORDER BY event_time DESC LIMIT 1",
            (employee_id,)
        ).fetchone()
        next_evt = "exit" if prev and prev["event_type"] == "entry" else "entry"
        db.execute(
            "INSERT INTO attendance_events(employee_id,event_type,event_time,confidence) VALUES(?,?,?,?)",
            (employee_id, next_evt, now.isoformat(), confidence)
        )
        emp = db.execute("SELECT first_name,last_name FROM employees WHERE id=?", (employee_id,)).fetchone()

    _last_seen[employee_id]   = now
    _last_status[employee_id] = next_evt

    name = f"{emp['first_name']} {emp['last_name']}" if emp else f"Employé #{employee_id}"
    t    = now.strftime("%H:%M")
    if next_evt == "entry" and t > LATE_AFTER:
        add_alert(employee_id, "Arrivée tardive", f"{name} est arrivé après {LATE_AFTER}.", "high")
    if t < WORKDAY_START or t > WORKDAY_END:
        add_alert(employee_id, "Hors horaires", f"Activité hors horaires pour {name}.", "medium")
    return next_evt


def compute_duration(employee_id: int, day=None) -> str:
    day = day or date.today()
    s   = datetime.combine(day, datetime.min.time()).isoformat()
    e   = datetime.combine(day + timedelta(days=1), datetime.min.time()).isoformat()
    with get_db() as db:
        evts = db.execute(
            "SELECT event_type,event_time FROM attendance_events "
            "WHERE employee_id=? AND event_time>=? AND event_time<? ORDER BY event_time",
            (employee_id, s, e)
        ).fetchall()
    total  = timedelta()
    entry  = None
    for ev in evts:
        t = datetime.fromisoformat(ev["event_time"])
        if ev["event_type"] == "entry":
            entry = t
        elif ev["event_type"] == "exit" and entry:
            total += t - entry
            entry = None
    if entry:
        total += datetime.now() - entry
    h, rem = divmod(int(total.total_seconds()), 3600)
    return f"{h}h{rem//60:02d}"


# ══════════════════════════════════════════════════════════════════════════════
#  Video streaming
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_frame(frame):
    global _recent_detections
    detections = []
    if not CV2_OK or _face_detector is None:
        return frame, detections

    height, width, _ = frame.shape
    
    # Downscale for ultra-fast detection
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    small_h, small_w, _ = small_frame.shape
    _face_detector.setInputSize((small_w, small_h))
    _, faces = _face_detector.detect(small_frame)

    if faces is not None:
        for face in faces:
            # Rescale face data back to original size for SFace alignment
            face = face / scale
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[0:4]
            name   = "Inconnu"
            color  = (0, 140, 255)
            status = "unknown"
            conf   = 0

            if _model_trained and _face_recognizer is not None:
                try:
                    aligned_face = _face_recognizer.alignCrop(frame, face)
                    embedding = _face_recognizer.feature(aligned_face)
                    
                    best_match_id = None
                    max_similarity = 0
                    
                    with _model_lock:
                        for emp_id, known_emb in _known_embeddings:
                            similarity = _face_recognizer.match(embedding, known_emb, cv2.FaceRecognizerSF_FR_COSINE)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_match_id = emp_id
                                
                        if max_similarity > SFACE_THRESHOLD and best_match_id is not None:
                            name   = _id_to_name.get(best_match_id, f"Emp #{best_match_id}")
                            # Scale confidence score mapping (threshold ~ 1.0)
                            conf   = min(100, max(0, round((max_similarity - SFACE_THRESHOLD) / (1.0 - SFACE_THRESHOLD) * 100, 1)))
                            event  = register_detection(best_match_id, conf / 100)
                            status = event or _last_status.get(best_match_id, "present")
                            color  = (86, 205, 119) if status == "entry" else (59, 130, 246)
                except Exception as e:
                    logger.error(f"Recognition error: {e}")

            if name == "Inconnu" and _model_trained:
                add_alert(None, "Visage inconnu", "Un visage non reconnu a été détecté.", "low")

            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            lbl_y = max(y - 32, 0)
            cv2.rectangle(frame, (x, lbl_y), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, name, (x+6, max(y-10, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if conf:
                cv2.putText(frame, f"{conf}%", (x+w-46, y+h-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            det = {"name": name, "status": status, "confidence": conf}
            detections.append(det)
            if name != "Inconnu":
                _recent_detections = [
                    {"name": name, "status": status, "time": datetime.now().strftime("%H:%M:%S")}
                ] + _recent_detections[:19]

    return frame, detections


def generate_frames():
    if not CV2_OK:
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
            time.sleep(1)
        return

    logger.info("Initializing camera feed using DirectShow backend for fast startup...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS,          30)
    try:
        while True:
            with _cam_lock:
                ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame, _ = _annotate_frame(frame)
            ok, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.04)
    finally:
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  Template filters
# ══════════════════════════════════════════════════════════════════════════════

@app.template_filter("dt")
def fmt_dt(value):
    if not value:
        return ""
    return datetime.fromisoformat(value).strftime("%d/%m/%Y %H:%M")


@app.template_filter("event_label")
def fmt_event(value):
    return "Entrée" if value == "entry" else "Sortie"


@app.template_filter("severity_label")
def fmt_severity(value):
    return {"high": "Critique", "medium": "Moyen", "low": "Faible"}.get(value, value)


# ══════════════════════════════════════════════════════════════════════════════
#  Auth routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("dashboard") if session.get("user_id") else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["name"]    = user["name"]
            session["role"]    = user["role"]
            return redirect(url_for("dashboard"))
        flash("Identifiants incorrects.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard")
@login_required
def dashboard():
    presence = current_presence()
    counts   = event_counts_today()
    with get_db() as db:
        alerts = db.execute("""
            SELECT a.*, e.first_name, e.last_name
            FROM alerts a LEFT JOIN employees e ON e.id=a.employee_id
            WHERE a.resolved=0 ORDER BY a.created_at DESC LIMIT 8
        """).fetchall()
        recent_events = db.execute("""
            SELECT ae.*, e.first_name, e.last_name, e.department, e.photo_path
            FROM attendance_events ae
            JOIN employees e ON e.id=ae.employee_id
            ORDER BY ae.event_time DESC LIMIT 12
        """).fetchall()
    return render_template("dashboard.html",
                           presence=presence, counts=counts,
                           alerts=alerts, recent_events=recent_events,
                           cv_ok=CV2_OK, sface_ok=SFACE_OK,
                           model_trained=_model_trained)


# ══════════════════════════════════════════════════════════════════════════════
#  Employees
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/employees")
@login_required
def employees():
    with get_db() as db:
        rows = db.execute("SELECT * FROM employees ORDER BY last_name,first_name").fetchall()
        # attach presence status
        result = []
        for emp in rows:
            last_evt = db.execute(
                "SELECT event_type FROM attendance_events WHERE employee_id=? ORDER BY event_time DESC LIMIT 1",
                (emp["id"],)
            ).fetchone()
            photo_count = db.execute(
                "SELECT COUNT(*) c FROM employee_photos WHERE employee_id=?", (emp["id"],)
            ).fetchone()["c"]
            result.append({
                "id": emp["id"], "first_name": emp["first_name"], "last_name": emp["last_name"],
                "department": emp["department"], "position": emp["position"],
                "email": emp["email"], "photo_path": emp["photo_path"],
                "status": last_evt["event_type"] if last_evt else "absent",
                "duration": compute_duration(emp["id"]),
                "photo_count": photo_count,
                "created_at": emp["created_at"],
            })
    return render_template("employees.html", employees=result, model_trained=_model_trained)


@app.route("/employees/new", methods=["GET", "POST"])
@login_required
@admin_required
def employee_new():
    if request.method == "POST":
        return _save_employee()
    return render_template("employee_form.html", employee=None)


@app.route("/employees/<int:eid>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def employee_edit(eid):
    with get_db() as db:
        emp = db.execute("SELECT * FROM employees WHERE id=?", (eid,)).fetchone()
        photos = db.execute(
            "SELECT * FROM employee_photos WHERE employee_id=? ORDER BY created_at DESC", (eid,)
        ).fetchall()
    if not emp:
        flash("Employé introuvable.", "warning")
        return redirect(url_for("employees"))
    if request.method == "POST":
        return _save_employee(eid)
    return render_template("employee_form.html", employee=emp, photos=photos)


def _save_employee(eid=None):
    first_name = request.form.get("first_name", "").strip()
    last_name  = request.form.get("last_name",  "").strip()
    department = request.form.get("department",  "").strip()
    position   = request.form.get("position",    "").strip()
    email      = request.form.get("email",       "").strip().lower()
    now        = datetime.now().isoformat()

    if not all([first_name, last_name, department, position, email]):
        flash("Tous les champs sont obligatoires.", "warning")
        return redirect(request.referrer or url_for("employees"))

    file       = request.files.get("photo")
    photo_path = None
    if file and file.filename and allowed_file(file.filename):
        fname      = f"{int(time.time())}_{secure_filename(file.filename)}"
        dest       = UPLOAD_DIR / fname
        file.save(dest)
        photo_path = f"uploads/employees/{fname}"

    burst_photos_str = request.form.get("burst_photos")

    with get_db() as db:
        if eid:
            cur = db.execute("SELECT photo_path FROM employees WHERE id=?", (eid,)).fetchone()
            db.execute(
                "UPDATE employees SET first_name=?,last_name=?,department=?,position=?,email=?,"
                "photo_path=?,updated_at=? WHERE id=?",
                (first_name, last_name, department, position, email,
                 photo_path or cur["photo_path"], now, eid)
            )
            flash("Employé mis à jour.", "success")
            new_id = eid
        else:
            cur = db.execute(
                "INSERT INTO employees(first_name,last_name,department,position,email,photo_path,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (first_name, last_name, department, position, email, photo_path, now, now)
            )
            new_id = cur.lastrowid
            flash("Employé ajouté.", "success")

    # Handle burst capture JSON array
    if burst_photos_str:
        try:
            burst_list = json.loads(burst_photos_str)
            if burst_list:
                process_burst_photos(new_id, burst_list)
        except Exception as e:
            logger.error(f"Failed to decode burst photos: {e}")
    elif photo_path:
        process_and_save_photos(new_id, photo_path)
    else:
        threading.Thread(target=train_model, daemon=True).start()
        
    return redirect(url_for("employees"))


@app.route("/employees/<int:eid>/delete", methods=["POST"])
@login_required
@admin_required
def employee_delete(eid):
    with get_db() as db:
        db.execute("DELETE FROM employees WHERE id=?", (eid,))
    threading.Thread(target=train_model, daemon=True).start()
    flash("Employé supprimé.", "success")
    return redirect(url_for("employees"))


@app.route("/employees/<int:eid>/add_photo", methods=["POST"])
@login_required
@admin_required
def employee_add_photo(eid):
    """Accept a base64 webcam capture or file upload as additional reference photo."""
    now = datetime.now()
    saved = False

    # ── webcam base64 ─────────────────────────────────────────────────────────
    b64 = request.form.get("webcam_data", "")
    if b64:
        try:
            header, data = b64.split(",", 1)
            img_bytes = base64.b64decode(data)
            fname = f"{int(now.timestamp())}_webcam.jpg"
            dest  = UPLOAD_DIR / fname
            dest.write_bytes(img_bytes)
            photo_path = f"uploads/employees/{fname}"
            with get_db() as db:
                db.execute(
                    "INSERT INTO employee_photos(employee_id,photo_path,label,created_at) VALUES(?,?,?,?)",
                    (eid, photo_path, "webcam", now.isoformat())
                )
            saved = True
        except Exception as exc:
            flash(f"Erreur lors de l'enregistrement: {exc}", "danger")

    # ── file upload ───────────────────────────────────────────────────────────
    file = request.files.get("photo")
    if not saved and file and file.filename and allowed_file(file.filename):
        fname = f"{int(now.timestamp())}_{secure_filename(file.filename)}"
        dest  = UPLOAD_DIR / fname
        file.save(dest)
        photo_path = f"uploads/employees/{fname}"
        with get_db() as db:
            db.execute(
                "INSERT INTO employee_photos(employee_id,photo_path,label,created_at) VALUES(?,?,?,?)",
                (eid, photo_path, "upload", now.isoformat())
            )
        saved = True

    if saved:
        flash("Photo de référence ajoutée. Augmentation et réentraînement en cours…", "success")
        process_and_save_photos(eid, photo_path)
    else:
        flash("Aucune photo fournie.", "warning")

    return redirect(url_for("employee_edit", eid=eid))


@app.route("/employees/<int:eid>/delete_photo/<int:pid>", methods=["POST"])
@login_required
@admin_required
def employee_delete_photo(eid, pid):
    with get_db() as db:
        db.execute("DELETE FROM employee_photos WHERE id=? AND employee_id=?", (pid, eid))
    threading.Thread(target=train_model, daemon=True).start()
    flash("Photo supprimée.", "success")
    return redirect(url_for("employee_edit", eid=eid))


# ══════════════════════════════════════════════════════════════════════════════
#  Video
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/video")
@login_required
def video():
    return render_template("video.html", cv_ok=CV2_OK, sface_ok=SFACE_OK,
                           model_trained=_model_trained)


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ══════════════════════════════════════════════════════════════════════════════
#  Analytics
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html")


# ══════════════════════════════════════════════════════════════════════════════
#  Reports
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/reports")
@login_required
def reports():
    with get_db() as db:
        events = db.execute("""
            SELECT ae.*, e.first_name, e.last_name, e.department
            FROM attendance_events ae
            JOIN employees e ON e.id=ae.employee_id
            ORDER BY ae.event_time DESC LIMIT 200
        """).fetchall()
    return render_template("reports.html", events=events)


@app.route("/reports/excel")
@login_required
def report_excel():
    try:
        import pandas as pd
    except ImportError:
        flash("Installez pandas et openpyxl pour exporter Excel.", "warning")
        return redirect(url_for("reports"))
    with get_db() as db:
        rows = db.execute("""
            SELECT e.first_name prenom, e.last_name nom, e.department departement,
                   ae.event_type evenement, ae.event_time heure, ae.confidence confiance
            FROM attendance_events ae
            JOIN employees e ON e.id=ae.employee_id
            ORDER BY ae.event_time DESC
        """).fetchall()
    df  = pd.DataFrame([dict(r) for r in rows])
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Présence")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="smartwork_rapport.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/reports/pdf")
@login_required
def report_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        flash("Installez reportlab pour exporter PDF.", "warning")
        return redirect(url_for("reports"))

    presence = current_presence()
    counts   = event_counts_today()
    buf      = io.BytesIO()
    c        = rl_canvas.Canvas(buf, pagesize=A4)
    W, H     = A4
    y        = H - 50

    c.setFillColorRGB(0.388, 0.4, 0.945)
    c.rect(0, H - 80, W, 80, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 52, "Smart Workplace — Rapport de présence")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 70, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

    y = H - 110
    c.setFillColorRGB(0.09, 0.12, 0.18)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Résumé du jour")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y,
        f"Présents: {presence['present']}   Absents: {presence['absent']}   "
        f"Taux: {presence['occupancy']}%   "
        f"Entrées: {counts['entry']}   Sorties: {counts['exit']}")
    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Historique des événements")
    y -= 20
    c.setFont("Helvetica", 9)

    with get_db() as db:
        rows = db.execute("""
            SELECT e.first_name, e.last_name, e.department, ae.event_type, ae.event_time
            FROM attendance_events ae
            JOIN employees e ON e.id=ae.employee_id
            ORDER BY ae.event_time DESC LIMIT 60
        """).fetchall()

    for row in rows:
        if y < 50:
            c.showPage(); y = H - 50
            c.setFont("Helvetica", 9)
        label = "Entrée" if row["event_type"] == "entry" else "Sortie"
        t     = datetime.fromisoformat(row["event_time"]).strftime("%d/%m/%Y %H:%M")
        c.drawString(40, y,
            f"{t}   {label}   {row['first_name']} {row['last_name']}   ({row['department']})")
        y -= 15

    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="smartwork_rapport.pdf",
                     mimetype="application/pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  JSON API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/presence")
@login_required
def api_presence():
    return jsonify(current_presence())


@app.route("/api/recent_detections")
@login_required
def api_recent_detections():
    return jsonify(_recent_detections[:10])


@app.route("/api/stats")
@login_required
def api_stats():
    s, e = today_bounds()
    with get_db() as db:
        daily = db.execute("""
            SELECT DATE(event_time) day, COUNT(*) count FROM attendance_events
            GROUP BY DATE(event_time) ORDER BY day DESC LIMIT 14
        """).fetchall()
        hourly = db.execute("""
            SELECT event_type, STRFTIME('%H',event_time) hour, COUNT(*) count
            FROM attendance_events WHERE event_time>=? AND event_time<?
            GROUP BY event_type,hour ORDER BY hour
        """, (s, e)).fetchall()
        departments = db.execute("""
            SELECT department, COUNT(*) count FROM employees
            GROUP BY department ORDER BY count DESC
        """).fetchall()
        weekly = db.execute("""
            SELECT DATE(event_time) day, COUNT(DISTINCT employee_id) unique_emp
            FROM attendance_events WHERE event_type='entry'
            GROUP BY DATE(event_time) ORDER BY day DESC LIMIT 7
        """).fetchall()
        top_emp = db.execute("""
            SELECT e.first_name||' '||e.last_name name,
                   COUNT(ae.id) total_events
            FROM attendance_events ae JOIN employees e ON e.id=ae.employee_id
            WHERE ae.event_type='entry'
            GROUP BY ae.employee_id ORDER BY total_events DESC LIMIT 6
        """).fetchall()
    return jsonify({
        "presence":    current_presence(),
        "today":       event_counts_today(),
        "daily":       [dict(r) for r in reversed(daily)],
        "hourly":      [dict(r) for r in hourly],
        "departments": [dict(r) for r in departments],
        "weekly":      [dict(r) for r in reversed(weekly)],
        "top_emp":     [dict(r) for r in top_emp],
    })


@app.route("/api/alerts")
@login_required
def api_alerts():
    with get_db() as db:
        rows = db.execute("""
            SELECT a.*, e.first_name, e.last_name FROM alerts a
            LEFT JOIN employees e ON e.id=a.employee_id
            WHERE a.resolved=0 ORDER BY a.created_at DESC LIMIT 20
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts/<int:aid>/resolve", methods=["POST"])
@login_required
def api_resolve_alert(aid):
    with get_db() as db:
        db.execute("UPDATE alerts SET resolved=1 WHERE id=?", (aid,))
    return jsonify({"ok": True})


@app.route("/api/retrain", methods=["POST"])
@login_required
@admin_required
def api_retrain():
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True, "message": "Réentraînement démarré."})


@app.route("/api/system_status")
@login_required
def api_system_status():
    return jsonify({
        "cv2_available":   CV2_OK,
        "sface_available": SFACE_OK,
        "model_trained":   _model_trained,
        "is_training":     _is_training,
        "employees_count": current_presence()["total"],
        "known_faces":     len(_id_to_name),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    init_face_recognition()
    threading.Thread(target=train_model, daemon=True).start()
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
