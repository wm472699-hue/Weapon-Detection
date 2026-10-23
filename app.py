import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from datetime import datetime
import base64
import io
import tempfile
import os
import time
import json
import hashlib
import re
from types import SimpleNamespace

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
ROBOFLOW_API_KEY = "c4rtJieiUdsZTiztYMgR"
ROBOFLOW_MODEL_ID = "gun-mmmct-sck1h"
ROBOFLOW_MODEL_VERSION = "1"
N8N_WEBHOOK_URL = "https://mehar333.app.n8n.cloud/webhook/weapon-alert"

ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"

SNAPSHOT_COOLDOWN_SECONDS = 5

ENABLE_EMAIL_ALERTS = False   # n8n is not running right now, so no webhook/email calls are made.
                               # Flip this back to True whenever n8n is running again.

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

VIDEO_CLIP_SECONDS = 4        # length of the video clip recorded when a weapon is detected

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ------------------------------------------------------------------
# CAMERA AUTO-CONNECT — built in directly (no separate file to lose/break).
# Tries the common RTSP path for each major camera brand and reports
# back whichever one actually opens and returns a real frame.
# ------------------------------------------------------------------
_COMMON_RTSP_PATHS = [
    ("Hikvision", "Streaming/Channels/101"),
    ("Dahua", "cam/realmonitor?channel=1&subtype=0"),
    ("CP Plus / Generic", "stream1"),
    ("Reolink", "h264Preview_01_main"),
    ("ONVIF / Generic", "onvif1"),
    ("Generic (live)", "live"),
    ("Root stream", ""),
]

def auto_connect_camera(ip, username, password, port=554, progress_callback=None):
    """Tries every common RTSP path against this camera and returns the first
    one that actually opens and delivers a frame. Returns an object with:
    .success (bool), .url (str or None), .tried (list of labels), .diagnostic (str)."""
    tried = []
    for brand_label, path in _COMMON_RTSP_PATHS:
        if progress_callback:
            progress_callback(f"Trying {brand_label} path...")
        url = f"rtsp://{username}:{password}@{ip}:{port}/{path}" if path else f"rtsp://{username}:{password}@{ip}:{port}/"
        tried.append(brand_label)
        try:
            cap = cv2.VideoCapture(url)
            opened = cap.isOpened()
            frame_ok = False
            if opened:
                ret, _ = cap.read()
                frame_ok = ret
            cap.release()
            if opened and frame_ok:
                if progress_callback:
                    progress_callback(f"✅ Connected using {brand_label} path.")
                return SimpleNamespace(success=True, url=url, tried=tried, diagnostic="")
        except Exception:
            pass

    diagnostic = (
        "None of the common camera paths responded. Double-check the IP address, "
        "username, and password, and make sure this computer and the camera are "
        "on the same WiFi/network."
    )
    return SimpleNamespace(success=False, url=None, tried=tried, diagnostic=diagnostic)

def load_users():
    """Returns a dict: { username: {"password": <hash>, "email": <email>} }.
    Transparently upgrades the old format (username -> hash string) if found."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            return {}
        upgraded = {}
        for uname, val in data.items():
            if isinstance(val, str):
                upgraded[uname] = {"password": val, "email": ""}
            else:
                upgraded[uname] = val
        return upgraded
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def find_username_by_email(users, email):
    email = email.strip().lower()
    for uname, val in users.items():
        if val.get("email", "").strip().lower() == email:
            return uname
    return None

# ------------------------------------------------------------------
# PERSON DETECTOR (built into OpenCV, free, runs locally, no API needed)
# ------------------------------------------------------------------
PERSON_DETECTION_AVAILABLE = False
PERSON_DETECTOR = None
try:
    PERSON_DETECTOR = cv2.HOGDescriptor()
    PERSON_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    PERSON_DETECTION_AVAILABLE = True
except Exception:
    PERSON_DETECTION_AVAILABLE = False

def detect_persons(image_bgr):
    """Returns a list of (x, y, w, h) boxes for detected people using OpenCV's HOG detector."""
    if not PERSON_DETECTION_AVAILABLE:
        return []
    small = cv2.resize(image_bgr, (0, 0), fx=0.6, fy=0.6)
    if small.shape[0] < 128 or small.shape[1] < 64:
        return []
    boxes, weights = PERSON_DETECTOR.detectMultiScale(
        small, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    scaled_boxes = []
    for (x, y, w, h) in boxes:
        scaled_boxes.append((int(x / 0.6), int(y / 0.6), int(w / 0.6), int(h / 0.6)))
    return scaled_boxes

# ------------------------------------------------------------------
# FACE DETECTOR + FACE DE-DUPLICATION
# ------------------------------------------------------------------
FACE_DETECTION_AVAILABLE = False
FACE_CASCADE = None
try:
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(_cascade_path)
    FACE_DETECTION_AVAILABLE = not FACE_CASCADE.empty()
except Exception:
    FACE_DETECTION_AVAILABLE = False

FACE_MATCH_THRESHOLD = 0.55

def detect_faces(image_bgr):
    if not FACE_DETECTION_AVAILABLE:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    return list(faces)

def face_signature(image_bgr, box):
    x, y, w, h = box
    x, y = max(x, 0), max(y, 0)
    face_crop = image_bgr[y:y + h, x:x + w]
    if face_crop.size == 0:
        return None
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (100, 100))
    face_gray = cv2.equalizeHist(face_gray)
    return face_gray

def is_same_face(sig_a, sig_b, threshold=FACE_MATCH_THRESHOLD):
    if sig_a is None or sig_b is None:
        return False
    score = cv2.matchTemplate(sig_a, sig_b, cv2.TM_CCOEFF_NORMED)[0][0]
    return score >= threshold

def find_matching_person_id(signature, known_faces, threshold=FACE_MATCH_THRESHOLD):
    for known in known_faces:
        if is_same_face(signature, known["signature"], threshold):
            return known["person_id"]
    return None

st.set_page_config(page_title="Argus AI — Weapon Detection", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------
# GLOBAL STYLE — professional dark security-tech look (navy + emerald)
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #0f1b17 0%, #060b09 60%);
    }

    /* ---------- Top brand bar ---------- */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 6px 26px 6px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 28px;
    }
    .brand-bar .logo {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0 !important;
        letter-spacing: 0.5px;
    }
    .brand-bar .logo span { color: #10b981; }
    .brand-bar .tagline {
        font-size: 13px;
        color: #64748b !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #0b2019 0%, #0e2b22 45%, #08201a 100%);
        border: 1px solid rgba(16,185,129,0.18);
        padding: 70px 40px;
        border-radius: 20px;
        text-align: center;
        color: #f1f5f9;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; left: -20%;
        width: 140%; height: 200%;
        background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 55%);
    }
    .hero .badge {
        display: inline-block;
        background: rgba(16,185,129,0.12);
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.35);
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 12.5px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 18px;
        position: relative;
        z-index: 1;
    }
    .hero h1 {
        font-size: 46px;
        font-weight: 800;
        margin: 6px 0 14px 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
        color: #f1f5f9 !important;
    }
    .hero h1 .accent { color: #10b981; }
    .hero p.sub {
        font-size: 17px;
        color: #94a3b8 !important;
        max-width: 640px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
        line-height: 1.55;
    }

    /* ---------- Stats row ---------- */
    .stat-box {
        text-align: center;
        padding: 6px 0;
    }
    .stat-box .num { font-size: 30px; font-weight: 800; color: #10b981; }
    .stat-box .lbl { font-size: 12.5px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

    /* ---------- Feature cards ---------- */
    .feature-card {
        background: linear-gradient(160deg, #0f1a15 0%, #0b1310 100%);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 28px 18px;
        color: #e2e8f0;
        text-align: center;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: #10b981;
        box-shadow: 0 12px 30px rgba(16,185,129,0.15);
    }
    .feature-card .icon-wrap {
        width: 54px; height: 54px;
        border-radius: 14px;
        background: rgba(16,185,129,0.1);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px auto;
        font-size: 26px;
    }
    .feature-card b { font-size: 15.5px; color: #f1f5f9 !important; }
    .feature-card .desc { font-size: 13px; opacity: 0.9; margin-top: 6px; display: block; line-height: 1.5; }

    /* ---------- Buttons ----------
       Every button — Logout, Clear Gallery, Start Live Monitoring, "Back",
       "Sign up" switch, etc. — gets an explicit dark background here.
       Without this, Streamlit's default white button background remains,
       and the earlier blanket text-color rule turns the label light too,
       producing an unreadable white-on-white box. */
    div[data-testid="stButton"] > button {
        background: #101b16 !important;
        color: #e5e9f0 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #10b981 !important;
        box-shadow: 0 8px 20px rgba(16,185,129,0.18);
    }
    div[data-testid="stButton"] > button:disabled {
        background: #0b120f !important;
        color: #4b5568 !important;
        border-color: rgba(148,163,184,0.12) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        border: none !important;
        color: white !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: #101b16 !important;
        color: #e5e9f0 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }

    /* ---------- Top nav login/signup buttons ---------- */
    .top-nav-row { margin-top: -10px; margin-bottom: 18px; }

    /* ---------- Auth card ---------- */
    .auth-title {
        text-align:center;
        color:#f1f5f9;
        font-weight:800;
        margin-bottom: 4px;
    }
    .auth-sub {
        text-align:center;
        color:#64748b !important;
        font-size: 13.5px;
        margin-bottom: 22px;
    }

    /* ---------- Trust bar ---------- */
    .trust-bar {
        display: flex;
        justify-content: center;
        gap: 36px;
        flex-wrap: wrap;
        padding: 18px 0 36px 0;
        border-bottom: 1px solid rgba(148,163,184,0.1);
        margin-bottom: 36px;
    }
    .trust-item {
        color: #64748b;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 7px;
        letter-spacing: 0.3px;
    }
    .trust-item b { color: #94a3b8; }

    /* ---------- Section heading ---------- */
    .section-heading {
        text-align: center;
        margin: 12px 0 36px 0;
    }
    .section-heading .eyebrow {
        color: #10b981 !important;
        font-size: 12.5px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .section-heading h2 {
        color: #f1f5f9 !important;
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0 6px 0;
    }
    .section-heading p {
        color: #64748b !important;
        font-size: 14.5px;
        max-width: 520px;
        margin: 0 auto;
    }

    /* ---------- How it works steps ---------- */
    .step-card {
        background: #0f1a15;
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 30px 22px;
        color: #e2e8f0;
        height: 100%;
        position: relative;
    }
    .step-card .step-num {
        position: absolute;
        top: 18px; right: 22px;
        font-size: 40px;
        font-weight: 800;
        color: rgba(16,185,129,0.14);
    }
    .step-card .step-icon { font-size: 30px; margin-bottom: 12px; }
    .step-card b { font-size: 16px; display: block; margin-bottom: 6px; color: #f1f5f9 !important; }
    .step-card .step-desc { font-size: 13.5px; color: #94a3b8; line-height: 1.55; }

    /* ---------- Closing CTA ---------- */
    .cta-banner {
        background: linear-gradient(120deg, #0b2019 0%, #103428 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 18px;
        padding: 46px 30px;
        text-align: center;
        margin: 46px 0 30px 0;
    }
    .cta-banner h3 { color: #f1f5f9 !important; font-size: 25px; font-weight: 800; margin-bottom: 8px; }
    .cta-banner p { color: #94a3b8 !important; font-size: 14.5px; margin-bottom: 4px; }

    .landing-footer {
        text-align: center;
        color: #475569;
        font-size: 12.5px;
        padding: 24px 0 10px 0;
        border-top: 1px solid rgba(148,163,184,0.1);
        margin-top: 10px;
    }

    /* ---------- Extra polish: consistent card shadow + section rhythm ---------- */
    .feature-card, .step-card {
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    div[data-testid="column"] { padding: 6px; }

    /* ================================================================
       READABILITY — applies everywhere, including the post-login
       dashboard (title, captions, alerts, expander, code, uploader,
       camera input, video/image, sidebar, inputs). Without this block
       Streamlit's own default text colors show through on the dark
       background and become nearly invisible.
       ================================================================ */

    /* Blanket rule first, then re-mute specific secondary text below */
    .stApp, .stApp * {
        color: #e5e9f0 !important;
    }
    .stApp small,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {
        color: #8b96a8 !important;
    }
    .stApp a { color: #34d399 !important; }

    /* Form labels */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    /* Text / number inputs */
    input[type="text"], input[type="password"], input[type="number"],
    div[data-baseweb="input"] input {
        background-color: #101b16 !important;
        color: #e5e9f0 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }
    input::placeholder { color: #64748b !important; opacity: 1 !important; }
    input[type="text"]:focus, input[type="password"]:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }
    div[data-testid="stTextInput"] button {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }

    /* Dropdown / selectbox */
    div[data-baseweb="select"] > div {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }
    ul[data-baseweb="menu"] { background: #101b16 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1310 !important;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    /* Radio / checkbox */
    div[data-testid="stRadio"] label p,
    div[data-testid="stCheckbox"] label p {
        color: #e5e9f0 !important;
    }

    /* st.info / success / warning / error boxes */
    div[data-testid="stAlert"], div[data-testid="stAlertContainer"] {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 10px !important;
    }

    /* st.code blocks (RTSP url, connection log) */
    div[data-testid="stCodeBlock"], pre {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 10px !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #0f1a15 !important;
        border: 1px solid rgba(148,163,184,0.15) !important;
        border-radius: 10px !important;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploader"] section {
        background: #101b16 !important;
        border: 1.5px dashed rgba(148,163,184,0.3) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"] button {
        background: #0f1a15 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }
    div[data-testid="stFileUploaderFile"] {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 8px !important;
    }

    /* Camera input */
    div[data-testid="stCameraInput"] {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCameraInput"] button {
        background: #10b981 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Image / video frames */
    div[data-testid="stImage"], div[data-testid="stVideo"], video {
        border-radius: 12px !important;
        border: 1px solid rgba(148,163,184,0.15) !important;
        overflow: hidden !important;
    }

    /* Top header bar (Deploy button strip) */
    header[data-testid="stHeader"] { background: #060b09 !important; }

    /* Form submit button */
    div[data-testid="stFormSubmitButton"] button {
        background: #101b16 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
    }

    /* Spinner text */
    div[data-testid="stSpinner"] p { color: #8b96a8 !important; }

    .auth-title { color: #f8fafc !important; font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# AUTHENTICATION
# ------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "landing"
if "current_user" not in st.session_state:
    st.session_state.current_user = None

def render_brand_bar():
    st.markdown("""
    <div class="brand-bar">
        <div class="logo">🛡️ Argus<span>AI</span></div>
        <div class="tagline" style="color:#94a3b8 !important;">Intelligent Threat Detection</div>
    </div>
    """, unsafe_allow_html=True)

def render_top_nav():
    """Brand bar plus Log In / Sign Up buttons right at the top of the
    marketing page — not just at the bottom — the way a real product
    landing page puts auth access in the header."""
    col_logo, col_gap, col_login, col_signup = st.columns([4, 3, 1, 1])
    with col_logo:
        st.markdown("""
        <div style="padding-top:18px;">
            <span style="font-size:22px;font-weight:700;color:#e2e8f0;letter-spacing:0.5px;">
                🛡️ Argus<span style="color:#10b981;">AI</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_login:
        st.markdown('<div style="padding-top:12px;">', unsafe_allow_html=True)
        if st.button("Log In", key="topnav_login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_signup:
        st.markdown('<div style="padding-top:12px;">', unsafe_allow_html=True)
        if st.button("Sign Up", key="topnav_signup", use_container_width=True, type="primary"):
            st.session_state.auth_view = "signup"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="border-bottom:1px solid rgba(148,163,184,0.15);margin:14px 0 28px 0;"></div>', unsafe_allow_html=True)

def render_landing():
    render_top_nav()
    st.markdown("""
    <div class="hero">
        <span class="badge">● Live &nbsp;·&nbsp; AI-Powered &nbsp;·&nbsp; Real-Time</span>
        <h1>Detect Threats <span class="accent">Before They Escalate</span></h1>
        <p class="sub" style="color:#cbd5e1 !important;">ArgusAI watches your cameras around the clock, instantly recognizes firearms and
        edged weapons, and alerts your team the moment something matters — so no one has to stare at a
        screen all day.</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [("99%+", "Detection Confidence"), ("<1s", "Alert Latency"),
             ("24/7", "Continuous Monitoring"), ("Any", "IP / CCTV Camera")]
    for col, (num, lbl) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f'<div class="stat-box"><div class="num">{num}</div><div class="lbl" style="color:#94a3b8 !important;">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="trust-bar">
        <div class="trust-item" style="color:#cbd5e1 !important;">🔒 <b style="color:#f1f5f9 !important;">End-to-end</b> account isolation</div>
        <div class="trust-item" style="color:#cbd5e1 !important;">📡 <b style="color:#f1f5f9 !important;">Brand-agnostic</b> camera support</div>
        <div class="trust-item" style="color:#cbd5e1 !important;">🧠 <b style="color:#f1f5f9 !important;">AI-trained</b> detection model</div>
        <div class="trust-item" style="color:#cbd5e1 !important;">⚡ <b style="color:#f1f5f9 !important;">Sub-second</b> response time</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">
        <div class="eyebrow" style="color:#10b981 !important;">Capabilities</div>
        <h2>Everything a modern security desk needs</h2>
        <p style="color:#cbd5e1 !important;">Built to plug straight into the cameras you already own — no new hardware required.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("📡", "Universal Camera Support", "Auto-connects to almost any IP / CCTV camera brand on your network"),
        ("🧑‍🤝‍🧑", "Smart Snapshot Capture", "One clean photo per unique person — never hundreds of duplicates"),
        ("🎬", "Automatic Evidence Clips", "Every confirmed detection is recorded as a short video automatically"),
        ("🔐", "Private, Secure Access", "Your own login — footage and alerts stay isolated to your account"),
    ]
    for col, (emoji, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="icon-wrap">{emoji}</div>'
                f'<b>{title}</b><span class="desc" style="color:#cbd5e1 !important;">{desc}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="section-heading" style="margin-top:56px;">
        <div class="eyebrow" style="color:#10b981 !important;">How it works</div>
        <h2>From camera to alert in three steps</h2>
        <p style="color:#cbd5e1 !important;">No complicated setup — connect once, and ArgusAI takes it from there.</p>
    </div>
    """, unsafe_allow_html=True)

    st1, st2, st3 = st.columns(3)
    steps = [
        ("📷", "1", "Connect Your Camera", "Point ArgusAI at your webcam, an existing CCTV / IP camera, or upload photos and video for one-off checks."),
        ("🧠", "2", "AI Scans Every Frame", "Each frame is analyzed in real time against a model trained specifically to recognize firearms and edged weapons."),
        ("🚨", "3", "Instant Alert & Evidence", "The moment a threat is confirmed, it's flagged on screen, snapshotted, and a short evidence clip is saved automatically."),
    ]
    for col, (icon, num, title, desc) in zip([st1, st2, st3], steps):
        with col:
            st.markdown(
                f'<div class="step-card"><div class="step-num">{num}</div>'
                f'<div class="step-icon">{icon}</div><b>{title}</b>'
                f'<div class="step-desc" style="color:#cbd5e1 !important;">{desc}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="cta-banner">
        <h3>Ready to start monitoring?</h3>
        <p style="color:#cbd5e1 !important;">Create a free account and connect your first camera in under two minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_a, col_b, col_r = st.columns([2, 1, 1, 2])
    with col_a:
        if st.button("🔑 Log In", use_container_width=True, type="primary"):
            st.session_state.auth_view = "login"
            st.rerun()
    with col_b:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.auth_view = "signup"
            st.rerun()

    st.markdown('<div class="landing-footer">ArgusAI · Built with Roboflow API + Streamlit + n8n</div>', unsafe_allow_html=True)

def render_login():
    render_brand_bar()
    st.markdown('<h2 class="auth-title">Welcome back</h2>', unsafe_allow_html=True)
    st.markdown('<p class="auth-sub" style="color:#cbd5e1 !important;">Log in to your ArgusAI monitoring dashboard</p>', unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        with st.container(border=True):
            identifier = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            if st.button("Log In", use_container_width=True, type="primary"):
                users = load_users()
                if not users:
                    st.error("No accounts exist yet — please create one first.")
                else:
                    uname = identifier.strip()
                    if uname not in users:
                        # maybe they typed their email instead of username
                        matched = find_username_by_email(users, identifier)
                        if matched:
                            uname = matched
                    if uname in users and users[uname]["password"] == hash_password(password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = uname
                        st.rerun()
                    else:
                        st.error("Invalid username/email or password.")
            st.write("")
            if st.button("Don't have an account? Sign up", use_container_width=True):
                st.session_state.auth_view = "signup"
                st.rerun()
            if st.button("⬅ Back", use_container_width=True):
                st.session_state.auth_view = "landing"
                st.rerun()

def render_signup():
    render_brand_bar()
    st.markdown('<h2 class="auth-title">Create your account</h2>', unsafe_allow_html=True)
    st.markdown('<p class="auth-sub" style="color:#cbd5e1 !important;">Set up secure access to your detection dashboard</p>', unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        with st.container(border=True):
            new_email = st.text_input("Email Address")
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                users = load_users()
                if not new_email or not new_username or not new_password:
                    st.error("Email, username, and password are all required.")
                elif not EMAIL_REGEX.match(new_email.strip()):
                    st.error("Please enter a valid email address.")
                elif new_username in users:
                    st.error("This username is already taken.")
                elif find_username_by_email(users, new_email):
                    st.error("An account with this email already exists.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    users[new_username] = {"password": hash_password(new_password), "email": new_email.strip()}
                    save_users(users)
                    st.success("Account created! You can now log in.")
                    st.session_state.auth_view = "login"
                    st.rerun()
            st.write("")
            if st.button("Already have an account? Log in", use_container_width=True):
                st.session_state.auth_view = "login"
                st.rerun()
            if st.button("⬅ Back", use_container_width=True, key="signup_back"):
                st.session_state.auth_view = "landing"
                st.rerun()

if not st.session_state.authenticated:
    if st.session_state.auth_view == "signup":
        render_signup()
    elif st.session_state.auth_view == "login":
        render_login()
    else:
        render_landing()
    st.stop()

st.title("🛡️ ArgusAI — Weapon Detection Dashboard")
st.write(f"Logged in as **{st.session_state.current_user}** — upload a photo, take a live picture, upload a video, or connect a live CCTV camera to detect weapon-like objects.")

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "last_saved_time" not in st.session_state:
    st.session_state.last_saved_time = {}
if "confirmed_rtsp_url" not in st.session_state:
    st.session_state.confirmed_rtsp_url = None
if "known_faces" not in st.session_state:
    st.session_state.known_faces = []
if "next_person_id" not in st.session_state:
    st.session_state.next_person_id = 1
if "active_recording" not in st.session_state:
    st.session_state.active_recording = None
if "camera_grid_sources" not in st.session_state:
    st.session_state.camera_grid_sources = []

# ------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------
def detect_weapon(image, confidence=40):
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    response = requests.post(
        ROBOFLOW_URL,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": confidence, "overlap": 30},
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return response.json()

def draw_boxes(image, predictions, draw_persons=True):
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if draw_persons:
        try:
            person_boxes = detect_persons(img_bgr)
            for (x, y, w, h) in person_boxes:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 150, 0), 2)
                cv2.putText(img_bgr, "Person", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
        except Exception:
            pass

    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = pred["class"]
        conf = pred["confidence"]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        text = f"{label} {conf*100:.1f}%"
        cv2.putText(img_bgr, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def send_alert(label, confidence, source="Streamlit Upload"):
    if not ENABLE_EMAIL_ALERTS:
        return None
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json={
                "label": label,
                "confidence": round(confidence, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source,
                "image_path": "uploaded_via_streamlit"
            },
            timeout=5
        )
        return response.json()
    except Exception as e:
        st.warning(f"Could not reach n8n webhook: {e}")
        return None

def stamp_timestamp_on_image(pil_image, dt):
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    text = dt.strftime("%d %B %Y, %I:%M:%S %p")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 1000)
    thickness = max(1, int(font_scale * 2))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 10
    bar_h = text_h + baseline + pad * 2

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    img_bgr = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)

    text_x = pad
    text_y = h - pad - baseline
    cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def maybe_save_snapshot(annotated_pil_image, label, confidence):
    now = time.time()
    last_time = st.session_state.last_saved_time.get(label, 0)
    if now - last_time >= SNAPSHOT_COOLDOWN_SECONDS:
        now_dt = datetime.now()
        stamped_image = stamp_timestamp_on_image(annotated_pil_image, now_dt)
        st.session_state.snapshots.insert(0, {
            "image": stamped_image,
            "label": label,
            "confidence": confidence,
            "time": now_dt.strftime("%d %B %Y, %I:%M:%S %p")
        })
        st.session_state.last_saved_time[label] = now
        st.session_state.snapshots = st.session_state.snapshots[:30]

def handle_weapon_frame(frame_bgr, annotated_pil, predictions, source_label):
    weapon_labels = ", ".join(sorted(set(p["class"] for p in predictions)))
    avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions) * 100

    faces = detect_faces(frame_bgr)

    if faces:
        for box in faces:
            signature = face_signature(frame_bgr, box)
            matched_id = find_matching_person_id(signature, st.session_state.known_faces)
            if matched_id is None and signature is not None:
                person_id = st.session_state.next_person_id
                st.session_state.next_person_id += 1
                st.session_state.known_faces.append({"person_id": person_id, "signature": signature})

                now_dt = datetime.now()
                stamped_image = stamp_timestamp_on_image(annotated_pil, now_dt)
                st.session_state.snapshots.insert(0, {
                    "image": stamped_image,
                    "label": f"Person {person_id} — {weapon_labels}",
                    "confidence": avg_confidence,
                    "time": now_dt.strftime("%d %B %Y, %I:%M:%S %p")
                })
                st.session_state.snapshots = st.session_state.snapshots[:30]
                send_alert(f"Person {person_id} — {weapon_labels}", avg_confidence, source=source_label)
    else:
        maybe_save_snapshot(annotated_pil, weapon_labels, avg_confidence)
        send_alert(weapon_labels, avg_confidence, source=source_label)

def start_video_recording(frame_bgr, label):
    h, w = frame_bgr.shape[:2]
    fname = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
    path = os.path.join(tempfile.gettempdir(), fname)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 8.0, (w, h))
    return {"writer": writer, "path": path, "frames_left": VIDEO_CLIP_SECONDS * 8, "label": label}

def finalize_video_recording(recording):
    recording["writer"].release()
    st.session_state.snapshots.insert(0, {
        "image": None,
        "video_path": recording["path"],
        "label": recording["label"],
        "confidence": None,
        "time": datetime.now().strftime("%d %B %Y, %I:%M:%S %p")
    })
    st.session_state.snapshots = st.session_state.snapshots[:30]

def show_gallery():
    if len(st.session_state.snapshots) == 0:
        st.info("No detections captured yet. Snapshots will appear here automatically.")
        return
    st.subheader(f"📸 Detection Gallery ({len(st.session_state.snapshots)} captured)")
    cols = st.columns(4)
    for i, snap in enumerate(st.session_state.snapshots):
        with cols[i % 4]:
            if snap.get("video_path"):
                st.video(snap["video_path"])
            else:
                st.image(snap["image"], use_container_width=True)
            conf_text = f"{snap['confidence']:.1f}%" if snap.get("confidence") is not None else "video clip"
            st.caption(f"**{snap['label']}** — {conf_text}\n\n{snap['time']}")

def process_and_show(image, conf_slider, source_label):
    with st.spinner("Analyzing..."):
        try:
            result = detect_weapon(image, confidence=conf_slider)
        except Exception as e:
            st.error(f"API Error: {e}")
            result = {"predictions": []}

    if "predictions" not in result:
        st.error("⚠️ Roboflow API returned an unexpected response (shown below). This usually means your API key, model ID, or free credits have an issue.")
        st.json(result)

    predictions = result.get("predictions", [])
    annotated_image_arr = draw_boxes(image, predictions)
    annotated_pil = Image.fromarray(annotated_image_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection Result")
        st.image(annotated_pil, use_container_width=True)

    if len(predictions) > 0:
        for pred in predictions:
            label = pred["class"]
            confidence = pred["confidence"] * 100
            st.error(f"🚨 WARNING: **{label}** detected! Confidence: **{confidence:.1f}%**")
            maybe_save_snapshot(annotated_pil, label, confidence)
            alert_response = send_alert(label, confidence, source=source_label)
            if alert_response:
                st.info("Alert sent via email ✅")
    else:
        st.success("✅ No weapon-like object detected.")

def run_live_stream(video_source, conf_slider, source_label, skip_frames=5):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("❌ Could not connect to the camera / CCTV stream. Check the URL, login, or camera index.")
        return

    st.markdown("### 🔴 Live Feed")
    scan_status = st.empty()
    stframe = st.empty()
    alert_placeholder = st.empty()
    stop_button = st.button("Stop Live Feed")
    gallery_placeholder = st.container()
    frame_count = 0

    while cap.isOpened():
        if stop_button:
            break
        ret, frame = cap.read()
        if not ret:
            st.warning("Stream ended or camera disconnected.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        try:
            result = detect_weapon(pil_frame, confidence=conf_slider)
        except Exception:
            result = {"predictions": []}

        predictions = result.get("predictions", [])
        annotated_arr = draw_boxes(pil_frame, predictions)
        annotated_pil = Image.fromarray(annotated_arr)

        stframe.image(annotated_pil, channels="RGB", use_container_width=True)

        if len(predictions) > 0:
            detected_items = ", ".join(sorted(set(p["class"] for p in predictions)))
            scan_status.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#3a0d0d,#1a0505);border:1px solid #ef4444;
                            border-radius:14px;padding:20px;text-align:center;">
                    <h2 style="color:#f87171;margin:0;letter-spacing:1px;">🚨 THREAT DETECTED</h2>
                    <p style="color:#fca5a5;font-size:16px;margin:6px 0 0 0;">Carrying: <b>{detected_items}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            for pred in predictions:
                label = pred["class"]
                confidence = pred["confidence"] * 100
                alert_placeholder.error(f"🚨 WARNING: **{label}** detected on {source_label}! Confidence: **{confidence:.1f}%**")
            handle_weapon_frame(frame, annotated_pil, predictions, source_label)

            if st.session_state.active_recording is None:
                st.session_state.active_recording = start_video_recording(frame, f"{source_label} — {detected_items}")
            rec = st.session_state.active_recording
            rec["writer"].write(frame)
            rec["frames_left"] -= 1
            if rec["frames_left"] <= 0:
                finalize_video_recording(rec)
                st.session_state.active_recording = None
        else:
            if st.session_state.active_recording is not None:
                finalize_video_recording(st.session_state.active_recording)
                st.session_state.active_recording = None
            scan_status.markdown(
                """
                <div style="background:linear-gradient(135deg,#062b1e,#031a12);border:1px solid #22c55e;
                            border-radius:14px;padding:20px;text-align:center;">
                    <h2 style="color:#4ade80;margin:0;letter-spacing:1px;">✅ AREA CLEAR</h2>
                    <p style="color:#86efac;font-size:15px;margin:6px 0 0 0;">No weapon detected — monitoring...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            alert_placeholder.empty()

        with gallery_placeholder:
            gallery_placeholder.empty()
            show_gallery()

        time.sleep(0.05)

    if st.session_state.active_recording is not None:
        finalize_video_recording(st.session_state.active_recording)
        st.session_state.active_recording = None

    cap.release()

def run_camera_grid(sources, conf_slider):
    caps = []
    for src in sources:
        caps.append({"name": src["name"], "cap": cv2.VideoCapture(src["source"]), "frame_count": 0})

    st.markdown("### 🔴 Live Grid Feed")
    stop_button = st.button("Stop Grid Monitoring")

    num_cols = 2
    rows_needed = (len(caps) + num_cols - 1) // num_cols
    frame_slots = {}
    status_slots = {}
    for row in range(rows_needed):
        cols = st.columns(num_cols)
        for c in range(num_cols):
            idx = row * num_cols + c
            if idx >= len(caps):
                break
            cam = caps[idx]
            with cols[c]:
                st.markdown(f"**📷 {cam['name']}**")
                status_slots[cam["name"]] = st.empty()
                frame_slots[cam["name"]] = st.empty()

    gallery_placeholder = st.container()

    while True:
        if stop_button:
            break

        any_alive = False
        for cam in caps:
            if not cam["cap"].isOpened():
                status_slots[cam["name"]].warning("Disconnected")
                continue
            ret, frame = cam["cap"].read()
            if not ret:
                status_slots[cam["name"]].warning("Stream ended")
                continue

            any_alive = True
            cam["frame_count"] += 1
            if cam["frame_count"] % 5 != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            try:
                result = detect_weapon(pil_frame, confidence=conf_slider)
            except Exception:
                result = {"predictions": []}

            predictions = result.get("predictions", [])
            annotated_arr = draw_boxes(pil_frame, predictions)
            annotated_pil = Image.fromarray(annotated_arr)
            frame_slots[cam["name"]].image(annotated_pil, channels="RGB", use_container_width=True)

            if predictions:
                status_slots[cam["name"]].error("🚨 THREAT DETECTED")
                handle_weapon_frame(frame, annotated_pil, predictions, cam["name"])
            else:
                status_slots[cam["name"]].success("✅ Clear")

        with gallery_placeholder:
            gallery_placeholder.empty()
            show_gallery()

        if not any_alive:
            st.warning("All camera streams have ended or disconnected.")
            break

        time.sleep(0.05)

    for cam in caps:
        cam["cap"].release()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Settings")
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.auth_view = "landing"
    st.rerun()

conf_slider = st.sidebar.slider("Confidence threshold (%)", 10, 100, 40)

mode = st.sidebar.radio(
    "Choose input type:",
    ["Upload Photo", "Live Camera (take picture)", "Upload Video", "Live CCTV / IP Camera"]
)

if st.sidebar.button("🗑️ Clear Gallery"):
    st.session_state.snapshots = []
    st.session_state.last_saved_time = {}
    st.session_state.known_faces = []
    st.session_state.next_person_id = 1

# ==================================================================
# MODE 1: UPLOAD PHOTO
# ==================================================================
if mode == "Upload Photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        process_and_show(image, conf_slider, "Photo Upload")
    st.markdown("---")
    show_gallery()

# ==================================================================
# MODE 2: LIVE CAMERA (single snapshot)
# ==================================================================
elif mode == "Live Camera (take picture)":
    camera_photo = st.camera_input("Take a live picture")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        process_and_show(image, conf_slider, "Live Camera")
    st.markdown("---")
    show_gallery()

# ==================================================================
# MODE 3: UPLOAD VIDEO
# ==================================================================
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        if st.button("Run Detection on Video"):
            run_live_stream(tfile.name, conf_slider, "Video Upload", skip_frames=10)
            os.unlink(tfile.name)
    else:
        st.markdown("---")
        show_gallery()

# ==================================================================
# MODE 4: LIVE CCTV / IP CAMERA (auto-connect, no brand guessing needed)
# ==================================================================
else:
    st.subheader("📹 Connect a CCTV / IP Camera")

    connection_type = st.radio(
        "How do you want to connect?",
        [
            "My laptop's own webcam",
            "A CCTV / IP Camera (auto-detect)",
            "I already have a full RTSP/stream URL",
            "Multiple Cameras (Grid View)",
        ],
        horizontal=False
    )

    final_source = "0"

    if connection_type == "My laptop's own webcam":
        st.info("Just click Start below — this will open your laptop's built-in camera.")
        final_source = "0"

    elif connection_type == "A CCTV / IP Camera (auto-detect)":
        st.write(
            "Enter only the camera's IP, username and password. "
            "The app will automatically test every common RTSP path "
            "(Hikvision, Dahua, CP Plus, Reolink, ONVIF, etc.) and use "
            "whichever one actually works — no need to know the brand or exact path."
        )

        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", value="admin")
            ip_address = st.text_input("Camera IP Address", placeholder="e.g. 192.168.1.64")
        with col2:
            password = st.text_input("Password", type="password")
            port = st.text_input("RTSP Port", value="554")

        if st.button("🔍 Test & Auto-Connect"):
            if not ip_address:
                st.error("Camera IP Address is required.")
            else:
                status_box = st.empty()
                log_lines = []

                def progress(msg):
                    log_lines.append(msg)
                    status_box.code("\n".join(log_lines[-8:]))

                with st.spinner("Testing RTSP paths against the camera..."):
                    try:
                        port_int = int(port) if str(port).strip().isdigit() else 554
                    except ValueError:
                        port_int = 554

                    result = auto_connect_camera(
                        ip=ip_address.strip(),
                        username=username.strip(),
                        password=password,
                        port=port_int,
                        progress_callback=progress,
                    )

                if result.success:
                    st.session_state.confirmed_rtsp_url = result.url
                    st.success(f"✅ Connected successfully. Working path found after {len(result.tried)} attempt(s).")
                    st.code(result.url.replace(password, "••••••") if password else result.url, language="text")
                else:
                    st.session_state.confirmed_rtsp_url = None
                    st.error(f"❌ Could not connect after trying {len(result.tried)} path(s).")
                    st.warning(result.diagnostic)

        if st.session_state.confirmed_rtsp_url:
            final_source = st.session_state.confirmed_rtsp_url
        else:
            final_source = None
            st.info("Run **Test & Auto-Connect** above first — Start will unlock once a working stream is confirmed.")

    elif connection_type == "I already have a full RTSP/stream URL":
        manual_url = st.text_input(
            "Paste your full RTSP/HTTP stream URL",
            placeholder="rtsp://username:password@192.168.1.64:554/..."
        )
        final_source = manual_url if manual_url else None

    else:  # Multiple Cameras (Grid View)
        st.write("Add each camera below, then start grid monitoring to watch all of them at once.")

        with st.form("add_camera_form", clear_on_submit=True):
            cam_name = st.text_input("Camera Name (e.g. 'Gate', 'Warehouse')")
            cam_type = st.selectbox("Camera Type", ["Auto-detect IP Camera", "Manual RTSP/URL", "This laptop's webcam"])
            cam_ip = st.text_input("IP Address (for auto-detect only)", placeholder="192.168.1.64")
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                cam_user = st.text_input("Username", value="admin")
            with gcol2:
                cam_pass = st.text_input("Password", type="password")
            cam_manual_url = st.text_input("Manual RTSP/HTTP URL (if using Manual option)")
            submitted = st.form_submit_button("➕ Add Camera to Grid")

            if submitted:
                if not cam_name:
                    st.error("Please give this camera a name.")
                elif cam_type == "This laptop's webcam":
                    st.session_state.camera_grid_sources.append({"name": cam_name, "source": 0})
                    st.success(f"Added '{cam_name}' (webcam).")
                elif cam_type == "Manual RTSP/URL":
                    if cam_manual_url:
                        st.session_state.camera_grid_sources.append({"name": cam_name, "source": cam_manual_url})
                        st.success(f"Added '{cam_name}'.")
                    else:
                        st.error("Paste a manual RTSP/HTTP URL.")
                else:
                    if cam_ip:
                        with st.spinner(f"Testing connection to '{cam_name}'..."):
                            result = auto_connect_camera(ip=cam_ip.strip(), username=cam_user.strip(), password=cam_pass, port=554)
                        if result.success:
                            st.session_state.camera_grid_sources.append({"name": cam_name, "source": result.url})
                            st.success(f"Connected and added '{cam_name}'.")
                        else:
                            st.error(f"Could not connect to '{cam_name}': {result.diagnostic}")
                    else:
                        st.error("Camera IP Address is required for auto-detect.")

        if st.session_state.camera_grid_sources:
            st.write("**Cameras in this grid:**")
            for i, cam in enumerate(st.session_state.camera_grid_sources):
                gcol1, gcol2 = st.columns([4, 1])
                gcol1.write(f"📷 {cam['name']}")
                if gcol2.button("Remove", key=f"remove_cam_{i}"):
                    st.session_state.camera_grid_sources.pop(i)
                    st.rerun()

            st.markdown("---")
            if st.button("▶️ Start Grid Monitoring"):
                run_camera_grid(st.session_state.camera_grid_sources, conf_slider)
        else:
            st.info("No cameras added to the grid yet.")

        final_source = None

    with st.expander("ℹ️ Why does the camera need to be on the same network?"):
        st.markdown("""
        - RTSP is a **local network protocol** — this app and the camera must be on the
          same WiFi/LAN for a direct IP connection to work.
        - If your camera is currently only reachable through a mobile app over
          mobile data (cloud/P2P mode), connect it to WiFi first and enable
          **RTSP / Local Network Login** in the camera's app settings — the local
          IP will then appear there.
        - Username/password here are the camera's **local login**, which is usually
          different from your cloud app account (Hik-Connect, Tapo, etc.).
        """)

    st.markdown("---")

    if connection_type != "Multiple Cameras (Grid View)":
        if st.button("Start Live Monitoring"):
            if final_source is not None:
                source = int(final_source) if str(final_source).strip().isdigit() else str(final_source).strip()
                run_live_stream(source, conf_slider, "CCTV Camera", skip_frames=5)
            else:
                st.error("Please confirm a working connection above first.")

    st.markdown("---")
    show_gallery()

st.markdown("---")
st.caption("ArgusAI — Final Bootcamp Project | Built with Roboflow API + Streamlit + n8n")