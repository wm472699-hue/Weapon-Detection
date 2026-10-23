# 🛡️ ArgusAI — AI-Based Weapon Detection System

ArgusAI is a real-time security monitoring web app that uses AI to detect weapons (firearms and edged weapons) in photos, videos, and live CCTV / IP camera feeds. When a threat is confirmed, it saves timestamped evidence automatically and tracks each unique person only once, so there are no duplicate alerts.

## ✨ Features

- **Multiple input modes**
  - Upload a photo
  - Take a live picture from the webcam
  - Upload a video and scan it frame by frame
  - Connect a live CCTV / IP camera
- **AI weapon detection** using a Roboflow-trained model, with an adjustable confidence threshold
- **Person detection** (OpenCV HOG) drawn alongside weapon detections
- **Smart face de-duplication** — one snapshot per unique person instead of hundreds of duplicates
- **Automatic evidence recording** — a short video clip is saved the moment a weapon is detected
- **Timestamped snapshots** — date and time are burned directly into every saved image
- **Detection gallery** with confidence scores and timestamps
- **Camera auto-connect** — tries the common RTSP paths for Hikvision, Dahua, CP Plus, Reolink, ONVIF and generic cameras, and uses whichever works
- **Multi-camera grid view** — monitor several cameras at once
- **Secure accounts** — sign up and log in with username or email; passwords are stored hashed (SHA-256)
- **Email alerts** via an n8n webhook (optional, disabled by default)
- **Dark, professional UI** with a landing page, login / signup flow, and monitoring dashboard

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Web app | Streamlit |
| Computer vision | OpenCV, NumPy, Pillow |
| Weapon detection | Roboflow Inference API |
| Alert automation | n8n (webhook) |
| Language | Python 3 |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/wm472699-hue/Weapon-Detection.git
cd Weapon-Detection
```

### 2. Install dependencies

```bash
pip install streamlit opencv-python numpy pillow requests
```

### 3. Configure your keys

Open `app.py` and set your own values in the CONFIG section:

```python
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
ROBOFLOW_MODEL_ID = "your-model-id"
ROBOFLOW_MODEL_VERSION = "1"
N8N_WEBHOOK_URL = "YOUR_N8N_WEBHOOK_URL"   # optional
ENABLE_EMAIL_ALERTS = False                # set True once n8n is running
```

> ⚠️ Never commit real API keys to a public repository.

### 4. Run the app

```bash
streamlit run app.py
```

On Windows, if `streamlit.exe` is blocked by an Application Control policy, use:

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## 📖 How to Use

1. Create an account on the Sign Up page, then log in.
2. Choose an input type from the sidebar: photo, live camera, video, or CCTV / IP camera.
3. For a CCTV camera, enter its IP address, username and password, then click **Test & Auto-Connect**.
4. Click **Start Live Monitoring**. Detections appear on screen and are saved to the gallery with a snapshot and a video clip.

> The computer and the camera must be on the same network for RTSP connections to work.

## 📁 Project Structure

```
Weapon-Detection/
├── app.py          # Main Streamlit application
├── users.json      # Local user accounts (created automatically, do not commit)
└── README.md
```

## 🔮 Future Improvements

- Cloud camera support (P2P / cloud-only cameras)
- Persistent storage for detection history
- SMS / push notification alerts
- Role-based access for multiple operators

## ⚠️ Disclaimer

This project is built for educational and research purposes as a bootcamp final project. It is not a certified security product and should not be the sole safeguard in any real-world safety system.

## 👤 Author

Built by **Muzammal Mehar**
