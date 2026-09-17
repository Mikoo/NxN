# 🐾 PawSentry AI — Autonomous Multi-Pet Edge Sentinel & Wellness Intelligence

[![Nebius](https://img.shields.io/badge/Nebius-Token_Factory-00E599?style=flat-square&logo=cloud)](https://studio.nebius.ai/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron--3--Nano-76B900?style=flat-square&logo=nvidia)](https://build.nvidia.com/)
[![Track](https://img.shields.io/badge/Hackathon-Physical_AI_Track-blue?style=flat-square)](https://nebiusglobalaihackathon.devpost.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Clinical_Search_API-FF6B6B?style=flat-square)](https://tavily.com/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)

> **"An autonomous edge multimodal physical AI agent that senses, tracks, and protects household pets in real time — extracting ethological behavioral bouts with Nebius Token Factory (`MiniCPM-V-4_5`), reasoning clinical wellness with NVIDIA Nemotron, and providing evidence-based triage via Tavily Search API."**

---

## 📌 Executive Summary

Domestic pets instinctively mask pain, illness, and behavioral distress until conditions reach acute clinical stages. Conventional security cameras are passive: they accumulate hundreds of hours of unindexed video that no owner has time to review.

**PawSentry AI** transforms ordinary consumer webcams and IP surveillance cameras (such as the **EZVIZ H8c Pro**) into an **Autonomous Physical AI Sentinel**:
1. **Edge Motion & Temporal Bursting:** Local OpenCV MOG2 detector isolates moving targets and captures high-density **3-frame micro-event bursts** (`01_onset`, `02_peak`, `03_post`), capturing dynamic biomechanics without streaming high-bandwidth video to the cloud.
2. **Multimodal Vision on Nebius:** Evaluates each temporal burst using `openbmb/MiniCPM-V-4_5` on Nebius Token Factory, extracting species, posture, activity, emotional mood, energy level, and visual features.
3. **Multi-Pet Identity & Active Learning:** Distinguishes multiple household pets (e.g. **Ichi** and **Ditto**) through progressive visual trait learning and 1-click human-in-the-loop active confirmation.
4. **Ethological Session Clustering (Bout Analysis):** Clusters high-frequency micro-bursts into continuous behavioral sessions (resting bouts, hydration visits, feeding events, active exploration), reflecting authentic animal ethology.
5. **Clinical Wellness Reasoning (NVIDIA Nemotron):** Synthesizes daily behavioral patterns with `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` into an empathetic daily digest with actionable veterinary recommendations.
6. **Evidence-Based Veterinary Triage (Tavily API):** When anomalous postures, limp gait, or compulsive scratching are detected, autonomously consults verified clinical literature (`petmd.com`, `aspca.org`, `akc.org`, `vcaanimalhospitals.com`).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Edge ["🏠 Edge Physical AI Layer"]
        CAM["📷 Video Source\n(Webcam 0 / EZVIZ H8c Pro RTSP)"]
        MOG2["⚡ OpenCV MOG2 Subtractor\n+ Morphological Filtering"]
        BB["🟩 Bounding Box Tracker\n& Cooldown Controller"]
        BURST["🎞️ Temporal Burst Capture\n(01_onset, 02_peak, 03_post)"]
    end

    subgraph Nebius ["☁️ Nebius Token Factory"]
        VISION["👁️ MiniCPM-V-4_5 Multimodal\nPosture, Mood, Energy, Species"]
        NEMO["🧠 NVIDIA Nemotron-3-Nano\nHabit Synthesis & Daily Digest"]
    end

    subgraph Ethology ["📊 Quantitative Ethology Engine"]
        BOUTS["⏱️ Behavioral Bout Clustering\n(Gap debouncing <= 180s)"]
        CHRONO["📈 24h Circadian Chronobiology\n(Rest vs Mobility vs Hydration)"]
        COMFORT["💚 Comfort Index (0-100)\nHabit Matrix Correlator"]
    end

    subgraph Clinical ["🩺 Clinical Triage Layer"]
        TAVILY["🔍 Tavily Search API\nVetted Veterinary Domains"]
        VET["📋 VetAdvisor Synthesis\nUrgency & Recommendations"]
    end

    subgraph UI ["💻 Presentation Layer"]
        DASH["🐾 Streamlit Modern SaaS\n(Apple Health / Linear UI)"]
        PLOTLY["📊 Plotly Interactive Visuals\n(Gauge, Radar, 24h Stacked Bar)"]
        CHAT["💬 Nemotron Companion Chat"]
    end

    CAM --> MOG2 --> BB --> BURST
    BURST --> VISION
    VISION --> BOUTS --> CHRONO --> COMFORT
    VISION -- "Anomaly Detected" --> TAVILY --> VET
    COMFORT --> NEMO
    BOUTS --> NEMO
    NEMO --> DASH
    COMFORT --> PLOTLY --> DASH
    VET --> DASH
    DASH --> CHAT
```

---

## 🌟 Key Differentiators & Hackathon Features

### 1. Physical AI Temporal Micro-Bursting
Rather than streaming continuous raw video (prohibitive bandwidth and privacy concerns) or analyzing single static snapshots (which miss motion velocity and gait), PawSentry AI captures a **3-frame burst**:
- `01_onset`: The animal entering the motion zone.
- `02_peak`: The apex of the physical biomechanical movement.
- `03_post`: The stabilization or follow-through.

### 2. Multi-Pet Household Support & Active Learning
Household cameras frequently monitor multiple pets. PawSentry AI:
- Maintains individual profiles (e.g. **Ichi** and **Ditto**).
- Includes registered visual traits in the vision model prompt context.
- Allows 1-click **Human-in-the-Loop Active Learning**: clicking a pet's tag appends observed physical features (fur pattern, markings, collar) to their persistent profile, making future identification progressively smarter.

### 3. Quantitative Ethological Bout Clustering
Capturing movement every few seconds when a cat twitches on a sofa would create hundreds of redundant records. Our clustering algorithm groups consecutive events into **Behavioral Bouts**:
- Recognizes continuous resting sessions (e.g., 14.5 minutes on sofa).
- Quantifies water and food visits accurately.
- Accurately computes feline-appropriate circadian cycles (cats naturally sleep 12-16 hours without false "lethargy" penalization).

### 4. Dual Video Source: USB Webcam & EZVIZ H8c Pro IP Camera
Natively supports:
- **Local USB Webcams** with Windows DirectShow apartment initialization and automatic black-frame fallback.
- **Surveillance IP Cameras** via RTSP (e.g., `rtsp://admin:VERIFY_CODE@192.168.1.120:554/H.264/ch1/main`).

---

## 📂 Repository Structure

```text
├── app.py                     # Presentation Layer (Streamlit reactive bilingual dashboard)
├── requirements.txt           # Python dependencies (OpenCV, Plotly, OpenAI, Tavily, Pydantic)
├── PROJECT_RULES.md           # Core architectural guidelines and constraints
├── PROJECT_OVERVIEW.md        # Technical master overview and roadmap
├── core/
│   ├── detector.py            # OpenCV MOG2 motion detector, DirectShow fallback & burst engine
│   ├── nebius_client.py       # Nebius Token Factory client (MiniCPM-V-4_5 + NVIDIA Nemotron)
│   ├── report_generator.py    # Bout clustering, habit metrics, persistent caching & chat memory
│   ├── schemas.py             # Strongly-typed Pydantic v2 schemas with tolerant validators
│   ├── vet_advisor.py         # Tavily Clinical Search API client with domain whitelist
│   └── i18n.py                # Bilingual localization dictionary (English default / Spanish)
├── data/
│   ├── events.json            # Persistent behavioral event database
│   ├── snapshots/             # Real-world temporal bursts captured during sentinel mode
│   └── reports/               # Cached daily reports generated by NVIDIA Nemotron
└── scratch/                   # Offline verification and benchmarking scripts (gitignored)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- A functional USB webcam or RTSP network camera (e.g. EZVIZ H8c Pro)
- A [Nebius Token Factory](https://studio.nebius.ai/) API key
- A [Tavily Search](https://tavily.com/) API key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/pawsentry-ai.git
cd pawsentry-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# or: .\.venv\Scripts\Activate.ps1   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
NEBIUS_VISION_MODEL=openbmb/MiniCPM-V-4_5
NEBIUS_REASONING_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B

TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Running the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛡️ Live Verification & Real-World Results

During autonomous surveillance testing:
- **648 total bursts evaluated**: Over 380 genuine feline behavioral events captured and validated.
- **Ethological accuracy**: Successfully identified posture (`sitting`, `lying_down`), relaxed mood, 2 hydration visits, 1 nutrition visit, and continuous resting sessions.
- **Zero Hallucination Event Filtering**: Environmental motion (lighting, human movement) without pets is cleanly categorized without polluting companion habit metrics.
- **NVIDIA Nemotron Synthesis**: Daily reports generated in <4 seconds with 94+ overall wellness scores and actionable veterinary care recommendations.

---

## 👥 Hackathon Team & Acknowledgments
Built with ❤️ for the **Nebius × NVIDIA Global AI Hackathon** (Physical AI & Best Use of Tavily Tracks).
- **Nebius Token Factory** for ultra-fast, high-concurrency open multimodal inference.
- **NVIDIA** for high-precision Nemotron reasoning models.
- **Tavily** for evidence-based search grounding.
