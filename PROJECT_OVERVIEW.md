# 🐾 PawSentry AI — Master Project Overview & System Architecture

> **"An edge multimodal AI agent for real-time pet behavioral monitoring and wellness analytics powered by NVIDIA Nemotron on Nebius open infrastructure."**

---

## 📌 1. Hackathon & Submission Overview
* **Event:** [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Devpost)
* **Prizes in scope:** +$50,000 USD Grand/Place Prizes + NVIDIA Jetson Orin Nano + **$3,000 USD Best Use of Tavily**.
* **Primary Track:** **Physical AI Track** *(Embodied and edge agents, IoT, camera sensing, temporal micro-events, hybrid edge/cloud inference)*.
* **Secondary Track Alignment:** **Best Apps and Agents Track** & **Personal AI**.
* **Presentation Language:** **English by Default** *(mandatory for global Devpost judging and 3-minute video presentation)* with multi-language support (English / Spanish).
* **Official Branding:** Validated 3:2 Hackathon Thumbnail generated and active on Devpost (`pawsentry_thumbnail.jpg`).

---

## 🎯 2. Core Value Proposition & Physical AI Differentiators

### The Challenge
Pets cannot verbally communicate pain, fatigue, or illness, and instinctively mask discomfort until acute stages. Traditional security cameras are passive and record hundreds of hours of raw footage without semantic comprehension.

### The PawSentry AI Solution
1. **Edge Sensing with Temporal Micro-Event Bursting:**
   - Instead of streaming continuous high-bandwidth video, a local OpenCV engine (`core/detector.py`) with MOG2 background subtraction and a circular pre-roll buffer captures **3-frame micro-event bursts** (`01_onset`, `02_peak`, `03_post`).
   - Enables NVIDIA Vision models to analyze physical trajectory, biomechanics (e.g. limping, sudden jumps, eating posture) rather than just static snapshots.
2. **Cloud Multimodal & Reasoning on Nebius Token Factory:**
   - **Vision Ingestion:** Serves `openbmb/MiniCPM-V-4_5` on Nebius for high-speed multi-frame classification of pet activity, posture, and mood into Pydantic v2 schemas.
   - **Reasoning Synthesis:** Serves `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (and `nvidia/Nemotron-3-Ultra-550b-a55b`) for deep habit correlation, comfort indexing, and empathetic narrative generation.
3. **Evidence-Based Veterinary Triage via Tavily API:**
   - When anomalous behavior is flagged (e.g. compulsive ear scratching, sudden >30% drop in activity), the system autonomously queries authoritative clinical domains (`petmd.com`, `aspca.org`, `akc.org`, `vcaanimalhospitals.com`), attaching verified sources and urgency levels.
4. **Habit Matrix & Proactive Alerts:**
   - Correlates water visits, food visits, estimated sleep/active hours, and calculates the **Comfort Index (0-100)** with alerts when metrics deviate significantly from baseline.

---

## 🏗️ 3. Architecture & Modular File Tree

Strictly decoupled layer architecture conforming to `PROJECT_RULES.md`:

```
g:/Users/mikit/Workspace/NxN/
├── .env                       # Live credentials (NEBIUS_API_KEY, TAVILY_API_KEY, models)
├── .env.example               # Clean sanitized template for public open-source submission
├── PROJECT_RULES.md           # Engineering guidelines (Strict typing, Pydantic v2, Mock fallback)
├── PROJECT_OVERVIEW.md        # Master documentation and status (This file)
├── README.md                  # Public-facing documentation for GitHub repository
├── requirements.txt           # Python dependencies (OpenCV, OpenAI, Tavily, Streamlit, Pydantic)
├── pawsentry_thumbnail.jpg    # Official 3:2 hackathon thumbnail
├── camera_test.py             # Hardware verification utility (Windows DirectShow, cams 0 & 1)
├── core/
│   ├── __init__.py
│   ├── schemas.py             # Strongly typed Pydantic v2 models (BehaviorAnalysis, DailyPetReport, etc.)
│   ├── detector.py            # Edge motion detector, MOG2, circular pre-roll & 3-frame burst engine
│   ├── nebius_client.py       # Nebius Token Factory client with NVIDIA Nemotron & MiniCPM-V (Mock & Live)
│   ├── vet_advisor.py         # Tavily Search API client with vetted veterinary domain filtering
│   └── report_generator.py    # Habit matrix calculator, daily report synthesis & chat memory
├── data/
│   ├── events.json            # Local persistent event log
│   └── snapshots/             # Timestamped captured micro-event bursts
└── app.py                     # Presentation Layer (Streamlit reactive dashboard)
```

---

## 🟢 4. Production Verification Status (Tested & Validated)

All credentials and models have been tested in production with live credits:
* **Nebius Token Factory:** Authenticated (`HTTP 200 OK`).
  * Vision Model: `openbmb/MiniCPM-V-4_5` — Tested live with real webcam frames.
  * Reasoning Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` — Tested live for narrative and chat.
  * Ultra Model: `nvidia/Nemotron-3-Ultra-550b-a55b` — Configured for deep reports.
* **Tavily Search API:** Authenticated (`HTTP 200 OK`).
  * Verified live queries returning medical sources from PetMD, GoodRx Pet, and Animal Hospitals.
* **Hardware & Edge:**
  * DirectShow captures on camera indices 0 and 1 validated on Windows.
  * 3-frame temporal burst (`onset`, `peak`, `post`) functioning with 0.1-3.0s cooldown.
* **Current App:**
  * Running locally on `http://localhost:8501`.

---

## 🎨 5. Next Stage: High-End UI/UX Redesign (Hand-off to Claude Sonnet)

### Objective
Redesign `app.py` into a world-class, competition-winning dashboard designed specifically for judges and demo videos.

### Key Requirements for the UI Redesign:
1. **Internationalization (i18n):**
   - **Default Language: English (`en`)**, with an instant toggle to Spanish (`es`).
   - All labels, metrics, cards, chat prompts, and system messages must support both languages smoothly (e.g. via a clean dictionary mapping in `core/i18n.py` or within `app.py`).
2. **Visual Identity & Design System:**
   - **Palette:** Cyber-emerald (`#10B981`), Tech Cyan (`#06B6D4`), Deep Slate/Navy (`#0B1329` and `#1E293B`), Alert Amber (`#F59E0B`), Critical Red (`#EF4444`).
   - **Aesthetics:** Glassmorphism (`backdrop-filter: blur(12px)`), high-contrast accessible typography, rounded modern cards (`16px`), subtle glows matching `pawsentry_thumbnail.jpg`.
3. **Module Layouts & Polish:**
   - **Live Physical AI HUD:** Visual frame with corner HUD reticles, live blinking status pill (`PHYSICAL AI SENSING: ACTIVE`), camera selector, and dual action triggers (Instant Burst vs Single Shot).
   - **Micro-Event Temporal Showcase:** Visual 3-frame carousel (`01 Onset` -> `02 Peak` -> `03 Post`) with smooth badges for Activity, Posture, Mood, and Energy Gauge.
   - **Habit Matrix Analytics:** Plotly or Altair interactive radial gauge / radar chart for Comfort Index, hydration frequency vs baseline, and sleep/activity breakdown.
   - **Clinical Triage Accordion:** Clean medical alert cards highlighting Tavily citations with clickable badge links to PetMD/ASPCA.
   - **Interactive Nemotron Chat:** Floating / streamlined chat panel with preset suggestion chips (*"How did Toby sleep today?"*, *"Did he drink enough water?"*, *"Any health anomalies detected?"*).
4. **Skills & Guidelines for the Design Model:**
   - Use `generative_ui` principles: self-contained clean CSS, CSS custom properties, semantic contrast, responsive containers.
   - Maintain strict separation of concerns: UI must remain in `app.py` and never import OpenCV internals or execute raw HTTP calls outside `core/`.

---
*Generated by Senior AI Architect & Lead Engineer for PawSentry AI.*
