"""
app.py — PawSentry AI · Autonomous Multi-Pet Edge Sentinel
Presentation layer only. Business and AI logic lives in core/.
Full bilingual support (English default for hackathon judges, Spanish switch).
"""
import sys, io, base64, time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.detector import PetMotionDetector
from core.i18n import t, t_val
from core.nebius_client import NebiusClient
from core.report_generator import ReportGenerator
from core.schemas import (
    BehaviorAnalysis,
    BehavioralBout,
    HabitMatrixMetrics,
    MoodType,
    PetActivityType,
    PetProfile,
    UrgencyLevel,
)
from core.vet_advisor import VetAdvisor

st.set_page_config(
    page_title="PawSentry AI — Multi-Pet Sentry",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Canvas ─────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
.stApp { background: #0F1117 !important; }
.block-container { padding: 0 2.2rem 3rem !important; max-width: 1340px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0F1117; }
::-webkit-scrollbar-thumb { background: #252D40; border-radius: 5px; }

/* ── Top nav ─────────────────────────────────────────────────── */
.topnav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 0 16px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  margin-bottom: 20px;
}
.brand-logo svg { display: block; }

/* ── Live pill ───────────────────────────────────────────────── */
@keyframes soft-pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 14px; border-radius: 999px;
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  border: 1px solid;
}
.pill-live { background: rgba(52,211,153,0.10); border-color: rgba(52,211,153,0.28); color: #34D399; }
.pill-mock { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.28); color: #F59E0B; }
.pill-dot  { width:6px; height:6px; border-radius:50%; background:currentColor; animation: soft-pulse 2s ease-in-out infinite; }
.pill-mock .pill-dot { animation: none; }

/* ── Settings expander ───────────────────────────────────────── */
div[data-testid="stExpander"] {
  background: #1A2035 !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
  margin-bottom: 18px !important;
}
div[data-testid="stExpander"] summary {
  font-size: 0.82rem !important; color: #7A8BAD !important; font-weight: 600 !important;
}

/* ── Profile card ─────────────────────────────────────────────── */
.profile-card {
  background: linear-gradient(145deg, #1E2235 0%, #1A1A2E 100%);
  border: 1px solid rgba(245,158,11,0.22);
  border-radius: 20px; padding: 22px;
  box-shadow: 0 4px 24px rgba(245,158,11,0.07), 0 1px 4px rgba(0,0,0,0.4);
  position: relative; overflow: hidden;
}
.profile-card::after {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 160px; height: 160px; border-radius: 50%;
  background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.profile-photo {
  width: 80px; height: 80px; border-radius: 50%;
  background: linear-gradient(135deg, #F59E0B 0%, #FB923C 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 2.2rem; flex-shrink: 0;
  border: 3px solid rgba(245,158,11,0.40);
  box-shadow: 0 0 0 6px rgba(245,158,11,0.07), 0 4px 12px rgba(245,158,11,0.15);
  overflow: hidden;
}
.profile-photo img { width:100%; height:100%; object-fit:cover; border-radius:50%; }
.profile-name  { font-size:1.3rem; font-weight:800; color:#EFF2F8; letter-spacing:-0.02em; line-height:1.2; }
.profile-meta  { font-size:0.76rem; color:#7A8BAD; margin-top:3px; }
.profile-desc  { font-size:0.73rem; color:#94A3B8; margin-top:4px; line-height:1.4; }
.profile-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.25);
  border-radius: 999px; padding: 3px 12px; margin-top: 8px;
  font-size: 0.66rem; font-weight: 700; color: #34D399;
  text-transform: uppercase; letter-spacing: 0.08em;
}
.profile-stats {
  display: flex; gap: 18px; margin-top: 16px; padding-top: 14px;
  border-top: 1px solid rgba(255,255,255,0.06);
  flex-wrap: wrap;
}
.pstat-val { font-size: 1.1rem; font-weight: 800; color: #F59E0B; }
.pstat-lbl { font-size: 0.6rem; color: #7A8BAD; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

/* ── KPI cards ────────────────────────────────────────────────── */
.kpi {
  background: #1A2035;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 22px 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.3);
  position: relative; overflow: hidden; height: 100%;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi:hover { border-color: rgba(245,158,11,0.25); box-shadow: 0 4px 20px rgba(245,158,11,0.08); }
.kpi-bg { position:absolute; right:14px; bottom:10px; font-size:2.8rem; opacity:0.055; pointer-events:none; }
.kpi-icon  { font-size:1.2rem; margin-bottom:10px; }
.kpi-label { font-size:0.67rem; font-weight:600; text-transform:uppercase; letter-spacing:0.10em; color:#7A8BAD; margin-bottom:5px; }
.kpi-value { font-size:1.9rem; font-weight:800; line-height:1; }
.kpi-unit  { font-size:0.95rem; font-weight:500; color:#7A8BAD; margin-left:2px; }
.kpi-sub   { font-size:0.71rem; color:#7A8BAD; margin-top:7px; }
.kpi-green .kpi-value { color:#34D399; }
.kpi-amber .kpi-value { color:#F59E0B; }
.kpi-peach .kpi-value { color:#FB923C; }
.kpi-lav   .kpi-value { color:#A78BFA; }

/* ── Sentry Control Banner ───────────────────────────────────── */
.sentry-panel {
  background: linear-gradient(145deg, #182033 0%, #131726 100%);
  border: 1px solid rgba(52,211,153,0.25);
  border-radius: 18px; padding: 18px 22px;
  margin-bottom: 18px; position: relative;
}
.sentry-panel-standby {
  background: linear-gradient(145deg, #1A2035 0%, #151929 100%);
  border: 1px solid rgba(255,255,255,0.08);
}
.sentry-status-bar {
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(0,0,0,0.35); border-radius: 10px;
  padding: 8px 14px; margin-top: 10px;
  font-size: 0.78rem; font-weight: 600; color: #CBD5E1;
}
.sentry-status-alert {
  background: rgba(245,158,11,0.18); border: 1px solid rgba(245,158,11,0.4);
  border-radius: 10px; padding: 10px 14px; margin-top: 10px;
  font-size: 0.85rem; font-weight: 700; color: #F59E0B; text-align: center;
}

/* ── Section titles ──────────────────────────────────────────── */
.stitle {
  font-size: 0.73rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.10em; color: #7A8BAD;
  margin: 22px 0 12px; display: flex; align-items: center; gap: 8px;
}
.stitle::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.06); }

/* ── Badges ──────────────────────────────────────────────────── */
.pb {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 6px;
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  margin-right: 5px; margin-bottom: 3px;
}
.pb-g   { background:rgba(52,211,153,0.12); color:#34D399; border:1px solid rgba(52,211,153,0.25); }
.pb-r   { background:rgba(251,113,133,0.12); color:#FB7185; border:1px solid rgba(251,113,133,0.25); }
.pb-a   { background:rgba(245,158,11,0.12); color:#F59E0B; border:1px solid rgba(245,158,11,0.25); }
.pb-v   { background:rgba(167,139,250,0.12); color:#A78BFA; border:1px solid rgba(167,139,250,0.25); }
.pb-c   { background:rgba(56,189,248,0.12);  color:#38BDF8; border:1px solid rgba(56,189,248,0.25); }
.pb-m   { background:rgba(255,255,255,0.06); color:#7A8BAD; border:1px solid rgba(255,255,255,0.09); }
.pb-pet { background:rgba(245,158,11,0.15); color:#FBBF24; border:1px solid rgba(245,158,11,0.35); }

/* ── Timeline items ──────────────────────────────────────────── */
.moment {
  background: #1A2035; border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px; padding: 16px; margin-bottom: 12px;
  transition: border-color 0.15s, transform 0.15s;
}
.moment:hover { border-color: rgba(245,158,11,0.22); transform: translateY(-1px); }
.moment-alert {
  border-color: rgba(251,113,133,0.30) !important;
  background: linear-gradient(135deg, #1F1B2A 0%, #1A2035 100%) !important;
}
.mbody { display: flex; flex-direction: column; justify-content: space-between; height: 100%; }
.mts   { font-size: 0.69rem; color: #7A8BAD; font-weight: 500; }
.mdiag { font-size: 0.95rem; font-weight: 700; color: #EFF2F8; margin: 4px 0 2px; }
.msub  { font-size: 0.77rem; color: #94A3B8; line-height: 1.4; }
.ebar  { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.elbl  { font-size: 0.65rem; color: #7A8BAD; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.etrack{ flex: 1; height: 4px; background: rgba(255,255,255,0.08); border-radius: 999px; overflow: hidden; max-width: 140px; }
.efill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #34D399, #F59E0B); }
.eval  { font-size: 0.70rem; font-weight: 700; color: #F59E0B; }
.alert-strip {
  background: rgba(251,113,133,0.08); border: 1px solid rgba(251,113,133,0.22);
  border-radius: 10px; padding: 9px 14px; margin-top: 10px;
  font-size: 0.76rem; color: #FCA5A5;
}

/* ── Active Tagging Pill Bar ─────────────────────────────────── */
.tag-bar {
  display: flex; align-items: center; gap: 8px; margin-top: 10px;
  padding: 10px 14px; background: rgba(245,158,11,0.06);
  border: 1px dashed rgba(245,158,11,0.28); border-radius: 12px;
}
.tag-title { font-size: 0.74rem; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Digest & Report cards ───────────────────────────────────── */
.rcard {
  background: #1A2035; border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px; padding: 20px; margin-bottom: 14px;
}
.rcard-title {
  font-size: 0.73rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #7A8BAD; margin-bottom: 12px;
}
.score-hero {
  background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(245,158,11,0.08) 100%);
  border: 1px solid rgba(52,211,153,0.25); border-radius: 20px;
  padding: 28px 24px; text-align: center;
}
.score-num { font-size: 4rem; font-weight: 900; color: #34D399; line-height: 1; letter-spacing: -0.03em; }
.score-lbl { font-size: 0.73rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em; color: #7A8BAD; margin-top: 6px; }
.ritem {
  display: flex; align-items: flex-start; gap: 10px;
  font-size: 0.81rem; color: #CBD5E1; line-height: 1.55;
  padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ritem:last-child { border-bottom: none; padding-bottom: 0; }
.rdot { width: 5px; height: 5px; border-radius: 50%; background: #F59E0B; margin-top: 7px; flex-shrink: 0; }
.rdot-r { background: #FB7185; }

/* ── Household Comparison Card ───────────────────────────────── */
.comp-card {
  background: #161C2C; border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 16px; margin-bottom: 14px;
}
.comp-title { font-size: 0.88rem; font-weight: 700; color: #EFF2F8; margin-bottom: 6px; }

/* ── Form and input styling ──────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
  color: #0F1117 !important; font-weight: 700 !important;
  border: none !important; border-radius: 11px !important;
  box-shadow: 0 4px 14px rgba(245,158,11,0.22) !important;
}
.stButton > button:not([kind="primary"]) {
  background: #1E2840 !important; border: 1px solid rgba(255,255,255,0.09) !important;
  color: #CBD5E1 !important; border-radius: 11px !important;
  font-weight: 500 !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: rgba(245,158,11,0.40) !important; color: #F59E0B !important;
}

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stTabs"] [role="tab"] { font-weight: 500 !important; color: #7A8BAD !important; font-size: 0.87rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #EFF2F8 !important; border-bottom: 2px solid #F59E0B !important; font-weight: 600 !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #1E2840 !important; border-color: rgba(255,255,255,0.09) !important;
  color: #EFF2F8 !important; border-radius: 10px !important;
}
.stSelectbox > div > div {
  background: #1E2840 !important; border-color: rgba(255,255,255,0.09) !important;
  color: #EFF2F8 !important; border-radius: 10px !important;
}
.stChatMessage[data-testid*="user"] {
  background: rgba(245,158,11,0.06) !important; border: 1px solid rgba(245,158,11,0.16) !important;
  border-radius: 14px !important;
}
.stChatMessage[data-testid*="assistant"] {
  background: #1A2035 !important; border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Multi-Pet Roster (Default to Ichi and Ditto)
if "pets" not in st.session_state:
    st.session_state.pets: List[PetProfile] = [
        PetProfile(
            pet_id="ichi",
            name="Ichi",
            species="Cat",
            breed="Domestic Cat",
            age_years=2,
            description="First cat in room. Edge vision learns visual traits upon confirmation.",
        ),
        PetProfile(
            pet_id="ditto",
            name="Ditto",
            species="Cat",
            breed="Domestic Cat",
            age_years=2,
            description="Second cat in room. Edge vision learns visual traits upon confirmation.",
        ),
    ]

if "active_pet_id" not in st.session_state:
    st.session_state.active_pet_id = "ichi"

if "camera_source_mode" not in st.session_state:
    st.session_state.camera_source_mode = "usb"
if "camera_index" not in st.session_state or st.session_state.camera_index != 0:
    st.session_state.camera_index = 0
if st.session_state.get("cam_index_select") == 1:
    st.session_state.cam_index_select = 0
    st.session_state.camera_index = 0
if "rtsp_url" not in st.session_state:
    st.session_state.rtsp_url = "rtsp://admin:VERIFICATION_CODE@192.168.1.100:554/H.264/ch1/main"
if "min_motion_area" not in st.session_state:
    st.session_state.min_motion_area = 3500
if "cooldown" not in st.session_state:
    st.session_state.cooldown = 3.0
if "sentry_active" not in st.session_state:
    st.session_state.sentry_active = False
if "last_sentry_burst" not in st.session_state:
    st.session_state.last_sentry_burst = []
if "last_sentry_analysis" not in st.session_state:
    st.session_state.last_sentry_analysis = None
if "timeline_filter" not in st.session_state:
    st.session_state.timeline_filter = "active"
if "chip_query" not in st.session_state:
    st.session_state.chip_query = ""

if "generator" not in st.session_state:
    st.session_state.nebius    = NebiusClient()
    st.session_state.vet       = VetAdvisor()
    st.session_state.generator = ReportGenerator(
        nebius_client=st.session_state.nebius,
        vet_advisor=st.session_state.vet,
    )

lang: str = st.session_state.lang

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": t("chat_greeting", lang=lang)}
    ]

def get_resolved_camera_source() -> int | str:
    """Retorna la fuente de camara configurada (indice int para USB o URL str para RTSP)."""
    if st.session_state.camera_source_mode == "rtsp":
        url = st.session_state.get("rtsp_url", "").strip()
        return url if url else 0
    return int(st.session_state.get("camera_index", 0))

# Ensure active pet is valid
active_pet: PetProfile = next(
    (p for p in st.session_state.pets if p.pet_id == st.session_state.active_pet_id),
    st.session_state.pets[0],
)

# ── HELPERS ───────────────────────────────────────────────────────────────────
_ICONS = {"Dog":"🐕","Cat":"🐈","Perro":"🐕","Gato":"🐈","Other":"🐾","Otro":"🐾"}
_ACT_ICONS = {
    PetActivityType.DRINKING: "💧",
    PetActivityType.EATING: "🍽️",
    PetActivityType.SLEEPING: "😴",
    PetActivityType.RESTING: "🛋️",
    PetActivityType.PLAYING: "🎾",
    PetActivityType.WALKING: "🐾",
    PetActivityType.SCRATCHING: "⚡",
    PetActivityType.PACING: "🔄",
    PetActivityType.ANXIOUS_BEHAVIOR: "⚠️",
    PetActivityType.UNKNOWN: "🐾",
}

def _pb(text: str, cls: str = "pb-m") -> str:
    return f"<span class='pb {cls}'>{text}</span>"

def _time_ago(ts: datetime) -> str:
    secs = int((datetime.now() - ts).total_seconds())
    if lang == "en":
        if secs < 60:   return f"{secs}s ago"
        if secs < 3600: return f"{secs//60}m ago"
        return f"{secs//3600}h {(secs%3600)//60}m ago"
    else:
        if secs < 60:   return f"hace {secs}s"
        if secs < 3600: return f"hace {secs//60}m"
        return f"hace {secs//3600}h {(secs%3600)//60}m"

_MOOD_CLS = {
    "relaxed":"pb-g","playful":"pb-a","alert":"pb-c",
    "anxious":"pb-r","lethargic":"pb-v","agitated":"pb-r",
}

def _get_pet_by_id(pid: Optional[str]) -> Optional[PetProfile]:
    if not pid:
        return None
    return next((p for p in st.session_state.pets if p.pet_id == pid), None)

# ── SVG LOGO ──────────────────────────────────────────────────────────────────
_SVG_LOGO = """
<div class="brand-logo" style="display:flex;align-items:center;gap:14px;">
  <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="pawGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%"   stop-color="#F59E0B"/>
        <stop offset="100%" stop-color="#FB923C"/>
      </linearGradient>
      <filter id="pawGlow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <circle cx="23" cy="23" r="22" fill="rgba(245,158,11,0.12)" stroke="rgba(245,158,11,0.28)" stroke-width="1"/>
    <ellipse cx="14" cy="15" rx="3.2" ry="4"   fill="url(#pawGrad)" opacity="0.9" filter="url(#pawGlow)"/>
    <ellipse cx="21" cy="12" rx="3"   ry="4.2" fill="url(#pawGrad)" opacity="0.9"/>
    <ellipse cx="28" cy="12" rx="3"   ry="4.2" fill="url(#pawGrad)" opacity="0.9"/>
    <ellipse cx="35" cy="15" rx="3.2" ry="4"   fill="url(#pawGrad)" opacity="0.9"/>
    <path d="M23 19 C17 19 12 24 13 30 C14 35 18 38 23 38 C28 38 32 35 33 30 C34 24 29 19 23 19Z"
          fill="url(#pawGrad)" opacity="0.95"/>
  </svg>
  <div>
    <div style="display:flex;align-items:baseline;gap:0;">
      <svg width="210" height="34" viewBox="0 0 210 34" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="textGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stop-color="#F59E0B"/>
            <stop offset="55%"  stop-color="#FB923C"/>
            <stop offset="100%" stop-color="#FBBF24"/>
          </linearGradient>
        </defs>
        <text x="0" y="26" font-family="Inter, system-ui, sans-serif" font-weight="800" font-size="26" letter-spacing="-0.5" fill="url(#textGrad)">PawSentry</text>
        <rect x="155" y="6" width="28" height="16" rx="8" fill="rgba(245,158,11,0.18)" stroke="rgba(245,158,11,0.38)" stroke-width="1"/>
        <text x="169" y="18" font-family="Inter, system-ui, sans-serif" font-weight="700" font-size="9.5" text-anchor="middle" letter-spacing="0.5" fill="#F59E0B">AI</text>
      </svg>
    </div>
    <div style="font-size:0.68rem;color:#7A8BAD;font-weight:400;margin-top:-4px;letter-spacing:0.02em;">
      Autonomous Multi-Pet Sentry &nbsp;·&nbsp; Nebius × NVIDIA Physical AI
    </div>
  </div>
</div>
"""

# ── TOP NAV ───────────────────────────────────────────────────────────────────
is_mock: bool = st.session_state.nebius.mock
pill_cls  = "pill pill-mock" if is_mock else "pill pill-live"
pill_text = t("status_mock", lang=lang).upper() if is_mock else "AI SENSING LIVE"

nav_l, nav_r = st.columns([4, 1.2])
with nav_l:
    st.markdown(f"<div class='topnav'>{_SVG_LOGO}</div>", unsafe_allow_html=True)
with nav_r:
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:flex-end;padding-top:20px;'>"
        f"<span class='{pill_cls}'><span class='pill-dot'></span>{pill_text}</span></div>",
        unsafe_allow_html=True,
    )

# ── ⚙️ SETTINGS EXPANDER (HARDWARE, CAMERAS, MODELS, I18N) ────────────────────
with st.expander(f"⚙️  {t('hardware_title', lang=lang)} & Settings", expanded=False):
    s1, s2, s3, s4 = st.columns([1, 1.5, 1, 1], gap="medium")

    with s1:
        st.markdown(f"**🌐 {t('lang_select_label', lang=lang)}**")
        lc = st.radio(
            "Language",
            ["English", "Español"],
            index=0 if lang == "en" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="lang_radio_select",
        )
        nl = "en" if lc == "English" else "es"
        if nl != st.session_state.lang:
            st.session_state.lang = nl
            if len(st.session_state.chat_history) == 1 and st.session_state.chat_history[0]["role"] == "assistant":
                st.session_state.chat_history[0]["content"] = t("chat_greeting", lang=nl)
            st.rerun()

    with s2:
        st.markdown(f"**📹 {t('cam_source_label', lang=lang)}**")
        cam_opts = [t("cam_source_usb", lang=lang), t("cam_source_rtsp", lang=lang)]
        cur_cam_idx = 0 if st.session_state.camera_source_mode == "usb" else 1
        sel_cam_mode = st.radio(
            "Mode",
            cam_opts,
            index=cur_cam_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="cam_mode_radio",
        )
        st.session_state.camera_source_mode = "usb" if sel_cam_mode == cam_opts[0] else "rtsp"

        if st.session_state.camera_source_mode == "usb":
            cam_options = [0, 1]
            st.session_state.camera_index = st.selectbox(
                t("camera_select_label", lang=lang),
                cam_options,
                index=0 if st.session_state.camera_index not in cam_options else cam_options.index(st.session_state.camera_index),
                format_func=lambda i: "Laptop Webcam (Camera 0 - Active 🟢)" if i == 0 else "NVIDIA Virtual Camera (Camera 1 - Black ⚠️)",
                key="cam_index_select",
            )
        else:
            st.session_state.rtsp_url = st.text_input(
                t("rtsp_url_label", lang=lang),
                value=st.session_state.rtsp_url,
                help=t("rtsp_helper", lang=lang),
            )

        with st.expander("📖 EZVIZ H8c Pro RTSP Setup Guide", expanded=False):
            st.markdown(
                """
                **How to connect your EZVIZ H8c Pro:**
                1. **Disable Encryption**: In the EZVIZ mobile app -> Camera Settings -> turn OFF *Video Encryption*.
                2. **Verification Code**: Locate the 6-letter uppercase code printed on the camera sticker (e.g. `ABCDEF`).
                3. **Camera Local IP**: Find the IP address in your Wi-Fi router or EZVIZ app (e.g. `192.168.1.55`).
                4. **RTSP Stream URL**:
                   `rtsp://admin:VERIFICATION_CODE@CAMERA_IP:554/H.264/ch1/main`
                """
            )

    with s3:
        st.markdown(f"**⏱️ Cooldown & Sensitivity**")
        st.session_state.cooldown = st.slider(
            t("cooldown_label", lang=lang), 1.0, 10.0,
            float(st.session_state.cooldown), 0.5
        )
        st.session_state.min_motion_area = st.slider(
            "Motion Threshold (px)", 1000, 8000,
            int(st.session_state.min_motion_area), 500,
            help="Minimum contour area to consider movement a pet"
        )

    with s4:
        st.markdown(f"**☁️ {t('cloud_title', lang=lang)}**")
        def _sl(mock: bool) -> str:
            dot = "🟡" if mock else "🟢"
            lbl = t("status_mock" if mock else "status_connected", lang=lang)
            col = "#F59E0B" if mock else "#34D399"
            return f"<span style='color:{col};font-size:0.8rem;font-weight:600;'>{dot} {lbl}</span>"
        st.markdown(
            f"Nebius:&nbsp;&nbsp;{_sl(is_mock)}<br>"
            f"Tavily:&nbsp;&nbsp;&nbsp;{_sl(st.session_state.vet.mock)}<br>"
            f"<span style='font-size:0.67rem;color:#7A8BAD;'>"
            f"Vision: <code>{st.session_state.nebius.vision_model.split('/')[-1]}</code></span>",
            unsafe_allow_html=True,
        )
        if st.button(t("toggle_mock_btn", lang=lang), width="stretch"):
            st.session_state.nebius.mock = not st.session_state.nebius.mock
            st.session_state.vet.mock    = not st.session_state.vet.mock
            st.rerun()

# ── MODAL DIALOGS (STREAMLIT 1.63+ NATIVE) ───────────────────────────────────
@st.dialog("➕ " + ("Register New Household Pet" if lang == "en" else "Registrar Nueva Mascota"))
def add_pet_dialog():
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        new_name = st.text_input("Pet Name" if lang=="en" else "Nombre de Mascota", placeholder="e.g. Luna", key="modal_new_name")
        new_species = st.selectbox("Species" if lang=="en" else "Especie", ["Dog", "Cat", "Other"] if lang=="en" else ["Perro", "Gato", "Otro"], key="modal_new_species")
        new_breed = st.text_input("Breed / Cross" if lang=="en" else "Raza / Cruce", placeholder="e.g. Domestic / Siamese", key="modal_new_breed")
    with f_c2:
        new_age = st.number_input("Age (years)" if lang=="en" else "Edad (años)", 0, 30, 2, key="modal_new_age")
        new_desc = st.text_input(
            "Visual Description (For NVIDIA Vision)" if lang=="en" else "Descripción Visual (Para NVIDIA Vision)",
            placeholder="e.g. Tabby cat, white socks on paws",
            key="modal_new_desc",
        )
        new_photo = st.file_uploader("Upload Photo" if lang=="en" else "Subir Foto", type=["jpg", "jpeg", "png"], key="modal_new_photo")

    if st.button("💾 " + ("Save Pet to Roster" if lang == "en" else "Guardar en Roster"), type="primary", width="stretch"):
        if new_name.strip():
            slug_id = new_name.lower().replace(" ", "_").strip()
            existing_ids = [p.pet_id for p in st.session_state.pets]
            if slug_id in existing_ids:
                slug_id = f"{slug_id}_{len(existing_ids)+1}"

            photo_b64 = None
            if new_photo:
                try:
                    img = Image.open(new_photo).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    photo_b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    photo_b64 = None

            norm_species = "Dog" if new_species in ("Dog", "Perro") else ("Cat" if new_species in ("Cat", "Gato") else "Other")
            new_profile = PetProfile(
                pet_id=slug_id,
                name=new_name.strip(),
                species=norm_species,
                breed=new_breed.strip() or None,
                age_years=int(new_age),
                description=new_desc.strip(),
                photo_b64=photo_b64,
            )
            st.session_state.pets.append(new_profile)
            st.session_state.active_pet_id = slug_id
            st.rerun()
        else:
            st.error("Please enter a pet name." if lang == "en" else "Por favor ingrese un nombre.")

@st.dialog("✏️ " + ("Edit Pet Profile" if lang == "en" else "Editar Perfil"))
def edit_pet_dialog():
    e_name = st.text_input("Name" if lang=="en" else "Nombre", value=active_pet.name, key=f"dlg_edit_name_{active_pet.pet_id}")
    sp_opts = ["Dog","Cat","Other"] if lang=="en" else ["Perro","Gato","Otro"]
    idx = sp_opts.index(active_pet.species) if active_pet.species in sp_opts else 1
    e_sp = st.selectbox("Species" if lang=="en" else "Especie", sp_opts, index=idx, key=f"dlg_edit_sp_{active_pet.pet_id}")
    e_breed = st.text_input("Breed / Cross" if lang=="en" else "Raza", value=active_pet.breed or "", key=f"dlg_edit_br_{active_pet.pet_id}")
    e_age = st.number_input("Age (years)" if lang=="en" else "Edad", 0, 30, active_pet.age_years, key=f"dlg_edit_age_{active_pet.pet_id}")
    e_desc = st.text_input("Visual Description (Used by Vision Model)" if lang=="en" else "Descripción Visual", value=active_pet.description, key=f"dlg_edit_desc_{active_pet.pet_id}")
    e_up = st.file_uploader("Update photo" if lang=="en" else "Actualizar Foto", type=["jpg","jpeg","png"], key=f"dlg_edit_photo_{active_pet.pet_id}")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("💾 " + ("Save Changes" if lang=="en" else "Guardar Cambios"), type="primary", width="stretch", key=f"dlg_save_{active_pet.pet_id}"):
            active_pet.name = e_name
            active_pet.species = "Dog" if e_sp in ("Dog","Perro") else ("Cat" if e_sp in ("Cat","Gato") else "Other")
            active_pet.breed = e_breed or None
            active_pet.age_years = int(e_age)
            active_pet.description = e_desc
            if e_up:
                try:
                    img = Image.open(e_up).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    active_pet.photo_b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    pass
            st.rerun()
    with c_s2:
        if len(st.session_state.pets) > 1:
            if st.button("🗑️ " + ("Remove" if lang=="en" else "Eliminar"), width="stretch", key=f"dlg_del_{active_pet.pet_id}"):
                st.session_state.pets = [p for p in st.session_state.pets if p.pet_id != active_pet.pet_id]
                st.session_state.active_pet_id = st.session_state.pets[0].pet_id
                st.rerun()


# ── 🐾 MULTI-PET ROSTER BAR (STREAMLIT NATIVE PILLS) ────────────────────────
st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)
roster_c1, roster_c2 = st.columns([4, 1.2], vertical_alignment="center")

with roster_c1:
    pet_id_list = [p.pet_id for p in st.session_state.pets]
    if st.session_state.active_pet_id not in pet_id_list:
        st.session_state.active_pet_id = pet_id_list[0]

    def _pet_label_fmt(pid: str) -> str:
        p_obj = next((p for p in st.session_state.pets if p.pet_id == pid), None)
        if not p_obj:
            return pid
        ico = _ICONS.get(p_obj.species, "🐾")
        return f"{ico} {p_obj.name}"

    selected_pet = st.pills(
        "Select Active Pet",
        options=pet_id_list,
        format_func=_pet_label_fmt,
        default=st.session_state.active_pet_id,
        key="pills_pet_selector",
        label_visibility="collapsed",
    )
    if selected_pet and selected_pet != st.session_state.active_pet_id:
        st.session_state.active_pet_id = selected_pet
        st.rerun()

with roster_c2:
    add_lbl = "➕ Add Pet" if lang == "en" else "➕ Añadir Mascota"
    if st.button(add_lbl, width="stretch"):
        add_pet_dialog()

# ── DATA FOR ACTIVE PET ───────────────────────────────────────────────────────
events_all   = [e for e in st.session_state.generator.get_events() if e.pet_detected]
events_pet   = [e for e in st.session_state.generator.get_events(pet_id=active_pet.pet_id) if e.pet_detected]
last_event   = events_pet[-1] if events_pet else None
evt_count    = len(events_pet)
alert_count  = sum(1 for e in events_pet if e.is_anomaly)
metrics: HabitMatrixMetrics = st.session_state.generator.compute_habit_metrics(pet_id=active_pet.pet_id)
pet_bouts    = st.session_state.generator.get_behavioral_bouts(pet_id=active_pet.pet_id)
pet_icon     = _ICONS.get(active_pet.species, "🐾")

# ── PET PROFILE CARD + KPI BENTO ─────────────────────────────────────────────
pc, k1, k2, k3 = st.columns([1.65, 1, 1, 1], gap="large")

with pc:
    if active_pet.photo_b64:
        try:
            photo_html = f"<img src='data:image/jpeg;base64,{active_pet.photo_b64}' />"
        except Exception:
            photo_html = f"<span>{pet_icon}</span>"
    else:
        photo_html = f"<span>{pet_icon}</span>"

    last_lbl = _time_ago(last_event.timestamp) if last_event and hasattr(last_event.timestamp,"strftime") else "—"
    mood_lbl = t_val("mood", last_event.mood.value, lang=lang) if last_event else "—"
    ac_color = "#FB7185" if alert_count else "#34D399"

    species_display = t_val("species", active_pet.species, lang=lang)
    age_unit = "yr" if lang == "en" else "años"
    monitoring_lbl = "ACTIVE TRACKING" if lang == "en" else "MONITOREO ACTIVO"
    events_lbl = "Pet Events" if lang == "en" else "Eventos"
    alerts_lbl = "Alerts" if lang == "en" else "Alertas"
    last_seen_lbl = "Last seen" if lang == "en" else "Última vez"
    mood_title = "Mood" if lang == "en" else "Ánimo"

    st.markdown(
        f"""<div class='profile-card'>
            <div style='display:flex;align-items:flex-start;gap:16px;'>
                <div class='profile-photo'>{photo_html}</div>
                <div style='flex:1;min-width:0;'>
                    <div class='profile-name'>{active_pet.name}</div>
                    <div class='profile-meta'>{species_display} &nbsp;·&nbsp; {active_pet.breed or 'Domestic'} &nbsp;·&nbsp; {active_pet.age_years} {age_unit}</div>
                    <div class='profile-desc'>{active_pet.description or "Vision tracker active for this profile."}</div>
                    <div class='profile-badge'><span>●</span> {monitoring_lbl}</div>
                </div>
            </div>
            <div class='profile-stats'>
                <div>
                    <div class='pstat-val'>{evt_count}</div>
                    <div class='pstat-lbl'>{events_lbl}</div>
                </div>
                <div>
                    <div class='pstat-val' style='color:{ac_color};'>{alert_count}</div>
                    <div class='pstat-lbl'>{alerts_lbl}</div>
                </div>
                <div>
                    <div class='pstat-val' style='font-size:0.88rem;color:#7A8BAD;'>{last_lbl}</div>
                    <div class='pstat-lbl'>{last_seen_lbl}</div>
                </div>
                <div>
                    <div class='pstat-val' style='font-size:0.88rem;color:#A78BFA;'>{mood_lbl}</div>
                    <div class='pstat-lbl'>{mood_title}</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if st.button(f"✏️ " + (f"Edit {active_pet.name}'s Profile" if lang == "en" else f"Editar Perfil de {active_pet.name}"), width="stretch", key=f"btn_open_edit_{active_pet.pet_id}"):
        edit_pet_dialog()

with k1:
    ck = "kpi-green" if metrics.comfort_index >= 70 else "kpi-peach"
    c_label = f"{t('comfort_index', lang=lang)} ({active_pet.name})"
    sleep_sub = f"{metrics.sleep_hours_estimated:.1f}h sleep" if lang == "en" else f"{metrics.sleep_hours_estimated:.1f}h sueño"
    st.markdown(
        f"""<div class='kpi {ck}'>
            <div class='kpi-bg'>💚</div>
            <div class='kpi-icon'>💚</div>
            <div class='kpi-label'>{c_label}</div>
            <div class='kpi-value'>{metrics.comfort_index:.0f}<span class='kpi-unit'>/100</span></div>
            <div class='kpi-sub'>{metrics.activity_variance_vs_baseline:+.1f}% {t('vs_baseline', lang=lang)} · {sleep_sub}</div>
        </div>""", unsafe_allow_html=True)

with k2:
    act_label = f"{t('active_hours', lang=lang)} ({active_pet.name})"
    meals_sub = f"{metrics.food_visits_count} meals · {metrics.water_visits_count} water visits" if lang == "en" else f"{metrics.food_visits_count} comidas · {metrics.water_visits_count} agua"
    st.markdown(
        f"""<div class='kpi kpi-amber'>
            <div class='kpi-bg'>⚡</div>
            <div class='kpi-icon'>⚡</div>
            <div class='kpi-label'>{act_label}</div>
            <div class='kpi-value'>{metrics.active_hours_estimated:.1f}<span class='kpi-unit'>h</span></div>
            <div class='kpi-sub'>{meals_sub}</div>
        </div>""", unsafe_allow_html=True)

with k3:
    last_label = "Last Detected" if lang == "en" else "Última Detección"
    lt = _time_ago(last_event.timestamp) if last_event and hasattr(last_event.timestamp,"strftime") else "—"
    la = t_val("activity", last_event.activity.value, lang=lang) if last_event else ("No data" if lang=="en" else "Sin datos")
    st.markdown(
        f"""<div class='kpi kpi-lav'>
            <div class='kpi-bg'>🕐</div>
            <div class='kpi-icon'>🕐</div>
            <div class='kpi-label'>{last_label} ({active_pet.name})</div>
            <div class='kpi-value' style='font-size:1.4rem;'>{lt}</div>
            <div class='kpi-sub'>{la}</div>
        </div>""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_monitor, tab_digest, tab_chat = st.tabs([
    f"📹  {t('tab_monitor', lang=lang)} & Timeline",
    f"📊  {t('tab_digest', lang=lang)} & Comparison",
    f"💬  {t('tab_chat', lang=lang)} ({active_pet.name})",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AUTONOMOUS SENTINEL MONITOR + MULTI-PET TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown(f"<div class='stitle'>📡 {t('sentry_title', lang=lang)}</div>", unsafe_allow_html=True)

    sentry_active = st.session_state.get("sentry_active", False)
    current_source = get_resolved_camera_source()
    source_label = f"Webcam {current_source}" if isinstance(current_source, int) else f"IP Camera ({str(current_source)[:28]}...)"

    # Sentinel Top Action Banner
    sentry_cls = "sentry-panel" if sentry_active else "sentry-panel sentry-panel-standby"
    st.markdown(
        f"""<div class='{sentry_cls}'>
            <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>
                <div>
                    <div style='font-size:1.05rem;font-weight:800;color:#EFF2F8;display:flex;align-items:center;gap:8px;'>
                        <span>{'🟢' if sentry_active else '⚪'}</span>
                        <span>{'AUTONOMOUS SENTRY: ACTIVE SCANNING' if sentry_active else 'AUTONOMOUS SENTRY: STANDBY'}</span>
                    </div>
                    <div style='font-size:0.77rem;color:#7A8BAD;margin-top:4px;'>
                        Source: <b style='color:#CBD5E1;'>{source_label}</b> &nbsp;·&nbsp;
                        Threshold: <b>{st.session_state.min_motion_area} px</b> &nbsp;·&nbsp;
                        Cooldown: <b>{st.session_state.cooldown:.1f}s</b>
                    </div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if current_source == 1:
        st.warning("⚠️ **Webcam 1 is NVIDIA Broadcast Virtual Camera and produces black frames.** Switch to your laptop's integrated camera (Camera 0):")
        if st.button("👉 Switch to Laptop Webcam (Camera 0)", type="primary", key="switch_to_cam0"):
            st.session_state.camera_index = 0
            st.session_state.cam_index_select = 0
            st.session_state.camera_source_mode = "usb"
            st.rerun()

    sc_col1, sc_col2 = st.columns([2.5, 1], gap="large")

    with sc_col2:
        st.markdown(
            f"""<div class='cap-hint' style='background:#1A2035;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;margin-bottom:12px;'>
                <strong style='color:#F59E0B;'>🛡️ Physical AI Autonomous Edge</strong><br>
                <span style='font-size:0.76rem;color:#CBD5E1;'>{t("sentry_desc", lang=lang)}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if not sentry_active:
            if st.button(t("sentry_start_btn", lang=lang), type="primary", width="stretch", key="btn_start_sentry"):
                st.session_state.sentry_active = True
                st.rerun()

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            tpreview = st.button("👁️ Test / Refresh Camera Preview", width="stretch", key="btn_test_preview")
            tburst  = st.button(f"⚡ {t('trigger_burst_btn', lang=lang)}", width="stretch", key="btn_manual_burst")
            tsingle = st.button(f"📸 {t('trigger_single_btn', lang=lang)}", width="stretch", key="btn_manual_snap")
        else:
            if st.button(t("sentry_stop_btn", lang=lang), type="primary", width="stretch", key="btn_stop_sentry"):
                st.session_state.sentry_active = False
                st.rerun()
            tpreview = False
            tburst = False
            tsingle = False

        rpanel = st.empty()

    with sc_col1:
        pslot = st.empty()
        hud_slot = st.empty()

        # ── ACTIVE SENTINEL LOOP ──────────────────────────────────────────────
        if sentry_active:
            det = PetMotionDetector(
                camera_source=current_source,
                min_area=int(st.session_state.min_motion_area),
                cooldown_seconds=float(st.session_state.cooldown),
                snapshots_dir=ROOT_DIR / "data" / "snapshots",
            )
            if not det.start():
                st.error(f"❌ Cannot connect to video source: {current_source}. Please verify your webcam index or EZVIZ RTSP URL in Settings.")
                st.session_state.sentry_active = False
            else:
                try:
                    # Stream frames smoothly; yield upon motion trigger or frame budget
                    for _ in range(60):
                        ret, frame = det.read_frame()
                        if not ret or frame is None:
                            break

                        motion_detected, fg_mask, bboxes = det.process_frame(frame)
                        annotated = frame.copy()

                        # Draw green bounding boxes for moving regions
                        for (x, y, w, h) in bboxes:
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (52, 211, 153), 2)
                            cv2.putText(
                                annotated, "PET MOTION", (x, max(18, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (52, 211, 153), 2
                            )

                        cd_rem = det.get_cooldown_remaining()
                        can_cap = det.can_capture()

                        # Check autonomous trigger condition
                        if motion_detected and can_cap:
                            hud_slot.markdown(
                                f"<div class='sentry-status-alert'>⚡ {t('sentry_status_motion', lang=lang)}</div>",
                                unsafe_allow_html=True,
                            )
                            burst = det.capture_micro_event_burst(frame, prefix="sentry")
                            if burst:
                                analysis = st.session_state.nebius.analyze_micro_event(
                                    image_paths=burst,
                                    pet_name=active_pet.name,
                                    registered_pets=st.session_state.pets,
                                    pet_id=active_pet.pet_id,
                                    lang=lang,
                                )
                                vr = st.session_state.generator.add_event(analysis, lang=lang)
                                st.session_state.last_sentry_burst = [str(p) for p in burst]
                                st.session_state.last_sentry_analysis = analysis
                                break
                        else:
                            mot_desc = f"🐾 {len(bboxes)} motion target(s)" if motion_detected else f"👁️ {t('sentry_status_scanning', lang=lang)}"
                            cd_desc = f"⏳ Cooldown: {cd_rem:.1f}s" if cd_rem > 0 else "🟢 Sensor Ready"
                            hud_slot.markdown(
                                f"<div class='sentry-status-bar'><span>{mot_desc}</span><span>{cd_desc}</span></div>",
                                unsafe_allow_html=True,
                            )

                        pslot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
                        time.sleep(0.02)
                finally:
                    det.stop()

                if st.session_state.get("sentry_active", False):
                    st.rerun()

        # ── LIVE PREVIEW TEST ─────────────────────────────────────────────────
        elif tpreview:
            with st.spinner("Connecting to camera..."):
                det = PetMotionDetector(
                    camera_source=current_source,
                    cooldown_seconds=0.1,
                    snapshots_dir=ROOT_DIR / "data" / "snapshots",
                )
                if det.start():
                    ret, frame = det.read_frame()
                    det.stop()
                    if ret and frame is not None:
                        cv2.imwrite(str(ROOT_DIR / "current_live_cam0.jpg"), frame)
                        cv2.imwrite(str(ROOT_DIR / "test_camera_0.jpg"), frame)
                        pslot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Camera {det.camera_source} · Live Preview (Brightness: {frame.mean():.1f})",
                                    width="stretch")
                        st.success(f"✅ Camera {det.camera_source} operational! Brightness: {frame.mean():.1f}/255")
                    else:
                        st.error("Cannot read frame from camera.")
                else:
                    st.error(f"Camera source {current_source} not available.")

        # ── MANUAL TRIGGER FALLBACK ───────────────────────────────────────────
        elif tburst or tsingle:
            with st.spinner("Opening camera..."):
                det = PetMotionDetector(
                    camera_source=current_source,
                    cooldown_seconds=0.1,
                    snapshots_dir=ROOT_DIR / "data" / "snapshots",
                )
                if det.start():
                    ret, frame = det.read_frame()
                    if ret and frame is not None:
                        cv2.imwrite(str(ROOT_DIR / "current_live_cam0.jpg"), frame)
                        cv2.imwrite(str(ROOT_DIR / "test_camera_0.jpg"), frame)
                        pslot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Camera · Live Frame", width="stretch")
                        if tburst:
                            with st.spinner(t("processing_burst", lang=lang)):
                                burst = det.capture_micro_event_burst(frame, prefix="manual")
                                det.stop()
                            if burst:
                                analysis = st.session_state.nebius.analyze_micro_event(
                                    image_paths=burst,
                                    pet_name=active_pet.name,
                                    registered_pets=st.session_state.pets,
                                    pet_id=active_pet.pet_id,
                                    lang=lang,
                                )
                                vr = st.session_state.generator.add_event(analysis, lang=lang)
                                st.session_state.last_sentry_burst = [str(p) for p in burst]
                                st.session_state.last_sentry_analysis = analysis
                                st.rerun()
                        else:
                            det.stop()
                            st.success("Single frame captured.")
                    else:
                        det.stop()
                        st.error("Cannot read frame from camera.")
                else:
                    st.error(f"Camera source {current_source} not available.")
        else:
            # Standby mode preview
            snap = ROOT_DIR / "current_live_cam0.jpg"
            if not snap.exists():
                snap = ROOT_DIR / "test_camera_0.jpg"
            if snap.exists():
                pslot.image(str(snap), caption=f"Camera {current_source} · Standby Preview", width="stretch")
            else:
                pslot.markdown(
                    "<div class='cam-ph'><div style='font-size:2.2rem;'>📷</div>"
                    "<div>Camera Standby. Click 'Start Autonomous Sentry' or 'Test / Refresh Camera Preview'.</div></div>",
                    unsafe_allow_html=True,
                )

    # ── LATEST EVENT & ACTIVE TAGGING CARD ────────────────────────────────────
    if st.session_state.last_sentry_analysis and st.session_state.last_sentry_burst:
        l_an: BehaviorAnalysis = st.session_state.last_sentry_analysis
        l_burst = st.session_state.last_sentry_burst

        st.markdown(f"<div class='stitle'>🎯 Latest Autonomous Detection & Human-In-The-Loop Tagging</div>", unsafe_allow_html=True)
        det_pet = _get_pet_by_id(l_an.pet_id)
        det_name = det_pet.name if det_pet else (l_an.pet_id or "Unassigned")
        det_icon = _ICONS.get(det_pet.species if det_pet else "Other", "🐾")
        det_species = t_val("species", det_pet.species if det_pet else l_an.pet_type, lang=lang)

        d_col1, d_col2 = st.columns([1.8, 1.2], gap="medium")
        with d_col1:
            f_cols = st.columns(min(3, len(l_burst)))
            _lbls = ["01 Onset", "02 Peak", "03 Post"]
            for bi, bp in enumerate(l_burst[:3]):
                if Path(bp).exists():
                    with f_cols[bi]:
                        st.image(bp, caption=_lbls[bi] if bi < 3 else f"Frame {bi+1}", width="stretch")

        with d_col2:
            st.markdown(
                f"""<div style='background:#1A2035;border:1px solid rgba(245,158,11,0.30);border-radius:14px;padding:16px;'>
                    <div style='font-size:0.68rem;font-weight:700;color:#F59E0B;text-transform:uppercase;'>🎯 AI Vision Inference</div>
                    <div style='font-size:1.15rem;font-weight:800;color:#EFF2F8;margin-top:2px;'>{det_icon} {det_name} ({det_species})</div>
                    <div style='font-size:0.75rem;color:#7A8BAD;margin-top:2px;'>Confidence: <b>{int(l_an.confidence*100)}%</b> &nbsp;·&nbsp; Energy: <b>{l_an.energy_level}/10</b></div>
                    <div style='font-size:0.78rem;color:#CBD5E1;margin-top:8px;line-height:1.4;'>{l_an.clinical_notes or 'Observation logged.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Human-in-the-Loop Active Learning Tagging Buttons
            st.markdown(
                f"<div style='font-size:0.75rem;font-weight:700;color:#F59E0B;margin-top:10px;text-transform:uppercase;'>"
                f"🏷️ Confirm Identity (Active Learning):</div>",
                unsafe_allow_html=True,
            )
            tag_cols = st.columns(len(st.session_state.pets))
            for ti, p in enumerate(st.session_state.pets):
                with tag_cols[ti]:
                    p_ico = _ICONS.get(p.species, "🐾")
                    if st.button(f"{p_ico} {p.name}", key=f"tag_btn_{p.pet_id}", width="stretch"):
                        # Assign last event to this pet
                        evs = st.session_state.generator.get_events()
                        if evs:
                            evs[-1].pet_id = p.pet_id
                        # Enrich profile traits with vision notes
                        if l_an.clinical_notes and l_an.clinical_notes not in p.description:
                            p.description = f"{p.description} | {l_an.clinical_notes[:70]}".strip()
                        st.session_state.active_pet_id = p.pet_id
                        st.success(t("tag_success", lang=lang, pet_name=p.name))
                        st.rerun()

    # ── TIMELINE CONTROLS & FILTER ────────────────────────────────────────────
    st.markdown(f"<div class='stitle'>🗂️ {t('timeline_header', lang=lang)}</div>", unsafe_allow_html=True)

    tf_c1, tf_c2, tf_c3 = st.columns([3, 3, 1.2])
    with tf_c1:
        cur_opt = f"{'Current:' if lang=='en' else 'Actual:'} {active_pet.name}"
        all_opt = "All Household Pets" if lang == "en" else "Todas las Mascotas"
        f_mode = st.radio(
            "Filter Events:" if lang == "en" else "Filtrar Eventos:",
            [cur_opt, all_opt],
            index=0 if st.session_state.timeline_filter == "active" else 1,
            horizontal=True,
            key="timeline_filter_choice",
        )
        st.session_state.timeline_filter = "active" if cur_opt in f_mode else "all"

    with tf_c3:
        if st.button(t("clear_history_btn", lang=lang), width="stretch"):
            target_del = active_pet.pet_id if st.session_state.timeline_filter == "active" else None
            st.session_state.generator.clear_events(pet_id=target_del)
            st.rerun()

    display_events = events_pet if st.session_state.timeline_filter == "active" else events_all

    if not display_events:
        st.markdown(
            f"<div class='empty-box'><div class='empty-icon'>🔭</div>"
            f"<div class='empty-txt'>{t('empty_timeline', lang=lang)}</div></div>",
            unsafe_allow_html=True)
    else:
        _bml = ["01 Onset","02 Peak","03 Post"]
        for ev_idx, ev in enumerate(reversed(display_events)):
            ia   = ev.is_anomaly
            acls = "moment-alert" if ia else ""
            ts   = ev.timestamp.strftime("%H:%M") if hasattr(ev.timestamp,"strftime") else ""
            ago  = _time_ago(ev.timestamp) if hasattr(ev.timestamp,"strftime") else ""
            act  = t_val("activity", ev.activity.value, lang=lang)
            mood = t_val("mood", ev.mood.value, lang=lang)
            pos  = t_val("posture", ev.posture.value, lang=lang)
            eng  = int(ev.energy_level*10)
            mpb  = _MOOD_CLS.get(ev.mood.value,"pb-m")
            apb  = "pb-r" if ia else "pb-a"
            sbadge = _pb(t("badge_alert", lang=lang),"pb-r") if ia else _pb(t("badge_normal", lang=lang),"pb-g")

            ev_pet = _get_pet_by_id(ev.pet_id)
            ev_pet_name = ev_pet.name if ev_pet else (ev.pet_id or "Pet")
            ev_pet_icon = _ICONS.get(ev_pet.species if ev_pet else "Other", "🐾")
            pet_badge = f"<span class='pb pb-pet'>{ev_pet_icon} {ev_pet_name}</span>"

            valid  = [p for p in (ev.image_paths or []) if Path(p).exists()]
            peak   = valid[1] if len(valid)>1 else (valid[0] if valid else None)

            st.markdown(f"<div class='moment {acls}'>", unsafe_allow_html=True)
            ic, bc = st.columns([1, 3.5])
            with ic:
                if peak:
                    st.image(str(peak), width="stretch")
                else:
                    st.markdown(
                        f"<div style='background:#1E2840;min-height:110px;display:flex;"
                        f"align-items:center;justify-content:center;font-size:1.8rem;'>{'⚠️' if ia else ev_pet_icon}</div>",
                        unsafe_allow_html=True)
            with bc:
                notes = ev.clinical_notes or ("No additional observations." if lang == "en" else "Sin observaciones adicionales.")
                energy_lbl = t("energy_label", lang=lang)
                st.markdown(
                    f"""<div class='mbody'>
                        <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:6px;'>
                            <div>{pet_badge}{sbadge}{_pb(act,apb)}{_pb(mood,mpb)}{_pb(pos,"pb-v")}</div>
                            <div class='mts'>{ts} · {ago}</div>
                        </div>
                        <div class='mdiag'>{act} ({ev_pet_name})</div>
                        <div class='msub'>{notes}</div>
                        <div class='ebar'>
                            <span class='elbl'>{energy_lbl}</span>
                            <div class='etrack'><div class='efill' style='width:{eng}%;'></div></div>
                            <span class='eval'>{ev.energy_level}/10</span>
                            <span class='elbl' style='margin-left:10px;'>Confidence {int(ev.confidence*100)}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Reassign pet dropdown (if multiple pets)
            if len(st.session_state.pets) > 1:
                with st.expander(f"🔄 Reassign Pet or View Frames (Current: {ev_pet_name})", expanded=False):
                    re_c1, re_c2 = st.columns([1.5, 3])
                    with re_c1:
                        target_names = [p.name for p in st.session_state.pets]
                        curr_name_idx = target_names.index(ev_pet_name) if ev_pet_name in target_names else 0
                        re_sel = st.selectbox("Assign to:" if lang=="en" else "Reasignar a:", target_names, index=curr_name_idx, key=f"reassign_{ev_idx}")
                        if st.button("Update Assignment" if lang=="en" else "Actualizar", key=f"btn_re_{ev_idx}"):
                            new_pid = next(p.pet_id for p in st.session_state.pets if p.name == re_sel)
                            real_idx = len(display_events) - 1 - ev_idx
                            st.session_state.generator.assign_event_pet(real_idx, new_pid)
                            st.rerun()
                    with re_c2:
                        if len(valid) >= 2:
                            bc2 = st.columns(min(3, len(valid)))
                            for bi, bp in enumerate(valid[:3]):
                                with bc2[bi]:
                                    st.image(str(bp), caption=_bml[bi] if bi<3 else f"Frame {bi+1}", width="stretch")

            if ia and ev.anomaly_reason:
                st.markdown(f"<div class='alert-strip'>⚠️ <strong>{ev.anomaly_reason}</strong></div>", unsafe_allow_html=True)
            st.markdown("</div><div style='height:4px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HABIT MATRIX, DAILY DIGEST & MULTI-PET COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_digest:
    st.markdown(f"<div class='stitle'>🌡️ {t('digest_header', lang=lang, pet_name=active_pet.name)}</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4, gap="large")
    _dkpi = [
        (d1, "💧", t("water_visits", lang=lang), str(metrics.water_visits_count), f"{metrics.hydration_score:.0f}/100 hydration" if lang == "en" else f"{metrics.hydration_score:.0f}/100 hidratación"),
        (d2, "🍽️", t("food_visits", lang=lang), str(metrics.food_visits_count), f"{metrics.nutrition_score:.0f}/100 appetite" if lang == "en" else f"{metrics.nutrition_score:.0f}/100 apetito"),
        (d3, "😴", t("sleep_hours", lang=lang), f"{metrics.sleep_hours_estimated:.1f}h", f"{metrics.rest_bouts_count} rest sessions" if lang == "en" else f"{metrics.rest_bouts_count} sesiones reposo"),
        (d4, "🏃", t("active_hours", lang=lang), f"{metrics.active_hours_estimated:.1f}h", f"{metrics.active_bouts_count} active sessions" if lang == "en" else f"{metrics.active_bouts_count} sesiones activas"),
    ]
    for col, icon, label, val, sub in _dkpi:
        with col:
            col.markdown(
                f"<div class='kpi kpi-amber'><div class='kpi-bg'>{icon}</div>"
                f"<div class='kpi-icon'>{icon}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value' style='font-size:1.65rem;'>{val}</div>"
                f"<div class='kpi-sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go

        gc, rc = st.columns(2, gap="large")
        with gc:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=metrics.comfort_index,
                delta={"reference": 85, "valueformat": ".0f"},
                title={"text": f"{t('comfort_index', lang=lang)} ({active_pet.name})", "font": {"size": 13, "color": "#7A8BAD"}},
                number={"font": {"size": 38, "color": "#34D399"}, "suffix": "/100"},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#7A8BAD"}},
                    "bar": {"color": "#34D399", "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(251,113,133,0.10)"},
                        {"range": [40, 70], "color": "rgba(251,146,60,0.08)"},
                        {"range": [70, 100], "color": "rgba(52,211,153,0.10)"},
                    ],
                    "threshold": {"line": {"color": "#F59E0B", "width": 3}, "thickness": 0.75, "value": 85},
                },
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=10),
                height=230,
            )
            st.plotly_chart(fig_g, width="stretch")

        with rc:
            _rl = [
                t("water_visits", lang=lang),
                t("food_visits", lang=lang),
                t("sleep_hours", lang=lang),
                t("active_hours", lang=lang),
                t("comfort_index", lang=lang),
            ]
            _rv = [
                metrics.hydration_score,
                metrics.nutrition_score,
                min(100.0, (metrics.sleep_hours_estimated / 16.0) * 100.0),
                metrics.mobility_score,
                metrics.comfort_index,
            ]
            fig_r = go.Figure(go.Scatterpolar(
                r=_rv + [_rv[0]],
                theta=_rl + [_rl[0]],
                fill="toself",
                fillcolor="rgba(245,158,11,0.07)",
                line=dict(color="#F59E0B", width=2),
                marker=dict(color="#34D399", size=6),
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#7A8BAD", size=9), gridcolor="rgba(255,255,255,0.04)"),
                    angularaxis=dict(tickfont=dict(color="#7A8BAD", size=10), gridcolor="rgba(255,255,255,0.04)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                height=230,
            )
            st.plotly_chart(fig_r, width="stretch")

        # ── 24-HOUR CIRCADIAN ACTIVITY DISTRIBUTION ───────────────────────────
        hours = list(range(24))
        rest_h = [0] * 24
        act_h = [0] * 24
        hydro_h = [0] * 24
        food_h = [0] * 24

        for ev in events_pet:
            h = ev.timestamp.hour
            if ev.activity in (PetActivityType.SLEEPING, PetActivityType.RESTING):
                rest_h[h] += 1
            elif ev.activity in (PetActivityType.PLAYING, PetActivityType.WALKING):
                act_h[h] += 1
            elif ev.activity == PetActivityType.DRINKING:
                hydro_h[h] += 1
            elif ev.activity == PetActivityType.EATING:
                food_h[h] += 1

        fig_chrono = go.Figure()
        fig_chrono.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=rest_h,
            name="Rest / Sleep" if lang == "en" else "Reposo / Sueño",
            marker_color="#818CF8",
        ))
        fig_chrono.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=act_h,
            name="Active / Play" if lang == "en" else "Activo / Juego",
            marker_color="#F59E0B",
        ))
        fig_chrono.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=hydro_h,
            name="Hydration" if lang == "en" else "Hidratación",
            marker_color="#38BDF8",
        ))
        fig_chrono.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=food_h,
            name="Nutrition" if lang == "en" else "Nutrición",
            marker_color="#34D399",
        ))

        chrono_title = f"📊 24-Hour Circadian Activity Distribution ({active_pet.name})" if lang == "en" else f"📊 Distribución Circadiana de Actividad en 24h ({active_pet.name})"
        fig_chrono.update_layout(
            barmode="stack",
            title={"text": chrono_title, "font": {"color": "#EFF2F8", "size": 13}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#7A8BAD", size=9), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(tickfont=dict(color="#7A8BAD", size=9), gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1", size=10)),
            margin=dict(l=20, r=20, t=40, b=20),
            height=230,
        )
        st.plotly_chart(fig_chrono, width="stretch")

    except ImportError:
        st.info("pip install plotly")

    # ── BEHAVIORAL BOUTS HISTORY ──────────────────────────────────────────────
    if pet_bouts:
        bout_exp_title = f"🐾 Ethological Behavioral Sessions ({len(pet_bouts)} bouts recorded today for {active_pet.name})" if lang == "en" else f"🐾 Sesiones Conductuales Etológicas ({len(pet_bouts)} sesiones hoy para {active_pet.name})"
        with st.expander(bout_exp_title, expanded=False):
            b_cols = st.columns([1.5, 2.5, 1.5, 1.2, 1.2])
            with b_cols[0]:
                st.markdown("<b style='color:#7A8BAD;'>Time Window</b>", unsafe_allow_html=True)
            with b_cols[1]:
                st.markdown("<b style='color:#7A8BAD;'>Behavior & Location</b>", unsafe_allow_html=True)
            with b_cols[2]:
                st.markdown("<b style='color:#7A8BAD;'>Duration</b>", unsafe_allow_html=True)
            with b_cols[3]:
                st.markdown("<b style='color:#7A8BAD;'>Frames</b>", unsafe_allow_html=True)
            with b_cols[4]:
                st.markdown("<b style='color:#7A8BAD;'>Mood</b>", unsafe_allow_html=True)

            for b in reversed(pet_bouts[-12:]):
                b_icon = _ACT_ICONS.get(b.activity, "🐾")
                b_act_name = t_val("activity", b.activity.value, lang=lang)
                b_loc = f" · {b.location_hint}" if b.location_hint else ""
                b_time = f"{b.start_time.strftime('%H:%M:%S')} - {b.end_time.strftime('%H:%M:%S')}" if b.start_time != b.end_time else b.start_time.strftime('%H:%M:%S')
                b_mood_name = t_val("mood", b.dominant_mood.value, lang=lang)

                c0, c1, c2, c3, c4 = st.columns([1.5, 2.5, 1.5, 1.2, 1.2])
                with c0:
                    st.caption(b_time)
                with c1:
                    st.markdown(f"<b>{b_icon} {b_act_name}</b>{b_loc}", unsafe_allow_html=True)
                with c2:
                    st.caption(f"{b.duration_minutes:.1f} min")
                with c3:
                    st.caption(f"{b.event_count} evts")
                with c4:
                    st.caption(b_mood_name)

    # ── MULTI-PET HOUSEHOLD COMPARISON (HACKATHON DIFFERENTIATOR) ─────────────
    if len(st.session_state.pets) > 1:
        comp_heading = "👥 Household Multi-Pet Health Comparison" if lang == "en" else "👥 Comparativa de Salud Multi-Mascota del Hogar"
        st.markdown(f"<div class='stitle'>{comp_heading}</div>", unsafe_allow_html=True)
        comp_cols = st.columns(len(st.session_state.pets))
        for ci, p_comp in enumerate(st.session_state.pets):
            p_metrics = st.session_state.generator.compute_habit_metrics(pet_id=p_comp.pet_id)
            p_evts = [e for e in st.session_state.generator.get_events(pet_id=p_comp.pet_id) if e.pet_detected]
            p_alerts = sum(1 for e in p_evts if e.is_anomaly)
            p_icon = _ICONS.get(p_comp.species, "🐾")
            p_species_display = t_val("species", p_comp.species, lang=lang)
            c_color = "#34D399" if p_metrics.comfort_index >= 70 else "#F59E0B"
            with comp_cols[ci]:
                comfort_sub = "Comfort" if lang == "en" else "Confort"
                if len(p_evts) == 0:
                    details_html = (
                        f"• {events_lbl}: <b>0</b> ({alerts_lbl}: <span style='color:#34D399;'>0</span>)<br>"
                        f"• <i>{'Awaiting live edge detections' if lang == 'en' else 'Esperando detecciones en vivo'}</i>"
                    )
                else:
                    details_html = (
                        f"• {events_lbl}: <b>{len(p_evts)}</b> ({alerts_lbl}: <span style='color:{'#FB7185' if p_alerts else '#34D399'};'>{p_alerts}</span>)<br>"
                        f"• Rest: <b>{p_metrics.sleep_hours_estimated:.1f}h</b> ({p_metrics.rest_bouts_count} bouts) · Active: <b>{p_metrics.active_hours_estimated:.1f}h</b><br>"
                        f"• Water: <b>{p_metrics.water_visits_count}</b> · Food: <b>{p_metrics.food_visits_count}</b>"
                    )
                st.markdown(
                    f"""<div class='comp-card'>
                        <div class='comp-title'>{p_icon} {p_comp.name} ({p_species_display})</div>
                        <div style='font-size:1.8rem;font-weight:800;color:{c_color};'>{p_metrics.comfort_index:.0f}<span style='font-size:0.9rem;color:#7A8BAD;'>/100 {comfort_sub}</span></div>
                        <div style='font-size:0.75rem;color:#CBD5E1;margin-top:8px;'>{details_html}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── DAILY AI REPORT ───────────────────────────────────────────────────────
    st.markdown(f"<div class='stitle'>📋 {t('report_title', lang=lang, date=datetime.now().strftime('%Y-%m-%d'))} ({active_pet.name})</div>", unsafe_allow_html=True)

    report_key = f"last_report_{active_pet.pet_id}"
    if report_key not in st.session_state:
        # Load from disk cache if available
        cached_rep = st.session_state.generator.get_cached_daily_report(active_pet.pet_id)
        if cached_rep:
            st.session_state[report_key] = cached_rep
        elif len(events_pet) >= 3:
            # Auto-generate if enough events exist
            st.session_state[report_key] = st.session_state.generator.generate_daily_report(
                pet_name=active_pet.name,
                pet_id=active_pet.pet_id,
                lang=lang,
            )

    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        if st.button(f"✨ {t('generate_digest_btn', lang=lang)} ({active_pet.name})", type="primary", width="stretch"):
            with st.spinner(t("generating_digest", lang=lang)):
                st.session_state[report_key] = st.session_state.generator.generate_daily_report(
                    pet_name=active_pet.name,
                    pet_id=active_pet.pet_id,
                    lang=lang,
                )
                st.rerun()
    with c_btn2:
        if report_key in st.session_state:
            rep_date = st.session_state[report_key].date
            st.caption(f"Status: 🟢 Synced ({rep_date})")

    if report_key in st.session_state:
        rep = st.session_state[report_key]
        sc, nar = st.columns([1, 3], gap="large")
        with sc:
            st.markdown(
                f"<div class='score-hero'><div class='score-num'>{rep.overall_wellness_score}</div>"
                f"<div class='score-lbl'>{t('wellness_score_label', lang=lang)} ({active_pet.name})</div>"
                f"<div style='font-size:0.68rem;color:#7A8BAD;margin-top:6px;'>{rep.date} · NVIDIA Nemotron</div></div>",
                unsafe_allow_html=True,
            )
        with nar:
            sum_title = "Nemotron Clinical Summary" if lang == "en" else "Resumen Clínico Nemotron"
            st.markdown(
                f"<div class='rcard'><div class='rcard-title'>🧠 {sum_title} ({active_pet.name})</div>"
                f"<div style='font-size:0.87rem;color:#CBD5E1;line-height:1.65;'>{rep.summary_narrative}</div></div>",
                unsafe_allow_html=True,
            )
        rh, ra = st.columns(2, gap="large")
        with rh:
            items = "".join(f"<div class='ritem'><div class='rdot'></div>{h}</div>" for h in rep.key_highlights)
            st.markdown(f"<div class='rcard'><div class='rcard-title'>🌟 {t('highlights_title', lang=lang)}</div>{items}</div>", unsafe_allow_html=True)
        with ra:
            acts = "".join(f"<div class='ritem'><div class='rdot'></div>{a}</div>" for a in rep.recommended_actions)
            st.markdown(f"<div class='rcard'><div class='rcard-title'>🩺 {t('actions_title', lang=lang)}</div>{acts}</div>", unsafe_allow_html=True)
        if rep.alerts:
            st.markdown(f"<div class='stitle'>⚠️ {t('alerts_title', lang=lang)}</div>", unsafe_allow_html=True)
            for al in rep.alerts:
                st.markdown(
                    f"<div class='rcard' style='border-color:rgba(251,113,133,0.25);background:rgba(251,113,133,0.04);'>"
                    f"<div class='ritem'><div class='rdot rdot-r'></div>{al}</div></div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEMOTRON VET CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        f"<div class='chat-intro'>"
        f"<div class='chat-title'>💬 {t('chat_header', lang=lang, pet_name=active_pet.name)}</div>"
        f"<div class='chat-sub'>{t('chat_caption', lang=lang)}</div></div>",
        unsafe_allow_html=True,
    )

    chips = [
        t("prompt_chip_1", lang=lang),
        t("prompt_chip_2", lang=lang),
        t("prompt_chip_3", lang=lang),
    ]
    st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#F59E0B;text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 6px;'>💡 Quick Questions:</div>", unsafe_allow_html=True)
    clicked_chip = st.pills("Quick Questions", chips, key=f"quick_chat_pills_{active_pet.pet_id}", label_visibility="collapsed")

    # Contenedor dedicado para mensajes (evita que los nuevos queden debajo del input)
    msg_container = st.container()
    with msg_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    uq: str = st.chat_input(t("chat_input_placeholder", lang=lang), key="chat_user_input")

    resolved_query = uq or (clicked_chip if clicked_chip else "")

    if resolved_query:
        st.session_state.chat_history.append({"role": "user", "content": resolved_query})
        with msg_container:
            with st.chat_message("user"):
                st.markdown(resolved_query)
            with st.chat_message("assistant"):
                with st.spinner(t("chat_analyzing", lang=lang)):
                    reply = st.session_state.generator.chat_with_agent(
                        resolved_query,
                        pet_name=active_pet.name,
                        pet_id=active_pet.pet_id,
                        lang=lang,
                        conversation_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
