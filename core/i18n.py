"""
core/i18n.py
Módulo de Internacionalización (i18n) para PawSentry AI.
Provee soporte bilingüe con inglés por defecto (para evaluación global y demo)
y conmutación fluida a español.
"""

from typing import Any, Dict

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "PawSentry AI",
        "app_subtitle": "Edge Multimodal Pet Monitoring & Behavioral Health Engine — Nebius x NVIDIA Hackathon",
        "profile_title": "Pet Profile",
        "pet_name_label": "Pet Name",
        "pet_species_label": "Species",
        "species_dog": "Dog",
        "species_cat": "Cat",
        "species_other": "Other",
        "hardware_title": "Hardware & Sensing",
        "camera_select_label": "Active Camera (Index)",
        "cooldown_label": "Capture Cooldown (seconds)",
        "cloud_title": "Cloud & Model Status",
        "nebius_status": "Nebius Token Factory",
        "vision_model": "Vision Model",
        "reasoning_model": "Reasoning Model",
        "tavily_status": "Tavily Search API",
        "status_connected": "Connected (Live)",
        "status_mock": "Mock Mode (Offline)",
        "toggle_mock_btn": "Toggle Mock / Live",
        "lang_select_label": "Interface Language",
        
        # Tabs
        "tab_monitor": "Live Edge Monitor",
        "tab_timeline": "Micro-Event Feed",
        "tab_digest": "Habit Matrix & Digest",
        "tab_chat": "Nemotron Vet Chat",
        
        # Tab 1: Monitor
        "monitor_header": "Real-time Video Sensing & Micro-Event Trigger",
        "physical_ai_badge": "PHYSICAL AI SENSING: ACTIVE",
        "physical_ai_info": "Edge Advantage: Triggering a micro-event captures a 3-frame temporal burst (onset, peak, post-roll) for NVIDIA multimodal evaluation of trajectory and biomechanics.",
        "trigger_burst_btn": "Capture & Analyze Micro-Event (3-Frame Burst)",
        "trigger_single_btn": "Capture Single Snapshot",
        "processing_burst": "Capturing 3-frame burst and streaming to Nebius Token Factory...",
        "burst_success": "Micro-event processed successfully! Activity: {activity} ({mood})",
        "clinical_alert": "Clinical Anomaly Flagged: {reason}",
        "vet_evidence": "Veterinary Evidence (Tavily): {summary}",
        "vet_sources": "Medical Sources: {sources}",
        
        # Tab 2: Timeline
        "timeline_header": "Historical Micro-Event Log",
        "clear_history_btn": "Clear History",
        "empty_timeline": "No events recorded yet. Trigger a capture from the Live Edge Monitor tab.",
        "badge_normal": "NORMAL",
        "badge_alert": "CLINICAL ALERT",
        "posture_label": "Posture",
        "energy_label": "Energy Level",
        
        # Tab 3: Digest & Habits
        "digest_header": "Habit Matrix & Wellness Digest for {pet_name}",
        "comfort_index": "Comfort Index",
        "water_visits": "Hydration (Visits)",
        "food_visits": "Nutrition (Visits)",
        "sleep_hours": "Sleep Duration",
        "active_hours": "Active Hours",
        "vs_baseline": "vs baseline",
        "generate_digest_btn": "Generate Daily Digest with NVIDIA Nemotron",
        "generating_digest": "NVIDIA Nemotron synthesizing daily logs and habit correlations...",
        "report_title": "Daily Health Report ({date})",
        "wellness_score_label": "Overall Wellness Score",
        "highlights_title": "Key Daily Highlights",
        "actions_title": "Recommended Care & Actions",
        "alerts_title": "Preventive Alerts",
        
        # Tab 4: Chat
        "chat_header": "Inquire About {pet_name}'s Health & Behavior",
        "chat_caption": "Powered by NVIDIA Nemotron with full context of today's habit matrix and Tavily clinical knowledge.",
        "chat_input_placeholder": "Ask anything (e.g., How was Toby's activity today? Did he drink enough water?)...",
        "chat_greeting": "Hello! I am PawSentry AI, your veterinary ethologist assistant powered by NVIDIA Nemotron. How can I assist you with your companion's wellness today?",
        "chat_analyzing": "Nemotron analyzing historical logs and veterinary context...",
        "prompt_chip_1": "How active was my pet today?",
        "prompt_chip_2": "Are there any health or stress alerts?",
        "prompt_chip_3": "Did my pet drink enough water today?",

        # Autonomous Sentinel & Camera
        "sentry_title": "Autonomous Edge Sentinel Mode",
        "sentry_desc": "Continuous background monitoring: OpenCV MOG2 detects pet motion, automatically captures a 3-frame burst, and identifies household members with Nebius Vision.",
        "sentry_start_btn": "🛡️ Start Autonomous Sentry",
        "sentry_stop_btn": "🛑 Stop Sentry Mode",
        "sentry_status_scanning": "Scanning room for pet movement...",
        "sentry_status_motion": "⚡ Pet motion detected! Firing 3-frame burst...",
        "sentry_status_cooldown": "Sensor cooling down ({sec:.1f}s remaining)...",
        "cam_source_label": "Camera Source Mode",
        "cam_source_usb": "💻 Local USB Webcam",
        "cam_source_rtsp": "🌐 IP Camera (EZVIZ H8c Pro RTSP)",
        "rtsp_url_label": "RTSP Stream URL (EZVIZ H8c Pro)",
        "rtsp_helper": "Format: rtsp://admin:{verification_code}@{camera_ip}:554/H.264/ch1/main",
        "tag_as_btn": "👉 Tag as {pet_name}",
        "tag_success": "Saved! Assigned to {pet_name} and updated profile notes."
    },
    "es": {
        "app_title": "PawSentry AI",
        "app_subtitle": "Motor Multimodal de Monitoreo de Mascotas y Salud Conductual — Nebius x NVIDIA Hackathon",
        "profile_title": "Perfil de la Mascota",
        "pet_name_label": "Nombre de la Mascota",
        "pet_species_label": "Especie",
        "species_dog": "Perro",
        "species_cat": "Gato",
        "species_other": "Otro",
        "hardware_title": "Hardware y Sensado",
        "camera_select_label": "Cámara Activa (Índice)",
        "cooldown_label": "Cooldown de Captura (segundos)",
        "cloud_title": "Estado en la Nube y Modelos",
        "nebius_status": "Nebius Token Factory",
        "vision_model": "Modelo Visión",
        "reasoning_model": "Modelo Razonamiento",
        "tavily_status": "Tavily Search API",
        "status_connected": "Conectado (En Vivo)",
        "status_mock": "Modo Mock (Offline)",
        "toggle_mock_btn": "Alternar Mock / Live",
        "lang_select_label": "Idioma de la Interfaz",
        
        # Tabs
        "tab_monitor": "Monitor en Vivo (Edge)",
        "tab_timeline": "Feed de Micro-Eventos",
        "tab_digest": "Matriz de Hábitos y Resumen",
        "tab_chat": "Chat Asistente Nemotron",
        
        # Tab 1: Monitor
        "monitor_header": "Sensado de Video en Tiempo Real y Trigger de Micro-Eventos",
        "physical_ai_badge": "SENSADO PHYSICAL AI: ACTIVO",
        "physical_ai_info": "Diferencial Edge: Disparar un micro-evento captura una ráfaga temporal de 3 fotogramas (inicio, cúspide y post-roll) para análisis de trayectoria y biomecánica en NVIDIA.",
        "trigger_burst_btn": "Capturar y Analizar Micro-Evento (Ráfaga 3 Fotos)",
        "trigger_single_btn": "Capturar Fotograma Simple",
        "processing_burst": "Capturando ráfaga de 3 fotogramas y enviando a Nebius Token Factory...",
        "burst_success": "¡Micro-evento procesado! Actividad: {activity} ({mood})",
        "clinical_alert": "Alerta Clínica Identificada: {reason}",
        "vet_evidence": "Evidencia Veterinaria (Tavily): {summary}",
        "vet_sources": "Fuentes Médicas: {sources}",
        
        # Tab 2: Timeline
        "timeline_header": "Bitácora Histórica de Micro-Eventos",
        "clear_history_btn": "Limpiar Historial",
        "empty_timeline": "Aún no hay eventos registrados. Dispara una captura en el Monitor en Vivo.",
        "badge_normal": "NORMAL",
        "badge_alert": "ALERTA CLÍNICA",
        "posture_label": "Postura",
        "energy_label": "Nivel de Energía",
        
        # Tab 3: Digest & Habits
        "digest_header": "Matriz de Hábitos y Resumen de Bienestar para {pet_name}",
        "comfort_index": "Índice de Confort",
        "water_visits": "Hidratación (Visitas)",
        "food_visits": "Alimentación (Visitas)",
        "sleep_hours": "Horas de Sueño",
        "active_hours": "Horas Activas",
        "vs_baseline": "vs línea base",
        "generate_digest_btn": "Generar Reporte Diario con NVIDIA Nemotron",
        "generating_digest": "NVIDIA Nemotron sintetizando bitácora y correlaciones de hábitos...",
        "report_title": "Reporte de Salud Diario ({date})",
        "wellness_score_label": "Puntuación de Bienestar General",
        "highlights_title": "Momentos Destacados del Día",
        "actions_title": "Cuidados y Acciones Sugeridas",
        "alerts_title": "Alertas Preventivas",
        
        # Tab 4: Chat
        "chat_header": "Consultas sobre el Estado y Salud de {pet_name}",
        "chat_caption": "Impulsado por NVIDIA Nemotron con contexto completo de la matriz de hábitos y literatura clínica de Tavily.",
        "chat_input_placeholder": "Pregunta lo que desees (ej. ¿Cómo estuvo Toby hoy? ¿Tomó suficiente agua?)...",
        "chat_greeting": "¡Hola! Soy PawSentry AI, tu asistente veterinario y etólogo impulsado por NVIDIA Nemotron. ¿Cómo puedo ayudarte hoy con el bienestar de tu compañero?",
        "chat_analyzing": "Nemotron analizando historial de eventos y literatura veterinaria...",
        "prompt_chip_1": "¿Qué tan activo estuvo hoy?",
        "prompt_chip_2": "¿Hay alguna alerta de salud o estrés?",
        "prompt_chip_3": "¿Tomó suficiente agua hoy?",

        # Modo Centinela Autónomo y Cámara
        "sentry_title": "Modo Centinela Autónomo en el Borde",
        "sentry_desc": "Vigilancia continua: OpenCV MOG2 detecta movimiento de mascotas, dispara automáticamente una ráfaga de 3 fotos e identifica integrantes con Nebius Vision.",
        "sentry_start_btn": "🛡️ Iniciar Centinela Autónomo",
        "sentry_stop_btn": "🛑 Detener Centinela",
        "sentry_status_scanning": "Escaneando habitación en busca de movimiento...",
        "sentry_status_motion": "⚡ ¡Movimiento de mascota detectado! Disparando ráfaga...",
        "sentry_status_cooldown": "Sensor en cooldown ({sec:.1f}s restantes)...",
        "cam_source_label": "Fuente de Cámara",
        "cam_source_usb": "💻 Cámara Web USB Local",
        "cam_source_rtsp": "🌐 Cámara IP (EZVIZ H8c Pro RTSP)",
        "rtsp_url_label": "URL del Stream RTSP (EZVIZ H8c Pro)",
        "rtsp_helper": "Formato: rtsp://admin:{código_verificación}@{ip_camara}:554/H.264/ch1/main",
        "tag_as_btn": "👉 Etiquetar como {pet_name}",
        "tag_success": "¡Guardado! Asignado a {pet_name} y notas de perfil actualizadas."
    }
}


def t(key: str, lang: str = "en", **kwargs: Any) -> str:
    """
    Obtiene la cadena traducida para la clave dada en el idioma indicado ('en' por defecto).
    Soporta formato con variables: t('burst_success', lang='en', activity='eating', mood='calm')
    """
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    template = lang_dict.get(key, TRANSLATIONS["en"].get(key, key))
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


ENUM_TRANSLATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "activity": {
        "en": {
            "eating": "Eating",
            "drinking": "Drinking",
            "sleeping": "Sleeping",
            "resting": "Resting",
            "playing": "Playing",
            "walking": "Walking",
            "scratching": "Scratching",
            "pacing": "Pacing",
            "anxious_behavior": "Anxious Behavior",
            "unknown": "Unknown Activity",
        },
        "es": {
            "eating": "Comiendo",
            "drinking": "Bebiendo Agua",
            "sleeping": "Durmiendo",
            "resting": "Descansando",
            "playing": "Jugando",
            "walking": "Caminando",
            "scratching": "Rascándose",
            "pacing": "Merodeando",
            "anxious_behavior": "Conducta Ansiosa",
            "unknown": "Actividad Desconocida",
        },
    },
    "mood": {
        "en": {
            "relaxed": "Relaxed",
            "playful": "Playful",
            "alert": "Alert",
            "anxious": "Anxious",
            "lethargic": "Lethargic",
            "agitated": "Agitated",
            "unknown": "Unknown Mood",
        },
        "es": {
            "relaxed": "Relajado",
            "playful": "Juguetón",
            "alert": "Alerta",
            "anxious": "Ansioso",
            "lethargic": "Letárgico",
            "agitated": "Agitado",
            "unknown": "Ánimo Desconocido",
        },
    },
    "posture": {
        "en": {
            "lying_down": "Lying Down",
            "sitting": "Sitting",
            "standing": "Standing",
            "crouched": "Crouched",
            "stretching": "Stretching",
            "awkward_or_limping": "Awkward / Limping",
            "unknown": "Unknown Posture",
        },
        "es": {
            "lying_down": "Acostado",
            "sitting": "Sentado",
            "standing": "De Pie",
            "crouched": "Agazapado",
            "stretching": "Estirándose",
            "awkward_or_limping": "Cojera / Postura Anómala",
            "unknown": "Postura Desconocida",
        },
    },
    "species": {
        "en": {
            "dog": "Dog",
            "cat": "Cat",
            "other": "Other",
        },
        "es": {
            "dog": "Perro",
            "cat": "Gato",
            "other": "Otro",
        },
    },
}


def t_val(category: str, value: str, lang: str = "en") -> str:
    """
    Traduce valores de enums (activity, mood, posture, species) al idioma seleccionado.
    """
    cat_dict = ENUM_TRANSLATIONS.get(category, {})
    lang_dict = cat_dict.get(lang, cat_dict.get("en", {}))
    clean_val = str(value).lower().strip().replace(" ", "_")
    if clean_val in lang_dict:
        return lang_dict[clean_val]
    # Fallback: title case
    return str(value).replace("_", " ").title()

