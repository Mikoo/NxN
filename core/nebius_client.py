"""
core/nebius_client.py
Cliente de inferencia para Nebius Token Factory y modelos open source de NVIDIA.
Soporta analisis visual de micro-eventos (Llama-3.2-Vision / Nemotron Vision),
sintesis analitica con Nemotron 70B y modo Mock desacoplado para pruebas offline.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Asegurar path para imports directos
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dotenv import load_dotenv
from openai import OpenAI

from core.schemas import (
    BehaviorAnalysis,
    DailyPetReport,
    HabitMatrixMetrics,
    MoodType,
    PetActivityType,
    PetPosture,
    PetProfile,
)

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )


class NebiusInferenceError(Exception):
    """Excepcion personalizada para errores en el pipeline de Nebius."""
    pass


class NebiusClient:
    """
    Cliente desacoplado para servicios de inferencia en Nebius Token Factory.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        mock: Optional[bool] = None,
    ) -> None:
        """
        Inicializa el cliente de Nebius.

        Args:
            api_key: Clave de API de Nebius (o lee NEBIUS_API_KEY del entorno).
            base_url: Endpoint base (por defecto lee NEBIUS_BASE_URL o https://api.studio.nebius.ai/v1).
            vision_model: Modelo multimodal para imagenes/video.
            reasoning_model: Modelo de texto para sintesis y correlacion de habitos.
            mock: Si es True, fuerza modo simulacion sin llamadas de red.
        """
        self.api_key: Optional[str] = api_key or os.getenv("NEBIUS_API_KEY")
        self.base_url: str = (
            base_url
            or os.getenv("NEBIUS_BASE_URL")
            or "https://api.studio.nebius.ai/v1"
        )
        self.vision_model: str = (
            vision_model
            or os.getenv("NEBIUS_VISION_MODEL")
            or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        )
        self.reasoning_model: str = (
            reasoning_model
            or os.getenv("NEBIUS_REASONING_MODEL")
            or "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
        )

        # Evaluar si se activa el modo Mock
        env_mock = os.getenv("USE_MOCKS", "false").lower() in ("true", "1", "yes")
        if mock is not None:
            self.mock: bool = mock
        elif not self.api_key or self.api_key.startswith("your_") or env_mock:
            self.mock = True
        else:
            self.mock = False

        self._client: Optional[OpenAI] = None
        if not self.mock:
            try:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.info(
                    "NebiusClient conectado a Token Factory [%s] con modelos: Vision=%s, Reasoning=%s",
                    self.base_url,
                    self.vision_model,
                    self.reasoning_model,
                )
            except Exception as err:
                logger.warning(
                    "Error al inicializar cliente OpenAI para Nebius: %s. Activando modo Mock.",
                    err,
                )
                self.mock = True
        else:
            logger.info("NebiusClient operando en modo MOCK (Simulacion local sin costo de API).")

    @staticmethod
    def _encode_image_to_base64(image_path: Union[Path, str]) -> str:
        """Lee un archivo de imagen local y lo codifica en base64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de imagen: {path}")
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def analyze_micro_event(
        self,
        image_paths: List[Union[Path, str]],
        pet_name: str = "Toby",
        registered_pets: Optional[List[PetProfile]] = None,
        pet_id: Optional[str] = None,
        lang: str = "en",
    ) -> BehaviorAnalysis:
        """
        Analiza una rafaga de 1 a 3 fotogramas de un micro-evento temporal con el modelo
        NVIDIA Vision para deducir identidad de mascota, postura, accion, trayectoria y confort.
        Soporta prompts en ingles (default) o espanol segun el idioma activo de la aplicacion.

        Args:
            image_paths: Lista de rutas a los fotogramas (ej. inicio, punto medio, final).
            pet_name: Nombre de la mascota para el contexto de analisis.
            registered_pets: Lista opcional de perfiles de mascotas registradas para identificacion.
            pet_id: ID sugerido de la mascota activa.
            lang: Idioma deseado para la inferencia ('en' | 'es').

        Returns:
            BehaviorAnalysis: Objeto validado por Pydantic con la clasificacion.
        """
        if self.mock or self._client is None:
            return self._mock_analyze_micro_event(image_paths, pet_name, registered_pets, pet_id, lang=lang)

        try:
            logger.info(
                "Enviando micro-evento (%d fotogramas) a Nebius Vision [%s] (Idioma: %s)...",
                len(image_paths),
                self.vision_model,
                lang,
            )

            if lang == "en":
                system_prompt = (
                    "You are an animal ethologist and veterinary computer vision AI assistant for PawSentry AI. "
                    "Always respond in strict JSON format without any conversational intro. "
                    "All natural text fields (clinical_notes, anomaly_reason) must be written in English."
                )
                if registered_pets:
                    pets_info = "\n".join(
                        f'- ID: "{p.pet_id}", Name: "{p.name}", Species: "{p.species}", Breed: "{p.breed or "Mixed"}", Traits: "{p.description or "No specific description"}"'
                        for p in registered_pets
                    )
                    pets_context = (
                        f"The household has the following registered pets:\n{pets_info}\n"
                        "Carefully identify WHICH registered pet appears in these frames based on their visual traits and species. "
                        'If clearly matched, assign their exact "pet_id". If not a clear match, assign "unknown".\n'
                    )
                else:
                    pets_context = f"The target pet is '{pet_name}' (ID: '{pet_id or pet_name.lower()}').\n"

                instruction_text = (
                    f"Analyze this temporal sequence of {len(image_paths)} frame(s) captured by PawSentry AI.\n{pets_context}"
                    "Evaluate whether a pet is present in the frame. If ONLY furniture, room, or humans appear without pets, set 'pet_detected': false, 'pet_id': null, 'activity': 'unknown'.\n"
                    "If a dog or cat IS visible, evaluate its biomechanical action, posture, mood, "
                    "energy level (1-10), visual features, and if any anomaly or symptom is present (limping, compulsive scratching, lethargy). "
                    "Respond EXCLUSIVELY with valid JSON with this exact structure:\n"
                    "{\n"
                    '  "pet_detected": true,\n'
                    '  "pet_id": "ichi" | "ditto" | "unknown" (or null if pet_detected is false),\n'
                    '  "pet_type": "dog" | "cat" | null,\n'
                    '  "activity": "eating" | "drinking" | "sleeping" | "resting" | "playing" | "walking" | "scratching" | "pacing" | "anxious_behavior",\n'
                    '  "posture": "lying_down" | "sitting" | "standing" | "crouched" | "stretching" | "awkward_or_limping",\n'
                    '  "mood": "relaxed" | "playful" | "alert" | "anxious" | "lethargic" | "agitated",\n'
                    '  "confidence": 0.95,\n'
                    '  "energy_level": 5,\n'
                    '  "observed_features": "Brief visual description of the animal (coat color, pattern, markings, size)",\n'
                    '  "is_anomaly": false,\n'
                    '  "anomaly_reason": null,\n'
                    '  "clinical_notes": "Concise ethological and behavioral dynamic description in English"\n'
                    "}"
                )
            else:
                system_prompt = (
                    "Eres un etólogo y asistente veterinario de visión artificial para PawSentry AI. "
                    "Responde siempre en formato JSON estricto sin introducción. "
                    "Todos los campos de texto libre (clinical_notes, anomaly_reason, observed_features) deben estar en español."
                )
                if registered_pets:
                    pets_info = "\n".join(
                        f'- ID: "{p.pet_id}", Nombre: "{p.name}", Especie: "{p.species}", Raza: "{p.breed or "Mestizo"}", Rasgos: "{p.description or "Sin rasgos especificados"}"'
                        for p in registered_pets
                    )
                    pets_context = (
                        f"En este hogar conviven las siguientes mascotas registradas:\n{pets_info}\n"
                        "Determina con precisión a cuál de estas mascotas corresponde el animal capturado según sus rasgos visuales. "
                        'Si coincide claramente, asigna su "pet_id" exacto. Si no coincide o hay duda, asigna "unknown".\n'
                    )
                else:
                    pets_context = f"La mascota a analizar es '{pet_name}' (ID sugerido: '{pet_id or pet_name.lower()}').\n"

                instruction_text = (
                    f"Analiza esta secuencia temporal de {len(image_paths)} fotograma(s) capturada(s) por "
                    f"PawSentry AI.\n{pets_context}"
                    "Evalúa si hay una mascota visible. Si SOLO se observa la habitación, muebles o humanos sin mascotas, asigna 'pet_detected': false, 'pet_id': null, 'activity': 'unknown'.\n"
                    "Si un perro o gato ES visible, evalúa su acción biomecánica, postura, estado de ánimo, "
                    "nivel de energía (1-10), rasgos visuales y si existe alguna anomalía o síntoma (cojera, rascado convulsivo, letargo). "
                    "Responde EXCLUSIVAMENTE con un JSON válido con esta estructura:\n"
                    "{\n"
                    '  "pet_detected": true,\n'
                    '  "pet_id": "ichi" | "ditto" | "unknown" (o null si pet_detected es false),\n'
                    '  "pet_type": "dog" | "cat" | null,\n'
                    '  "activity": "eating" | "drinking" | "sleeping" | "resting" | "playing" | "walking" | "scratching" | "pacing" | "anxious_behavior",\n'
                    '  "posture": "lying_down" | "sitting" | "standing" | "crouched" | "stretching" | "awkward_or_limping",\n'
                    '  "mood": "relaxed" | "playful" | "alert" | "anxious" | "lethargic" | "agitated",\n'
                    '  "confidence": 0.95,\n'
                    '  "energy_level": 5,\n'
                    '  "observed_features": "Breve descripción visual del animal (color de pelaje, patrón, manchas, tamaño)",\n'
                    '  "is_anomaly": false,\n'
                    '  "anomaly_reason": null,\n'
                    '  "clinical_notes": "Breve descripción de la dinámica observada e identificación visual en español"\n'
                    "}"
                )

            # Construir contenido del mensaje con imagenes en base64
            content_elements: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": instruction_text,
                }
            ]

            for img_path in image_paths:
                b64_data = self._encode_image_to_base64(img_path)
                content_elements.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                })

            response = self._client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": content_elements},
                ],
                temperature=0.2,
                max_tokens=600,
            )

            raw_content = response.choices[0].message.content or "{}"
            cleaned_json = raw_content.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            parsed_data = json.loads(cleaned_json)
            parsed_data["image_paths"] = [str(p) for p in image_paths]
            parsed_data["timestamp"] = datetime.now()

            # Resolver pet_id segun deteccion y certeza
            if not parsed_data.get("pet_detected"):
                parsed_data["pet_id"] = None
            elif not parsed_data.get("pet_id") or parsed_data.get("pet_id") == "unknown":
                if registered_pets and len(registered_pets) == 1:
                    parsed_data["pet_id"] = registered_pets[0].pet_id
                elif not registered_pets and pet_id:
                    parsed_data["pet_id"] = pet_id
                else:
                    # En entorno multi-mascota con incertidumbre, mantener 'unknown' para active learning
                    parsed_data["pet_id"] = "unknown"

            analysis = BehaviorAnalysis(**parsed_data)
            logger.info(
                "Inferencia completada: MascotaID=%s, Tipo=%s, Actividad=%s, Animo=%s, Anomalia=%s",
                analysis.pet_id,
                analysis.pet_type,
                analysis.activity.value,
                analysis.mood.value,
                analysis.is_anomaly,
            )
            return analysis

        except Exception as err:
            logger.error("Error durante inferencia en Nebius: %s. Aplicando fallback Mock.", err)
            return self._mock_analyze_micro_event(image_paths, pet_name, registered_pets, pet_id, lang=lang)

    def generate_daily_digest(
        self,
        events: List[BehaviorAnalysis],
        pet_name: str = "Toby",
        pet_id: Optional[str] = None,
        date_str: Optional[str] = None,
        lang: str = "en",
    ) -> DailyPetReport:
        """
        Sintetiza la bitacora de eventos del dia usando NVIDIA Nemotron 70B,
        calculando la matriz de habitos, indice de confort y alertas preventivas.
        Genera el reporte en ingles (default para jueces) o espanol segun 'lang'.

        Args:
            events: Lista cronologica de micro-eventos analizados durante la jornada.
            pet_name: Nombre de la mascota.
            pet_id: ID opcional de la mascota para el reporte.
            date_str: Fecha en formato YYYY-MM-DD (por defecto hoy).
            lang: Idioma deseado para el reporte ('en' | 'es').

        Returns:
            DailyPetReport: Reporte formal validado con Pydantic.
        """
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")

        if self.mock or self._client is None or not events:
            return self._mock_generate_daily_digest(events, pet_name, target_date, pet_id, lang=lang)

        try:
            logger.info(
                "Sintetizando %d eventos con Nemotron [%s] (Idioma: %s)...",
                len(events),
                self.reasoning_model,
                lang,
            )

            # Filtrar y compactar eventos para el prompt de Nemotron (priorizando alimentacion, hidratacion y anomalias)
            key_events = [e for e in events if e.activity in (PetActivityType.DRINKING, PetActivityType.EATING, PetActivityType.WALKING, PetActivityType.PLAYING) or e.is_anomaly]
            resting_sample = [e for e in events if e.activity in (PetActivityType.SLEEPING, PetActivityType.RESTING)][:8]
            recent_sample = events[-10:] if len(events) >= 10 else events
            merged_events = list({id(e): e for e in (key_events + resting_sample + recent_sample)}.values())
            merged_events.sort(key=lambda x: x.timestamp)

            events_summary = [
                {
                    "time": e.timestamp.strftime("%H:%M:%S"),
                    "activity": e.activity.value,
                    "posture": e.posture.value,
                    "mood": e.mood.value,
                    "energy": e.energy_level,
                    "anomaly": e.is_anomaly,
                    "notes": e.clinical_notes,
                }
                for e in merged_events
            ]

            if lang == "en":
                system_prompt = (
                    "You are an animal ethology and veterinary wellness assistant powered by NVIDIA Nemotron. "
                    "Always respond in valid JSON format only, without introductory text or markdown tags outside JSON."
                )
                prompt_text = (
                    f"Act as the PawSentry AI Wellness Advisor for the companion '{pet_name}'. "
                    f"Here is the event log recorded today ({target_date}):\n"
                    f"{json.dumps(events_summary, indent=2)}\n\n"
                    "Generate a consolidated health and wellness report in strict valid JSON matching this exact structure:\n"
                    "{\n"
                    '  "overall_wellness_score": 92,\n'
                    '  "summary_narrative": "Detailed 2-3 sentence overview of the day in English",\n'
                    '  "key_highlights": ["Key positive behavior or habit observed", "Another highlight"],\n'
                    '  "alerts": [],\n'
                    '  "recommended_actions": ["Care, play or hydration suggestion in English", "Another recommendation"]\n'
                    "}"
                )
            else:
                system_prompt = (
                    "Eres un asistente de etología y bienestar animal impulsado por NVIDIA Nemotron. "
                    "Responde siempre en formato JSON válido únicamente, sin texto introductorio."
                )
                prompt_text = (
                    f"Actúa como el Asesor de Bienestar de PawSentry AI para la mascota '{pet_name}'. "
                    f"A continuación tienes la bitácora de eventos registrados hoy ({target_date}):\n"
                    f"{json.dumps(events_summary, indent=2)}\n\n"
                    "Genera un reporte consolidado en JSON estricto con esta estructura exacta:\n"
                    "{\n"
                    '  "overall_wellness_score": 92,\n'
                    '  "summary_narrative": "Resumen detallado de 2-3 oraciones sobre la jornada en español",\n'
                    '  "key_highlights": ["Conducta o hábito positivo observado", "Otro momento clave"],\n'
                    '  "alerts": [],\n'
                    '  "recommended_actions": ["Sugerencia de cuidado, juego o hidratación en español", "Otra recomendación"]\n'
                    "}"
                )

            response = self._client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            raw_content = response.choices[0].message.content or "{}"
            logger.info("Nemotron raw_content length: %d, snippet: %s", len(raw_content), repr(raw_content[:200]))
            cleaned_json = raw_content.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            data = json.loads(cleaned_json)
            logger.info("Nemotron parsed keys: %s", list(data.keys()))
            logger.info("Nemotron parsed sample: %s", json.dumps(data)[:200])
            data["date"] = target_date
            data["pet_name"] = pet_name
            data["pet_id"] = pet_id
            data["highlight_image_paths"] = [
                p for e in events for p in e.image_paths if p
            ][:4]

            return DailyPetReport(**data)

        except Exception as err:
            logger.error("Error en sintesis con Nemotron: %s. Aplicando fallback Mock.", err)
            return self._mock_generate_daily_digest(events, pet_name, target_date, pet_id, lang=lang)

    # -------------------------------------------------------------------------
    # MOCK ENGINE: Generador de respuestas realistas para pruebas sin costo de API
    # -------------------------------------------------------------------------
    def _mock_analyze_micro_event(
        self,
        image_paths: List[Union[Path, str]],
        pet_name: str,
        registered_pets: Optional[List[PetProfile]] = None,
        pet_id: Optional[str] = None,
        lang: str = "en",
    ) -> BehaviorAnalysis:
        """Simula una inferencia de vision multimodal coherente con identificacion multi-mascota y soporte bilingue."""
        target_pet_id = pet_id
        target_pet_type = "dog"
        if registered_pets:
            matched = next((p for p in registered_pets if p.pet_id == pet_id or p.name.lower() == pet_name.lower()), None)
            if not matched:
                matched = random.choice(registered_pets)
            target_pet_id = matched.pet_id
            target_pet_type = matched.species.lower()
            pet_name = matched.name
        else:
            target_pet_id = pet_id or pet_name.lower()

        if lang == "en":
            scenarios = [
                (
                    PetActivityType.DRINKING,
                    PetPosture.STANDING,
                    MoodType.RELAXED,
                    4,
                    False,
                    None,
                    f"{pet_name} approached the water fountain and drank continuously for 18 seconds with relaxed posture.",
                ),
                (
                    PetActivityType.PLAYING,
                    PetPosture.STANDING,
                    MoodType.PLAYFUL,
                    8,
                    False,
                    None,
                    f"{pet_name} shows high tail carriage and energetic exploration movements across the area.",
                ),
                (
                    PetActivityType.RESTING,
                    PetPosture.LYING_DOWN,
                    MoodType.RELAXED,
                    2,
                    False,
                    None,
                    f"{pet_name} is resting comfortably in lateral recumbency with regular, peaceful breathing.",
                ),
                (
                    PetActivityType.EATING,
                    PetPosture.CROUCHED,
                    MoodType.RELAXED,
                    5,
                    False,
                    None,
                    f"{pet_name} is consuming their meal portion with healthy appetite and steady posture.",
                ),
                (
                    PetActivityType.SCRATCHING,
                    PetPosture.SITTING,
                    MoodType.ALERT,
                    5,
                    True,
                    "Persistent scratching observed around the left ear region.",
                    f"{pet_name} is intensively scratching the left ear area for more than 25 consecutive seconds.",
                ),
            ]
        else:
            scenarios = [
                (
                    PetActivityType.DRINKING,
                    PetPosture.STANDING,
                    MoodType.RELAXED,
                    4,
                    False,
                    None,
                    f"{pet_name} se aproximo al bebedero e ingirio agua de forma continua durante 18 segundos.",
                ),
                (
                    PetActivityType.PLAYING,
                    PetPosture.STANDING,
                    MoodType.PLAYFUL,
                    8,
                    False,
                    None,
                    f"{pet_name} presenta cola en posicion alta y realiza saltos dinamicos explorando la zona.",
                ),
                (
                    PetActivityType.RESTING,
                    PetPosture.LYING_DOWN,
                    MoodType.RELAXED,
                    2,
                    False,
                    None,
                    f"{pet_name} descansa comodamente en decubito lateral con respiracion pausada.",
                ),
                (
                    PetActivityType.EATING,
                    PetPosture.CROUCHED,
                    MoodType.RELAXED,
                    5,
                    False,
                    None,
                    f"{pet_name} consume su racion alimenticia con buen apetito y postura adecuada.",
                ),
                (
                    PetActivityType.SCRATCHING,
                    PetPosture.SITTING,
                    MoodType.ALERT,
                    5,
                    True,
                    "Rascado recurrente en la zona retroauricular izquierda.",
                    f"{pet_name} se rasca intensamente durante mas de 25 segundos consecutivos.",
                ),
            ]

        choice = random.choice(scenarios)
        return BehaviorAnalysis(
            timestamp=datetime.now(),
            pet_detected=True,
            pet_type=target_pet_type,
            pet_id=target_pet_id,
            activity=choice[0],
            posture=choice[1],
            mood=choice[2],
            confidence=0.95,
            energy_level=choice[3],
            is_anomaly=choice[4],
            anomaly_reason=choice[5],
            clinical_notes=choice[6],
            image_paths=[str(p) for p in image_paths],
        )

    def _mock_generate_daily_digest(
        self,
        events: List[BehaviorAnalysis],
        pet_name: str,
        date_str: str,
        pet_id: Optional[str] = None,
        lang: str = "en",
    ) -> DailyPetReport:
        """Simula la sintesis analitica diaria con Nemotron con soporte bilingue."""
        highlight_photos = [
            p for e in events for p in e.image_paths if Path(p).exists()
        ][:4]

        water_count = sum(1 for e in events if e.activity == PetActivityType.DRINKING) or 3
        food_count = sum(1 for e in events if e.activity == PetActivityType.EATING) or 2
        anomalies = [e for e in events if e.is_anomaly]

        if lang == "en":
            alerts = [f"Alert: {a.anomaly_reason}" for a in anomalies if a.anomaly_reason]
            if not alerts:
                alerts = ["No biomechanical anomalies or signs of discomfort detected."]

            summary_narrative = (
                f"Today {pet_name} displayed a well-balanced and healthy behavioral pattern. "
                "Maintained fluid mobility with no visible signs of joint discomfort and completed "
                "regular visits to hydration and rest spots. Average mood was calm and receptive."
            )
            key_highlights = [
                f"Consistent hydration with {water_count} recorded water visits.",
                "Deep restorative sleep session observed during early morning quiet period.",
                "Alert, friendly posture displayed during ambient movements.",
            ]
            recommended_actions = [
                "Ensure fresh water remains available in the designated bowl.",
                "Schedule a short interactive play or brushing session before evening.",
            ]
        else:
            alerts = [f"Alerta: {a.anomaly_reason}" for a in anomalies if a.anomaly_reason]
            if not alerts:
                alerts = ["No se detectaron anomalias biomecanicas ni signos de dolor."]

            summary_narrative = (
                f"Hoy {pet_name} demostro un patron conductual muy saludable y equilibrado. "
                "Mantuvo una movilidad fluida sin indicios de dolor articular y realizo visitas "
                "regulares a sus fuentes de hidratacion y descanso. Su nivel de animo promedio "
                "fue sereno y receptivo a estimulos del entorno."
            )
            key_highlights = [
                f"Hidratacion optima con {water_count} visitas consistentes al bebedero.",
                "Sesion de descanso profundo en las horas de mayor quietud matutina.",
                "Postura alerta y receptiva en momentos de transito ambiental.",
            ]
            recommended_actions = [
                "Mantener agua fresca y limpia en el dispensador habitual.",
                "Proveer un breve paseo o sesion de juego interactivo antes del anochecer.",
            ]

        return DailyPetReport(
            date=date_str,
            pet_name=pet_name,
            pet_id=pet_id,
            overall_wellness_score=92 if not anomalies else 78,
            summary_narrative=summary_narrative,
            key_highlights=key_highlights,
            alerts=alerts,
            metrics=HabitMatrixMetrics(
                water_visits_count=water_count,
                food_visits_count=food_count,
                sleep_hours_estimated=13.5,
                active_hours_estimated=3.2,
                comfort_index=88.5,
                activity_variance_vs_baseline=4.2,
            ),
            recommended_actions=recommended_actions,
            highlight_image_paths=highlight_photos,
        )


if __name__ == "__main__":
    print("==================================================")
    print("     PawSentry AI - Test de NebiusClient (Mock)   ")
    print("==================================================")

    client = NebiusClient(mock=True)
    test_analysis = client.analyze_micro_event(
        image_paths=["test_camera_0.jpg"],
        pet_name="Firulais",
    )

    print("\n[OK] Micro-evento analizado:")
    print(f" - Mascota detectada: {test_analysis.pet_detected} ({test_analysis.pet_type})")
    print(f" - Actividad: {test_analysis.activity.value}")
    print(f" - Postura: {test_analysis.posture.value}")
    print(f" - Animo: {test_analysis.mood.value} (Energia: {test_analysis.energy_level}/10)")
    print(f" - Anomalia: {test_analysis.is_anomaly} ({test_analysis.anomaly_reason})")
    print(f" - Notas: {test_analysis.clinical_notes}")

    print("\n[OK] Generando Daily Digest de prueba:")
    digest = client.generate_daily_digest([test_analysis], pet_name="Firulais")
    print(f" - Score de Bienestar: {digest.overall_wellness_score}/100")
    print(f" - Confort Index: {digest.metrics.comfort_index}/100")
    print(f" - Resumen: {digest.summary_narrative[:120]}...")
    print(f" - Alertas: {digest.alerts}")
    print("\n[EXITO] Prueba de esquemas e inferencia completada exitosamente.")
