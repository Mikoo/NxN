"""
core/schemas.py
Esquemas de datos fuertemente tipados con Pydantic v2 para PawSentry AI.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PetActivityType(str, Enum):
    EATING = "eating"
    DRINKING = "drinking"
    SLEEPING = "sleeping"
    RESTING = "resting"
    PLAYING = "playing"
    WALKING = "walking"
    SCRATCHING = "scratching"
    PACING = "pacing"
    ANXIOUS_BEHAVIOR = "anxious_behavior"
    UNKNOWN = "unknown"


class PetPosture(str, Enum):
    LYING_DOWN = "lying_down"
    SITTING = "sitting"
    STANDING = "standing"
    CROUCHED = "crouched"
    STRETCHING = "stretching"
    AWKWARD_OR_LIMPING = "awkward_or_limping"
    UNKNOWN = "unknown"


class MoodType(str, Enum):
    RELAXED = "relaxed"
    PLAYFUL = "playful"
    ALERT = "alert"
    ANXIOUS = "anxious"
    LETHARGIC = "lethargic"
    AGITATED = "agitated"
    UNKNOWN = "unknown"


class PetProfile(BaseModel):
    """Perfil registrado de una mascota en el hogar."""
    pet_id: str = Field(..., description="ID único o slug de la mascota (ej. 'firulais', 'luna')")
    name: str = Field(..., description="Nombre de la mascota")
    species: str = Field(default="dog", description="Especie: 'dog' | 'cat' | 'other'")
    breed: Optional[str] = Field(default=None, description="Raza de la mascota")
    age_years: int = Field(default=3, ge=0, le=40, description="Edad en años")
    description: str = Field(default="", description="Descripción física para visión (ej: 'Golden Retriever grande, pelaje dorado claro')")
    photo_b64: Optional[str] = Field(default=None, description="Foto de perfil en Base64")
    created_at: datetime = Field(default_factory=datetime.now)


class BehaviorAnalysis(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    pet_detected: bool = Field(default=False)
    pet_type: Optional[str] = Field(default="unknown")
    pet_id: Optional[str] = Field(default=None, description="ID de la mascota identificada")
    activity: PetActivityType = Field(default=PetActivityType.UNKNOWN)
    posture: PetPosture = Field(default=PetPosture.UNKNOWN)
    mood: MoodType = Field(default=MoodType.UNKNOWN)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    energy_level: int = Field(default=5, ge=1, le=10)
    is_anomaly: bool = Field(default=False)
    anomaly_reason: Optional[str] = Field(default=None)
    clinical_notes: Optional[str] = Field(default=None)
    observed_features: Optional[str] = Field(default=None, description="Rasgos visuales específicos observados (color de pelaje, patrón, manchas)")
    image_paths: List[str] = Field(default_factory=list)

    @field_validator("activity", mode="before")
    @classmethod
    def clean_activity(cls, v: Any) -> Any:
        if v is None or v == "":
            return PetActivityType.UNKNOWN
        if isinstance(v, str):
            v_low = v.lower().strip()
            mapping = {
                "sitting": PetActivityType.RESTING,
                "standing": PetActivityType.RESTING,
                "lying": PetActivityType.RESTING,
                "lying down": PetActivityType.RESTING,
                "running": PetActivityType.PLAYING,
                "drinking water": PetActivityType.DRINKING,
                "eating food": PetActivityType.EATING,
            }
            if v_low in mapping:
                return mapping[v_low]
            valid_vals = {e.value for e in PetActivityType}
            if v_low in valid_vals:
                return v_low
            return PetActivityType.UNKNOWN
        return v

    @field_validator("posture", mode="before")
    @classmethod
    def clean_posture(cls, v: Any) -> Any:
        if v is None or v == "":
            return PetPosture.UNKNOWN
        if isinstance(v, str):
            v_low = v.lower().strip()
            valid_vals = {e.value for e in PetPosture}
            if v_low in valid_vals:
                return v_low
            return PetPosture.UNKNOWN
        return v

    @field_validator("mood", mode="before")
    @classmethod
    def clean_mood(cls, v: Any) -> Any:
        if v is None or v == "":
            return MoodType.UNKNOWN
        if isinstance(v, str):
            v_low = v.lower().strip()
            mapping = {
                "neutral": MoodType.RELAXED,
                "calm": MoodType.RELAXED,
                "happy": MoodType.PLAYFUL,
                "sleepy": MoodType.RELAXED,
                "stressed": MoodType.ANXIOUS,
                "scared": MoodType.ANXIOUS,
                "nervous": MoodType.ANXIOUS,
                "tired": MoodType.LETHARGIC,
            }
            if v_low in mapping:
                return mapping[v_low]
            valid_vals = {e.value for e in MoodType}
            if v_low in valid_vals:
                return v_low
            return MoodType.UNKNOWN
        return v

    @field_validator("energy_level", mode="before")
    @classmethod
    def clean_energy(cls, v: Any) -> Any:
        if v is None:
            return 5
        try:
            val = int(v)
            return max(1, min(10, val))
        except (ValueError, TypeError):
            return 5


class BehavioralBout(BaseModel):
    activity: PetActivityType
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    event_count: int
    dominant_mood: MoodType = MoodType.RELAXED
    location_hint: Optional[str] = None


class HabitMatrixMetrics(BaseModel):
    water_visits_count: int = Field(default=0, ge=0)
    food_visits_count: int = Field(default=0, ge=0)
    sleep_hours_estimated: float = Field(default=0.0, ge=0.0, le=24.0)
    active_hours_estimated: float = Field(default=0.0, ge=0.0, le=24.0)
    comfort_index: float = Field(default=85.0, ge=0.0, le=100.0)
    activity_variance_vs_baseline: float = Field(default=0.0)
    total_bouts_count: int = Field(default=0, ge=0)
    rest_bouts_count: int = Field(default=0, ge=0)
    active_bouts_count: int = Field(default=0, ge=0)
    hydration_score: float = Field(default=90.0, ge=0.0, le=100.0)
    nutrition_score: float = Field(default=90.0, ge=0.0, le=100.0)
    mobility_score: float = Field(default=90.0, ge=0.0, le=100.0)


class DailyPetReport(BaseModel):
    date: str = Field(...)
    pet_name: str = Field(default="Mi Mascota")
    pet_id: Optional[str] = Field(default=None)
    overall_wellness_score: int = Field(default=85, ge=0, le=100)
    summary_narrative: str = Field(...)
    key_highlights: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    metrics: HabitMatrixMetrics = Field(default_factory=HabitMatrixMetrics)
    recommended_actions: List[str] = Field(default_factory=list)
    highlight_image_paths: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalizar narrativa de resumen
            if "summary_narrative" not in data or not data["summary_narrative"]:
                for k in ["summary", "narrative", "overview", "description", "resumen"]:
                    if k in data and data[k]:
                        data["summary_narrative"] = str(data[k])
                        break
                if "summary_narrative" not in data:
                    data["summary_narrative"] = "Behavioral patterns and wellness routine recorded normally today."

            # Normalizar score de bienestar
            if "overall_wellness_score" not in data or data["overall_wellness_score"] is None:
                for k in ["wellness_score", "score", "overall_score", "puntaje"]:
                    if k in data and data[k] is not None:
                        try:
                            data["overall_wellness_score"] = int(data[k])
                        except Exception:
                            pass
                        break
                if "overall_wellness_score" not in data:
                    data["overall_wellness_score"] = 85

            # Normalizar highlights
            if "key_highlights" not in data:
                for k in ["highlights", "positive_moments", "momentos_clave"]:
                    if k in data and isinstance(data[k], list):
                        data["key_highlights"] = [str(x) for x in data[k]]
                        break

            # Normalizar acciones recomendadas
            if "recommended_actions" not in data:
                for k in ["recommendations", "actions", "suggestions", "acciones"]:
                    if k in data and isinstance(data[k], list):
                        data["recommended_actions"] = [str(x) for x in data[k]]
                        break

            # Normalizar alertas
            if "alerts" not in data:
                for k in ["warnings", "clinical_alerts", "alertas"]:
                    if k in data and isinstance(data[k], list):
                        data["alerts"] = [str(x) for x in data[k]]
                        break

        return data


class UrgencyLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    URGENT = "urgent"


class VetConsultationResult(BaseModel):
    query: str
    urgency: UrgencyLevel = UrgencyLevel.LOW
    summary: str
    potential_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    clinical_confidence: float = Field(default=0.88, ge=0.0, le=1.0)

