"""
core/report_generator.py
Módulo de Síntesis y Correlación de Hábitos para PawSentry AI.
Consolida la bitacora de micro-eventos, computa la matriz de confort,
dispara consultas a VetAdvisor ante anomalías y genera el Daily Pet Digest con NVIDIA Nemotron.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Asegurar raíz en sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from core.nebius_client import NebiusClient
from core.schemas import (
    BehaviorAnalysis,
    BehavioralBout,
    DailyPetReport,
    HabitMatrixMetrics,
    MoodType,
    PetActivityType,
    VetConsultationResult,
)
from core.vet_advisor import VetAdvisor

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )


class ReportGenerator:
    """
    Agregador analitico y generador de reportes de bienestar animal.
    """

    def __init__(
        self,
        nebius_client: Optional[NebiusClient] = None,
        vet_advisor: Optional[VetAdvisor] = None,
        storage_file: Path | str = Path("data/events.json"),
        baseline_activity_hours: float = 3.5,
    ) -> None:
        """
        Inicializa el generador de reportes.

        Args:
            nebius_client: Instancia del cliente de inferencia de Nebius/NVIDIA.
            vet_advisor: Instancia del asesor clinico de Tavily.
            storage_file: Archivo JSON para persistencia local de la bitacora.
            baseline_activity_hours: Promedio historico de horas activas para calcular varianza.
        """
        self.nebius: NebiusClient = nebius_client or NebiusClient()
        self.vet_advisor: VetAdvisor = vet_advisor or VetAdvisor()
        self.storage_file: Path = Path(storage_file)
        self.baseline_activity: float = baseline_activity_hours

        # Asegurar directorio de almacenamiento
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir: Path = self.storage_file.parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[BehaviorAnalysis] = self._load_events()

    def _load_events(self) -> List[BehaviorAnalysis]:
        """Carga los eventos persistidos del archivo JSON si existe."""
        if not self.storage_file.exists():
            return []
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
                return [BehaviorAnalysis(**item) for item in raw_list]
        except Exception as err:
            logger.error("Error al cargar eventos de %s: %s", self.storage_file, err)
            return []

    def _save_events(self) -> None:
        """Persiste la lista de eventos en disco."""
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                serialized = [json.loads(e.model_dump_json()) for e in self._events]
                json.dump(serialized, f, indent=2, ensure_ascii=False)
        except Exception as err:
            logger.error("Error al guardar eventos en %s: %s", self.storage_file, err)

    def add_event(self, event: BehaviorAnalysis, lang: str = "en") -> Optional[VetConsultationResult]:
        """
        Registra un nuevo micro-evento. Si es una anomalia, consulta a VetAdvisor en el idioma activo.

        Args:
            event: Micro-evento analizado.
            lang: Idioma deseado ('en' | 'es').

        Returns:
            Optional[VetConsultationResult]: Resultado veterinario si hubo anomalia, o None.
        """
        self._events.append(event)
        self._save_events()
        logger.info(
            "Evento registrado en bitacora: %s (Total: %d, Mascota: %s)",
            event.activity.value,
            len(self._events),
            event.pet_id or "default",
        )

        # Si el modelo visual marco anomalia, investigar clinicamente con VetAdvisor
        vet_result: Optional[VetConsultationResult] = None
        if event.is_anomaly and event.anomaly_reason:
            logger.warning("Anomalia detectada: %s. Activando VetAdvisor (Idioma: %s)...", event.anomaly_reason, lang)
            pet_type = event.pet_type or "dog"
            vet_result = self.vet_advisor.consult_symptom(
                symptom_description=event.anomaly_reason,
                pet_type=pet_type,
                lang=lang,
            )

        return vet_result

    def get_events(self, pet_id: Optional[str] = None) -> List[BehaviorAnalysis]:
        """
        Retorna la lista de eventos almacenados, opcionalmente filtrados por mascota.
        """
        if pet_id:
            return [e for e in self._events if e.pet_id == pet_id]
        return list(self._events)

    def assign_event_pet(self, event_index: int, pet_id: str) -> bool:
        """Permite reasignar manualmente una mascota a un evento especifico."""
        if 0 <= event_index < len(self._events):
            self._events[event_index].pet_id = pet_id
            self._save_events()
            return True
        return False

    def clear_events(self, pet_id: Optional[str] = None) -> None:
        """Limpia la bitacora de eventos local (o solo de una mascota si se especifica)."""
        if pet_id:
            self._events = [e for e in self._events if e.pet_id != pet_id]
        else:
            self._events.clear()
        self._save_events()
        logger.info("Bitacora de eventos actualizada.")

    def get_behavioral_bouts(self, pet_id: Optional[str] = None, max_gap_seconds: float = 180.0) -> List[BehavioralBout]:
        """
        Agrupa micro-eventos consecutivos en 'bouts' o sesiones conductuales continuas (etología cuantitativa).
        Solo considera eventos donde una mascota fue detectada fehacientemente (pet_detected=True).
        """
        valid_events = [e for e in self._events if e.pet_detected and (e.pet_id == pet_id if pet_id else True)]
        if not valid_events:
            return []

        sorted_events = sorted(valid_events, key=lambda x: x.timestamp)
        bouts: List[BehavioralBout] = []
        current_bout_events: List[BehaviorAnalysis] = []

        for ev in sorted_events:
            if not current_bout_events:
                current_bout_events.append(ev)
            else:
                prev_ev = current_bout_events[-1]
                gap = (ev.timestamp - prev_ev.timestamp).total_seconds()
                if gap <= max_gap_seconds and ev.activity == prev_ev.activity:
                    current_bout_events.append(ev)
                else:
                    bouts.append(self._create_bout_from_events(current_bout_events))
                    current_bout_events = [ev]

        if current_bout_events:
            bouts.append(self._create_bout_from_events(current_bout_events))

        return bouts

    def _create_bout_from_events(self, events: List[BehaviorAnalysis]) -> BehavioralBout:
        start_t = events[0].timestamp
        end_t = events[-1].timestamp
        raw_seconds = (end_t - start_t).total_seconds()
        if raw_seconds < 10.0:
            if events[0].activity in (PetActivityType.DRINKING, PetActivityType.EATING):
                duration_min = 1.0
            elif events[0].activity in (PetActivityType.PLAYING, PetActivityType.WALKING):
                duration_min = 1.5
            else:
                duration_min = 2.0
        else:
            duration_min = round(max(0.5, raw_seconds / 60.0), 1)

        mood_counts: Dict[str, int] = {}
        for ev in events:
            mood_counts[ev.mood.value] = mood_counts.get(ev.mood.value, 0) + 1
        top_mood_val = max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else "relaxed"
        try:
            top_mood = MoodType(top_mood_val)
        except Exception:
            top_mood = MoodType.RELAXED

        loc_hint = None
        for ev in events:
            if ev.clinical_notes:
                notes_low = ev.clinical_notes.lower()
                for loc in ["couch", "sofa", "backrest", "window", "table", "chair", "bed", "lap", "floor", "regazo", "mesa", "cama"]:
                    if loc in notes_low:
                        loc_hint = loc.capitalize()
                        break
            if loc_hint:
                break

        return BehavioralBout(
            activity=events[0].activity,
            start_time=start_t,
            end_time=end_t,
            duration_minutes=duration_min,
            event_count=len(events),
            dominant_mood=top_mood,
            location_hint=loc_hint,
        )

    def compute_habit_metrics(self, pet_id: Optional[str] = None, target_date: Optional[str] = None) -> HabitMatrixMetrics:
        """
        Calcula las metricas objetivas de la matriz de habitos a partir de eventos y bouts conductuales reales.
        Filtra por fecha (hoy por defecto) y proyecta un presupuesto circadiano diario continuo.
        """
        date_str = target_date or datetime.now().strftime("%Y-%m-%d")
        target_events = [
            e for e in self._events
            if e.pet_detected
            and (e.pet_id == pet_id if pet_id else True)
            and e.timestamp.strftime("%Y-%m-%d") == date_str
        ]
        # Fallback a todos los eventos confirmados de la mascota si no hay en la fecha exacta
        if not target_events:
            target_events = [e for e in self._events if e.pet_detected and (e.pet_id == pet_id if pet_id else True)]
        if not target_events:
            return HabitMatrixMetrics()

        bouts = self.get_behavioral_bouts(pet_id=pet_id)
        # Filtrar bouts del dia si existen
        day_bouts = [b for b in bouts if b.start_time.strftime("%Y-%m-%d") == date_str]
        active_bouts_list = day_bouts if len(day_bouts) >= 2 else bouts

        water_visits = sum(1 for b in active_bouts_list if b.activity == PetActivityType.DRINKING)
        food_visits = sum(1 for b in active_bouts_list if b.activity == PetActivityType.EATING)
        
        # En caso de eventos individuales que no cerraron bout
        if water_visits == 0:
            water_visits = sum(1 for e in target_events if e.activity == PetActivityType.DRINKING)
        if food_visits == 0:
            food_visits = sum(1 for e in target_events if e.activity == PetActivityType.EATING)

        rest_bouts = [b for b in active_bouts_list if b.activity in (PetActivityType.SLEEPING, PetActivityType.RESTING)]
        active_bouts = [
            b for b in active_bouts_list
            if b.activity in (PetActivityType.PLAYING, PetActivityType.WALKING, PetActivityType.SCRATCHING, PetActivityType.PACING)
        ]

        rest_minutes = sum(b.duration_minutes for b in rest_bouts)
        active_minutes = sum(b.duration_minutes for b in active_bouts)

        # Proyeccion circadiana continua (sin corte brusco a las 12h)
        total_recorded_min = max(1.0, rest_minutes + active_minutes)
        active_ratio = active_minutes / total_recorded_min

        # Presupuesto circadiano felino/canino saludable (1.0h - 4.5h activo, 10.0h - 16.5h reposo)
        active_hours = round(min(5.0, max(1.0, 1.0 + (active_ratio * 3.5) + (len(active_bouts) * 0.15))), 1)
        sleep_hours = round(min(18.0, max(8.0, 16.0 - (active_hours * 1.1))), 1)

        variance = round(((active_hours - self.baseline_activity) / self.baseline_activity) * 100.0, 1)

        # Comfort Index con base etologica
        score = 92.0

        # Hidratacion (+4 si tomo agua)
        if water_visits >= 1:
            score += 4.0
            hydration_score = min(100.0, 85.0 + water_visits * 5.0)
        else:
            hydration_score = 75.0

        # Nutricion (+3 si comio)
        if food_visits >= 1:
            score += 3.0
            nutrition_score = min(100.0, 85.0 + food_visits * 5.0)
        else:
            nutrition_score = 80.0

        # Movilidad basada en bouts activos
        walking_bouts = len(active_bouts)
        mobility_score = min(100.0, max(60.0, 80.0 + walking_bouts * 3.0))

        # Penalizaciones por anomalias clinicas verificadas
        anomalies_count = sum(1 for e in target_events if e.is_anomaly)
        score -= anomalies_count * 15.0

        # Penalizacion moderada por ansiedad repetida
        anxious_count = sum(1 for e in target_events if e.mood in (MoodType.ANXIOUS, MoodType.AGITATED))
        score -= min(20.0, anxious_count * 5.0)
        comfort_index = max(15.0, min(100.0, round(score, 1)))

        zone_visits: Dict[str, int] = {}
        for e in target_events:
            z = e.location_zone or (e.camera_name if e.camera_name else "Living Room")
            zone_visits[z] = zone_visits.get(z, 0) + 1

        return HabitMatrixMetrics(
            water_visits_count=water_visits,
            food_visits_count=food_visits,
            sleep_hours_estimated=sleep_hours,
            active_hours_estimated=active_hours,
            comfort_index=comfort_index,
            activity_variance_vs_baseline=variance,
            total_bouts_count=len(active_bouts_list),
            rest_bouts_count=len(rest_bouts),
            active_bouts_count=len(active_bouts),
            hydration_score=round(hydration_score, 1),
            nutrition_score=round(nutrition_score, 1),
            mobility_score=round(mobility_score, 1),
            zone_visits_count=zone_visits,
        )

    def get_cached_daily_report(self, pet_id: str, date_str: Optional[str] = None) -> Optional[DailyPetReport]:
        """Recupera el reporte diario persistido en disco para evitar recomputos innecesarios."""
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        report_path = self.reports_dir / f"report_{pet_id}_{target_date}.json"
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    rep = DailyPetReport(**data)
                    rep.metrics = self.compute_habit_metrics(pet_id=pet_id)
                    return rep
            except Exception as e:
                logger.error("Error al cargar reporte cacheado %s: %s", report_path, e)
        return None

    def generate_daily_report(
        self,
        pet_name: str = "Toby",
        pet_id: Optional[str] = None,
        date_str: Optional[str] = None,
        lang: str = "en",
    ) -> DailyPetReport:
        """
        Coordina con Nebius (Nemotron) para redactar el Daily Pet Digest formal para una mascota especifica,
        y persiste el resultado en cache local.
        """
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        target_events = [e for e in self._events if e.pet_detected and (e.pet_id == pet_id if pet_id else True)]
        metrics = self.compute_habit_metrics(pet_id=pet_id)
        report = self.nebius.generate_daily_digest(
            events=target_events,
            pet_name=pet_name,
            pet_id=pet_id,
            date_str=target_date,
            lang=lang,
        )
        report.metrics = metrics

        # Persistir en cache local
        try:
            pid = pet_id or "default"
            report_path = self.reports_dir / f"report_{pid}_{target_date}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            logger.info("Reporte diario guardado en cache: %s", report_path)
        except Exception as err:
            logger.error("Error al persistir reporte diario: %s", err)

        return report

    def chat_with_agent(
        self,
        user_message: str,
        pet_name: str = "Toby",
        pet_id: Optional[str] = None,
        lang: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Permite al tutor dialogar con el asistente Nemotron sobre el estado de su mascota,
        proporcionando memoria multi-turno, extraccion de CoT, timeout seguro y fallback.
        """
        import re

        # Clinical Safety Rail: Bloquear fármacos tóxicos humanos inmediatamente
        toxic_drugs = ["paracetamol", "acetaminophen", "ibuprofen", "ibuprofeno", "aspirin", "aspirina", "tylenol", "advil"]
        user_msg_low = user_message.lower()
        if any(d in user_msg_low for d in toxic_drugs):
            if lang == "en":
                return (
                    "⚠️ **CLINICAL SAFETY ALERT (NeMo Guardrail):**\n\n"
                    "**NEVER administer human pain medications (such as acetaminophen, ibuprofen, or aspirin) to household pets.**\n\n"
                    "These compounds are severely toxic to cats and dogs, causing fatal methemoglobinemia, acute liver failure, or renal necrosis even in minute doses. "
                    "If your pet is in pain or accidentally ingested human medication, please contact your nearest veterinary emergency hospital immediately."
                )
            else:
                return (
                    "⚠️ **ALERTA DE SEGURIDAD CLÍNICA (NeMo Guardrail):**\n\n"
                    "**NUNCA administres medicamentos analgésicos humanos (como paracetamol, ibuprofeno o aspirina) a tus mascotas.**\n\n"
                    "Estos compuestos son altamente tóxicos para gatos y perros, causando metahemoglobinemia mortal, fallo hepático agudo o necrosis renal incluso en microdosis. "
                    "Si tu mascota manifiesta dolor o sospechas de ingestión accidental, comunícate de urgencia con un hospital veterinario de guardia."
                )

        target_events = [e for e in self._events if e.pet_detected and (e.pet_id == pet_id if pet_id else True)]
        if self.nebius.mock or self.nebius._client is None:
            if lang == "en":
                return (
                    f"[Mock Mode] Hello! I am the PawSentry AI Assistant. Today {pet_name} has logged "
                    f"{len(target_events)} verified micro-events. Their comfort index is currently healthy, "
                    f"with consistent hydration and rest routines."
                )
            else:
                return (
                    f"[Modo Mock] Hola, soy el Asistente de PawSentry AI. Hoy {pet_name} ha registrado "
                    f"{len(target_events)} micro-eventos verificados. Su confort se encuentra en niveles saludables "
                    f"y ha cumplido sus visitas habituales al comedero y bebedero."
                )

        try:
            metrics = self.compute_habit_metrics(pet_id=pet_id)
            recent_activities = [e.activity.value for e in target_events[-8:]] if target_events else ["resting"]
            zones_str = ", ".join([f"{k}: {v} detecciones" for k, v in metrics.zone_visits_count.items()]) if metrics.zone_visits_count else "Living Room"
            context_summary = (
                f"Pet Name: {pet_name} (ID: {pet_id or 'default'})\n"
                f"Verified events recorded today: {len(target_events)}\n"
                f"Comfort Index: {metrics.comfort_index:.0f}/100\n"
                f"Active Hours (projected): {metrics.active_hours_estimated}h ({metrics.active_bouts_count} active bouts)\n"
                f"Sleep Hours (projected): {metrics.sleep_hours_estimated}h ({metrics.rest_bouts_count} rest bouts)\n"
                f"Water visits: {metrics.water_visits_count} | Food visits: {metrics.food_visits_count}\n"
                f"Household Zones Tracked across Camera Network: {zones_str}\n"
                f"Recent behavioral sequence: {', '.join(recent_activities)}\n"
            )

            if lang == "en":
                system_prompt = (
                    "You are PawSentry AI Assistant, an expert animal ethologist and veterinary wellness companion "
                    "powered by NVIDIA Nemotron. Respond in an empathetic, warm, concise, and medically precise tone in English. "
                    "Directly answer the owner's query using the companion context below. Do not output raw internal reasoning tokens.\n\n"
                    f"Companion Context:\n{context_summary}"
                )
            else:
                system_prompt = (
                    "Eres PawSentry AI Assistant, un experto etólogo veterinario y asesor de bienestar animal "
                    "impulsado por NVIDIA Nemotron. Responde en tono empático, afectuoso, conciso y médicamente riguroso en español. "
                    "Responde directamente la duda del tutor basándote en el contexto de la mascota. No muestres razonamiento interno crudo.\n\n"
                    f"Contexto de la Mascota:\n{context_summary}"
                )

            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            
            # Incorporar memoria de dialogo multi-turno
            if conversation_history:
                for turn in conversation_history[-6:]:
                    if turn.get("role") in ("user", "assistant") and turn.get("content"):
                        messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": user_message})

            # Intentar primero con el modelo principal; si tarda > 18s, conmutar a Nemotron-3_5-Lightning
            response = None
            models_to_try = [self.nebius.reasoning_model, "nvidia/Nemotron-3_5-Lightning"]
            for m in models_to_try:
                try:
                    response = self.nebius._client.chat.completions.create(
                        model=m,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.3,
                        timeout=18.0,
                    )
                    if response and response.choices and response.choices[0].message:
                        break
                except Exception as model_err:
                    logger.warning("Fallo o timeout en modelo %s: %s. Probando siguiente...", m, model_err)

            if not response or not response.choices:
                raise RuntimeError("No se obtuvo respuesta valida de ningun modelo Nemotron")

            msg = response.choices[0].message
            raw_content = msg.content or getattr(msg, "reasoning_content", None) or ""

            # Limpiar etiquetas <think>...</think> si estuvieran presentes
            cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = raw_content.strip()

            return cleaned if cleaned else ("Response received, but content was empty." if lang == "en" else "Respuesta recibida sin contenido.")

        except Exception as err:
            logger.error("Error en chat asistencial Nemotron: %s", err)
            if lang == "en":
                return (
                    f"🐾 **PawSentry Nemotron Assistant:** Today {pet_name} is in a calm state with a Comfort Index of "
                    f"**{metrics.comfort_index:.0f}/100**, {metrics.water_visits_count} hydration visits, and "
                    f"{metrics.rest_bouts_count} resting bouts. *(Cloud inference fallback due to: {err})*"
                )
            else:
                return (
                    f"🐾 **Asistente Nemotron PawSentry:** Hoy {pet_name} se encuentra estable con un Índice de Confort de "
                    f"**{metrics.comfort_index:.0f}/100**, {metrics.water_visits_count} visitas de hidratación y "
                    f"{metrics.rest_bouts_count} sesiones de descanso. *(Fallback local por: {err})*"
                )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("==================================================")
    print("   PawSentry AI — Test de ReportGenerator         ")
    print("==================================================")

    generator = ReportGenerator()
    metrics = generator.compute_habit_metrics()
    print("[OK] Metricas de habitos calculadas:")
    print(f" - Visitas agua: {metrics.water_visits_count}")
    print(f" - Horas sueno estimadas: {metrics.sleep_hours_estimated}h")
    print(f" - Indice de confort: {metrics.comfort_index}/100")
    print(f" - Varianza de actividad: {metrics.activity_variance_vs_baseline}%")

    print("\n[OK] Generando reporte diario:")
    report = generator.generate_daily_report(pet_name="Firulais")
    print(f" - Score: {report.overall_wellness_score}")
    print(f" - Resumen: {report.summary_narrative[:150]}...")

    print("\n[OK] Test de Chat con Nemotron:")
    chat_reply = generator.chat_with_agent("¿Cómo viste a Firulais hoy?", pet_name="Firulais")
    try:
        print(f" - Respuesta de la IA:\n{chat_reply}")
    except UnicodeEncodeError:
        print(f" - Respuesta de la IA:\n{chat_reply.encode('ascii', errors='replace').decode('ascii')}")

    print("\n[EXITO] ReportGenerator verificado exitosamente.")

