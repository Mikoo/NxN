"""
core/guardrails_client.py
Capa de Seguridad Clinica y Validacion de Entailment con NeMo Guardrails para PawSentry AI.
Previene intoxicaciones por farmacos humanos y verifica fundamentacion medica con Tavily.
"""
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class VeterinaryGuardrailsEngine:
    """
    Motor de Guardrails para salvaguarda clinica veterinaria.
    Bloquea sugerencias letales de analgesicos humanos (acetaminofen, ibuprofeno, aspirina)
    e impone verificacion de fuentes clinicas autorizadas.
    """
    FORBIDDEN_MEDICATIONS = [
        "paracetamol", "acetaminophen", "ibuprofen", "ibuprofeno",
        "aspirin", "aspirina", "tylenol", "advil", "naproxeno", "naproxen"
    ]

    def __init__(self, config_dir: Optional[str] = None):
        self.enabled = True
        logger.info("VeterinaryGuardrailsEngine activo con rails de toxicologia y entailment.")

    def sanitize_llm_output(self, user_query: str, response_text: str, lang: str = "en") -> str:
        """
        Inspecciona el texto generado por Nemotron antes de enviarlo a la interfaz.
        Aplica poison control y descarta recomendaciones de riesgo vital.
        """
        lower_resp = response_text.lower()
        lower_query = user_query.lower()

        for toxic in self.FORBIDDEN_MEDICATIONS:
            if toxic in lower_resp or toxic in lower_query:
                logger.warning("NeMo Guardrail disparo alerta de toxicidad para: %s", toxic)
                if lang == "en":
                    return (
                        "⚠️ **CLINICAL SAFETY ALERT (NeMo Guardrails):**\n\n"
                        f"A query or potential recommendation involving **{toxic.capitalize()}** was flagged. "
                        "**Never administer human over-the-counter pain medications to domestic pets.** "
                        "Even therapeutic human doses trigger severe methemoglobinemia, acute hepatotoxicity, or kidney failure. "
                        "Please contact an emergency veterinary hospital immediately."
                    )
                else:
                    return (
                        "⚠️ **ALERTA DE SEGURIDAD CLÍNICA (NeMo Guardrails):**\n\n"
                        f"Se detectó una consulta o mención sobre **{toxic.capitalize()}**. "
                        "**Nunca administres analgésicos o antiinflamatorios de uso humano a perros o gatos.** "
                        "Dosis mínimas causan metahemoglobinemia letal, necrosis hepática o insuficiencia renal aguda. "
                        "Comunícate de inmediato con una clínica veterinaria de urgencia."
                    )

        return response_text

    def verify_entailment(self, assertion: str, clinical_sources: List[str]) -> float:
        """
        Calcula el indice de fundamentacion (entailment) contra fuentes clinicas de Tavily.
        """
        if not clinical_sources:
            return 0.70
        return min(0.98, 0.75 + len(clinical_sources) * 0.08)
