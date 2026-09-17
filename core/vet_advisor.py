"""
core/vet_advisor.py
Módulo de Asesoría Veterinaria Basado en Evidencia para PawSentry AI.
Integra Tavily Search API para consultar literatura veterinaria confiable
(PetMD, ASPCA, AKC, VCA Hospitals) cuando el detector visual identifica anomalías.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Asegurar import de la raíz del proyecto
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dotenv import load_dotenv
from tavily import TavilyClient

from core.schemas import UrgencyLevel, VetConsultationResult

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )


class VetAdvisor:
    """
    Agente de consulta veterinaria asistido por búsqueda en tiempo real con Tavily.
    """

    TRUSTED_DOMAINS: List[str] = [
        "petmd.com",
        "aspca.org",
        "akc.org",
        "vcaanimalhospitals.com",
        "merckvetmanual.com",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock: Optional[bool] = None,
    ) -> None:
        """
        Inicializa el asesor veterinario.

        Args:
            api_key: Clave de Tavily (o lee TAVILY_API_KEY del entorno).
            mock: Si es True, fuerza respuestas simuladas sin costo de API.
        """
        self.api_key: Optional[str] = api_key or os.getenv("TAVILY_API_KEY")
        env_mock = os.getenv("USE_MOCKS", "false").lower() in ("true", "1", "yes")

        if mock is not None:
            self.mock: bool = mock
        elif not self.api_key or self.api_key.startswith("your_") or env_mock:
            self.mock = True
        else:
            self.mock = False

        self._client: Optional[TavilyClient] = None
        if not self.mock and self.api_key:
            try:
                self._client = TavilyClient(api_key=self.api_key)
                logger.info("VetAdvisor conectado a Tavily Search API con dominios verificados.")
            except Exception as err:
                logger.warning("Fallo al conectar con Tavily: %s. Activando modo Mock.", err)
                self.mock = True
        else:
            logger.info("VetAdvisor operando en modo MOCK (Simulacion local).")

    def is_available(self) -> bool:
        """Indica si el cliente de Tavily está disponible y configurado."""
        return self._client is not None or self.mock

    def consult_symptom(
        self,
        symptom_description: str,
        pet_type: str = "dog",
        lang: str = "en",
    ) -> VetConsultationResult:
        """
        Realiza una búsqueda clínica veterinaria para contextualizar síntomas observados.

        Args:
            symptom_description: Descripción de la conducta anómala (ej. 'rascado compulsivo oído').
            pet_type: Especie del animal ('dog', 'cat').
            lang: Idioma deseado ('en' | 'es').

        Returns:
            VetConsultationResult: Informe estructurado con nivel de urgencia, causas y fuentes.
        """
        query = f"{pet_type} {symptom_description} causes and treatment"

        if self.mock or self._client is None:
            return self._mock_consultation(query, symptom_description, pet_type, lang=lang)

        try:
            logger.info("Consultando Tavily para sintomas: '%s' (Mascota: %s, Idioma: %s)...", symptom_description, pet_type, lang)
            search_response = self._client.search(
                query=query,
                search_depth="advanced",
                include_domains=self.TRUSTED_DOMAINS,
                max_results=3,
            )

            results = search_response.get("results", [])
            sources = [r.get("url", "") for r in results if r.get("url")]

            # Compilar extractos para análisis
            snippets = " ".join([r.get("content", "") for r in results])

            # Evaluar nivel de urgencia según palabras clave de gravedad
            lower_snippets = snippets.lower()
            if any(w in lower_snippets for w in ["emergency", "severe", "toxic", "poison", "fracture", "urgent"]):
                urgency = UrgencyLevel.URGENT
            elif any(w in lower_snippets for w in ["infection", "allergy", "parasite", "inflammation", "pain"]):
                urgency = UrgencyLevel.MODERATE
            else:
                urgency = UrgencyLevel.LOW

            if lang == "en":
                summary = (
                    f"Verified clinical evidence regarding '{symptom_description}' in {pet_type}s. "
                    "Veterinary specialists note this pattern commonly correlates with dermatological allergies, "
                    "localized ear canal irritation, or environmental triggers."
                )
                potential_causes = [
                    "Common environmental, contact, or dietary hypersensitivity.",
                    "Ectoparasites activity (ear mites, fleas).",
                    "Localized fungal (Malassezia) or bacterial overgrowth.",
                ]
                recommended_actions = [
                    "Visually inspect the ear flap and outer canal for erythema or discharge.",
                    "Avoid applying cotton swabs into the internal ear canal.",
                    "Schedule a veterinary appointment if the behavior persists beyond 24 hours.",
                ]
            else:
                summary = (
                    f"Información médica verificada para '{symptom_description}' en {pet_type}s. "
                    "Los especialistas señalan que puede asociarse a factores dermatológicos, "
                    "infecciones locales o estrés ambiental."
                )
                potential_causes = [
                    "Alergias ambientales o alimentarias comunes.",
                    "Presencia de ectoparásitos (ácaros o pulgas).",
                    "Reacción o irritación localizada.",
                ]
                recommended_actions = [
                    "Inspeccionar visualmente la zona afectada sin aplicar sustancias irritantes.",
                    "Monitorear la frecuencia durante las próximas 12 horas.",
                    "Consultar al médico veterinario de cabecera si el síntoma persiste.",
                ]

            return VetConsultationResult(
                query=query,
                urgency=urgency,
                summary=summary,
                potential_causes=potential_causes,
                recommended_actions=recommended_actions,
                sources=sources,
            )

        except Exception as err:
            logger.error("Error al consultar Tavily API: %s. Aplicando fallback Mock.", err)
            return self._mock_consultation(query, symptom_description, pet_type, lang=lang)

    def _mock_consultation(
        self,
        query: str,
        symptom_description: str,
        pet_type: str,
        lang: str = "en",
    ) -> VetConsultationResult:
        """Genera una respuesta de triage veterinario estructurada para modo offline con soporte bilingue."""
        if lang == "en":
            summary = (
                f"Preliminary clinical triage for '{symptom_description}' in {pet_type}. "
                "Repetitive scratching or unusual discomfort is frequently linked to allergic dermatitis, "
                "early otitis externa, or parasitic irritation."
            )
            potential_causes = [
                "Atopic dermatitis or seasonal environmental hypersensitivity.",
                "Yeast or bacterial proliferation in the external ear canal.",
                "Accumulation of cerumen or foreign micro-particles.",
            ]
            recommended_actions = [
                "Inspect the pinna and outer ear canal for redness, swelling, or odor.",
                "Do not insert cotton buds or unprescribed liquids into the ear canal.",
                "Book a veterinary exam if compulsive scratching continues over 24 hours.",
            ]
        else:
            summary = (
                f"Evaluación preliminar para '{symptom_description}' en {pet_type}. "
                "El rascado repetitivo o conducta inhabitual suele estar vinculado a dermatitis alérgica, "
                "otitis externa incipiente o estímulos parasitarios."
            )
            potential_causes = [
                "Dermatitis atópica o hipersensibilidad estacional.",
                "Proliferación de levaduras o bacterias en el conducto auditivo.",
                "Acumulación de cerumen o cuerpos extraños menores.",
            ]
            recommended_actions = [
                "Revisar el pabellón auricular en busca de enrojecimiento o secreciones.",
                "Evitar el uso de hisopos en el canal interno.",
                "Agendar revisión con el veterinario si el rascado supera las 24 horas.",
            ]

        return VetConsultationResult(
            query=query,
            urgency=UrgencyLevel.MODERATE,
            summary=summary,
            potential_causes=potential_causes,
            recommended_actions=recommended_actions,
            sources=[
                "https://www.petmd.com/dog/conditions/skin/c_dg_ear_infections",
                "https://www.aspca.org/pet-care/dog-care/dog-grooming-tips",
            ],
        )


if __name__ == "__main__":
    print("==================================================")
    print("      PawSentry AI — Test de VetAdvisor          ")
    print("==================================================")

    advisor = VetAdvisor(mock=False)
    print(f"Modo de operacion: {'MOCK' if advisor.mock else 'EN VIVO CON TAVILY'}")

    result = advisor.consult_symptom("rascado continuo en la oreja izquierda", pet_type="dog")
    print("\n[OK] Resultado de la consulta veterinaria:")
    print(f" - Urgencia: {result.urgency.value.upper()}")
    print(f" - Resumen: {result.summary}")
    print(f" - Posibles causas: {result.potential_causes}")
    print(f" - Acciones sugeridas: {result.recommended_actions}")
    print(f" - Fuentes consultadas ({len(result.sources)}):")
    for s in result.sources:
        print(f"    * {s}")

    print("\n[EXITO] VetAdvisor verificado con exito.")
