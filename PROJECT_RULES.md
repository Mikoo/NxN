# 📐 PawSentry AI — Engineering Guidelines & Architecture Rules

Este documento establece las directrices arquitectónicas y estándares de calidad de software que rigen el desarrollo de **PawSentry AI** para el **Nebius x NVIDIA Global AI Hackathon (Track: Physical AI)**.

---

## 1. Tipado Estricto (Strict Type Hinting)
* **100% de cobertura de tipos**: Todas las funciones, métodos, firmas de clases y atributos de retorno deben incluir type hints estándar (	yping / Python 3.10+ types como list[str], dict[str, Any], Optional[Path]).
* No se permiten parámetros sin tipo (def process(data) está prohibido; usar def process(data: np.ndarray) -> DetectionResult:).
* Las estructuras complejas de datos deben modelarse mediante clases de datos o esquemas Pydantic.

## 2. Modularidad Desacoplada y Separación de Responsabilidades
* **Arquitectura en Capas**:
  * core/: Contiene el 100% de la lógica de dominio, captura de video, algoritmos de detección, llamadas a APIs externas (Nebius Token Factory, Tavily) y generación de reportes.
  * pp.py (UI): **Capa de presentación pura** (Streamlit). No debe contener transformaciones de imagen con OpenCV, llamadas directas HTTP a Nebius ni prompts de IA incrustados. La UI consume exclusivamente servicios e interfaces expuestas por core/.
  * data/: Almacenamiento local para snapshots, logs y base de datos de eventos (data/snapshots/, data/events.json).
* Las dependencias deben inyectarse mediante constructores para facilitar testing unitario y mocking.

## 3. Patrón Mock / Fallback Obligatorio
* Todos los clientes de servicios externos (NebiusClient, TavilyClient, etc.) deben soportar de forma nativa el modo simulación mediante:
  1. Parámetro explícito en el constructor: mock: bool = False.
  2. Variable de entorno global: USE_MOCKS=true en .env.
* **Objetivo**: Permitir ejecutar tests de integración, validar la UI y realizar demos completas en local sin agotar la cuota de créditos ni depender de latencia o conectividad externa.
* Los mocks deben retornar instancias válidas de los modelos Pydantic con datos realistas y coherentes.

## 4. Validación de Esquemas con Pydantic v2
* Ningún componente debe intercambiar diccionarios genéricos no tipados para resultados de IA.
* Toda inferencia visual y síntesis médica debe serializarse y deserializarse a través de modelos Pydantic v2 (ej. PetDetectionEvent, BehaviorAnalysis, DailyReport, VetConsultationResult).
* Los modelos deben incluir validaciones de rango (ej. puntuación de bienestar entre 0 y 100) y descripciones en cada campo (Field(...)).

## 5. Manejo de Excepciones y Logging Estructurado
* **Prohibido el uso de print() en módulos de core/**: Se debe emplear el módulo logging de Python configurado con formato uniforme:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
* Todas las operaciones de I/O (cámara, lectura de archivos, llamadas de red) deben estar protegidas con bloques 	ry...except específicos, registrando trazas con logger.error(..., exc_info=True) y lanzando excepciones de dominio cuando sea necesario.
* Fallos en la cámara o en la red no deben crashear la aplicación: se deben degradar graciosamente informando al usuario.

---
*Directrices obligatorias para el ciclo de vida del proyecto PawSentry AI.*
