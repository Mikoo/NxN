# 🐾 PawSentry AI — Resumen Ejecutivo y Estado del Proyecto

> **"An edge multimodal AI agent for real-time pet behavioral monitoring and wellness analytics powered by NVIDIA Nemotron on Nebius open infrastructure."**

---

## 📌 1. Información General del Evento
* **Evento:** [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Devpost)
* **Premios en juego:** +,000 USD en efectivo + dispositivos NVIDIA Jetson Orin Nano + ,000 USD Premio "Best Use of Tavily".
* **Track Principal:** **Physical AI Track** *(Sensing y agentes en el mundo real, cámaras/IoT, inferencia en el borde y nube)*.
* **Track Secundario / Alternativo:** **Best Apps and Agents Track** & **Personal AI**.
* **Estado de Inscripción:** ✅ **Inscrito con éxito** en Devpost y en el programa **Nebius Builders**.
* **Créditos conseguidos:**  USD en créditos para Nebius Token Factory +  USD en Tavily API.

---

## 🎯 2. ¿De qué trata nuestro Proyecto? (Propuesta de Valor)

### El Problema
Millones de personas dejan a sus mascotas solas durante el día y se preocupan por su bienestar. Los animales no pueden comunicarse verbalmente y por instinto evolutivo tienden a enmascarar o disimular el dolor, el estrés y la enfermedad hasta que la condición se vuelve crítica. Las cámaras de seguridad convencionales son pasivas: graban horas de video inútil que nadie tiene tiempo de revisar y no entienden lo que ocurre.

### Nuestra Solución: PawSentry AI
**PawSentry AI** es un agente inteligente físico y multimodal que convierte cualquier cámara cotidiana (webcam, cámara USB o IP) en un **asistente de salud y diario interactivo para tu mascota**:

1. **Monitoreo Edge Inteligente:** No envía video continuo a la nube (lo cual sería costoso e ineficiente). Detecta movimiento y presencia localmente, extrayendo solo fotogramas o micro-clips cuando ocurre un evento de interés (alimentación, hidratación, descanso, agitación, rascado, maullidos/ladridos en la puerta).
2. **Razonamiento Multimodal con NVIDIA Nemotron en Nebius:** Envía las escenas clave a **Nebius Token Factory**, donde modelos multimodales de NVIDIA interpretan el lenguaje corporal, postura, nivel de energía y conducta del animal.
3. **Validación Veterinaria con Tavily Search:** Si se detecta un patrón anómalo (ejemplo: letargo inusual, rascado compulsivo, caída del 40% en consumo de agua), el agente investiga automáticamente en fuentes veterinarias confiables usando **Tavily API**.
4. **Pet Daily Digest & Álbum de Momentos:** Al final del día (o a demanda), genera un resumen con:
   - Puntuación de bienestar y nivel de actividad.
   - Bitácora horaria de hábitos.
   - Las mejores fotos/momentos destacados del día con descripciones empáticas.
   - Chat interactivo: *"¿Cómo viste a Firulais hoy? ¿Comió bien?"*.

---

## 🏗️ 3. Arquitectura y Stack Tecnológico

`
 [ Cámara Web / USB / IP ]
           │
           ▼
 [ Detección Local / Edge (Python + OpenCV) ]
    * Filtro de movimiento y cambios de escena
    * Captura de fotogramas nítidos
           │
           ▼
 [ Inferencia en la Nube (Nebius Token Factory) ]
    * NVIDIA Multimodal / Vision (Llama-3.2-Vision / Nemotron Vision)
    * Análisis postural, conductual y emocional estructurado (JSON)
           │
           ├──► [ Consulta Veterinaria / Síntomas: Tavily Search API ]
           │
           ▼
 [ Motor de Síntesis y Razonamiento ]
    * NVIDIA Nemotron 70B / 3 Ultra (Vía Nebius API)
    * Correlación de eventos diarios y redacción del "Daily Digest"
           │
           ▼
 [ Interfaz de Usuario: Streamlit Dashboard ]
    * Monitor en vivo y timeline de eventos
    * Métricas de salud (sueño, actividad, ingesta)
    * Galería de momentos destacados y Chat Asistente
`

---

## ✅ 4. Todo lo que hemos realizado hasta ahora

1. **Registro y Setup en el Hackathon:**
   - Creación del proyecto en Devpost bajo el nombre **PawSentry AI**.
   - Registro en **Nebius Builders Program** y obtención de créditos de inferencia y Tavily.
2. **Identidad Visual / Branding:**
   - Generación del **Thumbnail oficial en ratio 3:2** (pawsentry_thumbnail.jpg) con estética Physical AI / HUD holográfico.
   - Subida y configuración del thumbnail en la plataforma Devpost.
3. **Entorno de Desarrollo Local:**
   - Verificación de Python 3.13 en el sistema.
   - Creación del entorno virtual aislado .venv.
   - Instalación completa de librerías (opencv-python, openai, streamlit, 	avily-python, pydantic, python-dotenv).
4. **Configuración y Seguridad:**
   - Creación de .env.example con las variables para Nebius Token Factory, NVIDIA models y Tavily.
5. **Validación de Hardware / Cámaras:**
   - Desarrollo y ejecución de camera_test.py.
   - Detección exitosa de **2 cámaras operativas** en Windows con backend DirectShow y aceleración NVIDIA.
   - Captura y verificación de fotogramas de prueba (	est_camera_0.jpg y 	est_camera_1.jpg).

---

## 🚀 5. Roadmap de Próximos Pasos

* [ ] **Módulo Edge (core/detector.py):** Lógica para detectar presencia de la mascota y guardar snapshots solo cuando ocurre actividad relevante.
* [ ] **Cliente Nebius (core/nebius_client.py):** Integración con la API de Nebius Token Factory para enviar fotos y recibir respuestas estructuradas con modelos de NVIDIA (con soporte mock/fallback para desarrollo offline).
* [ ] **Motor de Reportes y Tavily (core/report_generator.py):** Algoritmo que procesa el historial del día, consulta a Tavily ante dudas médicas y redacta el resumen de salud.
* [ ] **Dashboard Web (pp.py):** Aplicación interactiva en Streamlit para visualizar la cámara, el timeline con fotos, gráficos de actividad y el chat.
* [ ] **Preparación para la entrega:** Grabación del video demo de 3 minutos (mostrando la cámara apuntando a la mascota/escena + funcionamiento de la app) y repositorio con licencia Open Source (MIT/Apache 2.0).

---
*Archivo generado automáticamente para el equipo de desarrollo de PawSentry AI.*
