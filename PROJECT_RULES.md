# PawSentry AI — Agent Instructions &amp; Code Standards

## Tech Stack

- Python 3.13 + Virtual Environment (.venv)

- OpenCV (cv2) for edge camera processing

- Nebius Token Factory (OpenAI-compatible client) + NVIDIA models

- Tavily Python SDK for veterinary retrieval

- Streamlit for dashboard UI

- Pydantic v2 for data schemas and validation

## Architecture Rules

1. All core logic belongs in `core/`. Do not put business or ML logic inside Streamlit UI files.

2. Every external service `nebius_[client.py](http://client.py)`, `tavily_[client.py](http://client.py)`) MUST support a `mock=True` flag for offline testing.

3. Type hints `typing`) are mandatory on all function definitions.

4. Use Pydantic models for any structured LLM response.

## Common Commands

- Activate venv: `.\.venv\Scripts\Activate.ps1`

- Run camera test: `python camera_[test.py](http://test.py)`

- Run UI: `streamlit run [app.py](http://app.py)`

- Run linting: `ruff check .` (or `flake8`)