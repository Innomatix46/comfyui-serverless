# Repository Guidelines

## Project Structure & Module Organization
- `src/api/`: FastAPI app (`main.py`, `middleware.py`, `routers/`).
- `src/services/`: Business logic (workflow, auth, storage, model, webhook).
- `src/models/`: SQLAlchemy models and Pydantic schemas.
- `src/config/`, `src/core/`, `src/database/`: settings, DB session, Alembic.
- `tests/`: `unit/`, `integration/`, `load/` with fixtures and config.
- `scripts/`: setup, deploy, helpers. `docker/`, `docs/` for infra and docs.

## Build, Test, and Development Commands
- Setup: `chmod +x scripts/setup.sh && ./scripts/setup.sh`
- Run API (dev): `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
- Celery worker: `celery -A src.services.workflow.celery_app worker --loglevel=info`
- Tests (all): `pytest --cov=src` (coverage threshold 80%).
- Focused tests: `pytest tests/unit -m "unit and not slow"`.
- Docker (local stack): `docker-compose -f docker-compose.test.yml up -d`
- Build test image: `docker build -t comfyui-serverless:test -f docker/Dockerfile.test .`

## Coding Style & Naming Conventions
- Python: PEP 8 with type hints; 100-char lines.
- Format: `black src tests`; Imports: `isort src tests` (profile=black).
- Lint: `flake8 src tests --max-line-length=100`; Types: `mypy src`.
- Security lint: `bandit -r src/`; Dependencies audit: `safety check`.
- Names: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio`); markers: `unit`, `integration`, `api`, `performance`, `slow`, etc.
- Layout: place unit tests in `tests/unit/test_*.py`; integration in `tests/integration/`.
- Coverage: keep `--cov=src --cov-branch` ≥ 80%; add tests for new/changed code.
- Use fixtures in `tests/fixtures/`; mock external systems; avoid real S3/Redis unless marked.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat: add webhook retries`, `fix: prevent redis leak`).
- Branches: `feature/<scope>`, `fix/<scope>`, `chore/<scope>`.
- PRs: clear description, linked issue, test plan, screenshots/logs for API changes, and updated docs.
- CI must pass (lint, type check, tests, security scans); maintain or improve coverage.

## Security & Configuration Tips
- Start from `.env.example` → `.env`; never commit secrets. Set `SECRET_KEY`, DB/Redis URLs, S3 keys.
- Use HTTPS in production; restrict CORS; rotate credentials.
- Verify Postgres/Redis availability locally; run Alembic from `src/database` when schemas change.
