# diffUI v0.2 scaffold

`SPEC.md` に基づいた Self-Host First の最小実装です。  
Compose を正本として、以下のサービスを起動します。

- `frontend` (Next.js)
- `api` (FastAPI)
- `worker` (Celery)
- `postgres`
- `redis`
- `minio`
- `migrate` (Alembic one-shot)

## Quickstart

```bash
docker compose up --build
```

## CI

- GitHub Actions: `.github/workflows/ci.yml`
- `quality`:
  - backend compile check
  - frontend build
- `acceptance`:
  - `AUTH_ENABLED=false` と `AUTH_ENABLED=true` の両方で
  - `scripts/acceptance_smoke.py` を実行

## Endpoints

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`
- MinIO Console: `http://localhost:9001`
- Health: `GET /healthz`, `GET /readyz`

## Implemented scope

- Core tables and constraints from `SPEC.md`:
  - `runs`, `artifacts`, `feedback_items`, `citations`
  - `issues`, `issue_evidences`
  - `source_docs`, `source_chunks`
  - `run_stages` (idempotency)
  - `search_requests`, `search_results`
- API endpoints:
  - Runs, Issues, Source presign/ingest, Pipeline kickoff, Search, Health
  - `GET /runs/{run_id}/stages` で stageの成否・attempt・errorを確認可能
  - `POST /auth/token`（JWT発行）
- Worker tasks:
  - ingest, low draft, search, feedback(with citation), apply feedback, high final
  - detect_change_spans, derive_issues_from_changes, attach_evidence_to_issues, full pipeline
- Citation detail:
  - `GET /citations/{citation_id}` で根拠スニペットと原文URLを取得
- Evidence-first rule:
  - Issue is displayed only when evidence(s) exist

## Notes

- LLM integration is currently stubbed for local validation.
- `ingest_source_doc` は MinIOから実データを取得し、text/json/html/pdf を抽出してチャンク化します。
- Browserから直接S3へPUT/GETするため `S3_PUBLIC_ENDPOINT` を利用します（compose既定: `http://localhost:9000`）。
- 認証は `AUTH_ENABLED` で有効化し、Bearer JWTでAPIを保護します。

## Acceptance smoke

```bash
python scripts/acceptance_smoke.py
```
