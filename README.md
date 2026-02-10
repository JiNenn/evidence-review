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
  - 同一入力の別Runで issue-evidence 選定シグネチャが再現することも検証

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
  - `GET /runs/{run_id}/stages` で stageの成否・attempt・failure_type・failure_detail・errorを確認可能
  - `GET /runs/{run_id}/issues?include_hidden=true` は監査用途（AUTH有効時は監査権限のみ）
  - `GET /runs/{run_id}/audit/issues` は監査UI用の専用導線（hidden含む）
  - `POST /auth/token`（JWT発行）
  - `POST /runs/{run_id}/sources/presign-put` は `source_doc_id/object_key` を返し、この時点で `source_docs` が作成される
  - `POST /runs/{run_id}/sources/ingest` は `source_doc_id` 主導で実行される
- Worker tasks:
  - ingest, low draft, search, feedback(with citation), apply feedback, high final
  - detect_change_spans, derive_issues_from_changes, attach_evidence_to_issues, full pipeline
- Citation detail:
  - `GET /citations/{citation_id}` で根拠スニペットと原文URLを取得
- Evidence-first rule:
  - Issue is displayed only when evidence(s) exist
  - `status=hidden` の issue は API/UI の一覧表示対象外
  - evidence は `citation -> source_doc_id/chunk_id/loc` まで辿れる（`span` は任意）
  - `GET /issues/{issue_id}` では `selection` を明示返却し、互換のため `citation.span.selection` も当面維持する
  - `attach_evidence_to_issues` は `SearchResult.score` 重み付き（0.7）+ lexical一致度（0.3）で根拠候補を選定し、`citation.span.selection` に選定理由を残す
  - evidence不足時は `search_chunks` を固定戦略で再実行（strict -> relaxed -> vector fallback, 最大3回）してから `blocked_evidence` 判定

## Notes

- LLM integration is currently stubbed for local validation.
- `ingest_source_doc` は MinIOから実データを取得し、text/json/html/pdf を抽出してチャンク化します。
- Browserから直接S3へPUT/GETするため `S3_PUBLIC_ENDPOINT` を利用します（compose既定: `http://localhost:9000`）。
- 認証は `AUTH_ENABLED` で有効化し、Bearer JWTでAPIを保護します。
- Frontend は `Run作成 -> Sourceアップロード/取り込み -> Pipeline実行 -> Stages/Issues確認` の流れを1画面で実行できます。
- Frontend 主画面は再実行操作を持たず、復旧操作は `http://localhost:3000/debug` の Run単位導線で行います。
- `run.status` は `success / success_partial / blocked_evidence / failed_system / failed_legacy` を区別します。
- `attempt` は同一 `idempotency_key` に対する再試行回数です。
- presignのみで放置された orphan `source_docs` はTTL超過時に自動クリーンアップされます（`SOURCE_DOC_ORPHAN_TTL_HOURS`）。

## Acceptance smoke

```bash
python scripts/acceptance_smoke.py
```
