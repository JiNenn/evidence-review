# B'アーキテクチャ設計書（Self-Host First / 根拠付きFB比較UX）v0.1

## 0. 文書の目的

本書は、以下のUXを成立させるための **Self-Host First** 前提のアーキテクチャ（B'案）を定義する。

* 高性能モデルが生成した「完成版」
* 低性能モデルのアウトプットに対して、根拠（ソース）に基づくFBで改善された「改善版」
* それらの差分から **レビュー論点（Issue）** を抽出し、優先度付きで提示する
* 論点は参照したソースに紐づき、**ワンクリックで根拠（該当スニペット/原文）を閲覧可能**

また、本プロジェクトの配布形態として「ローカル完結型OSS配布」と「将来的なオンプレ/K8s展開」を前提とし、`docker compose up` を正本とする。

---

## 1. ゴール / 非ゴール

### 1.1 ゴール

* **Self-host**：ユーザーの手元1台で再現可能（composeが正本）
* **Stateless**：Web/API/Workerはステートレス（永続はDB/S3互換へ）
* **Out-of-Process**：重い処理は必ずWorkerで非同期実行
* **Idempotent**：ジョブは冪等（重複実行で不整合を起こさない）
* **Evidence-first**：論点（Issue）は根拠（Citation）なしではUIに出さない（品質の骨格）
* **Issue UX**：3者比較の差分を内部で処理し、ユーザーには論点中心で提示する

### 1.2 非ゴール（v0.1時点）

* リアルタイム共同編集
* 高度な権限管理（組織・チーム等）
* GPU必須の推論基盤（オンプレでのGPU最適化は将来）
* 初期からの分散トレーシング/メトリクスフルセット（最低限は実装）

---

## 2. 前提：Self-Host First Development Standard (v2.1) 反映

* Docker ComposeをSingle Source of Truth
* `latest`禁止（イメージは固定タグ）
* 設定はすべてENV（`.env.example`必須）
* SQLite等のファイルDBは避け、PostgreSQLを標準
* バイナリはS3互換（MinIO）に保存（ローカルパス直書き禁止）
* 重い処理はWorkerで実行
* すべての非同期ジョブは冪等
* ログはstdout/stderr（構造化JSON推奨）
* `/healthz` `/readyz` を実装

---

## 3. 全体アーキテクチャ（B'案）

### 3.1 サービス構成（docker compose単位）

* **frontend**：Next.js（比較UI、アップロードUI、閲覧UI）
* **api**：FastAPI（Run管理、ジョブ投入、署名付きURL発行、閲覧API、認可）
* **worker**：Celery（推論、RAG、抽出、探索、差分生成、評価）
* **redis**：Celery broker / キャッシュ（将来：rate-limit等）
* **postgres**：正本DB（Run/Artifact/Feedback/Citation/SourceChunk 等）
* **minio**：S3互換オブジェクトストレージ（PDF/スナップショット/成果物ファイル）
* **migrate**：Alembic（one-shot マイグレーション）

### 3.2 役割分担の原則

* **frontend**：描画と操作のみ。LLMや探索の実処理は行わない。
* **api**：短時間で返せる処理のみ（認可/DB読み書き/ジョブ投入/署名付きURL）。
* **worker**：重い処理の唯一の実行主体（抽出、探索、生成、比較、評価）。
* **永続はDB/S3へ**：コンテナFSに依存しない。

### 3.3 ネットワーク境界

* frontend ⇄ api（HTTP）
* api ⇄ postgres/redis/minio（内部ネットワーク）
* worker ⇄ postgres/redis/minio（内部ネットワーク）
* browser ⇄ minio（presigned URL経由で直PUT/GET）

---

## 4. 主要ユースケースとフロー

### 4.1 ソース取り込み（Evidence基盤の構築）

1. frontend → api：アップロード準備要求（ファイル名/Content-Type等）
2. api：`source_docs` を先に作成し、`source_doc_id` を発行
3. api → frontend：MinIOの **presigned PUT URL** と `source_doc_id` / `object_key` を返す
4. browser → minio：ファイルを直接アップロード
5. frontend → api：`source_doc_id` 指定で ingest 要求
6. api → worker：`ingest_source_doc(run_id, source_doc_id)` を投入
7. worker：抽出→チャンク化→（必要に応じて埋め込み）→DBへ保存

**補足**

* `object_key` は `runs/{run_id}/sources/{source_doc_id}/raw/{filename}` を必須規約とする
* ingest は `source_doc_id` を主キーとして扱い、`object_key` は `source_docs` から参照する

### 4.2 生成パイプライン（Issue抽出主導）

* `generate_low_draft(run_id)`
* `generate_feedback_with_citations(run_id, target=low_draft)`（改善版生成の補助）
* `apply_feedback(run_id, base=low_draft) -> improved`
* `generate_high_final(run_id)`
* `detect_change_spans(run_id, low_draft, improved, high_final)`
* `derive_issues_from_changes(run_id)`（論点化・要約・優先度付け）
* `attach_evidence_to_issues(run_id)`（根拠付与）

**補足**

* `diff artifact` はデバッグ用途で保持するが、UXの主導オブジェクトは `issues` とする。

### 4.3 探索（glob/grep/snippet化）

* 原則：**Worker実装**
* データソース：基本は **取り込み済みSourceChunkのDB検索**
* 例外：MinIOオブジェクト一覧（疑似glob）はWorkerがS3 Listで実現
* 検索の入出力を永続化し、FBや再現性に利用する

---

## 5. データモデル（v0.1）

※詳細DDLは別章で拡張する（ここでは概念と主要キー/制約を定義）。

### 5.1 Run（実行セッション）

* `runs`

  * `id (uuid)`
  * `created_at`
  * `task_prompt (text)`
  * `status (enum: pending, running, success, success_partial, blocked_evidence, failed_system, failed_legacy)`
  * `metadata (jsonb)`

**運用規約**

* 旧値 `failed` は移行時に `failed_legacy` へ正規化し、APIは `failed` を返さない
* 失敗詳細は `run_stages.failure_type / failure_detail` を正とする

### 5.2 Artifact（成果物）

* `artifacts`

  * `id (uuid)`
  * `run_id (fk)`
  * `kind (enum: low_draft, improved, high_final, diff, ...)`
  * `format (enum: markdown, json, ...)`
  * `content_text (text, nullable)` ※小さい場合
  * `content_object_key (text, nullable)` ※大きい場合はMinIO
  * `version (int)`
  * `created_at`

### 5.3 Issue / IssueEvidence（根拠付き論点）

* `issues`

  * `id (uuid)`
  * `run_id (fk)`
  * `fingerprint (text)` ※再現性のための安定キー
  * `title (text)`
  * `summary (text)`
  * `severity (int)`
  * `confidence (float)`
  * `status (enum: open, resolved, hidden, ...)`
  * `metadata (jsonb)`
  * `created_at` / `updated_at`

* `issue_evidences`

  * `id (uuid)`
  * `issue_id (fk)`
  * `before_excerpt (text, nullable)`
  * `after_excerpt (text, nullable)`
  * `citation_id (fk)` ※原文到達可能な根拠
  * `loc (jsonb, optional)` ※キャッシュ用loc（参照元の正本は `source_chunks.loc`）

* `citations`（共通根拠）

  * `id (uuid)`
  * `feedback_id (fk, nullable)` ※従来FBとの互換のためnullable
  * `source_doc_id (fk)`
  * `chunk_id (fk)`
  * `span (jsonb, optional)` ※ハイライト情報。欠けても `loc` があれば可

**制約（重要）**

* `issues` は **issue_evidences が1件以上** あるものだけを UI表示対象とする。
* `fingerprint` は同趣旨Issueの安定キーとして、最低限 `change_type + phase + 正規化before/after要約` を入力に含める
  （Evidence解決結果 `source_doc_id/chunk_id` は含めない）
* loc解決の優先順位は `source_chunks.loc` を正本とし、`issue_evidences.loc` はキャッシュとして扱う

### 5.4 SourceDoc / SourceChunk（ソースとチャンク）

* `source_docs`

  * `id (uuid)`
  * `run_id (fk)`
  * `title`
  * `origin (enum: upload, url, ... )`
  * `object_key (text)` ※MinIO
  * `content_type`
  * `created_at`

* `source_chunks`

  * `id (uuid)`
  * `source_doc_id (fk)`
  * `chunk_index (int)`
  * `text (text)`
  * `loc (jsonb)` ※ page / offset / line 等
  * `embedding (vector/nullable)` ※将来pgvector等

### 5.5 RunStage（冪等・リトライ前提の実行管理）

* `run_stages`

  * `id (uuid)`
  * `run_id (fk)`
  * `stage_name (text)`
  * `idempotency_key (text)`
  * `status (enum: pending, running, success, failed)`
  * `failure_type (enum: system_error, evidence_insufficient, validation_error, nullable)`
  * `failure_detail (jsonb, nullable)` ※関連IDや再試行判断用の情報
  * `attempt (int)`
  * `started_at` / `finished_at`
  * `output_ref (jsonb)` ※作成artifact_id等

**制約（重要）**

* UNIQUE `(idempotency_key)` を正とする
* Workerは開始時に「既にsuccessなら即return」する。
* `stage_name` は分類用であり、同一 `run_id` 内で複数行を許容する（入力が異なる再実行を許可）
* `attempt` は **同一 `idempotency_key` に対する試行回数** とする（再試行は同一行を更新）

### 5.6 SearchRequest / SearchResult（探索の永続化）

* `search_requests`

  * `id (uuid)`
  * `run_id (fk)`
  * `query (text)`
  * `filters (jsonb)`
  * `mode (enum: keyword, regex, vector)`
  * `status`
  * `created_at`

* `search_results`

  * `id (uuid)`
  * `search_id (fk)`
  * `rank (int)`
  * `source_doc_id (fk)`
  * `chunk_id (fk)`
  * `snippet (text)`
  * `score (float)`
  * `payload (jsonb)`

**冪等キー例**

* `hash(run_id + query + filters + mode)` を `search_requests` に保持しUNIQUE

---

## 6. API設計（概要）

### 6.1 認証

* v0.1：JWTベース（Self-hostで扱いやすい）
* すべてENVで設定（秘密鍵/有効期限）
* 認証トークン発行：`POST /auth/token`

### 6.2 主要エンドポイント

* Runs

  * `POST /runs`（run作成）
  * `GET /runs/{run_id}`（run取得）
  * `GET /runs/{run_id}/stages`（stage状態/失敗理由の取得）
  * `GET /runs/{run_id}/issues`（主導API）
  * `GET /runs/{run_id}/issues?include_hidden=true`（監査/デバッグ用、権限制御あり）
  * `GET /runs/{run_id}/audit/issues`（監査UI向け、hidden含む）
  * `GET /issues/{issue_id}`（論点詳細 + 根拠）
  * `GET /runs/{run_id}/artifacts`（デバッグ用途）
  * `GET /runs/{run_id}/feedback`（互換用途）

* Source Upload

  * `POST /runs/{run_id}/sources/presign-put`（presigned PUT発行 + `source_doc_id/object_key` 返却）
  * `POST /runs/{run_id}/sources/ingest`（`source_doc_id` 指定でworker投入）
  * `GET /sources/{source_doc_id}/presign-get`
  * presignのみで放置された `source_docs` はTTL超過後に自動クリーンアップ

* Pipeline

  * `POST /runs/{run_id}/pipeline`（low/high/feedback/apply/change-span/issue/evidence を一括投入）

* Search

  * `POST /runs/{run_id}/search`（検索要求→worker投入）
  * `GET /search/{search_id}`（結果取得）

* Health

  * `GET /healthz`
  * `GET /readyz`（DB/Redis/MinIO接続確認）

---

## 7. Worker設計（ジョブ/パイプライン）

### 7.1 ジョブ一覧（v0.1）

* `ingest_source_doc(run_id, source_doc_id)`
* `generate_low_draft(run_id, model_cfg)`
* `search_chunks(run_id, query, filters, mode)`
* `generate_feedback_with_citations(run_id, target_artifact_id)`
* `apply_feedback(run_id, base_artifact_id)`
* `generate_high_final(run_id, model_cfg)`
* `detect_change_spans(run_id, artifact_ids)`
* `derive_issues_from_changes(run_id)`
* `attach_evidence_to_issues(run_id)`

### 7.2 リトライと冪等

* すべてのジョブは `run_stages` により冪等制御
* 外部API/LLM呼び出しはリトライ戦略（指数バックオフ + 最大回数）
* 成果物は **先にDBで「このstageの成果はこれ」** を確定してから書き込む（重複を防ぐ）
* 冪等判定は `idempotency_key` を正とし、同一 `stage_name` の多版実行を許容する
* `idempotency_key` の最低構成は
  `hash(run_id + stage_name + stage_input_fingerprint + prompt_template_version + model_cfg)` とする
* `stage_input_fingerprint` は stage入力実体（artifact_id/version, source_doc_id, search条件など）を正規化して含める

### 7.3 Evidence-first（論点表示条件）

* `attach_evidence_to_issues` は以下を満たすこと

  * UI表示対象 issue ごとに `issue_evidences >= 1`
  * issue_evidence が参照する citation の chunk が実在し、`loc` で原文へ戻れる
  * `GET /issues/{issue_id}` は evidence ごとに明示 `selection` を返す
    （互換期間は `citation.span.selection` も併存）
  * 根拠候補の選定は `SearchResult.score` を主（重み0.7）に、論点テキストとの lexical一致度（重み0.3）を合成した
    `combined_score` で決定する（SearchResultが無い場合は lexical一致度のみ）
  * 同点時は `search_rank -> chunk_index -> chunk_id` で決定し、再実行時の再現性を担保する
  * `span` は任意（存在する場合のみ妥当性チェック）
* 根拠を付与できない issue は `hidden` 扱いとし、UI表示しない
* 根拠不足により全 issue が hidden になる場合は `run.status=blocked_evidence` とし、
  `run_stages.failure_type=evidence_insufficient` を記録する
* `evidence_insufficient` の再試行戦略（既定3回）は次で固定する
  1回目：`keyword` + strict filters（`min_score>=0.02`, `top_k=50`）
  2回目：`keyword` + relaxed filters（`min_score>=0.0`, `top_k=100`）
  3回目：`vector` + fallback filters（`min_score>=0.0`, `top_k=120`）
* 各再試行で発行した `search_request_id` と条件差分は `run_stages.failure_detail` に記録する
* `run_stages.failure_detail` には運用者向け `summary` を必須で含め、
  `hidden_reason_counts` 等の集約情報で一目で原因把握できる形にする
* 再試行を使い切っても不足する場合のみ `blocked_evidence` で確定する

---

## 8. 差分（diff）設計（補助）

### 8.1 位置づけ

* `diff` は論点抽出の内部入力・デバッグ用成果物とする。
* ユーザー向け主画面は `issues` 一覧であり、raw diff 閲覧は必須導線にしない。

### 8.2 diff成果物

* `diff` は artifactとして保存（json）
* 推奨フォーマット：`[{status, before_excerpt, after_excerpt, score, phase}]`
* 必要時のみ詳細比較画面で参照する

---

## 9. オブジェクトストレージ（MinIO）設計

### 9.1 key命名規約

* `runs/{run_id}/sources/{source_doc_id}/raw/{filename}`
* `runs/{run_id}/sources/{source_doc_id}/extracted/{artifact}.txt`
* `runs/{run_id}/artifacts/{artifact_id}/{version}.{ext}`

### 9.2 署名付きURL運用

* アップロード：ブラウザ→MinIO（PUT）
* 閲覧：ブラウザ→MinIO（GET）
* APIキーはブラウザに渡さない

---

## 10. 観測性（最小要件）

* ログ：stdoutへJSON

  * `timestamp, service, level, run_id, stage, msg, extra`
* Health

  * `/healthz`：プロセス生存のみ
  * `/readyz`：DB/Redis/MinIO接続
* 将来拡張

  * OpenTelemetry（分散トレース）
  * Prometheus（メトリクス）

---

## 11. デプロイ / Compose運用

### 11.1 compose基本方針

* composeが正本
* すべて固定タグ
* `migrate` は one-shot（web/workerから勝手にmigrateしない）
* `depends_on` だけでなく、readyチェックが通るまで起動順序を担保（起動スクリプト/ヘルス）

### 11.2 ENV（.env.example）

* `DATABASE_URL`
* `REDIS_URL`
* `S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET`
* `SOURCE_DOC_ORPHAN_TTL_HOURS`
* `JWT_SECRET, JWT_EXPIRES_IN`
* `AUTH_ENABLED, AUTH_DEV_USER, AUTH_DEV_PASSWORD`
* `LLM_PROVIDER, LLM_API_KEY, MODEL_HIGH, MODEL_LOW`（キーはSecrets扱い）

---

## 12. K8s移植の見通し

* composeのサービス境界をそのままDeploymentへ
* `migrate` は Job
* `worker` は HPA対象（キュー深さでスケール）
* `minio` は外部S3に差し替え可能

---

## 13. リスクと対策（初期）

* **冪等漏れ**：run_stages＋UNIQUE制約で強制
* **探索の暴走**：regexは実行制限、タイムアウト、上限件数
* **ソースの再現性**：取り込み時にスナップショット（object_key固定）
* **コスト増**：モデル呼び出し回数をrun_stageで可視化し、キャッシュ/再利用

---

## 14. 次に書く（v0.1→v0.2の作業）

1. 具体DDL（テーブル定義、UNIQUE制約、インデックス、FTS/pgvector方針）
2. パイプラインのステージ表（stage_name / input / output / idempotency_key）
3. APIのOpenAPI草案（`issues` 主導のリクエスト/レスポンススキーマ）
4. compose雛形（サービス、ヘルスチェック、migrateの実行方式）

---

## 15. 受け入れ契約（最低ライン）

* `GET /runs/{run_id}/issues` は `issue_evidences>=1` の Issue のみ返す
* 各 evidence は `citation_id` から `(source_doc_id, chunk_id, loc)` に到達できる
* `include_hidden=true` のときのみ hidden Issue を取得できる
* 全Issueがhiddenになった場合は `run.status=blocked_evidence` かつ対応stageに
  `failure_type=evidence_insufficient` が残る
* 同一入力の再実行では成果物が増殖せず、`idempotency_key` により再利用される
