import hashlib
import io
import json
import math
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger
from pypdf import PdfReader
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.celery_app import celery_app
from app.config import get_settings
from app.database import session_scope
from app.llm_client import (
    LLMClientConfigError,
    LLMClientError,
    LLMClientResponseError,
    LLMClientTransientError,
    chat_complete,
    provider_is_stub,
)
from app.models import (
    Artifact,
    ArtifactFormat,
    ArtifactKind,
    Citation,
    FeedbackItem,
    Issue,
    IssueEvidence,
    IssueStatus,
    Run,
    RunStatus,
    SearchRequest,
    SearchMode,
    SearchResult,
    SearchStatus,
    SourceChunk,
    SourceDoc,
    StageFailureType,
)
from app.s3_client import get_object_bytes, put_object_bytes
from app.stage import StageFailureError, execute_stage

logger = get_task_logger(__name__)
settings = get_settings()
PROMPT_TEMPLATE_VERSION = "v2"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def hash_key(*values: Any) -> str:
    raw = "|".join(json.dumps(value, sort_keys=True, default=str) for value in values)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def digest_rows(rows: list[tuple[Any, ...]]) -> str:
    """Build a deterministic compact signature from ordered DB rows."""
    hasher = hashlib.sha256()
    for row in rows:
        for value in row:
            if isinstance(value, datetime):
                serialized = value.isoformat()
            elif isinstance(value, uuid.UUID):
                serialized = str(value)
            else:
                serialized = str(value)
            hasher.update(serialized.encode("utf-8"))
            hasher.update(b"|")
        hasher.update(b";")
    return hasher.hexdigest()


def stage_idempotency_key(
    *,
    run_id: str,
    stage_name: str,
    stage_input_fingerprint: Dict[str, Any] | None = None,
    model_cfg: Dict[str, Any] | None = None,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> str:
    payload = {
        "run_id": run_id,
        "stage_name": stage_name,
        "stage_input_fingerprint": stage_input_fingerprint or {},
        "prompt_template_version": prompt_template_version,
        "model_cfg": model_cfg or {},
    }
    return hash_key(payload)


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp932", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[page {idx + 1}]\n{page_text}")
    return "\n\n".join(pages)


def strip_html(html: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", without_scripts)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(raw: bytes, *, object_key: str, content_type: str) -> str:
    lowered_ct = (content_type or "").lower()
    ext = Path(object_key).suffix.lower()

    if "pdf" in lowered_ct or ext == ".pdf":
        return extract_pdf_text(raw)

    text = decode_text(raw)
    if "application/json" in lowered_ct or ext == ".json":
        try:
            payload = json.loads(text)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text
    if "text/html" in lowered_ct or ext in {".html", ".htm"}:
        return strip_html(text)
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            split = normalized.rfind("\n", start, end)
            if split > start + 120:
                end = split
        piece = normalized[start:end].strip()
        if piece:
            chunks.append({"text": piece, "loc": {"start_offset": start, "end_offset": end}})
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def artifact_version(db, run_id: uuid.UUID, kind: ArtifactKind) -> int:
    max_version = (
        db.query(func.max(Artifact.version))
        .filter(Artifact.run_id == run_id, Artifact.kind == kind)
        .scalar()
    )
    return (max_version or 0) + 1


def create_artifact(
    db,
    run_id: uuid.UUID,
    kind: ArtifactKind,
    format_: ArtifactFormat,
    content_text: Optional[str] = None,
    content_object_key: Optional[str] = None,
) -> Artifact:
    item = Artifact(
        run_id=run_id,
        kind=kind,
        format=format_,
        content_text=content_text,
        content_object_key=content_object_key,
        version=artifact_version(db, run_id, kind),
    )
    db.add(item)
    db.flush()
    return item


def get_run_or_raise(db, run_id: uuid.UUID) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    return run


def latest_artifact_by_kind(db, run_id: uuid.UUID, kind: ArtifactKind) -> Artifact:
    item = (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id, Artifact.kind == kind)
        .order_by(Artifact.created_at.desc())
        .first()
    )
    if item is None:
        raise ValueError(f"artifact missing for kind={kind.value}")
    return item


def excerpt(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:limit]


def multiline_excerpt(text: str, *, max_chars: int = 5000, max_lines: int = 140) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw:
        return ""
    lines = raw.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars]
    return clipped.strip()


def first_focus_line(text: str, limit: int = 96) -> str:
    for line in (text or "").splitlines():
        stripped = re.sub(r"^[#>\-\+\*\d\.\)\( ]+", "", line.strip())
        if not stripped:
            continue
        return stripped[:limit]
    return ""


def issue_title_from_span(status: str, *, before_text: str, after_text: str) -> str:
    focus = first_focus_line(after_text) or first_focus_line(before_text) or "差分箇所"
    if status == "modified":
        return f"{focus} の変更意図と影響範囲を確認する"
    if status == "added":
        return f"{focus} の追加内容が妥当か確認する"
    if status == "removed":
        return f"{focus} の削除影響を確認する"
    return f"{focus} に関する論点を確認する"


def issue_summary_from_span(*, before_preview: str, after_preview: str) -> str:
    if before_preview and after_preview:
        return f"before: {before_preview} / after: {after_preview}"
    if after_preview:
        return f"after: {after_preview}"
    if before_preview:
        return f"before: {before_preview}"
    return "差分の具体内容を確認し、根拠付きで妥当性を判断してください。"


def normalize_fingerprint_text(text: str) -> str:
    cleaned = " ".join((text or "").lower().split())
    return cleaned[:512]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def normalized_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def char_ngram_counts(text: str, n: int = 3) -> Dict[str, int]:
    normalized = normalized_text(text).replace(" ", "")
    if not normalized:
        return {}
    if len(normalized) < n:
        return {normalized: 1}
    counts: Dict[str, int] = {}
    for i in range(len(normalized) - n + 1):
        gram = normalized[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def cosine_similarity_from_counts(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, value in a.items():
        dot += float(value * b.get(key, 0))
    if dot <= 0.0:
        return 0.0
    norm_a = math.sqrt(sum(float(v * v) for v in a.values()))
    norm_b = math.sqrt(sum(float(v * v) for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def vector_similarity_score(query: str, text: str) -> float:
    query_counts = char_ngram_counts(query)
    text_counts = char_ngram_counts(text)
    return cosine_similarity_from_counts(query_counts, text_counts)


def ngram_key_set(text: str, n: int = 3) -> set[str]:
    return set(char_ngram_counts(text, n=n).keys())


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union_size = len(a.union(b))
    if union_size == 0:
        return 0.0
    return len(a.intersection(b)) / float(union_size)


def clamp_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(parsed, 1.0))


def lexical_match_score(hint_text: str, chunk_text: str) -> float:
    normalized_hint = " ".join((hint_text or "").lower().split())
    normalized_chunk = " ".join((chunk_text or "").lower().split())
    if not normalized_hint or not normalized_chunk:
        return 0.0

    hint_tokens = tokens(normalized_hint)
    chunk_tokens = tokens(normalized_chunk)
    token_score = 0.0
    if hint_tokens and chunk_tokens:
        overlap = len(hint_tokens.intersection(chunk_tokens))
        token_recall = overlap / max(len(hint_tokens), 1)
        token_precision = overlap / max(len(chunk_tokens), 1)
        token_score = 0.75 * token_recall + 0.25 * token_precision

    # Token overlap may be weak for some scripts; keep a short literal-match boost.
    prefix = normalized_hint[:24]
    prefix_hit = 1.0 if prefix and prefix in normalized_chunk else 0.0
    return max(0.0, min(1.0, 0.85 * token_score + 0.15 * prefix_hit))


def choose_best_chunk(
    candidates: list[Dict[str, Any]],
    hint_text: str,
) -> Dict[str, Any] | None:
    if not candidates:
        return None

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        chunk: SourceChunk = candidate["chunk"]
        lexical_score = lexical_match_score(hint_text, chunk.text or "")
        has_search_score = candidate.get("search_score") is not None
        search_score = clamp_score(candidate.get("search_score")) if has_search_score else 0.0
        if has_search_score:
            search_weight = 0.7
            lexical_weight = 0.3
        else:
            search_weight = 0.0
            lexical_weight = 1.0
        combined = search_weight * search_score + lexical_weight * lexical_score
        rank_order = int(candidate["search_rank"]) if candidate.get("search_rank") is not None else 10**9

        scored.append(
            {
                **candidate,
                "rank_order": rank_order,
                "selection_detail": {
                    "version": "search_score_weighted_v1",
                    "candidate_source": candidate.get("candidate_source", "run_chunks"),
                    "combined_score": round(combined, 6),
                    "search_score": round(search_score, 6) if has_search_score else None,
                    "lexical_score": round(lexical_score, 6),
                    "weights": {"search_score": search_weight, "lexical_score": lexical_weight},
                    "search_rank": rank_order if rank_order < 10**9 else None,
                },
            }
        )

    scored.sort(
        key=lambda row: (
            -row["selection_detail"]["combined_score"],
            -(row["selection_detail"]["search_score"] or 0.0),
            -row["selection_detail"]["lexical_score"],
            row["rank_order"],
            row["chunk"].chunk_index,
            str(row["chunk"].id),
        )
    )
    return scored[0]


def collect_source_context(db, run_id: uuid.UUID, *, limit_chunks: int = 4, max_chars: int = 2600) -> str:
    rows = (
        db.query(SourceChunk, SourceDoc)
        .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
        .filter(SourceDoc.run_id == run_id)
        .order_by(SourceDoc.created_at.asc(), SourceChunk.chunk_index.asc())
        .limit(limit_chunks)
        .all()
    )
    if not rows:
        return "（ソースチャンクなし）"

    parts: list[str] = []
    total = 0
    for chunk, source in rows:
        text = (chunk.text or "").strip()
        if not text:
            continue
        piece = f"[{source.title} / chunk {chunk.chunk_index}]\n{text[:900]}"
        next_total = total + len(piece) + 2
        if next_total > max_chars and parts:
            break
        parts.append(piece)
        total = next_total
    return "\n\n".join(parts) if parts else "（ソースチャンクなし）"


def source_chunk_fingerprint(db, run_id: uuid.UUID) -> str:
    rows = (
        db.query(SourceDoc.id, SourceChunk.id, SourceChunk.chunk_index)
        .join(SourceChunk, SourceChunk.source_doc_id == SourceDoc.id)
        .filter(SourceDoc.run_id == run_id)
        .order_by(SourceDoc.created_at.asc(), SourceChunk.chunk_index.asc())
        .all()
    )
    if not rows:
        return "no_source_chunks"
    return digest_rows(rows)


def artifact_ref_fingerprint(item: Artifact | None) -> str:
    if item is None:
        return "none"
    return hash_key(str(item.id), item.version, item.created_at.isoformat())


def feedback_rows_fingerprint(feedbacks: List[FeedbackItem]) -> str:
    if not feedbacks:
        return "no_feedback"
    rows = [
        (fb.id, fb.target_artifact_id, fb.text, fb.category, fb.severity, fb.created_at)
        for fb in feedbacks
    ]
    return digest_rows(rows)


def collect_feedback_context(
    db,
    feedbacks: List[FeedbackItem],
    *,
    max_chars: int = 2600,
) -> str:
    if not feedbacks:
        return "（フィードバックなし）"

    parts: list[str] = []
    total = 0
    for idx, fb in enumerate(feedbacks, start=1):
        citation_row = (
            db.query(Citation, SourceChunk, SourceDoc)
            .join(SourceChunk, SourceChunk.id == Citation.chunk_id)
            .join(SourceDoc, SourceDoc.id == Citation.source_doc_id)
            .filter(Citation.feedback_id == fb.id)
            .order_by(SourceChunk.chunk_index.asc())
            .first()
        )
        if citation_row is None:
            evidence_text = "（引用なし）"
        else:
            _, chunk, source = citation_row
            snippet = " ".join((chunk.text or "").split())[:220]
            evidence_text = f"{source.title} / chunk {chunk.chunk_index}: {snippet}"
        piece = (
            f"{idx}. 指摘: {fb.text}\n"
            f"   category={fb.category}, severity={fb.severity}\n"
            f"   根拠: {evidence_text}"
        )
        next_total = total + len(piece) + 2
        if next_total > max_chars and parts:
            break
        parts.append(piece)
        total = next_total
    return "\n\n".join(parts) if parts else "（フィードバックなし）"


def llm_text_or_stub(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    stub_text: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    if provider_is_stub():
        return stub_text
    try:
        text = chat_complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMClientConfigError as exc:
        raise StageFailureError(
            "LLM 設定エラーのため生成できませんでした",
            failure_type=StageFailureType.validation_error,
            failure_detail={
                "summary": "LLM設定を確認してください（provider / api key / base_url）",
                "reason": str(exc),
            },
        ) from exc
    except LLMClientTransientError as exc:
        raise RuntimeError(f"LLM 一時障害のため再試行します: {exc}") from exc
    except LLMClientResponseError as exc:
        raise StageFailureError(
            "LLM 応答形式エラーのため生成できませんでした",
            failure_type=StageFailureType.system_error,
            failure_detail={
                "summary": "LLM応答を解釈できませんでした",
                "reason": str(exc),
            },
        ) from exc
    except LLMClientError as exc:
        raise RuntimeError(f"LLM 生成で失敗しました: {exc}") from exc
    text = text.strip()
    if not text:
        raise StageFailureError(
            "LLM 生成結果が空のため処理を続行できません",
            failure_type=StageFailureType.system_error,
            failure_detail={"summary": "LLMの出力が空文字でした"},
        )
    return text


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def ingest_source_doc(
    self,
    run_id: str,
    source_doc_id: str,
    object_key: Optional[str] = None,
) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    source_doc_uuid = uuid.UUID(source_doc_id)
    stage_name = "ingest_source_doc"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"source_doc_id": source_doc_id, "object_key": object_key or ""},
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            source_doc = db.get(SourceDoc, source_doc_uuid)
            if source_doc is None:
                raise ValueError("source_doc not found for ingest")
            if source_doc.run_id != run_uuid:
                raise ValueError("source_doc does not belong to run")
            if object_key and source_doc.object_key != object_key:
                source_doc.object_key = object_key

            db.query(SourceChunk).filter(SourceChunk.source_doc_id == source_doc.id).delete()

            raw_bytes = get_object_bytes(source_doc.object_key)
            extracted_text = extract_text(
                raw_bytes,
                object_key=source_doc.object_key,
                content_type=source_doc.content_type,
            )
            if not extracted_text.strip():
                raise RuntimeError("no extractable text content")

            extracted_key = f"runs/{run_uuid}/sources/{source_doc.id}/extracted/extracted.txt"
            put_object_bytes(
                extracted_key,
                extracted_text.encode("utf-8"),
                content_type="text/plain; charset=utf-8",
            )

            chunks = chunk_text(extracted_text)
            for idx, chunk in enumerate(chunks):
                db.add(
                    SourceChunk(
                        source_doc_id=source_doc.id,
                        chunk_index=idx,
                        text=chunk["text"],
                        loc={**chunk["loc"], "chunk_index": idx},
                    )
                )
            return {
                "source_doc_id": str(source_doc.id),
                "chunks": len(chunks),
                "extracted_object_key": extracted_key,
            }

        result = execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        )
        return result.output_ref


@celery_app.task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_low_draft(self, run_id: str, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "generate_low_draft"

    with session_scope() as db:
        run = get_run_or_raise(db, run_uuid)
        source_fp = source_chunk_fingerprint(db, run_uuid)
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "task": "low_draft",
                "task_prompt": hash_key(run.task_prompt or ""),
                "source_chunk_fp": source_fp,
            },
            model_cfg=model_cfg or {},
        )

        def work() -> Dict[str, Any]:
            model_name = (model_cfg or {}).get("model", settings.model_low)
            source_context = collect_source_context(db, run_uuid)
            generated_text = llm_text_or_stub(
                model=model_name,
                system_prompt=(
                    "あなたは文書作成アシスタントです。"
                    "根拠に基づいた実務向けの下書きを日本語で作成してください。"
                    "見出しと箇条書きで簡潔に構造化してください。"
                ),
                user_prompt=(
                    "タスク:\n"
                    f"{run.task_prompt}\n\n"
                    "根拠スニペット:\n"
                    f"{source_context}\n\n"
                    "出力要件:\n"
                    "- Markdownのみ\n"
                    "- 先頭は '# Low Draft' で開始\n"
                    "- 冒頭に短い要約を置く\n"
                    "- 具体的な箇条書きを含める\n"
                ),
                stub_text=(
                    f"# Low Draft\n\nモデル: {model_name}\n\n"
                    f"タスク:\n{run.task_prompt}\n\n- 下書き項目1\n- 下書き項目2\n"
                ),
                temperature=0.3,
                max_tokens=1200,
            )
            if not generated_text.lstrip().startswith("#"):
                generated_text = f"# Low Draft\n\n{generated_text}"
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.low_draft,
                ArtifactFormat.markdown,
                content_text=generated_text,
            )
            run.status = RunStatus.running
            return {"artifact_id": str(artifact.id)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def search_chunks(
    self,
    run_id: str,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    mode: str = "keyword",
    search_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "search_chunks"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={
            "query": query.strip(),
            "filters": filters or {},
            "mode": mode,
            "search_id": search_id,
        },
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            search_req = db.get(SearchRequest, uuid.UUID(search_id)) if search_id else None
            if search_req:
                search_req.status = SearchStatus.running

            filter_payload = filters or {}
            try:
                top_k = int(filter_payload.get("top_k", 50))
            except (TypeError, ValueError):
                top_k = 50
            top_k = max(1, min(top_k, 200))
            try:
                min_score = float(filter_payload.get("min_score", 0.0))
            except (TypeError, ValueError):
                min_score = 0.0
            min_score = max(0.0, min(min_score, 1.0))

            q = query.strip()
            if not q:
                if search_req:
                    search_req.status = SearchStatus.failed
                raise ValueError("query is required")
            if len(q) > 512:
                if search_req:
                    search_req.status = SearchStatus.failed
                raise ValueError("query too long")

            rows = (
                db.query(SourceChunk, SourceDoc)
                .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
                .filter(SourceDoc.run_id == run_uuid)
                .all()
            )
            scored: List[Dict[str, Any]] = []
            rx = None
            if mode == "regex":
                if len(q) > 256:
                    if search_req:
                        search_req.status = SearchStatus.failed
                    raise ValueError("regex query too long")
                try:
                    rx = re.compile(q)
                except re.error as exc:
                    if search_req:
                        search_req.status = SearchStatus.failed
                    raise ValueError(f"invalid regex query: {exc}") from exc

            for chunk, source_doc in rows:
                text = chunk.text or ""
                score = 0.0
                score_reason = "no_match"
                if mode == "keyword":
                    if q.lower() in text.lower():
                        score = len(q) / max(len(text), 1)
                        score_reason = "keyword_substring"
                elif mode == "regex":
                    if rx and rx.search(text):
                        score = 1.0
                        score_reason = "regex_match"
                else:
                    score = vector_similarity_score(q, text)
                    score_reason = "char_ngram_cosine"

                if score > min_score:
                    scored.append(
                        {
                            "source_doc_id": source_doc.id,
                            "chunk_id": chunk.id,
                            "snippet": text[:240],
                            "score": float(score),
                            "payload": {
                                "loc": chunk.loc,
                                "mode": mode,
                                "score_reason": score_reason,
                                "min_score": min_score,
                            },
                        }
                    )

            scored.sort(key=lambda item: item["score"], reverse=True)
            if search_req:
                db.query(SearchResult).filter(SearchResult.search_id == search_req.id).delete()
                for rank, item in enumerate(scored[:top_k], start=1):
                    db.add(
                        SearchResult(
                            search_id=search_req.id,
                            rank=rank,
                            source_doc_id=item["source_doc_id"],
                            chunk_id=item["chunk_id"],
                            snippet=item["snippet"],
                            score=item["score"],
                            payload=item["payload"],
                        )
                    )
                search_req.status = SearchStatus.success

            return {"result_count": len(scored[:top_k]), "top_k": top_k, "min_score": min_score}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_feedback_with_citations(
    self,
    run_id: str,
    target_artifact_id: str,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    target_uuid = uuid.UUID(target_artifact_id)
    stage_name = "generate_feedback_with_citations"

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)
        target = db.get(Artifact, target_uuid)
        if target is None:
            raise ValueError("target artifact not found")
        selected_chunk = (
            db.query(SourceChunk, SourceDoc)
            .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
            .filter(SourceDoc.run_id == run_uuid)
            .order_by(SourceChunk.chunk_index.asc())
            .first()
        )
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "target_artifact_id": target_artifact_id,
                "target_artifact_fp": artifact_ref_fingerprint(target),
                "source_chunk_fp": source_chunk_fingerprint(db, run_uuid),
                "selected_chunk_id": str(selected_chunk[0].id) if selected_chunk else "none",
            },
            model_cfg=model_cfg or {},
        )

        def work() -> Dict[str, Any]:
            chunk = selected_chunk
            if chunk is None:
                raise StageFailureError(
                    "根拠付きフィードバック生成に必要な根拠チャンクがありません",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={
                        "run_id": str(run_uuid),
                        "required": "source_chunk",
                        "summary": "根拠チャンクがないため根拠付きフィードバックを生成できません",
                    },
                )

            source_chunk, source_doc = chunk
            model_name = (model_cfg or {}).get("model", settings.model_low)
            generated_feedback = llm_text_or_stub(
                model=model_name,
                system_prompt=(
                    "あなたはレビュー担当者です。"
                    "提示された根拠に直接結びつく、具体的で実行可能な改善指摘を日本語で1文だけ返してください。"
                ),
                user_prompt=(
                    "対象ドラフト抜粋:\n"
                    f"{(target.content_text or '')[:1600]}\n\n"
                    "根拠抜粋:\n"
                    f"{(source_chunk.text or '')[:1200]}\n\n"
                    "制約:\n"
                    "- 必ず1文のみ\n"
                    "- 何を追記/修正/明確化するかを明示\n"
                ),
                stub_text="根拠文書に照らして、主張の具体例を1つ追加してください。",
                temperature=0.2,
                max_tokens=220,
            )
            feedback_text = re.sub(r"\s+", " ", generated_feedback).strip()
            if not feedback_text:
                raise StageFailureError(
                    "フィードバック生成結果が空です",
                    failure_type=StageFailureType.system_error,
                    failure_detail={"summary": "feedback generation が空文字を返しました"},
                )
            feedback_text = feedback_text[:400]
            feedback = FeedbackItem(
                run_id=run_uuid,
                target_artifact_id=target.id,
                text=feedback_text,
                category="evidence",
                severity=2,
            )
            db.add(feedback)
            db.flush()
            citation = Citation(
                feedback_id=feedback.id,
                source_doc_id=source_doc.id,
                chunk_id=source_chunk.id,
                span={"loc": source_chunk.loc, "scope": "feedback"},
            )
            db.add(citation)
            db.flush()

            citation_count = db.query(Citation).filter(Citation.feedback_id == feedback.id).count()
            if citation_count < 1:
                raise StageFailureError(
                    "フィードバックには最低1件の引用が必要です",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={
                        "feedback_id": str(feedback.id),
                        "summary": "フィードバック生成は完了しましたが引用が0件でした",
                    },
                )

            return {"feedback_id": str(feedback.id), "citation_count": citation_count}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def apply_feedback(
    self,
    run_id: str,
    base_artifact_id: str,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    base_uuid = uuid.UUID(base_artifact_id)
    stage_name = "apply_feedback"

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)
        base = db.get(Artifact, base_uuid)
        if base is None:
            raise ValueError("base artifact not found")
        feedbacks = (
            db.query(FeedbackItem)
            .filter(FeedbackItem.run_id == run_uuid, FeedbackItem.target_artifact_id == base.id)
            .order_by(FeedbackItem.created_at.asc(), FeedbackItem.id.asc())
            .all()
        )
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "base_artifact_id": base_artifact_id,
                "base_artifact_fp": artifact_ref_fingerprint(base),
                "feedback_fp": feedback_rows_fingerprint(feedbacks),
                "feedback_count": len(feedbacks),
            },
            model_cfg=model_cfg or {},
        )

        def work() -> Dict[str, Any]:
            model_name = (model_cfg or {}).get("model", settings.model_low)
            base_text = base.content_text or ""
            feedback_context = collect_feedback_context(db, feedbacks)
            generated_text = llm_text_or_stub(
                model=model_name,
                system_prompt=(
                    "あなたは編集者です。"
                    "ベース原稿に対して、提示されたフィードバックを自然に反映した改稿版を日本語で作成してください。"
                    "内容の整合性と根拠性を保ち、冗長な重複を避けてください。"
                ),
                user_prompt=(
                    "ベース原稿:\n"
                    f"{base_text[:7000]}\n\n"
                    "反映対象フィードバック:\n"
                    f"{feedback_context}\n\n"
                    "出力要件:\n"
                    "- Markdownのみ\n"
                    "- 先頭は '# Improved Draft' で開始\n"
                    "- 既存構成をできるだけ維持しつつ改善を反映\n"
                    "- 末尾に「## 反映メモ」を置き、反映した指摘を箇条書きで要約\n"
                ),
                stub_text=(
                    "# Improved Draft\n\n"
                    f"{base_text}\n\n"
                    "## 反映メモ\n"
                    + ("\n".join(f"- {fb.text}" for fb in feedbacks) if feedbacks else "- 反映対象のフィードバックはありません")
                    + "\n"
                ),
                temperature=0.2,
                max_tokens=1800,
            )
            if not generated_text.lstrip().startswith("#"):
                generated_text = f"# Improved Draft\n\n{generated_text}"
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.improved,
                ArtifactFormat.markdown,
                content_text=generated_text,
            )
            return {"artifact_id": str(artifact.id)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_high_final(self, run_id: str, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "generate_high_final"

    with session_scope() as db:
        run = get_run_or_raise(db, run_uuid)
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "task": "high_oneshot",
                "task_prompt": hash_key(run.task_prompt or ""),
                "source_chunk_fp": source_chunk_fingerprint(db, run_uuid),
            },
            model_cfg=model_cfg or {},
        )

        def work() -> Dict[str, Any]:
            model_name = (model_cfg or {}).get("model", settings.model_high)
            source_context = collect_source_context(db, run_uuid)
            generated_text = llm_text_or_stub(
                model=model_name,
                system_prompt=(
                    "あなたはシニアエディタです。"
                    "与えられたタスクと根拠のみを使って、一発で完成度の高い日本語ドラフトを作成してください。"
                    "構成は読み手が比較しやすいように見出しと箇条書きを含めてください。"
                ),
                user_prompt=(
                    "タスク:\n"
                    f"{(run.task_prompt or '')[:5000]}\n\n"
                    "根拠スニペット:\n"
                    f"{source_context}\n\n"
                    "出力要件:\n"
                    "- Markdownのみ\n"
                    "- 先頭は '# High One-shot' で開始\n"
                    "- 具体的で判断可能な表現を維持\n"
                ),
                stub_text=(
                    f"# High One-shot\n\nモデル: {model_name}\n\n"
                    f"タスク:\n{run.task_prompt}\n\n"
                    "## 要点\n- 根拠に基づく一発生成のドラフトです。\n"
                ),
                temperature=0.2,
                max_tokens=1600,
            )
            if not generated_text.lstrip().startswith("#"):
                generated_text = f"# High One-shot\n\n{generated_text}"
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.high_final,
                ArtifactFormat.markdown,
                content_text=generated_text,
            )
            run.status = RunStatus.running
            return {"artifact_id": str(artifact.id)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


def block_diff(a_text: str, b_text: str) -> List[Dict[str, Any]]:
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    matcher = SequenceMatcher(a=a_lines, b=b_lines)
    result: List[Dict[str, Any]] = []
    block_idx = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        block_idx += 1
        status_map = {"equal": "unchanged", "replace": "modified", "insert": "added", "delete": "removed"}
        result.append(
            {
                "block_id": f"b{block_idx}",
                "status": status_map.get(tag, tag),
                "a_text": "\n".join(a_lines[i1:i2]),
                "b_text": "\n".join(b_lines[j1:j2]),
                "score": matcher.ratio(),
            }
        )
    return result


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def build_diffs(self, run_id: str, artifact_ids: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "build_diffs"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"artifact_ids": artifact_ids or {}},
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def latest(kind: ArtifactKind) -> Artifact:
            item = (
                db.query(Artifact)
                .filter(Artifact.run_id == run_uuid, Artifact.kind == kind)
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if item is None:
                raise ValueError(f"artifact missing for kind={kind.value}")
            return item

        def work() -> Dict[str, Any]:
            if artifact_ids:
                low = db.get(Artifact, uuid.UUID(artifact_ids["low_draft"]))
                improved = db.get(Artifact, uuid.UUID(artifact_ids["improved"]))
                high = db.get(Artifact, uuid.UUID(artifact_ids["high_final"]))
                if not low or not improved or not high:
                    raise ValueError("artifact_ids contain unknown artifact")
            else:
                low = latest(ArtifactKind.low_draft)
                improved = latest(ArtifactKind.improved)
                high = latest(ArtifactKind.high_final)

            low_vs_improved = block_diff(low.content_text or "", improved.content_text or "")
            improved_vs_high = block_diff(improved.content_text or "", high.content_text or "")
            payload = {"low_vs_improved": low_vs_improved, "improved_vs_high": improved_vs_high}
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.diff,
                ArtifactFormat.json,
                content_text=json.dumps(payload, ensure_ascii=False),
            )
            return {"artifact_id": str(artifact.id), "blocks": len(low_vs_improved) + len(improved_vs_high)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def run_full_pipeline(self, run_id: str, model_low: Optional[str] = None, model_high: Optional[str] = None) -> Dict[str, Any]:
    low_result = generate_low_draft.run(run_id, {"model": model_low or settings.model_low})
    low_artifact_id = low_result["artifact_id"]

    feedback_result = generate_feedback_with_citations.run(
        run_id,
        low_artifact_id,
        {"model": model_high or settings.model_high},
    )

    improved_result = apply_feedback.run(
        run_id,
        low_artifact_id,
        {"model": model_low or settings.model_low},
    )
    improved_artifact_id = improved_result["artifact_id"]

    high_result = generate_high_final.run(run_id, {"model": model_high or settings.model_high})
    high_artifact_id = high_result["artifact_id"]

    span_result = detect_change_spans.run(
        run_id,
        {
            "low_draft": low_artifact_id,
            "improved": improved_artifact_id,
            "high_final": high_artifact_id,
        },
    )
    issue_result = derive_issues_from_changes.run(run_id)
    run_uuid = uuid.UUID(run_id)
    with session_scope() as db:
        run = get_run_or_raise(db, run_uuid)
        retry_query = (run.task_prompt or "").strip() or "evidence"

    # Retry strategy for evidence不足:
    # 1) keyword + strict filters
    # 2) keyword + relaxed filters + larger top_k
    evidence_retry_plan = [
        {"attempt": 1, "mode": "keyword", "filters": {"strategy": "strict", "top_k": 50, "min_score": 0.02}},
        {"attempt": 2, "mode": "keyword", "filters": {"strategy": "relaxed", "top_k": 100, "min_score": 0.0}},
        {"attempt": 3, "mode": "vector", "filters": {"strategy": "vector_fallback", "top_k": 120, "min_score": 0.0}},
    ]
    retry_search_ids: list[str] = []
    evidence_retry_count = 0
    while True:
        retry_context = {
            "retry_attempt": evidence_retry_count,
            "retry_search_ids": retry_search_ids,
        }
        try:
            evidence_result = attach_evidence_to_issues.run(run_id, retry_context=retry_context)
            break
        except StageFailureError as exc:
            if exc.failure_type != StageFailureType.evidence_insufficient:
                raise
            if evidence_retry_count >= len(evidence_retry_plan):
                raise StageFailureError(
                    str(exc),
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={
                        **(exc.failure_detail or {}),
                        "retry_search_ids": retry_search_ids,
                        "retry_strategy": [row["filters"] for row in evidence_retry_plan],
                        "retry_exhausted": True,
                    },
                ) from exc

            step = evidence_retry_plan[evidence_retry_count]
            evidence_retry_count += 1

            with session_scope() as db:
                search_key = hash_key("evidence_retry_search", run_id, step, retry_query[:256])
                search_req = SearchRequest(
                    run_id=run_uuid,
                    query=retry_query[:256],
                    filters=step["filters"],
                    mode=SearchMode(step["mode"]),
                    status=SearchStatus.pending,
                    idempotency_key=search_key,
                )
                try:
                    db.add(search_req)
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    search_req = db.query(SearchRequest).filter(SearchRequest.idempotency_key == search_key).one()

                search_id = str(search_req.id)

            search_chunks.run(
                run_id,
                query=retry_query[:256],
                filters=step["filters"],
                mode=step["mode"],
                search_id=search_id,
            )
            retry_search_ids.append(search_id)

    return {
        "low_artifact_id": low_artifact_id,
        "feedback_id": feedback_result["feedback_id"],
        "improved_artifact_id": improved_artifact_id,
        "high_artifact_id": high_artifact_id,
        "high_oneshot_artifact_id": high_artifact_id,
        "diff_artifact_id": span_result["artifact_id"],
        "issue_count": issue_result["issue_count"],
        "issue_with_evidence_count": evidence_result["issue_with_evidence_count"],
        "evidence_retry_count": evidence_retry_count,
        "evidence_retry_search_ids": retry_search_ids,
    }


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def detect_change_spans(
    self,
    run_id: str,
    artifact_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "detect_change_spans"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"artifact_ids": artifact_ids or {}},
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            if artifact_ids:
                low = db.get(Artifact, uuid.UUID(artifact_ids["low_draft"]))
                improved = db.get(Artifact, uuid.UUID(artifact_ids["improved"]))
                high = db.get(Artifact, uuid.UUID(artifact_ids["high_final"]))
                if not low or not improved or not high:
                    raise ValueError("artifact_ids contain unknown artifact")
            else:
                low = latest_artifact_by_kind(db, run_uuid, ArtifactKind.low_draft)
                improved = latest_artifact_by_kind(db, run_uuid, ArtifactKind.improved)
                high = latest_artifact_by_kind(db, run_uuid, ArtifactKind.high_final)

            low_vs_improved = block_diff(low.content_text or "", improved.content_text or "")
            improved_vs_high = block_diff(improved.content_text or "", high.content_text or "")

            spans: list[dict[str, Any]] = []
            for idx, span in enumerate(low_vs_improved):
                if span["status"] == "unchanged":
                    continue
                before_raw = multiline_excerpt(span.get("a_text", ""))
                after_raw = multiline_excerpt(span.get("b_text", ""))
                spans.append(
                    {
                        "span_id": f"li-{idx + 1}",
                        "phase": "low_to_improved",
                        "status": span["status"],
                        "before_excerpt": before_raw,
                        "after_excerpt": after_raw,
                        "before_preview": excerpt(before_raw),
                        "after_preview": excerpt(after_raw),
                        "score": float(span.get("score", 0.0)),
                    }
                )
            for idx, span in enumerate(improved_vs_high):
                if span["status"] == "unchanged":
                    continue
                before_raw = multiline_excerpt(span.get("a_text", ""))
                after_raw = multiline_excerpt(span.get("b_text", ""))
                spans.append(
                    {
                        "span_id": f"ih-{idx + 1}",
                        "phase": "improved_to_high",
                        "status": span["status"],
                        "before_excerpt": before_raw,
                        "after_excerpt": after_raw,
                        "before_preview": excerpt(before_raw),
                        "after_preview": excerpt(after_raw),
                        "score": float(span.get("score", 0.0)),
                    }
                )

            payload = {
                "spans": spans,
                "debug_diff": {
                    "low_vs_improved": low_vs_improved,
                    "improved_vs_high": improved_vs_high,
                },
            }
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.diff,
                ArtifactFormat.json,
                content_text=json.dumps(payload, ensure_ascii=False),
            )
            return {"artifact_id": str(artifact.id), "span_count": len(spans)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def derive_issues_from_changes(self, run_id: str) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "derive_issues_from_changes"

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)
        diff_artifact = latest_artifact_by_kind(db, run_uuid, ArtifactKind.diff)
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "diff_artifact_id": str(diff_artifact.id),
                "diff_artifact_version": diff_artifact.version,
                "diff_artifact_created_at": (
                    diff_artifact.created_at.isoformat() if diff_artifact.created_at else None
                ),
            },
        )

        def work() -> Dict[str, Any]:
            payload = json.loads(diff_artifact.content_text or "{}")
            spans = payload.get("spans", [])

            db.query(Issue).filter(Issue.run_id == run_uuid).delete(synchronize_session=False)
            db.flush()

            created = 0
            dedup_merged_count = 0
            dedup_similarity_threshold = max(0.0, min(1.0, float(settings.issue_dedup_similarity_threshold)))
            groups: list[dict[str, Any]] = []
            for idx, span in enumerate(spans):
                status = span.get("status", "modified")
                title_prefix = {
                    "modified": "変更検知",
                    "added": "追加検知",
                    "removed": "削除検知",
                }.get(status, "論点候補")
                before_excerpt = span.get("before_excerpt") or ""
                after_excerpt = span.get("after_excerpt") or ""
                before_preview = span.get("before_preview") or excerpt(before_excerpt)
                after_preview = span.get("after_preview") or excerpt(after_excerpt)
                severity = 3 if status in {"modified", "removed"} else 2
                confidence = min(0.99, max(0.4, float(span.get("score", 0.5))))
                phase = span.get("phase", "")
                normalized_before = normalize_fingerprint_text(before_excerpt)
                normalized_after = normalize_fingerprint_text(after_excerpt)
                dedup_basis_text = f"{phase}\n{status}\n{normalized_before}\n{normalized_after}"
                gram_set = ngram_key_set(dedup_basis_text)
                if not gram_set:
                    gram_set = {f"phase={phase}|status={status}|empty"}

                best_group: dict[str, Any] | None = None
                best_similarity = 0.0
                for group in groups:
                    if group["phase"] != phase or group["status"] != status:
                        continue
                    similarity = jaccard_similarity(gram_set, group["gram_set"])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_group = group

                if best_group and best_similarity >= dedup_similarity_threshold:
                    dedup_merged_count += 1
                    best_group["gram_set"] = best_group["gram_set"].union(gram_set)
                    best_group["dedup_count"] += 1
                    best_group["span_ids"].append(span.get("span_id"))
                    best_group["severity"] = max(best_group["severity"], severity)
                    best_group["confidence"] = max(best_group["confidence"], confidence)
                    if len(after_excerpt) > len(best_group["after_excerpt"]):
                        best_group["after_excerpt"] = after_excerpt
                    if len(before_excerpt) > len(best_group["before_excerpt"]):
                        best_group["before_excerpt"] = before_excerpt
                    if len(after_preview) > len(best_group["after_preview"]):
                        best_group["after_preview"] = after_preview
                    if len(before_preview) > len(best_group["before_preview"]):
                        best_group["before_preview"] = before_preview
                    best_group["title"] = issue_title_from_span(
                        status=best_group["status"],
                        before_text=best_group["before_excerpt"],
                        after_text=best_group["after_excerpt"],
                    )
                    best_group["summary"] = issue_summary_from_span(
                        before_preview=best_group["before_preview"],
                        after_preview=best_group["after_preview"],
                    )
                    continue

                groups.append(
                    {
                        "phase": phase,
                        "status": status,
                        "title_prefix": title_prefix,
                        "title": issue_title_from_span(
                            status=status,
                            before_text=before_excerpt,
                            after_text=after_excerpt,
                        ),
                        "summary": issue_summary_from_span(
                            before_preview=before_preview,
                            after_preview=after_preview,
                        ),
                        "severity": severity,
                        "confidence": confidence,
                        "before_excerpt": before_excerpt,
                        "after_excerpt": after_excerpt,
                        "before_preview": before_preview,
                        "after_preview": after_preview,
                        "primary_span_id": span.get("span_id"),
                        "span_ids": [span.get("span_id")],
                        "gram_set": gram_set,
                        "dedup_count": 1,
                        "first_idx": idx,
                    }
                )

            for group_idx, group in enumerate(groups, start=1):
                fingerprint_basis = sorted(group["gram_set"])[:512]
                fingerprint = hash_key(
                    "issue-fingerprint-v3",
                    group["phase"],
                    group["status"],
                    fingerprint_basis,
                )
                db.add(
                    Issue(
                        run_id=run_uuid,
                        fingerprint=fingerprint[:120],
                        title=(group.get("title") or f"{group['title_prefix']} #{group_idx}")[:255],
                        summary=(group["summary"] or "")[:2000],
                        severity=group["severity"],
                        confidence=group["confidence"],
                        status=IssueStatus.open,
                        metadata_={
                            "phase": group["phase"],
                            "span_id": group["primary_span_id"],
                            "status": group["status"],
                            "before_excerpt": group["before_excerpt"],
                            "after_excerpt": group["after_excerpt"],
                            "before_preview": group.get("before_preview", ""),
                            "after_preview": group.get("after_preview", ""),
                            "merged_span_ids": [sid for sid in group["span_ids"] if sid][:50],
                            "dedup_count": group["dedup_count"],
                            "fingerprint_basis": {
                                "version": "v3",
                                "phase": group["phase"],
                                "status": group["status"],
                                "dedup_similarity_threshold": dedup_similarity_threshold,
                            },
                        },
                    )
                )
                created += 1

            return {
                "issue_count": created,
                "dedup_merged_count": dedup_merged_count,
                "dedup_similarity_threshold": dedup_similarity_threshold,
            }

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def attach_evidence_to_issues(self, run_id: str, retry_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "attach_evidence_to_issues"

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)
        issue_rows = (
            db.query(Issue.id, Issue.fingerprint, Issue.status, Issue.updated_at)
            .filter(Issue.run_id == run_uuid)
            .order_by(Issue.id.asc())
            .all()
        )
        chunk_rows_sig = (
            db.query(SourceChunk.id, SourceChunk.source_doc_id, SourceChunk.chunk_index)
            .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
            .filter(SourceDoc.run_id == run_uuid)
            .order_by(SourceChunk.id.asc())
            .all()
        )
        existing_evidence_rows = (
            db.query(IssueEvidence.id, IssueEvidence.issue_id, IssueEvidence.citation_id)
            .join(Issue, IssueEvidence.issue_id == Issue.id)
            .filter(Issue.run_id == run_uuid)
            .order_by(IssueEvidence.id.asc())
            .all()
        )
        retry_search_ids = (retry_context or {}).get("retry_search_ids") or []
        latest_retry_search_id = str(retry_search_ids[-1]) if retry_search_ids else None
        latest_retry_search_uuid: uuid.UUID | None = None
        if latest_retry_search_id:
            try:
                latest_retry_search_uuid = uuid.UUID(latest_retry_search_id)
            except ValueError:
                latest_retry_search_uuid = None
        retry_search_result_rows = []
        if latest_retry_search_uuid is not None:
            retry_search_result_rows = (
                db.query(SearchResult.chunk_id, SearchResult.rank, SearchResult.score)
                .filter(SearchResult.search_id == latest_retry_search_uuid)
                .order_by(SearchResult.rank.asc(), SearchResult.chunk_id.asc())
                .all()
            )
        idempotency_key = stage_idempotency_key(
            run_id=run_id,
            stage_name=stage_name,
            stage_input_fingerprint={
                "issue_signature_hash": digest_rows(issue_rows),
                "issue_count": len(issue_rows),
                "source_chunk_signature_hash": digest_rows(chunk_rows_sig),
                "source_chunk_count": len(chunk_rows_sig),
                "existing_evidence_signature_hash": digest_rows(existing_evidence_rows),
                "existing_evidence_count": len(existing_evidence_rows),
                "latest_retry_search_id": latest_retry_search_id,
                "latest_retry_search_result_signature_hash": digest_rows(retry_search_result_rows),
                "latest_retry_search_result_count": len(retry_search_result_rows),
            },
        )

        def work() -> Dict[str, Any]:
            issues = (
                db.query(Issue)
                .filter(Issue.run_id == run_uuid)
                .order_by(Issue.severity.desc(), Issue.created_at.asc())
                .all()
            )
            preferred_search_id = latest_retry_search_id
            if latest_retry_search_uuid is not None:
                chunk_candidates = [
                    {
                        "chunk": chunk,
                        "source_doc": source_doc,
                        "search_score": score,
                        "search_rank": rank,
                        "candidate_source": "search_result",
                    }
                    for chunk, source_doc, score, rank in (
                        db.query(SourceChunk, SourceDoc, SearchResult.score, SearchResult.rank)
                        .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
                        .join(SearchResult, SearchResult.chunk_id == SourceChunk.id)
                        .filter(SourceDoc.run_id == run_uuid, SearchResult.search_id == latest_retry_search_uuid)
                        .order_by(SearchResult.rank.asc(), SourceChunk.id.asc())
                        .all()
                    )
                ]
            else:
                chunk_candidates = []
            if not chunk_candidates:
                chunk_candidates = [
                    {
                        "chunk": chunk,
                        "source_doc": source_doc,
                        "search_score": None,
                        "search_rank": None,
                        "candidate_source": "run_chunks",
                    }
                    for chunk, source_doc in (
                        db.query(SourceChunk, SourceDoc)
                        .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
                        .filter(SourceDoc.run_id == run_uuid)
                        .order_by(SourceDoc.created_at.asc(), SourceChunk.chunk_index.asc())
                        .all()
                    )
                ]
            run = get_run_or_raise(db, run_uuid)

            attached = 0
            hidden_issue_ids: list[str] = []
            hidden_reason_counts: dict[str, int] = {}
            hidden_issue_reasons: list[dict[str, str]] = []

            def record_hidden(issue_id: uuid.UUID, reason: str) -> None:
                hidden_issue_ids.append(str(issue_id))
                hidden_reason_counts[reason] = hidden_reason_counts.get(reason, 0) + 1
                hidden_issue_reasons.append({"issue_id": str(issue_id), "reason": reason})

            for issue in issues:
                existing = db.query(IssueEvidence).filter(IssueEvidence.issue_id == issue.id).count()
                if existing > 0:
                    issue.status = IssueStatus.open
                    attached += 1
                    continue

                hint = f"{issue.title}\n{issue.summary}"
                before_excerpt = issue.metadata_.get("before_excerpt", "")
                after_excerpt = issue.metadata_.get("after_excerpt", "")
                best = choose_best_chunk(chunk_candidates, f"{hint}\n{before_excerpt}\n{after_excerpt}")
                best_chunk = best["chunk"] if best else None
                if best_chunk is None:
                    issue.status = IssueStatus.hidden
                    record_hidden(issue.id, "no_candidate_chunk")
                    continue
                if not best_chunk.loc:
                    issue.status = IssueStatus.hidden
                    record_hidden(issue.id, "missing_chunk_loc")
                    continue

                source_doc = best.get("source_doc") if best else None
                if source_doc is None:
                    issue.status = IssueStatus.hidden
                    record_hidden(issue.id, "missing_source_doc")
                    continue

                citation = Citation(
                    feedback_id=None,
                    source_doc_id=source_doc.id,
                    chunk_id=best_chunk.id,
                    span={
                        "loc": best_chunk.loc,
                        "scope": "issue",
                        "selection": (best or {}).get("selection_detail"),
                        "search_id": preferred_search_id,
                    },
                )
                db.add(citation)
                db.flush()

                evidence = IssueEvidence(
                    issue_id=issue.id,
                    citation_id=citation.id,
                    before_excerpt=before_excerpt or None,
                    after_excerpt=after_excerpt or None,
                    loc=best_chunk.loc,
                )
                db.add(evidence)
                issue.status = IssueStatus.open
                attached += 1

            issue_total = len(issues)
            hidden_count = len(hidden_issue_ids)
            if issue_total == 0:
                run.status = RunStatus.success
            elif attached == 0:
                summary = (
                    "all issues hidden after evidence attachment: "
                    f"issue_total={issue_total}, hidden_count={hidden_count}, "
                    f"reason_counts={hidden_reason_counts}, search_id={preferred_search_id or '-'}"
                )
                raise StageFailureError(
                    "all issues are hidden because evidence attachment failed",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={
                        "summary": summary,
                        "issue_total": issue_total,
                        "hidden_issue_ids": hidden_issue_ids,
                        "hidden_issue_reasons": hidden_issue_reasons[:50],
                        "hidden_reason_counts": hidden_reason_counts,
                        "source_chunk_count": len(chunk_candidates),
                        "selection_strategy": "search_score_weighted_v1",
                        "selection_search_id": preferred_search_id,
                        "retry_context": retry_context or {},
                    },
                )
            elif attached < issue_total:
                run.status = RunStatus.success_partial
            else:
                run.status = RunStatus.success

            return {
                "issue_with_evidence_count": attached,
                "issue_total": issue_total,
                "hidden_issue_count": hidden_count,
                "hidden_reason_counts": hidden_reason_counts,
                "selection_strategy": "search_score_weighted_v1",
                "selection_search_id": preferred_search_id,
                "retry_context": retry_context or {},
            }

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref
