import hashlib
import io
import json
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
PROMPT_TEMPLATE_VERSION = "v1"


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


def normalize_fingerprint_text(text: str) -> str:
    cleaned = " ".join((text or "").lower().split())
    return cleaned[:512]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


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


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_low_draft(self, run_id: str, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "generate_low_draft"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"task": "low_draft"},
        model_cfg=model_cfg or {},
    )

    with session_scope() as db:
        run = get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.low_draft,
                ArtifactFormat.markdown,
                content_text=(
                    f"# Low Draft\n\nModel: {(model_cfg or {}).get('model', settings.model_low)}\n\n"
                    f"Prompt:\n{run.task_prompt}\n\n- Draft bullet 1\n- Draft bullet 2\n"
                ),
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
                if mode == "keyword":
                    if q.lower() in text.lower():
                        score = len(q) / max(len(text), 1)
                elif mode == "regex":
                    if rx and rx.search(text):
                        score = 1.0
                else:
                    score = 0.1 if q.lower() in text.lower() else 0.0

                if score > min_score:
                    scored.append(
                        {
                            "source_doc_id": source_doc.id,
                            "chunk_id": chunk.id,
                            "snippet": text[:240],
                            "score": float(score),
                            "payload": {"loc": chunk.loc},
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


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_feedback_with_citations(self, run_id: str, target_artifact_id: str) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    target_uuid = uuid.UUID(target_artifact_id)
    stage_name = "generate_feedback_with_citations"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"target_artifact_id": target_artifact_id},
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            target = db.get(Artifact, target_uuid)
            if target is None:
                raise ValueError("target artifact not found")

            chunk = (
                db.query(SourceChunk, SourceDoc)
                .join(SourceDoc, SourceChunk.source_doc_id == SourceDoc.id)
                .filter(SourceDoc.run_id == run_uuid)
                .order_by(SourceChunk.chunk_index.asc())
                .first()
            )
            if chunk is None:
                raise StageFailureError(
                    "no source chunk found for citation-backed feedback",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={"run_id": str(run_uuid), "required": "source_chunk"},
                )

            source_chunk, source_doc = chunk
            feedback = FeedbackItem(
                run_id=run_uuid,
                target_artifact_id=target.id,
                text="根拠文書に照らして、主張の具体例を1つ追加してください。",
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
                    "feedback must include at least one citation",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={"feedback_id": str(feedback.id)},
                )

            return {"feedback_id": str(feedback.id), "citation_count": citation_count}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def apply_feedback(self, run_id: str, base_artifact_id: str) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    base_uuid = uuid.UUID(base_artifact_id)
    stage_name = "apply_feedback"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"base_artifact_id": base_artifact_id},
    )

    with session_scope() as db:
        get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            base = db.get(Artifact, base_uuid)
            if base is None:
                raise ValueError("base artifact not found")

            feedbacks = (
                db.query(FeedbackItem)
                .filter(FeedbackItem.run_id == run_uuid, FeedbackItem.target_artifact_id == base.id)
                .all()
            )
            applied_lines = [f"- {fb.text}" for fb in feedbacks]
            improved_text = (base.content_text or "") + "\n\n## Improvements from Feedback\n" + "\n".join(applied_lines)
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.improved,
                ArtifactFormat.markdown,
                content_text=improved_text,
            )
            return {"artifact_id": str(artifact.id)}

        return execute_stage(
            db,
            run_id=run_uuid,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            fn=work,
        ).output_ref


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_high_final(self, run_id: str, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_uuid = uuid.UUID(run_id)
    stage_name = "generate_high_final"
    idempotency_key = stage_idempotency_key(
        run_id=run_id,
        stage_name=stage_name,
        stage_input_fingerprint={"task": "high_final"},
        model_cfg=model_cfg or {},
    )

    with session_scope() as db:
        run = get_run_or_raise(db, run_uuid)

        def work() -> Dict[str, Any]:
            source = (
                db.query(Artifact)
                .filter(Artifact.run_id == run_uuid, Artifact.kind == ArtifactKind.improved)
                .order_by(Artifact.created_at.desc())
                .first()
            )
            body = source.content_text if source else run.task_prompt
            artifact = create_artifact(
                db,
                run_uuid,
                ArtifactKind.high_final,
                ArtifactFormat.markdown,
                content_text=(
                    f"# High Final\n\nModel: {(model_cfg or {}).get('model', settings.model_high)}\n\n"
                    f"{body}\n\n## Final polish\n- Added concise summary.\n"
                ),
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

    feedback_result = generate_feedback_with_citations.run(run_id, low_artifact_id)

    improved_result = apply_feedback.run(run_id, low_artifact_id)
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
                spans.append(
                    {
                        "span_id": f"li-{idx + 1}",
                        "phase": "low_to_improved",
                        "status": span["status"],
                        "before_excerpt": excerpt(span.get("a_text", "")),
                        "after_excerpt": excerpt(span.get("b_text", "")),
                        "score": float(span.get("score", 0.0)),
                    }
                )
            for idx, span in enumerate(improved_vs_high):
                if span["status"] == "unchanged":
                    continue
                spans.append(
                    {
                        "span_id": f"ih-{idx + 1}",
                        "phase": "improved_to_high",
                        "status": span["status"],
                        "before_excerpt": excerpt(span.get("a_text", "")),
                        "after_excerpt": excerpt(span.get("b_text", "")),
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
            for idx, span in enumerate(spans):
                status = span.get("status", "modified")
                title_prefix = {
                    "modified": "変更検知",
                    "added": "追加検知",
                    "removed": "削除検知",
                }.get(status, "論点候補")
                before_excerpt = span.get("before_excerpt") or ""
                after_excerpt = span.get("after_excerpt") or ""
                summary = f"{title_prefix}: {after_excerpt or before_excerpt}"
                severity = 3 if status in {"modified", "removed"} else 2
                confidence = min(0.99, max(0.4, float(span.get("score", 0.5))))
                phase = span.get("phase", "")
                fingerprint = hash_key(
                    "issue-fingerprint-v2",
                    phase,
                    status,
                    normalize_fingerprint_text(before_excerpt),
                    normalize_fingerprint_text(after_excerpt),
                )

                db.add(
                    Issue(
                        run_id=run_uuid,
                        fingerprint=fingerprint[:120],
                        title=f"{title_prefix} #{idx + 1}",
                        summary=summary[:2000],
                        severity=severity,
                        confidence=confidence,
                        status=IssueStatus.open,
                        metadata_={
                            "phase": span.get("phase"),
                            "span_id": span.get("span_id"),
                            "status": status,
                            "before_excerpt": before_excerpt,
                            "after_excerpt": after_excerpt,
                            "fingerprint_basis": {
                                "version": "v2",
                                "phase": phase,
                                "status": status,
                            },
                        },
                    )
                )
                created += 1

            return {"issue_count": created}

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
                if best_chunk is None or not best_chunk.loc:
                    issue.status = IssueStatus.hidden
                    hidden_issue_ids.append(str(issue.id))
                    continue

                source_doc = best.get("source_doc") if best else None
                if source_doc is None:
                    issue.status = IssueStatus.hidden
                    hidden_issue_ids.append(str(issue.id))
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
                raise StageFailureError(
                    "all issues are hidden because evidence attachment failed",
                    failure_type=StageFailureType.evidence_insufficient,
                    failure_detail={
                        "issue_total": issue_total,
                        "hidden_issue_ids": hidden_issue_ids,
                        "source_chunk_count": len(chunk_candidates),
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
