import hashlib
import json
import re
import uuid
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import redis
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app import tasks
from app.auth import AuthContext, authenticate_user, issue_access_token, require_auth
from app.config import get_settings
from app.database import get_db
from app.llm_client import (
    LLMClientConfigError,
    LLMClientError,
    LLMClientResponseError,
    LLMClientTransientError,
    chat_complete,
    provider_is_stub,
)
from app.logging import configure_logging
from app.models import (
    Artifact,
    ArtifactKind,
    Citation,
    FeedbackItem,
    Issue,
    IssueEvidence,
    Run,
    RunStage,
    RunStageAttempt,
    SearchRequest,
    SearchStatus,
    SourceChunk,
    SourceDoc,
    SourceOrigin,
    IssueStatus,
)
from app.s3_client import ensure_bucket_exists, get_s3_client, object_exists, presign_get, presign_put
from app.schemas import (
    ArtifactResponse,
    AuthTokenRequest,
    AuthTokenResponse,
    CompareRowResponse,
    CitationDetailResponse,
    FeedbackResponse,
    IngestRequest,
    IngestResponse,
    IssueCompareContextResponse,
    IssueDetailResponse,
    IssueEvidenceResponse,
    IssueWritingGuidanceResponse,
    IssueResponse,
    PipelineRequest,
    PipelineResponse,
    PresignGetResponse,
    PresignPutRequest,
    PresignPutResponse,
    RetrySummary,
    RunCompareResponse,
    RunCreate,
    RunListItemResponse,
    RunMetricsResponse,
    RunResponse,
    ScoreHistogramBucket,
    SelectionScoreSummary,
    RunStageAttemptResponse,
    RunStageResponse,
    SearchRequestCreate,
    SearchResponse,
    SourceChunkPreviewResponse,
    SourceDocSummaryResponse,
)

settings = get_settings()
configure_logging(service="api")
app = FastAPI(title="diffUI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_checks() -> None:
    ensure_bucket_exists()


def search_idempotency_key(run_id: uuid.UUID, query: str, filters: Dict[str, Any], mode: str) -> str:
    raw = json.dumps({"run_id": str(run_id), "query": query, "filters": filters, "mode": mode}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_run_status_for_api(status_value):
    if getattr(status_value, "value", None) == "failed":
        return "failed_legacy"
    return status_value


def run_public_title(run: Run) -> str:
    title = run.metadata_.get("title") if isinstance(run.metadata_, dict) else None
    if isinstance(title, str) and title.strip():
        return title.strip()[:120]
    collapsed = " ".join((run.task_prompt or "").split())
    return (collapsed or "(untitled)")[:120]


def build_object_key(run_id: uuid.UUID, source_doc_id: uuid.UUID, filename: str) -> str:
    safe = Path(filename).name
    return f"runs/{run_id}/sources/{source_doc_id}/raw/{safe}"


def cleanup_orphan_source_docs(db: Session) -> int:
    """
    Remove stale source_docs created by presign but never uploaded/ingested.
    """
    cutoff = tasks.now_utc() - timedelta(hours=settings.source_doc_orphan_ttl_hours)
    candidates = (
        db.query(SourceDoc)
        .outerjoin(SourceChunk, SourceChunk.source_doc_id == SourceDoc.id)
        .filter(SourceChunk.id.is_(None), SourceDoc.created_at < cutoff)
        .all()
    )
    removed = 0
    for source_doc in candidates:
        if not object_exists(source_doc.object_key):
            db.delete(source_doc)
            removed += 1
    if removed > 0:
        db.flush()
    return removed


def ensure_audit_visibility_allowed(include_hidden: bool, subject: AuthContext) -> None:
    if not include_hidden:
        return
    if not settings.auth_enabled:
        return
    if not subject.has_any_role(("audit", "admin")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="include_hidden requires audit privilege")


def parse_selection_score(selection: Any) -> float | None:
    if not isinstance(selection, dict):
        return None
    raw = selection.get("combined_score")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def evidence_selection_score(evidence: IssueEvidence) -> float | None:
    citation = evidence.citation
    if citation is None or not isinstance(citation.span, dict):
        return None
    return parse_selection_score(citation.span.get("selection"))


def percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    ratio = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * ratio


def make_score_histogram(values: List[float]) -> List[ScoreHistogramBucket]:
    ranges = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]
    buckets: List[ScoreHistogramBucket] = []
    for start, end in ranges:
        if end >= 1.0:
            count = sum(1 for score in values if start <= score <= end)
        else:
            count = sum(1 for score in values if start <= score < end)
        buckets.append(ScoreHistogramBucket(range_start=start, range_end=end, count=count))
    return buckets


def build_issue_rows(
    db: Session,
    run_id: uuid.UUID,
    *,
    include_hidden: bool,
    include_zero_evidence: bool,
) -> List[IssueResponse]:
    issues = (
        db.query(Issue)
        .options(joinedload(Issue.evidences).joinedload(IssueEvidence.citation))
        .filter(Issue.run_id == run_id)
        .order_by(Issue.severity.desc(), Issue.created_at.asc())
        .all()
    )
    rows: List[IssueResponse] = []
    for issue in issues:
        evidence_count = len(issue.evidences)
        if not include_hidden and issue.status.value == "hidden":
            continue
        if not include_zero_evidence and evidence_count < 1:
            continue
        top_score = None
        for evidence in issue.evidences:
            score = evidence_selection_score(evidence)
            if score is None:
                continue
            if top_score is None or score > top_score:
                top_score = score
        rows.append(
            IssueResponse(
                id=issue.id,
                run_id=issue.run_id,
                title=issue.title,
                summary=issue.summary,
                severity=issue.severity,
                confidence=issue.confidence,
                status=issue.status,
                evidence_count=evidence_count,
                top_evidence_score=top_score,
                created_at=issue.created_at,
                updated_at=issue.updated_at,
            )
        )
    return rows


def to_issue_evidence_response(
    db: Session,
    evidence: IssueEvidence,
    *,
    task_prompt: str,
    include_guidance: bool,
) -> IssueEvidenceResponse | None:
    citation = db.get(Citation, evidence.citation_id)
    if citation is None:
        return None
    source_doc = db.get(SourceDoc, citation.source_doc_id)
    chunk = db.get(SourceChunk, citation.chunk_id)
    if source_doc is None or chunk is None:
        return None
    resolved_loc = chunk.loc or evidence.loc or {}
    selection = None
    if isinstance(citation.span, dict):
        candidate = citation.span.get("selection")
        if isinstance(candidate, dict):
            selection = candidate
    guidance: IssueWritingGuidanceResponse | None = None
    if include_guidance:
        if isinstance(citation.span, dict):
            cached = citation.span.get("writing_guidance")
            if isinstance(cached, dict):
                try:
                    guidance = IssueWritingGuidanceResponse(**cached)
                except Exception:
                    guidance = None
        if guidance is None:
            guidance = generate_issue_writing_guidance(
                task_prompt=task_prompt,
                before_text=evidence.before_excerpt or "",
                after_text=evidence.after_excerpt or "",
            )
    return IssueEvidenceResponse(
        id=evidence.id,
        issue_id=evidence.issue_id,
        citation_id=evidence.citation_id,
        source_doc_id=citation.source_doc_id,
        chunk_id=citation.chunk_id,
        citation_span=citation.span,
        selection=selection,
        before_excerpt=evidence.before_excerpt,
        after_excerpt=evidence.after_excerpt,
        loc=resolved_loc,
        source_title=source_doc.title,
        source_presigned_url=presign_get(source_doc.object_key),
        chunk_text=chunk.text,
        chunk_loc=chunk.loc,
        writing_guidance=guidance,
    )


def normalize_inline_text(text: str, *, limit: int = 220) -> str:
    value = " ".join((text or "").split())
    return value[:limit]


def dedup_queries(values: List[str], *, max_items: int = 12) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized = normalize_inline_text(raw)
        key = normalized.lower()
        if len(key) < 8:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= max_items:
            break
    return result


def first_nonempty_line(text: str, *, limit: int = 180) -> str:
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def fallback_issue_writing_guidance(before_text: str, after_text: str) -> IssueWritingGuidanceResponse:
    left_example = first_nonempty_line(before_text) or "現状の意図を1行で明確に書く。"
    right_example = first_nonempty_line(after_text) or first_nonempty_line(before_text) or "変更後の判断条件を1行で書く。"
    return IssueWritingGuidanceResponse(
        left_point="左の書き方のポイント: 現状の意図と前提を端的に示す。",
        left_example=f"具体例: {left_example}",
        right_point="右の書き方のポイント: 結論先行で変更後の判断条件と実行内容を示す。",
        right_example=f"具体例: {right_example}",
        strategy="fallback",
    )


def parse_json_object_from_text(raw_text: str) -> Dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def generate_issue_writing_guidance(
    *,
    task_prompt: str,
    before_text: str,
    after_text: str,
) -> IssueWritingGuidanceResponse:
    fallback = fallback_issue_writing_guidance(before_text, after_text)
    if provider_is_stub():
        return fallback

    try:
        raw = chat_complete(
            model=settings.model_high,
            system_prompt=(
                "あなたは日本語テクニカルライターです。"
                "与えられた差分の left(現状) と right(改善案) について、"
                "書き方のポイント(抽象)と具体例(具体)を作成してください。"
                "出力はJSONのみ。余計な文章は出力しないこと。"
            ),
            user_prompt=(
                "task_prompt:\n"
                f"{(task_prompt or '')[:500]}\n\n"
                "left(before):\n"
                f"{(before_text or '')[:1200]}\n\n"
                "right(after):\n"
                f"{(after_text or '')[:1200]}\n\n"
                "JSON schema:\n"
                "{\n"
                '  "left_point": "左の書き方ポイント(1文)",\n'
                '  "left_example": "左の具体例(1文)",\n'
                '  "right_point": "右の書き方ポイント(1文)",\n'
                '  "right_example": "右の具体例(1文)"\n'
                "}\n"
                "制約:\n"
                "- それぞれ日本語で簡潔に\n"
                "- left/right の意味を混同しない\n"
                "- 一文が長すぎない\n"
            ),
            temperature=0.2,
            max_tokens=280,
        )
    except (LLMClientConfigError, LLMClientTransientError, LLMClientResponseError, LLMClientError):
        return fallback

    parsed = parse_json_object_from_text(raw)
    if parsed is None:
        return fallback

    def pick(key: str, default_value: str) -> str:
        value = parsed.get(key)
        if not isinstance(value, str):
            return default_value
        normalized = " ".join(value.split())
        return normalized[:220] if normalized else default_value

    return IssueWritingGuidanceResponse(
        left_point=pick("left_point", fallback.left_point),
        left_example=pick("left_example", fallback.left_example),
        right_point=pick("right_point", fallback.right_point),
        right_example=pick("right_example", fallback.right_example),
        strategy="ai",
    )


def build_compare_rows(left_text: str, right_text: str) -> List[CompareRowResponse]:
    left_lines = (left_text or "").splitlines()
    right_lines = (right_text or "").splitlines()
    matcher = SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)

    rows: List[CompareRowResponse] = []
    left_no = 1
    right_no = 1
    row_no = 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_block = left_lines[i1:i2]
        right_block = right_lines[j1:j2]

        if tag == "equal":
            for idx in range(len(left_block)):
                rows.append(
                    CompareRowResponse(
                        row_no=row_no,
                        kind="equal",
                        left_line_no=left_no,
                        left_text=left_block[idx],
                        right_line_no=right_no,
                        right_text=right_block[idx],
                    )
                )
                left_no += 1
                right_no += 1
                row_no += 1
            continue

        block_len = max(len(left_block), len(right_block))
        for idx in range(block_len):
            has_left = idx < len(left_block)
            has_right = idx < len(right_block)
            if tag == "replace":
                kind = "replace" if has_left and has_right else ("delete" if has_left else "insert")
            elif tag == "delete":
                kind = "delete"
            else:
                kind = "insert"

            rows.append(
                CompareRowResponse(
                    row_no=row_no,
                    kind=kind,
                    left_line_no=left_no if has_left else None,
                    left_text=left_block[idx] if has_left else "",
                    right_line_no=right_no if has_right else None,
                    right_text=right_block[idx] if has_right else "",
                )
            )
            if has_left:
                left_no += 1
            if has_right:
                right_no += 1
            row_no += 1

    return rows


def to_ingest_status(chunk_count: int) -> str:
    return "ready" if chunk_count > 0 else "pending"


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(db: Session = Depends(get_db)) -> Dict[str, str]:
    checks = {"database": "ok", "redis": "ok", "s3": "ok"}

    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    try:
        redis.from_url(settings.redis_url).ping()
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    try:
        client = get_s3_client()
        client.head_bucket(Bucket=settings.s3_bucket)
    except Exception as exc:
        checks["s3"] = f"error: {exc}"

    if any(value != "ok" for value in checks.values()):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=checks)
    return {"status": "ready"}


@app.post("/auth/token", response_model=AuthTokenResponse)
def auth_token(payload: AuthTokenRequest) -> AuthTokenResponse:
    if not settings.auth_enabled:
        roles = ("admin", "audit", "viewer")
        return AuthTokenResponse(access_token=issue_access_token("anonymous", roles), roles=list(roles))
    ctx = authenticate_user(payload.username, payload.password)
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    return AuthTokenResponse(access_token=issue_access_token(ctx.subject, ctx.roles), roles=list(ctx.roles))


@app.get("/runs", response_model=List[RunListItemResponse])
def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[RunListItemResponse]:
    runs = (
        db.query(Run)
        .order_by(Run.created_at.desc())
        .limit(limit)
        .all()
    )
    rows: List[RunListItemResponse] = []
    for run in runs:
        source_count = db.query(func.count(SourceDoc.id)).filter(SourceDoc.run_id == run.id).scalar() or 0
        issue_count = db.query(func.count(Issue.id)).filter(Issue.run_id == run.id).scalar() or 0
        visible_issue_count = (
            db.query(func.count(func.distinct(Issue.id)))
            .join(IssueEvidence, IssueEvidence.issue_id == Issue.id)
            .filter(Issue.run_id == run.id, Issue.status != IssueStatus.hidden)
            .scalar()
            or 0
        )
        rows.append(
            RunListItemResponse(
                id=run.id,
                created_at=run.created_at,
                title=run_public_title(run),
                status=normalize_run_status_for_api(run.status),
                source_count=int(source_count),
                issue_count=int(issue_count),
                visible_issue_count=int(visible_issue_count),
            )
        )
    return rows


@app.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
) -> RunResponse:
    run = Run(task_prompt=payload.task_prompt, metadata_=payload.metadata)
    db.add(run)
    db.commit()
    db.refresh(run)
    return RunResponse(
        id=run.id,
        created_at=run.created_at,
        task_prompt=run.task_prompt,
        status=normalize_run_status_for_api(run.status),
        metadata=run.metadata_,
    )


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> RunResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return RunResponse(
        id=run.id,
        created_at=run.created_at,
        task_prompt=run.task_prompt,
        status=normalize_run_status_for_api(run.status),
        metadata=run.metadata_,
    )


@app.get("/runs/{run_id}/sources", response_model=List[SourceDocSummaryResponse])
def list_run_sources(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[SourceDocSummaryResponse]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    rows = (
        db.query(SourceDoc, func.count(SourceChunk.id).label("chunk_count"))
        .outerjoin(SourceChunk, SourceChunk.source_doc_id == SourceDoc.id)
        .filter(SourceDoc.run_id == run_id)
        .group_by(SourceDoc.id)
        .order_by(SourceDoc.created_at.asc())
        .all()
    )
    return [
        SourceDocSummaryResponse(
            id=source_doc.id,
            run_id=source_doc.run_id,
            title=source_doc.title,
            content_type=source_doc.content_type,
            object_key=source_doc.object_key,
            chunk_count=int(chunk_count or 0),
            ingest_status=to_ingest_status(int(chunk_count or 0)),
            created_at=source_doc.created_at,
        )
        for source_doc, chunk_count in rows
    ]


@app.get("/runs/{run_id}/stages", response_model=List[RunStageResponse])
def list_run_stages(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[RunStageResponse]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return (
        db.query(RunStage)
        .filter(RunStage.run_id == run_id)
        .order_by(RunStage.started_at.asc().nullslast(), RunStage.stage_name.asc(), RunStage.attempt.asc())
        .all()
    )


@app.get("/runs/{run_id}/stage-attempts", response_model=List[RunStageAttemptResponse])
def list_run_stage_attempts(
    run_id: uuid.UUID,
    stage_name: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[RunStageAttempt]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    q = db.query(RunStageAttempt).filter(RunStageAttempt.run_id == run_id)
    if stage_name:
        q = q.filter(RunStageAttempt.stage_name == stage_name)
    return (
        q.order_by(
            RunStageAttempt.started_at.asc(),
            RunStageAttempt.stage_name.asc(),
            RunStageAttempt.attempt_no.asc(),
        )
        .all()
    )


@app.get("/runs/{run_id}/metrics", response_model=RunMetricsResponse)
def get_run_metrics(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> RunMetricsResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    issues = (
        db.query(Issue)
        .options(joinedload(Issue.evidences).joinedload(IssueEvidence.citation))
        .filter(Issue.run_id == run_id)
        .all()
    )
    issue_total = len(issues)
    hidden_issue_count = sum(1 for issue in issues if issue.status.value == "hidden")
    visible_issue_count = issue_total - hidden_issue_count
    hidden_rate = round(hidden_issue_count / issue_total, 6) if issue_total > 0 else 0.0

    selection_scores: List[float] = []
    for issue in issues:
        for evidence in issue.evidences:
            score = evidence_selection_score(evidence)
            if score is not None:
                selection_scores.append(score)

    avg_score = round(sum(selection_scores) / len(selection_scores), 6) if selection_scores else None
    min_score = round(min(selection_scores), 6) if selection_scores else None
    max_score = round(max(selection_scores), 6) if selection_scores else None
    p50_score = percentile(selection_scores, 0.5)
    p90_score = percentile(selection_scores, 0.9)

    attempts = (
        db.query(RunStageAttempt)
        .filter(RunStageAttempt.run_id == run_id)
        .order_by(RunStageAttempt.run_stage_id.asc(), RunStageAttempt.attempt_no.asc())
        .all()
    )
    per_stage_attempts: Dict[str, List[RunStageAttempt]] = {}
    for row in attempts:
        per_stage_attempts.setdefault(str(row.run_stage_id), []).append(row)

    retried_stage_count = 0
    retry_success_count = 0
    retry_failure_count = 0
    for rows in per_stage_attempts.values():
        if len(rows) <= 1:
            continue
        retried_stage_count += 1
        if any(row.status.value == "success" for row in rows):
            retry_success_count += 1
        else:
            retry_failure_count += 1

    retry_cases = retry_success_count + retry_failure_count
    retry_success_rate = round(retry_success_count / retry_cases, 6) if retry_cases > 0 else None

    return RunMetricsResponse(
        run_id=run_id,
        issue_total=issue_total,
        visible_issue_count=visible_issue_count,
        hidden_issue_count=hidden_issue_count,
        hidden_rate=hidden_rate,
        selection_score=SelectionScoreSummary(
            count=len(selection_scores),
            avg=avg_score,
            min=min_score,
            max=max_score,
            p50=round(p50_score, 6) if p50_score is not None else None,
            p90=round(p90_score, 6) if p90_score is not None else None,
            histogram=make_score_histogram(selection_scores),
        ),
        retry=RetrySummary(
            retried_stage_count=retried_stage_count,
            retry_success_count=retry_success_count,
            retry_failure_count=retry_failure_count,
            retry_success_rate=retry_success_rate,
        ),
    )


@app.get("/runs/{run_id}/artifacts", response_model=List[ArtifactResponse])
def list_artifacts(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[Artifact]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id)
        .order_by(Artifact.created_at.asc())
        .all()
    )


@app.get("/runs/{run_id}/issues", response_model=List[IssueResponse])
def list_issues(
    run_id: uuid.UUID,
    include_hidden: bool = Query(default=False),
    db: Session = Depends(get_db),
    subject: AuthContext = Depends(require_auth),
) -> List[IssueResponse]:
    ensure_audit_visibility_allowed(include_hidden, subject)
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    return build_issue_rows(db, run_id, include_hidden=include_hidden, include_zero_evidence=include_hidden)


@app.get("/runs/{run_id}/audit/issues", response_model=List[IssueResponse])
def list_audit_issues(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    subject: AuthContext = Depends(require_auth),
) -> List[IssueResponse]:
    ensure_audit_visibility_allowed(True, subject)
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return build_issue_rows(db, run_id, include_hidden=True, include_zero_evidence=True)


@app.get("/issues/{issue_id}", response_model=IssueDetailResponse)
def get_issue(
    issue_id: uuid.UUID,
    include_guidance: bool = Query(default=False),
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> IssueDetailResponse:
    issue = (
        db.query(Issue)
        .options(joinedload(Issue.evidences))
        .filter(Issue.id == issue_id)
        .one_or_none()
    )
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="issue not found")
    run = db.get(Run, issue.run_id)
    task_prompt = run.task_prompt if run is not None else ""

    evidence_rows: List[IssueEvidenceResponse] = []
    for evidence in issue.evidences:
        row = to_issue_evidence_response(
            db,
            evidence,
            task_prompt=task_prompt,
            include_guidance=include_guidance,
        )
        if row is not None:
            evidence_rows.append(row)

    return IssueDetailResponse(
        id=issue.id,
        run_id=issue.run_id,
        title=issue.title,
        summary=issue.summary,
        severity=issue.severity,
        confidence=issue.confidence,
        status=issue.status,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
        evidences=evidence_rows,
    )


@app.get("/issues/{issue_id}/compare-context", response_model=IssueCompareContextResponse)
def get_issue_compare_context(
    issue_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> IssueCompareContextResponse:
    issue = (
        db.query(Issue)
        .options(joinedload(Issue.evidences))
        .filter(Issue.id == issue_id)
        .one_or_none()
    )
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="issue not found")

    before_candidates = [ev.before_excerpt or "" for ev in issue.evidences]
    after_candidates = [ev.after_excerpt or "" for ev in issue.evidences]
    before_queries = dedup_queries(before_candidates, max_items=10)
    after_queries = dedup_queries(after_candidates + [issue.summary], max_items=10)
    highlight_queries = dedup_queries(
        [issue.title, issue.summary, *before_queries, *after_queries],
        max_items=16,
    )

    return IssueCompareContextResponse(
        issue_id=issue.id,
        run_id=issue.run_id,
        before_queries=before_queries,
        after_queries=after_queries,
        highlight_queries=highlight_queries,
    )


@app.get("/runs/{run_id}/compare", response_model=RunCompareResponse)
def get_run_compare(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> RunCompareResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    left_artifact = (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id, Artifact.kind == ArtifactKind.high_final)
        .order_by(Artifact.created_at.desc())
        .first()
    )
    right_artifact = (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id, Artifact.kind == ArtifactKind.improved)
        .order_by(Artifact.created_at.desc())
        .first()
    )

    if left_artifact is None or right_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="compare artifacts are not ready (requires high_final and improved)",
        )

    rows = build_compare_rows(left_artifact.content_text or "", right_artifact.content_text or "")
    return RunCompareResponse(
        run_id=run_id,
        left_label="High model one-shot",
        right_label="Low draft + high feedback",
        left_artifact_id=left_artifact.id,
        right_artifact_id=right_artifact.id,
        rows=rows,
    )


@app.get("/runs/{run_id}/feedback", response_model=List[FeedbackResponse])
def list_feedback(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[FeedbackItem]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    items = (
        db.query(FeedbackItem)
        .options(joinedload(FeedbackItem.citations))
        .filter(FeedbackItem.run_id == run_id)
        .order_by(FeedbackItem.created_at.asc())
        .all()
    )
    return [item for item in items if len(item.citations) >= 1]


@app.post("/runs/{run_id}/sources/presign-put", response_model=PresignPutResponse)
def presign_source_upload(
    run_id: uuid.UUID,
    payload: PresignPutRequest,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> PresignPutResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    cleanup_orphan_source_docs(db)

    source_doc_id = uuid.uuid4()
    object_key = build_object_key(run_id, source_doc_id, payload.filename)
    source_doc = SourceDoc(
        id=source_doc_id,
        run_id=run_id,
        title=Path(payload.filename).name,
        origin=SourceOrigin.upload,
        object_key=object_key,
        content_type=payload.content_type,
    )
    url = presign_put(object_key, payload.content_type)
    db.add(source_doc)
    db.commit()
    return PresignPutResponse(
        source_doc_id=source_doc_id,
        object_key=object_key,
        url=url,
        headers={"Content-Type": payload.content_type},
    )


@app.post("/runs/{run_id}/sources/ingest", response_model=IngestResponse)
def ingest_source(
    run_id: uuid.UUID,
    payload: IngestRequest,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> IngestResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    source_doc = db.get(SourceDoc, payload.source_doc_id)
    if source_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_doc not found")
    if source_doc.run_id != run_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_doc belongs to another run")

    if payload.title is not None:
        source_doc.title = payload.title
    if payload.origin is not None:
        source_doc.origin = payload.origin
    if payload.object_key is not None:
        source_doc.object_key = payload.object_key
    if payload.content_type is not None:
        source_doc.content_type = payload.content_type

    if not object_exists(source_doc.object_key):
        db.delete(source_doc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="source object not found; source_doc removed as orphan",
        )

    db.commit()
    db.refresh(source_doc)

    task = tasks.ingest_source_doc.delay(
        str(run_id),
        source_doc_id=str(source_doc.id),
        object_key=source_doc.object_key,
    )
    return IngestResponse(source_doc_id=source_doc.id, task_id=task.id)


@app.get("/sources/{source_doc_id}/presign-get", response_model=PresignGetResponse)
def presign_source_get(
    source_doc_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> PresignGetResponse:
    source_doc = db.get(SourceDoc, source_doc_id)
    if source_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_doc not found")
    return PresignGetResponse(object_key=source_doc.object_key, url=presign_get(source_doc.object_key))


@app.get("/sources/{source_doc_id}/chunks", response_model=List[SourceChunkPreviewResponse])
def list_source_chunks(
    source_doc_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> List[SourceChunkPreviewResponse]:
    source_doc = db.get(SourceDoc, source_doc_id)
    if source_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_doc not found")
    rows = (
        db.query(SourceChunk)
        .filter(SourceChunk.source_doc_id == source_doc_id)
        .order_by(SourceChunk.chunk_index.asc())
        .limit(limit)
        .all()
    )
    return [
        SourceChunkPreviewResponse(
            id=row.id,
            source_doc_id=row.source_doc_id,
            chunk_index=row.chunk_index,
            text_excerpt=(row.text or "")[:360],
            loc=row.loc or {},
        )
        for row in rows
    ]


@app.post("/runs/{run_id}/pipeline", response_model=PipelineResponse)
def start_pipeline(
    run_id: uuid.UUID,
    payload: PipelineRequest,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> PipelineResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    task = tasks.run_full_pipeline.delay(str(run_id), payload.model_low, payload.model_high)
    return PipelineResponse(enqueued=[task.id])


@app.post("/runs/{run_id}/search", response_model=SearchResponse, status_code=status.HTTP_202_ACCEPTED)
def create_search(
    run_id: uuid.UUID,
    payload: SearchRequestCreate,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> SearchResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="query is required")
    if len(q) > 512:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="query too long")
    if payload.mode.value == "regex":
        if len(q) > 256:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="regex query too long")
        try:
            re.compile(q)
        except re.error as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"invalid regex query: {exc}",
            ) from exc

    idempotency_key = search_idempotency_key(run_id, q, payload.filters, payload.mode.value)
    search = SearchRequest(
        run_id=run_id,
        query=q,
        filters=payload.filters,
        mode=payload.mode,
        status=SearchStatus.pending,
        idempotency_key=idempotency_key,
    )
    try:
        db.add(search)
        db.commit()
        db.refresh(search)
    except IntegrityError:
        db.rollback()
        search = db.query(SearchRequest).filter(SearchRequest.idempotency_key == idempotency_key).one()

    tasks.search_chunks.delay(str(run_id), q, payload.filters, payload.mode.value, str(search.id))
    db.refresh(search)
    return search


@app.get("/search/{search_id}", response_model=SearchResponse)
def get_search(
    search_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> SearchRequest:
    search = (
        db.query(SearchRequest)
        .options(joinedload(SearchRequest.results))
        .filter(SearchRequest.id == search_id)
        .one_or_none()
    )
    if search is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="search not found")
    return search


@app.get("/citations/{citation_id}", response_model=CitationDetailResponse)
def get_citation_detail(
    citation_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: AuthContext = Depends(require_auth),
) -> CitationDetailResponse:
    citation = db.get(Citation, citation_id)
    if citation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="citation not found")

    source_doc = db.get(SourceDoc, citation.source_doc_id)
    chunk = db.get(SourceChunk, citation.chunk_id)
    if source_doc is None or chunk is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="citation references missing source")

    return CitationDetailResponse(
        id=citation.id,
        feedback_id=citation.feedback_id,
        source_doc_id=citation.source_doc_id,
        chunk_id=citation.chunk_id,
        span=citation.span,
        source_title=source_doc.title,
        source_object_key=source_doc.object_key,
        source_presigned_url=presign_get(source_doc.object_key),
        chunk_text=chunk.text,
        chunk_loc=chunk.loc,
    )
