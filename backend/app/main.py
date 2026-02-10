import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import redis
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app import tasks
from app.auth import issue_access_token, require_auth
from app.config import get_settings
from app.database import get_db
from app.logging import configure_logging
from app.models import (
    Artifact,
    Citation,
    FeedbackItem,
    Issue,
    IssueEvidence,
    Run,
    RunStage,
    SearchRequest,
    SearchStatus,
    SourceChunk,
    SourceDoc,
)
from app.s3_client import ensure_bucket_exists, get_s3_client, presign_get, presign_put
from app.schemas import (
    ArtifactResponse,
    AuthTokenRequest,
    AuthTokenResponse,
    CitationDetailResponse,
    FeedbackResponse,
    IngestRequest,
    IngestResponse,
    IssueDetailResponse,
    IssueEvidenceResponse,
    IssueResponse,
    PipelineRequest,
    PipelineResponse,
    PresignGetResponse,
    PresignPutRequest,
    PresignPutResponse,
    RunCreate,
    RunResponse,
    RunStageResponse,
    SearchRequestCreate,
    SearchResponse,
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


def build_object_key(run_id: uuid.UUID, filename: str) -> str:
    safe = Path(filename).name
    source_doc_id = uuid.uuid4()
    return f"runs/{run_id}/sources/{source_doc_id}/raw/{safe}"


def to_issue_evidence_response(db: Session, evidence: IssueEvidence) -> IssueEvidenceResponse | None:
    citation = db.get(Citation, evidence.citation_id)
    if citation is None:
        return None
    source_doc = db.get(SourceDoc, citation.source_doc_id)
    chunk = db.get(SourceChunk, citation.chunk_id)
    if source_doc is None or chunk is None:
        return None
    return IssueEvidenceResponse(
        id=evidence.id,
        issue_id=evidence.issue_id,
        citation_id=evidence.citation_id,
        before_excerpt=evidence.before_excerpt,
        after_excerpt=evidence.after_excerpt,
        loc=evidence.loc,
        source_title=source_doc.title,
        source_presigned_url=presign_get(source_doc.object_key),
        chunk_text=chunk.text,
        chunk_loc=chunk.loc,
    )


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
        return AuthTokenResponse(access_token=issue_access_token("anonymous"))
    if payload.username != settings.auth_dev_user or payload.password != settings.auth_dev_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    return AuthTokenResponse(access_token=issue_access_token(payload.username))


@app.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> RunResponse:
    run = Run(task_prompt=payload.task_prompt, metadata_=payload.metadata)
    db.add(run)
    db.commit()
    db.refresh(run)
    return RunResponse(
        id=run.id,
        created_at=run.created_at,
        task_prompt=run.task_prompt,
        status=run.status,
        metadata=run.metadata_,
    )


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> RunResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return RunResponse(
        id=run.id,
        created_at=run.created_at,
        task_prompt=run.task_prompt,
        status=run.status,
        metadata=run.metadata_,
    )


@app.get("/runs/{run_id}/stages", response_model=List[RunStageResponse])
def list_run_stages(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> List[RunStage]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return (
        db.query(RunStage)
        .filter(RunStage.run_id == run_id)
        .order_by(RunStage.stage_name.asc())
        .all()
    )


@app.get("/runs/{run_id}/artifacts", response_model=List[ArtifactResponse])
def list_artifacts(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
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
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> List[IssueResponse]:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    issues = (
        db.query(Issue)
        .options(joinedload(Issue.evidences))
        .filter(Issue.run_id == run_id)
        .order_by(Issue.severity.desc(), Issue.created_at.asc())
        .all()
    )
    rows: List[IssueResponse] = []
    for issue in issues:
        evidence_count = len(issue.evidences)
        if evidence_count < 1:
            continue
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
                created_at=issue.created_at,
                updated_at=issue.updated_at,
            )
        )
    return rows


@app.get("/issues/{issue_id}", response_model=IssueDetailResponse)
def get_issue(
    issue_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> IssueDetailResponse:
    issue = (
        db.query(Issue)
        .options(joinedload(Issue.evidences))
        .filter(Issue.id == issue_id)
        .one_or_none()
    )
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="issue not found")

    evidence_rows: List[IssueEvidenceResponse] = []
    for evidence in issue.evidences:
        row = to_issue_evidence_response(db, evidence)
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


@app.get("/runs/{run_id}/feedback", response_model=List[FeedbackResponse])
def list_feedback(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
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
    _subject: str = Depends(require_auth),
) -> PresignPutResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    object_key = build_object_key(run_id, payload.filename)
    url = presign_put(object_key, payload.content_type)
    return PresignPutResponse(object_key=object_key, url=url, headers={"Content-Type": payload.content_type})


@app.post("/runs/{run_id}/sources/ingest", response_model=IngestResponse)
def ingest_source(
    run_id: uuid.UUID,
    payload: IngestRequest,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> IngestResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    title = payload.title or Path(payload.object_key).name
    source_doc = SourceDoc(
        run_id=run_id,
        title=title,
        origin=payload.origin,
        object_key=payload.object_key,
        content_type=payload.content_type,
    )
    db.add(source_doc)
    db.commit()
    db.refresh(source_doc)

    task = tasks.ingest_source_doc.delay(str(run_id), payload.object_key, str(source_doc.id))
    return IngestResponse(source_doc_id=source_doc.id, task_id=task.id)


@app.get("/sources/{source_doc_id}/presign-get", response_model=PresignGetResponse)
def presign_source_get(
    source_doc_id: uuid.UUID,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
) -> PresignGetResponse:
    source_doc = db.get(SourceDoc, source_doc_id)
    if source_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_doc not found")
    return PresignGetResponse(object_key=source_doc.object_key, url=presign_get(source_doc.object_key))


@app.post("/runs/{run_id}/pipeline", response_model=PipelineResponse)
def start_pipeline(
    run_id: uuid.UUID,
    payload: PipelineRequest,
    db: Session = Depends(get_db),
    _subject: str = Depends(require_auth),
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
    _subject: str = Depends(require_auth),
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
    _subject: str = Depends(require_auth),
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
    _subject: str = Depends(require_auth),
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
