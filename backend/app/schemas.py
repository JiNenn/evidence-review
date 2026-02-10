import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import (
    ArtifactFormat,
    ArtifactKind,
    IssueStatus,
    RunStatus,
    SearchMode,
    SearchStatus,
    SourceOrigin,
    StageFailureType,
    StageStatus,
)


class RunCreate(BaseModel):
    task_prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthTokenRequest(BaseModel):
    username: str
    password: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RunResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    task_prompt: str
    status: RunStatus
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True


class ArtifactResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    kind: ArtifactKind
    format: ArtifactFormat
    content_text: Optional[str]
    content_object_key: Optional[str]
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


class CitationResponse(BaseModel):
    id: uuid.UUID
    feedback_id: Optional[uuid.UUID]
    source_doc_id: uuid.UUID
    chunk_id: uuid.UUID
    span: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class CitationDetailResponse(BaseModel):
    id: uuid.UUID
    feedback_id: Optional[uuid.UUID]
    source_doc_id: uuid.UUID
    chunk_id: uuid.UUID
    span: Optional[Dict[str, Any]]
    source_title: str
    source_object_key: str
    source_presigned_url: str
    chunk_text: str
    chunk_loc: Dict[str, Any]


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    target_artifact_id: uuid.UUID
    text: str
    category: str
    severity: int
    created_at: datetime
    citations: List[CitationResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class IssueEvidenceResponse(BaseModel):
    id: uuid.UUID
    issue_id: uuid.UUID
    citation_id: uuid.UUID
    source_doc_id: uuid.UUID
    chunk_id: uuid.UUID
    citation_span: Optional[Dict[str, Any]]
    before_excerpt: Optional[str]
    after_excerpt: Optional[str]
    loc: Dict[str, Any]
    source_title: str
    source_presigned_url: str
    chunk_text: str
    chunk_loc: Dict[str, Any]


class IssueResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    title: str
    summary: str
    severity: int
    confidence: float
    status: IssueStatus
    evidence_count: int
    created_at: datetime
    updated_at: datetime


class IssueDetailResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    title: str
    summary: str
    severity: int
    confidence: float
    status: IssueStatus
    created_at: datetime
    updated_at: datetime
    evidences: List[IssueEvidenceResponse] = Field(default_factory=list)


class PresignPutRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"


class PresignPutResponse(BaseModel):
    source_doc_id: uuid.UUID
    object_key: str
    url: str
    method: str = "PUT"
    headers: Dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    source_doc_id: uuid.UUID
    object_key: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    origin: Optional[SourceOrigin] = None


class IngestResponse(BaseModel):
    source_doc_id: uuid.UUID
    task_id: str


class PresignGetResponse(BaseModel):
    object_key: str
    url: str
    method: str = "GET"


class PipelineRequest(BaseModel):
    model_low: Optional[str] = None
    model_high: Optional[str] = None


class PipelineResponse(BaseModel):
    enqueued: List[str]


class SearchRequestCreate(BaseModel):
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    mode: SearchMode = SearchMode.keyword


class SearchResultItem(BaseModel):
    id: uuid.UUID
    rank: int
    source_doc_id: uuid.UUID
    chunk_id: uuid.UUID
    snippet: str
    score: float
    payload: Dict[str, Any]

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    query: str
    filters: Dict[str, Any]
    mode: SearchMode
    status: SearchStatus
    idempotency_key: str
    created_at: datetime
    results: List[SearchResultItem] = Field(default_factory=list)

    class Config:
        from_attributes = True


class StageResponse(BaseModel):
    status: StageStatus
    output_ref: Dict[str, Any] = Field(default_factory=dict)


class RunStageResponse(BaseModel):
    stage_name: str
    status: StageStatus
    failure_type: Optional[StageFailureType]
    failure_detail: Optional[Dict[str, Any]]
    attempt: int
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    output_ref: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True
