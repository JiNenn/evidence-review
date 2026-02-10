import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    success_partial = "success_partial"
    blocked_evidence = "blocked_evidence"
    failed_system = "failed_system"
    failed_legacy = "failed_legacy"
    # legacy DB compatibility; API should normalize this to failed_legacy.
    failed = "failed"


class ArtifactKind(str, enum.Enum):
    low_draft = "low_draft"
    improved = "improved"
    high_final = "high_final"
    diff = "diff"


class ArtifactFormat(str, enum.Enum):
    markdown = "markdown"
    json = "json"
    text = "text"


class StageStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


class StageFailureType(str, enum.Enum):
    system_error = "system_error"
    evidence_insufficient = "evidence_insufficient"
    validation_error = "validation_error"


class SourceOrigin(str, enum.Enum):
    upload = "upload"
    url = "url"
    other = "other"


class SearchMode(str, enum.Enum):
    keyword = "keyword"
    regex = "regex"
    vector = "vector"


class SearchStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


class IssueStatus(str, enum.Enum):
    open = "open"
    resolved = "resolved"
    hidden = "hidden"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    task_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, name="run_status"), nullable=False, default=RunStatus.pending)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")
    feedback_items = relationship("FeedbackItem", back_populates="run", cascade="all, delete-orphan")
    source_docs = relationship("SourceDoc", back_populates="run", cascade="all, delete-orphan")
    run_stages = relationship("RunStage", back_populates="run", cascade="all, delete-orphan")
    search_requests = relationship("SearchRequest", back_populates="run", cascade="all, delete-orphan")
    issues = relationship("Issue", back_populates="run", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_run_kind_created_at", "run_id", "kind", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[ArtifactKind] = mapped_column(Enum(ArtifactKind, name="artifact_kind"), nullable=False)
    format: Mapped[ArtifactFormat] = mapped_column(Enum(ArtifactFormat, name="artifact_format"), nullable=False)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    run = relationship("Run", back_populates="artifacts")


class FeedbackItem(Base):
    __tablename__ = "feedback_items"
    __table_args__ = (
        Index("ix_feedback_items_run_created_at", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    target_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    run = relationship("Run", back_populates="feedback_items")
    citations = relationship("Citation", back_populates="feedback", cascade="all, delete-orphan")


class SourceDoc(Base):
    __tablename__ = "source_docs"
    __table_args__ = (
        Index("ix_source_docs_run_created_at", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    origin: Mapped[SourceOrigin] = mapped_column(Enum(SourceOrigin, name="source_origin"), nullable=False, default=SourceOrigin.upload)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False, default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    run = relationship("Run", back_populates="source_docs")
    chunks = relationship("SourceChunk", back_populates="source_doc", cascade="all, delete-orphan")


class SourceChunk(Base):
    __tablename__ = "source_chunks"
    __table_args__ = (
        UniqueConstraint("source_doc_id", "chunk_index", name="uq_source_chunks_doc_chunk_index"),
        Index("ix_source_chunks_doc_chunk_index", "source_doc_id", "chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_docs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    loc: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    embedding: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    source_doc = relationship("SourceDoc", back_populates="chunks")


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        Index("ix_citations_feedback", "feedback_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feedback_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feedback_items.id", ondelete="CASCADE"), nullable=True
    )
    source_doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_docs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_chunks.id", ondelete="CASCADE"), nullable=False
    )
    span: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    feedback = relationship("FeedbackItem", back_populates="citations")
    issue_evidences = relationship("IssueEvidence", back_populates="citation", cascade="all, delete-orphan")


class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = (
        UniqueConstraint("run_id", "fingerprint", name="uq_issues_run_fingerprint"),
        Index("ix_issues_run_created_at", "run_id", "created_at"),
        Index("ix_issues_run_severity", "run_id", "severity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    status: Mapped[IssueStatus] = mapped_column(Enum(IssueStatus, name="issue_status"), nullable=False, default=IssueStatus.open)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    run = relationship("Run", back_populates="issues")
    evidences = relationship("IssueEvidence", back_populates="issue", cascade="all, delete-orphan")


class IssueEvidence(Base):
    __tablename__ = "issue_evidences"
    __table_args__ = (
        Index("ix_issue_evidences_issue_created_at", "issue_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    issue_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("issues.id", ondelete="CASCADE"), nullable=False)
    citation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("citations.id", ondelete="CASCADE"), nullable=False
    )
    before_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    after_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    loc: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    issue = relationship("Issue", back_populates="evidences")
    citation = relationship("Citation", back_populates="issue_evidences")


class RunStage(Base):
    __tablename__ = "run_stages"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_run_stages_idempotency_key"),
        Index("ix_run_stages_run_status", "run_id", "status"),
        Index("ix_run_stages_run_stage_name", "run_id", "stage_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    stage_name: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[StageStatus] = mapped_column(Enum(StageStatus, name="stage_status"), nullable=False, default=StageStatus.pending)
    failure_type: Mapped[StageFailureType | None] = mapped_column(
        Enum(StageFailureType, name="stage_failure_type"),
        nullable=True,
    )
    failure_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    output_ref: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run = relationship("Run", back_populates="run_stages")


class SearchRequest(Base):
    __tablename__ = "search_requests"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_search_requests_idempotency_key"),
        Index("ix_search_requests_run_created_at", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    filters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    mode: Mapped[SearchMode] = mapped_column(Enum(SearchMode, name="search_mode"), nullable=False, default=SearchMode.keyword)
    status: Mapped[SearchStatus] = mapped_column(
        Enum(SearchStatus, name="search_status"), nullable=False, default=SearchStatus.pending
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    run = relationship("Run", back_populates="search_requests")
    results = relationship("SearchResult", back_populates="search", cascade="all, delete-orphan")


class SearchResult(Base):
    __tablename__ = "search_results"
    __table_args__ = (
        UniqueConstraint("search_id", "rank", name="uq_search_results_search_rank"),
        Index("ix_search_results_search_rank", "search_id", "rank"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("search_requests.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    source_doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_docs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_chunks.id", ondelete="CASCADE"), nullable=False
    )
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    search = relationship("SearchRequest", back_populates="results")
