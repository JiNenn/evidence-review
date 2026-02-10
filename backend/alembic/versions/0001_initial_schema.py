"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-08 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


run_status = sa.Enum("pending", "running", "success", "failed", name="run_status")
artifact_kind = sa.Enum("low_draft", "improved", "high_final", "diff", name="artifact_kind")
artifact_format = sa.Enum("markdown", "json", "text", name="artifact_format")
source_origin = sa.Enum("upload", "url", "other", name="source_origin")
stage_status = sa.Enum("pending", "running", "success", "failed", name="stage_status")
search_mode = sa.Enum("keyword", "regex", "vector", name="search_mode")
search_status = sa.Enum("pending", "running", "success", "failed", name="search_status")


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("task_prompt", sa.Text(), nullable=False),
        sa.Column("status", run_status, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", artifact_kind, nullable=False),
        sa.Column("format", artifact_format, nullable=False),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_object_key", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifacts_run_kind_created_at", "artifacts", ["run_id", "kind", "created_at"])

    op.create_table(
        "source_docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("origin", source_origin, nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_source_docs_run_created_at", "source_docs", ["run_id", "created_at"])

    op.create_table(
        "source_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("loc", sa.JSON(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["source_doc_id"], ["source_docs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_doc_id", "chunk_index", name="uq_source_chunks_doc_chunk_index"),
    )
    op.create_index("ix_source_chunks_doc_chunk_index", "source_chunks", ["source_doc_id", "chunk_index"])

    op.create_table(
        "feedback_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_artifact_id"], ["artifacts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_items_run_created_at", "feedback_items", ["run_id", "created_at"])

    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feedback_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("span", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["feedback_id"], ["feedback_items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_doc_id"], ["source_docs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["source_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_citations_feedback", "citations", ["feedback_id"])

    op.create_table(
        "run_stages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("status", stage_status, nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("output_ref", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_run_stages_idempotency_key"),
        sa.UniqueConstraint("run_id", "stage_name", name="uq_run_stages_run_stage"),
    )
    op.create_index("ix_run_stages_run_status", "run_stages", ["run_id", "status"])

    op.create_table(
        "search_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("filters", sa.JSON(), nullable=False),
        sa.Column("mode", search_mode, nullable=False),
        sa.Column("status", search_status, nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_search_requests_idempotency_key"),
    )
    op.create_index("ix_search_requests_run_created_at", "search_requests", ["run_id", "created_at"])

    op.create_table(
        "search_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("search_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("source_doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["search_id"], ["search_requests.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_doc_id"], ["source_docs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["source_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("search_id", "rank", name="uq_search_results_search_rank"),
    )
    op.create_index("ix_search_results_search_rank", "search_results", ["search_id", "rank"])


def downgrade() -> None:
    op.drop_index("ix_search_results_search_rank", table_name="search_results")
    op.drop_table("search_results")
    op.drop_index("ix_search_requests_run_created_at", table_name="search_requests")
    op.drop_table("search_requests")
    op.drop_index("ix_run_stages_run_status", table_name="run_stages")
    op.drop_table("run_stages")
    op.drop_index("ix_citations_feedback", table_name="citations")
    op.drop_table("citations")
    op.drop_index("ix_feedback_items_run_created_at", table_name="feedback_items")
    op.drop_table("feedback_items")
    op.drop_index("ix_source_chunks_doc_chunk_index", table_name="source_chunks")
    op.drop_table("source_chunks")
    op.drop_index("ix_source_docs_run_created_at", table_name="source_docs")
    op.drop_table("source_docs")
    op.drop_index("ix_artifacts_run_kind_created_at", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_table("runs")

    bind = op.get_bind()
    for enum_type in [
        search_status,
        search_mode,
        stage_status,
        source_origin,
        artifact_format,
        artifact_kind,
        run_status,
    ]:
        enum_type.drop(bind, checkfirst=True)
