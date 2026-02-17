"""add run_stage_attempts for detailed stage execution history

Revision ID: 0007_run_stage_attempts
Revises: 0006_issue_ev_loc_nullable
Create Date: 2026-02-11 21:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0007_run_stage_attempts"
down_revision = "0006_issue_ev_loc_nullable"
branch_labels = None
depends_on = None


stage_status = postgresql.ENUM(
    "pending",
    "running",
    "success",
    "failed",
    name="stage_status",
    create_type=False,
)


def upgrade() -> None:
    op.create_table(
        "run_stage_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_stage_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(length=128), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", stage_status, nullable=False),
        sa.Column(
            "failure_type",
            postgresql.ENUM(
                "system_error",
                "evidence_insufficient",
                "validation_error",
                name="stage_failure_type",
                create_type=False,
            ),
            nullable=True,
        ),
        sa.Column("failure_detail", sa.JSON(), nullable=True),
        sa.Column("output_ref", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_stage_id"], ["run_stages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_stage_id", "attempt_no", name="uq_run_stage_attempts_stage_attempt_no"),
    )
    op.create_index(
        "ix_run_stage_attempts_run_stage",
        "run_stage_attempts",
        ["run_id", "stage_name", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_run_stage_attempts_stage_attempt",
        "run_stage_attempts",
        ["run_stage_id", "attempt_no"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_run_stage_attempts_stage_attempt", table_name="run_stage_attempts")
    op.drop_index("ix_run_stage_attempts_run_stage", table_name="run_stage_attempts")
    op.drop_table("run_stage_attempts")
