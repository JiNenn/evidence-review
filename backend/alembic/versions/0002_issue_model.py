"""issue model and evidence-first expansion

Revision ID: 0002_issue_model
Revises: 0001_initial_schema
Create Date: 2026-02-10 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_issue_model"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


issue_status = postgresql.ENUM("open", "resolved", "hidden", name="issue_status", create_type=False)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'issue_status') THEN
                CREATE TYPE issue_status AS ENUM ('open', 'resolved', 'hidden');
            END IF;
        END$$;
        """
    )

    op.alter_column("citations", "feedback_id", existing_type=postgresql.UUID(as_uuid=True), nullable=True)

    op.create_table(
        "issues",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", issue_status, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "fingerprint", name="uq_issues_run_fingerprint"),
    )
    op.create_index("ix_issues_run_created_at", "issues", ["run_id", "created_at"])
    op.create_index("ix_issues_run_severity", "issues", ["run_id", "severity"])

    op.create_table(
        "issue_evidences",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("issue_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("citation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("before_excerpt", sa.Text(), nullable=True),
        sa.Column("after_excerpt", sa.Text(), nullable=True),
        sa.Column("loc", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["citation_id"], ["citations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_issue_evidences_issue_created_at", "issue_evidences", ["issue_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_issue_evidences_issue_created_at", table_name="issue_evidences")
    op.drop_table("issue_evidences")
    op.drop_index("ix_issues_run_severity", table_name="issues")
    op.drop_index("ix_issues_run_created_at", table_name="issues")
    op.drop_table("issues")

    op.execute("DELETE FROM citations WHERE feedback_id IS NULL")
    op.alter_column("citations", "feedback_id", existing_type=postgresql.UUID(as_uuid=True), nullable=False)

    op.execute("DROP TYPE IF EXISTS issue_status")
