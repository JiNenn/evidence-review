"""stage failure classification and run status expansion

Revision ID: 0003_stage_failure_contract
Revises: 0002_issue_model
Create Date: 2026-02-10 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_stage_failure_contract"
down_revision = "0002_issue_model"
branch_labels = None
depends_on = None


stage_failure_type = postgresql.ENUM(
    "system_error",
    "evidence_insufficient",
    "validation_error",
    name="stage_failure_type",
    create_type=False,
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'success_partial'
            ) THEN
                ALTER TYPE run_status ADD VALUE 'success_partial';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'blocked_evidence'
            ) THEN
                ALTER TYPE run_status ADD VALUE 'blocked_evidence';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'failed_system'
            ) THEN
                ALTER TYPE run_status ADD VALUE 'failed_system';
            END IF;
        END$$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'stage_failure_type') THEN
                CREATE TYPE stage_failure_type AS ENUM ('system_error', 'evidence_insufficient', 'validation_error');
            END IF;
        END$$;
        """
    )

    op.add_column("run_stages", sa.Column("failure_type", stage_failure_type, nullable=True))
    op.add_column("run_stages", sa.Column("failure_detail", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("run_stages", "failure_detail")
    op.drop_column("run_stages", "failure_type")
    op.execute("DROP TYPE IF EXISTS stage_failure_type")
    # Added run_status values are intentionally kept because PostgreSQL enum value removal is not safe.
