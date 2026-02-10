"""add failed_legacy run_status and migrate old failed rows

Revision ID: 0005_run_status_failed_legacy
Revises: 0004_stage_idem_key
Create Date: 2026-02-10 14:00:00.000000
"""

from alembic import op


revision = "0005_run_status_failed_legacy"
down_revision = "0004_stage_idem_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'failed'
            ) AND NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'failed_legacy'
            ) THEN
                ALTER TYPE run_status RENAME VALUE 'failed' TO 'failed_legacy';
            ELSIF NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'run_status' AND e.enumlabel = 'failed_legacy'
            ) THEN
                ALTER TYPE run_status ADD VALUE 'failed_legacy';
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    # enum value removal is intentionally not performed.
    pass
