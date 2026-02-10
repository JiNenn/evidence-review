"""make issue_evidences.loc nullable (cache field)

Revision ID: 0006_issue_ev_loc_nullable
Revises: 0005_run_status_failed_legacy
Create Date: 2026-02-10 15:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_issue_ev_loc_nullable"
down_revision = "0005_run_status_failed_legacy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "issue_evidences",
        "loc",
        existing_type=sa.JSON(),
        nullable=True,
    )


def downgrade() -> None:
    op.execute("UPDATE issue_evidences SET loc = '{}'::json WHERE loc IS NULL")
    op.alter_column(
        "issue_evidences",
        "loc",
        existing_type=sa.JSON(),
        nullable=False,
    )
