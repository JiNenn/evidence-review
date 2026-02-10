"""use idempotency_key as primary uniqueness for run_stages

Revision ID: 0004_stage_idem_key
Revises: 0003_stage_failure_contract
Create Date: 2026-02-10 13:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_stage_idem_key"
down_revision = "0003_stage_failure_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_run_stages_run_stage", "run_stages", type_="unique")
    op.create_index("ix_run_stages_run_stage_name", "run_stages", ["run_id", "stage_name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_run_stages_run_stage_name", table_name="run_stages")
    op.create_unique_constraint("uq_run_stages_run_stage", "run_stages", ["run_id", "stage_name"])
