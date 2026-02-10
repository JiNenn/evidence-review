from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import RunStage, StageStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StageExecution:
    already_succeeded: bool
    output_ref: Dict[str, Any]


def _persist_failed_stage(
    *,
    run_id,
    stage_name: str,
    idempotency_key: str,
    error: Exception,
) -> None:
    """
    Persist failed status in an independent transaction so outer rollback cannot erase it.
    """
    failed_at = utc_now()
    with SessionLocal() as failure_db:
        stage = (
            failure_db.query(RunStage)
            .filter(RunStage.run_id == run_id, RunStage.stage_name == stage_name)
            .one_or_none()
        )
        if stage is None:
            stage = RunStage(
                run_id=run_id,
                stage_name=stage_name,
                idempotency_key=idempotency_key,
                status=StageStatus.failed,
                attempt=1,
                started_at=failed_at,
                finished_at=failed_at,
                output_ref={"error": str(error)},
            )
            failure_db.add(stage)
        else:
            stage.idempotency_key = idempotency_key
            stage.status = StageStatus.failed
            stage.attempt = max(stage.attempt, 1)
            if stage.started_at is None:
                stage.started_at = failed_at
            stage.finished_at = failed_at
            stage.output_ref = {"error": str(error)}
        failure_db.commit()


def execute_stage(
    db: Session,
    *,
    run_id,
    stage_name: str,
    idempotency_key: str,
    fn: Callable[[], Dict[str, Any]],
) -> StageExecution:
    stage = db.query(RunStage).filter(RunStage.run_id == run_id, RunStage.stage_name == stage_name).one_or_none()
    if stage and stage.status == StageStatus.success:
        return StageExecution(already_succeeded=True, output_ref=stage.output_ref or {})

    if stage is None:
        stage = RunStage(
            run_id=run_id,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            status=StageStatus.pending,
            attempt=0,
        )
        db.add(stage)
        db.flush()

    stage.idempotency_key = idempotency_key
    stage.status = StageStatus.running
    stage.attempt += 1
    stage.started_at = utc_now()
    stage.finished_at = None
    db.flush()

    try:
        output_ref = fn()
    except Exception as exc:
        # Release current transaction first so failed-stage upsert does not conflict on unique keys.
        db.rollback()
        _persist_failed_stage(
            run_id=run_id,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            error=exc,
        )
        raise

    stage.status = StageStatus.success
    stage.finished_at = utc_now()
    stage.output_ref = output_ref
    db.flush()
    return StageExecution(already_succeeded=False, output_ref=output_ref)
