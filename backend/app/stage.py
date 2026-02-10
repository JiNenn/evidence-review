import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Run, RunStage, RunStatus, StageFailureType, StageStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StageExecution:
    already_succeeded: bool
    output_ref: Dict[str, Any]


class StageFailureError(Exception):
    def __init__(
        self,
        message: str,
        *,
        failure_type: StageFailureType,
        failure_detail: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.failure_detail = failure_detail or {}


def classify_failure(exc: Exception) -> tuple[StageFailureType, Dict[str, Any]]:
    if isinstance(exc, StageFailureError):
        detail = dict(exc.failure_detail)
        detail.setdefault("exception", exc.__class__.__name__)
        return exc.failure_type, detail
    if isinstance(exc, ValueError):
        return StageFailureType.validation_error, {"exception": exc.__class__.__name__}
    return StageFailureType.system_error, {"exception": exc.__class__.__name__}


def _advisory_lock_idempotency(session: Session, idempotency_key: str) -> None:
    """
    Serialize updates for the same idempotency_key to avoid attempt counter races.
    """
    raw = hashlib.sha256(idempotency_key.encode("utf-8")).digest()
    lock_key = int.from_bytes(raw[:8], byteorder="big", signed=False)
    if lock_key >= 2**63:
        lock_key -= 2**64
    session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": lock_key})


def _persist_failed_stage(
    *,
    run_id,
    stage_name: str,
    idempotency_key: str,
    error: Exception,
    failure_type: StageFailureType,
    failure_detail: Dict[str, Any] | None,
) -> None:
    """
    Persist failed status in an independent transaction so outer rollback cannot erase it.
    """
    failed_at = utc_now()
    with SessionLocal() as failure_db:
        _advisory_lock_idempotency(failure_db, idempotency_key)
        stage = (
            failure_db.query(RunStage)
            .filter(RunStage.idempotency_key == idempotency_key)
            .one_or_none()
        )
        if stage is None:
            stage = RunStage(
                run_id=run_id,
                stage_name=stage_name,
                idempotency_key=idempotency_key,
                status=StageStatus.failed,
                failure_type=failure_type,
                failure_detail={**(failure_detail or {}), "error": str(error)},
                attempt=1,
                started_at=failed_at,
                finished_at=failed_at,
                output_ref={"error": str(error)},
            )
            failure_db.add(stage)
        else:
            if stage.run_id != run_id or stage.stage_name != stage_name:
                raise ValueError("idempotency_key collision across run_id/stage_name")
            stage.idempotency_key = idempotency_key
            stage.status = StageStatus.failed
            stage.failure_type = failure_type
            stage.failure_detail = {**(failure_detail or {}), "error": str(error)}
            stage.attempt = (stage.attempt or 0) + 1
            if stage.started_at is None:
                stage.started_at = failed_at
            stage.finished_at = failed_at
            stage.output_ref = {"error": str(error)}

        run = failure_db.get(Run, run_id)
        if run is not None:
            if failure_type == StageFailureType.evidence_insufficient:
                run.status = RunStatus.blocked_evidence
            else:
                run.status = RunStatus.failed_system
        failure_db.commit()


def execute_stage(
    db: Session,
    *,
    run_id,
    stage_name: str,
    idempotency_key: str,
    fn: Callable[[], Dict[str, Any]],
) -> StageExecution:
    _advisory_lock_idempotency(db, idempotency_key)
    # attempt is defined as retry count for the same idempotency_key row.
    stage = db.query(RunStage).filter(RunStage.idempotency_key == idempotency_key).one_or_none()
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
    elif stage.run_id != run_id or stage.stage_name != stage_name:
        raise ValueError("idempotency_key collision across run_id/stage_name")

    stage.idempotency_key = idempotency_key
    stage.status = StageStatus.running
    stage.failure_type = None
    stage.failure_detail = None
    stage.attempt += 1
    stage.started_at = utc_now()
    stage.finished_at = None
    db.flush()

    try:
        output_ref = fn()
    except Exception as exc:
        failure_type, failure_detail = classify_failure(exc)
        # Release current transaction first so failed-stage upsert does not conflict on unique keys.
        db.rollback()
        _persist_failed_stage(
            run_id=run_id,
            stage_name=stage_name,
            idempotency_key=idempotency_key,
            error=exc,
            failure_type=failure_type,
            failure_detail=failure_detail,
        )
        raise

    stage.status = StageStatus.success
    stage.failure_type = None
    stage.failure_detail = None
    stage.finished_at = utc_now()
    stage.output_ref = output_ref
    db.flush()
    return StageExecution(already_succeeded=False, output_ref=output_ref)
