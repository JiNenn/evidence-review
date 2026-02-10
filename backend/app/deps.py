from typing import Generator

from sqlalchemy.orm import Session

from app.database import get_db


def db_session() -> Generator[Session, None, None]:
    yield from get_db()
