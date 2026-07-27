"""SQLAlchemy declarative base — single source of truth for all ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
