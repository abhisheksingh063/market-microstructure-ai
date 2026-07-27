"""Alembic environment configuration.

Uses our application's models for autogenerate support.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from core.config import settings
from database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Use the configured database URL (sync variant for migrations)
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite+aiosqlite"):
    db_url = db_url.replace("+aiosqlite", "")
elif db_url.startswith("postgresql+asyncpg"):
    db_url = db_url.replace("+asyncpg", "+psycopg2")

config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    kwargs: dict = {}
    if db_url.startswith("sqlite"):
        kwargs["render_as_batch"] = True
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **kwargs,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        kwargs: dict = {}
        if db_url.startswith("sqlite"):
            kwargs["render_as_batch"] = True
        context.configure(connection=connection, target_metadata=target_metadata, **kwargs)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
