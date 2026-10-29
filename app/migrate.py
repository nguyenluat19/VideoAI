"""Run Alembic migrations programmatically (called on startup)."""
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from .db import engine

ROOT = Path(__file__).resolve().parent.parent


def upgrade_db() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    url = engine.url.render_as_string(hide_password=False).replace("%", "%%")
    cfg.set_main_option("sqlalchemy.url", url)
    tables = set(inspect(engine).get_table_names())
    # A database created by the pre-Alembic version (create_all) already has the 0001 schema.
    if "users" in tables and "alembic_version" not in tables:
        command.stamp(cfg, "0001")
    command.upgrade(cfg, "head")
