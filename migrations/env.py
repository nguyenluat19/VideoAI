from alembic import context
from sqlalchemy import create_engine

from app.config import settings

config = context.config
url = config.get_main_option("sqlalchemy.url") or settings.database_url
is_sqlite = url.startswith("sqlite")


def run() -> None:
    engine = create_engine(url, connect_args={"timeout": 30} if is_sqlite else {})
    with engine.connect() as conn:
        # batch mode lets SQLite emulate ALTER TABLE operations
        context.configure(connection=conn, render_as_batch=is_sqlite, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
