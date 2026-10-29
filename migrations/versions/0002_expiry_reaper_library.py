"""Webhook log, generation heartbeat/share flag, library preview + source link

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "webhook_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("transaction_code", sa.String(32), server_default=""),
        sa.Column("amount", sa.Float, server_default="0"),
        sa.Column("content", sa.Text, server_default=""),
        sa.Column("result", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    with op.batch_alter_table("generation_history") as b:
        b.add_column(sa.Column("updated_at", sa.DateTime(timezone=True)))
        b.add_column(sa.Column("share_to_library", sa.Boolean, nullable=False, server_default=sa.true()))
    op.execute("UPDATE generation_history SET updated_at = created_at WHERE updated_at IS NULL")
    with op.batch_alter_table("shared_video_library") as b:
        b.add_column(sa.Column("preview_video_url", sa.Text, server_default=""))
        b.add_column(sa.Column("source_generation_id", sa.Integer, nullable=True))
        b.create_unique_constraint("uq_library_source_generation", ["source_generation_id"])


def downgrade() -> None:
    with op.batch_alter_table("shared_video_library") as b:
        b.drop_constraint("uq_library_source_generation", type_="unique")
        b.drop_column("source_generation_id")
        b.drop_column("preview_video_url")
    with op.batch_alter_table("generation_history") as b:
        b.drop_column("share_to_library")
        b.drop_column("updated_at")
    op.drop_table("webhook_log")
