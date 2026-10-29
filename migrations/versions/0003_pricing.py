"""Admin-editable pricing: pricing_items, pricing_log, app_settings

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pricing_items",
        sa.Column("key", sa.String(96), primary_key=True),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.Column("updated_by", sa.Integer, nullable=True),
    )
    op.create_table(
        "pricing_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(96), nullable=False),
        sa.Column("old_price", sa.Float),
        sa.Column("new_price", sa.Float),
        sa.Column("old_enabled", sa.Boolean),
        sa.Column("new_enabled", sa.Boolean),
        sa.Column("admin_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.String(255), nullable=False),
    )


def downgrade() -> None:
    for t in ("app_settings", "pricing_log", "pricing_items"):
        op.drop_table(t)
