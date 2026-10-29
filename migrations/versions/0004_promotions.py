"""Top-up promotions: promotions table + bonus columns on transactions

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "promotions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("tiers", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.Integer, nullable=True),
    )
    with op.batch_alter_table("transactions") as b:
        b.add_column(sa.Column("bonus_credits", sa.Float, nullable=False, server_default="0"))
        b.add_column(sa.Column("promo_name", sa.String(120), nullable=False, server_default=""))


def downgrade() -> None:
    with op.batch_alter_table("transactions") as b:
        b.drop_column("promo_name")
        b.drop_column("bonus_credits")
    op.drop_table("promotions")
