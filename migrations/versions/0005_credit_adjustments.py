"""Admin credit adjustments ledger

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "credit_adjustments",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("admin_id", sa.Integer, nullable=True),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("balance_after", sa.Float, nullable=False),
        sa.Column("reason", sa.String(200), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_credit_adjustments_user", "credit_adjustments", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_credit_adjustments_user", "credit_adjustments")
    op.drop_table("credit_adjustments")
