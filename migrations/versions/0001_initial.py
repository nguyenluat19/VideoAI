"""Initial schema: users, shared_video_library, transactions, generation_history

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("credits_balance", sa.Float, nullable=False, server_default="0"),
        sa.Column("role", sa.String(16), nullable=False, server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "shared_video_library",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("category", sa.String(32), server_default="full"),
        sa.Column("base_video_url", sa.Text, server_default=""),
        sa.Column("thumbnail_url", sa.Text, server_default=""),
        sa.Column("template_schema", sa.JSON),
        sa.Column("status", sa.String(16), nullable=False, server_default="private"),
        sa.Column("creator_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("usage_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("amount_vnd", sa.Float, nullable=False),
        sa.Column("credits_added", sa.Float, nullable=False),
        sa.Column("payment_method", sa.String(32), server_default="vietqr"),
        sa.Column("transaction_code", sa.String(32), nullable=False, unique=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "generation_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("template_id", sa.Integer, sa.ForeignKey("shared_video_library.id"), nullable=False),
        sa.Column("render_type", sa.String(32), nullable=False),
        sa.Column("input_data", sa.JSON),
        sa.Column("output_video_url", sa.Text, server_default=""),
        sa.Column("credits_cost", sa.Float, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    for t in ("generation_history", "transactions", "shared_video_library", "users"):
        op.drop_table(t)
