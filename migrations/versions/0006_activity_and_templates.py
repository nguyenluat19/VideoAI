"""Activity logs, user category, and title for generations

Revision ID: 0006
Revises: 0005
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Activity log table
    op.create_table(
        "activity_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("ip", sa.String(64), server_default="", nullable=False),
        sa.Column("details", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_activity_logs_user", "activity_logs", ["user_id"])

    # 2. Add columns to generation_history
    with op.batch_alter_table("generation_history") as batch:
        batch.add_column(sa.Column("user_category", sa.String(64), server_default="Mặc định", nullable=False))
        batch.add_column(sa.Column("title", sa.String(255), server_default="", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("generation_history") as batch:
        batch.drop_column("title")
        batch.drop_column("user_category")
    op.drop_index("ix_activity_logs_user", "activity_logs")
    op.drop_table("activity_logs")
