"""Add performance indexes on frequently queried columns.

Revision ID: 0007
Revises: 0006
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Generation history: queried by user_id (history page) and status (reaper)
    op.create_index("ix_gen_user_id", "generation_history", ["user_id"])
    op.create_index("ix_gen_status", "generation_history", ["status"])
    op.create_index("ix_gen_user_status", "generation_history", ["user_id", "status"])
    op.create_index("ix_gen_template_id", "generation_history", ["template_id"])

    # Transactions: queried by user_id (my transactions) and status (admin filter)
    op.create_index("ix_tx_user_id", "transactions", ["user_id"])
    op.create_index("ix_tx_status", "transactions", ["status"])
    op.create_index("ix_tx_code", "transactions", ["transaction_code"])

    # Shared video library: queried by status (public library) and creator_id
    op.create_index("ix_lib_status", "shared_video_library", ["status"])
    op.create_index("ix_lib_creator", "shared_video_library", ["creator_id"])

    # Credit adjustments: queried by user_id
    op.create_index("ix_adj_user_id", "credit_adjustments", ["user_id"])

    # Webhook logs: queried by result
    op.create_index("ix_whl_result", "webhook_log", ["result"])

    # Pricing items: primary key is key, already indexed


def downgrade() -> None:
    for name in ("ix_whl_result", "ix_adj_user_id", "ix_lib_creator", "ix_lib_status",
                 "ix_tx_code", "ix_tx_status", "ix_tx_user_id",
                 "ix_gen_template_id", "ix_gen_user_status", "ix_gen_status", "ix_gen_user_id"):
        op.drop_index(name)
