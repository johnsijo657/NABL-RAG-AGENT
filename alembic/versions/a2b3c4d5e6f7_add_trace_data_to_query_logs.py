"""add trace_data to query_logs

Revision ID: a2b3c4d5e6f7
Revises: 9a1b2c3d4e5f
Create Date: 2026-09-04 10:58:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2b3c4d5e6f7'
down_revision: Union[str, Sequence[str], None] = '9a1b2c3d4e5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('query_logs', sa.Column('trace_data', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('query_logs', 'trace_data')
