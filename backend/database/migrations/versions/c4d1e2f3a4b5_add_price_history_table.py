"""add_price_history_table

Revision ID: c4d1e2f3a4b5
Revises: 56bcd0e15759
Create Date: 2026-08-24 21:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4d1e2f3a4b5'
down_revision: Union[str, Sequence[str], None] = '56bcd0e15759'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'price_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('simulation_id', sa.Integer(), nullable=False),
        sa.Column('trade_id', sa.String(length=64), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('price_history', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_price_history_simulation_id'), ['simulation_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_price_history_trade_id'), ['trade_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_price_history_timestamp'), ['timestamp'], unique=False)
        batch_op.create_index('ix_price_history_sim_time', ['simulation_id', 'timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('price_history', schema=None) as batch_op:
        batch_op.drop_index('ix_price_history_sim_time')
        batch_op.drop_index(batch_op.f('ix_price_history_timestamp'))
        batch_op.drop_index(batch_op.f('ix_price_history_trade_id'))
        batch_op.drop_index(batch_op.f('ix_price_history_simulation_id'))

    op.drop_table('price_history')

