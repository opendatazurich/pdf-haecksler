"""init

Revision ID: e74278ee5526
Revises:
Create Date: 2022-03-09 16:35:46.265586

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e74278ee5526'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('username', sa.String, nullable=False),
        sa.Column('password', sa.String, nullable=False)
    )


def downgrade():
    op.drop_table('users')
