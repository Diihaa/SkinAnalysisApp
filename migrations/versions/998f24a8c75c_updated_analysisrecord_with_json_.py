"""Updated AnalysisRecord with JSON probabilities and filename

Revision ID: 998f24a8c75c
Revises: 
Create Date: 2025-10-19 10:45:43.418778

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '998f24a8c75c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Modify `probabilities` column to JSON
    with op.batch_alter_table('analysis_records', schema=None) as batch_op:
        batch_op.alter_column(
            'probabilities',
            existing_type=mysql.TEXT(),
            type_=mysql.JSON(),
            existing_nullable=True
        )
        # Add new filename column
        batch_op.add_column(sa.Column('filename', sa.String(length=255), nullable=True))


def downgrade():
    # Revert back to TEXT and drop filename
    with op.batch_alter_table('analysis_records', schema=None) as batch_op:
        batch_op.alter_column(
            'probabilities',
            existing_type=mysql.JSON(),
            type_=mysql.TEXT(),
            existing_nullable=True
        )
        batch_op.drop_column('filename')
