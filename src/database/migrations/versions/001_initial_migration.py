"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-08-31 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('username', sa.String(100), unique=True, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_superuser', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )
    
    # Create API keys table
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('key_hash', sa.String(255), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('last_used', sa.DateTime(timezone=True)),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    op.create_index('idx_api_keys_user_active', 'api_keys', ['user_id', 'is_active'])
    
    # Create workflow_executions table
    op.create_table('workflow_executions',
        sa.Column('id', sa.String(36), primary_key=True),  # UUID
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('workflow_definition', postgresql.JSON(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='workflowstatus'), default='PENDING', index=True),
        sa.Column('priority', sa.Enum('LOW', 'NORMAL', 'HIGH', name='priority'), default='NORMAL', index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('timeout_at', sa.DateTime(timezone=True)),
        sa.Column('outputs', postgresql.JSON()),
        sa.Column('error_message', sa.Text()),
        sa.Column('logs', postgresql.JSON()),
        sa.Column('webhook_url', sa.String(500)),
        sa.Column('metadata', postgresql.JSON()),
        sa.Column('queue_position', sa.Integer()),
        sa.Column('worker_id', sa.String(100)),
        sa.Column('execution_stats', postgresql.JSON())
    )
    op.create_index('idx_workflow_status_created', 'workflow_executions', ['status', 'created_at'])
    op.create_index('idx_workflow_user_status', 'workflow_executions', ['user_id', 'status'])
    op.create_index('idx_workflow_priority_created', 'workflow_executions', ['priority', 'created_at'])
    
    # Create models table
    op.create_table('models',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('name', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('type', sa.Enum('CHECKPOINT', 'LORA', 'EMBEDDING', 'VAE', 'CONTROLNET', 'UPSCALER', name='modeltype'), nullable=False, index=True),
        sa.Column('version', sa.String(50)),
        sa.Column('description', sa.Text()),
        sa.Column('file_path', sa.String(500)),
        sa.Column('file_size', sa.Integer()),
        sa.Column('file_hash', sa.String(64)),
        sa.Column('download_url', sa.String(500)),
        sa.Column('is_available', sa.Boolean(), default=False, index=True),
        sa.Column('is_loading', sa.Boolean(), default=False),
        sa.Column('is_downloading', sa.Boolean(), default=False),
        sa.Column('download_progress', sa.Float()),
        sa.Column('last_used', sa.DateTime(timezone=True)),
        sa.Column('usage_count', sa.Integer(), default=0),
        sa.Column('memory_usage_mb', sa.Float()),
        sa.Column('metadata', postgresql.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )
    op.create_index('idx_models_type_available', 'models', ['type', 'is_available'])
    op.create_index('idx_models_last_used', 'models', ['last_used'])
    
    # Create file_uploads table
    op.create_table('file_uploads',
        sa.Column('id', sa.String(36), primary_key=True),  # UUID
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('original_filename', sa.String(255), nullable=False),
        sa.Column('content_type', sa.String(100), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('storage_type', sa.String(20), default='s3'),
        sa.Column('is_uploaded', sa.Boolean(), default=False),
        sa.Column('is_processed', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('accessed_at', sa.DateTime(timezone=True)),
        sa.Column('metadata', postgresql.JSON())
    )
    op.create_index('idx_files_user_created', 'file_uploads', ['user_id', 'created_at'])
    op.create_index('idx_files_expires', 'file_uploads', ['expires_at'])
    
    # Create execution_logs table
    op.create_table('execution_logs',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('workflow_executions.id'), nullable=False),
        sa.Column('level', sa.String(20), nullable=False, index=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('component', sa.String(100)),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('metadata', postgresql.JSON())
    )
    op.create_index('idx_logs_execution_time', 'execution_logs', ['execution_id', 'timestamp'])
    op.create_index('idx_logs_level_time', 'execution_logs', ['level', 'timestamp'])
    
    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('cpu_usage_percent', sa.Float()),
        sa.Column('memory_usage_percent', sa.Float()),
        sa.Column('disk_usage_percent', sa.Float()),
        sa.Column('gpu_usage_percent', sa.Float()),
        sa.Column('gpu_memory_usage_percent', sa.Float()),
        sa.Column('gpu_temperature', sa.Float()),
        sa.Column('active_executions', sa.Integer(), default=0),
        sa.Column('queue_size', sa.Integer(), default=0),
        sa.Column('total_executions', sa.Integer(), default=0),
        sa.Column('failed_executions', sa.Integer(), default=0),
        sa.Column('average_execution_time', sa.Float()),
        sa.Column('loaded_models_count', sa.Integer(), default=0),
        sa.Column('model_memory_usage_mb', sa.Float(), default=0)
    )
    op.create_index('idx_metrics_timestamp', 'system_metrics', ['timestamp'])
    
    # Create webhook_logs table
    op.create_table('webhook_logs',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('workflow_executions.id'), nullable=False),
        sa.Column('webhook_url', sa.String(500), nullable=False),
        sa.Column('request_payload', postgresql.JSON()),
        sa.Column('response_status', sa.Integer()),
        sa.Column('response_body', sa.Text()),
        sa.Column('response_time_ms', sa.Float()),
        sa.Column('is_successful', sa.Boolean(), default=False),
        sa.Column('attempt_number', sa.Integer(), default=1),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('delivered_at', sa.DateTime(timezone=True)),
        sa.Column('next_retry_at', sa.DateTime(timezone=True))
    )
    op.create_index('idx_webhooks_execution', 'webhook_logs', ['execution_id'])
    op.create_index('idx_webhooks_retry', 'webhook_logs', ['next_retry_at', 'is_successful'])


def downgrade() -> None:
    op.drop_table('webhook_logs')
    op.drop_table('system_metrics')
    op.drop_table('execution_logs')
    op.drop_table('file_uploads')
    op.drop_table('models')
    op.drop_table('workflow_executions')
    op.drop_table('api_keys')
    op.drop_table('users')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS workflowstatus')
    op.execute('DROP TYPE IF EXISTS priority')
    op.execute('DROP TYPE IF EXISTS modeltype')