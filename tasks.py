import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlopsg24"
PYTHON_VERSION = "3.12"
#NOTE: belongs in config

# Project commands
@task
def api(ctx:Context) -> None:
    """open a FastAPI"""
    ctx.run(f"uv run uvicorn --reload --port 8000 src.{PROJECT_NAME}.api:app", echo=True, pty=not WINDOWS)

@task
def frontend(ctx: Context) -> None:
    """open a streamlit frontend"""
    ctx.run(f"uv run streamlit run src/{PROJECT_NAME}/frontend.py", echo=True, pty=not WINDOWS)


@task
def lfrontend(ctx:Context) -> None:
    """open a localhosted streamlit frontend"""
    ctx.run(f"uv run streamlit run src/{PROJECT_NAME}/frontend.py -- --localhost", echo=True, pty=not WINDOWS)

@task
def monitor(ctx:Context) -> None:
    """create data drift monitoring reports"""
    ctx.run(f"uv run src/{PROJECT_NAME}/data_drift.py", echo=True, pty=not WINDOWS)

@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data_preprocessed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("cd docs && uv run mkdocs build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("cd docs && uv run mkdocs build serve", echo=True, pty=not WINDOWS)
