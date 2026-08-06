"""Rollback: reactivate a previously trained (but not currently active)
model version. Older versions are never deleted, only deactivated, so
rollback is always possible."""
from __future__ import annotations

from rich.console import Console

from app.database.models import ModelVersion
from app.database.repositories import ModelVersionRepository
from app.database.session import session_scope

console = Console()


def rollback_to_version(version_label: str) -> bool:
    with session_scope() as session:
        all_versions = ModelVersionRepository.list_all(session)
        target = next((v for v in all_versions if v.version_label == version_label), None)
        if target is None:
            console.print(f"[red]No model version found with label '{version_label}'.[/red]")
            return False

        for v in all_versions:
            if v.model_type == target.model_type:
                v.is_active = False
        target.is_active = True

    console.print(f"[green]Rolled back {target.model_type} to version {version_label}.[/green]")
    return True


def list_versions() -> list[ModelVersion]:
    with session_scope() as session:
        return ModelVersionRepository.list_all(session)
