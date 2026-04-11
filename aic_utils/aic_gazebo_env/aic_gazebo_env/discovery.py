"""Executable and workspace discovery helpers for the Gazebo runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable


@dataclass(frozen=True)
class ExecutableResolution:
    """Resolved executable metadata with discovery diagnostics."""

    requested_name: str
    resolved_path: str | None
    searched_locations: tuple[str, ...]
    discovered_setup_script: str | None
    setup_explanation: str | None
    status: str | None
    setup_status: str | None


@dataclass(frozen=True)
class SetupScriptDiscovery:
    """Nearby workspace setup-script discovery result."""

    script_path: str | None
    explanation: str | None
    searched_locations: tuple[str, ...]


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root from a starting path."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "docs").is_dir() and (candidate / "aic_utils").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start or Path(__file__)}")


def discover_setup_script(
    *,
    repo_root: Path | None = None,
    extra_search_roots: Iterable[Path] = (),
) -> SetupScriptDiscovery:
    """Find a relevant colcon/ament setup script near this repo or common overlays."""
    repo = find_repo_root(repo_root or Path(__file__))
    search_roots: list[Path] = []
    for candidate in (repo, *repo.parents, *extra_search_roots):
        if candidate not in search_roots:
            search_roots.append(candidate)
    for overlay in (
        Path("/tmp/ws_overlay"),
        Path("/ws_aic"),
        Path.home() / "ws_aic",
    ):
        if overlay not in search_roots:
            search_roots.append(overlay)

    searched_locations: list[str] = []
    for root in search_roots:
        for relative in ("install/setup.bash", "install/local_setup.bash"):
            candidate = root / relative
            searched_locations.append(str(candidate))
            if candidate.exists():
                if root == repo:
                    explanation = f"found repo-local workspace setup script at {candidate}"
                elif root in repo.parents:
                    explanation = f"found parent-workspace setup script at {candidate}"
                else:
                    explanation = f"found nearby overlay setup script at {candidate}"
                return SetupScriptDiscovery(
                    script_path=str(candidate),
                    explanation=explanation,
                    searched_locations=tuple(searched_locations),
                )

    return SetupScriptDiscovery(
        script_path=None,
        explanation=None,
        searched_locations=tuple(searched_locations),
    )


def resolve_gz_executable(
    executable: str = "gz",
    *,
    repo_root: Path | None = None,
) -> ExecutableResolution:
    """Resolve the Gazebo CLI executable path with setup diagnostics."""
    setup = discover_setup_script(repo_root=repo_root)
    candidates: list[str] = []
    searched_locations: list[str] = []
    if executable:
        candidates.append(executable)
    if executable == "gz":
        for candidate in (
            shutil.which("gz"),
            "/usr/bin/gz",
            "/usr/local/bin/gz",
            "/snap/bin/gz",
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        resolved = _resolve_candidate(candidate)
        searched_locations.append(str(candidate))
        if resolved is not None:
            return ExecutableResolution(
                requested_name=executable,
                resolved_path=resolved,
                searched_locations=tuple(searched_locations),
                discovered_setup_script=setup.script_path,
                setup_explanation=setup.explanation,
                status=None,
                setup_status=None,
            )

    setup_status = (
        "workspace_setup_script_found_but_not_sourced"
        if setup.script_path is not None
        else "no_workspace_setup_script_found"
    )
    return ExecutableResolution(
        requested_name=executable,
        resolved_path=None,
        searched_locations=tuple((*searched_locations, *setup.searched_locations)),
        discovered_setup_script=setup.script_path,
        setup_explanation=setup.explanation,
        status="gz_not_found",
        setup_status=setup_status,
    )


def resolve_transport_helper_executable(
    helper_executable: str | None = None,
    *,
    repo_root: Path | None = None,
) -> ExecutableResolution:
    """Resolve the transport helper from explicit, env, PATH, or workspace installs."""
    repo = find_repo_root(repo_root or Path(__file__))
    setup = discover_setup_script(repo_root=repo)

    requested_name = helper_executable or "aic_gz_transport_bridge"
    searched_locations: list[str] = []
    candidates: list[str] = []

    explicit_helper = helper_executable is not None
    if explicit_helper:
        candidates.append(helper_executable)
    else:
        for candidate in (
            os.environ.get("AIC_GZ_TRANSPORT_BRIDGE_EXECUTABLE"),
            shutil.which("aic_gz_transport_bridge"),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    if not explicit_helper:
        for workspace_root in _candidate_workspace_roots(repo):
            candidate = (
                workspace_root
                / "install"
                / "aic_gazebo_transport_bridge"
                / "lib"
                / "aic_gazebo_transport_bridge"
                / "aic_gz_transport_bridge"
            )
            candidate_text = str(candidate)
            if candidate_text not in candidates:
                candidates.append(candidate_text)

    for candidate in candidates:
        searched_locations.append(str(candidate))
        resolved = _resolve_candidate(candidate)
        if resolved is not None:
            return ExecutableResolution(
                requested_name=requested_name,
                resolved_path=resolved,
                searched_locations=tuple(searched_locations),
                discovered_setup_script=setup.script_path,
                setup_explanation=setup.explanation,
                status=None,
                setup_status=None,
            )

    setup_status = (
        "workspace_setup_script_found_but_not_sourced"
        if setup.script_path is not None
        else "no_workspace_setup_script_found"
    )

    return ExecutableResolution(
        requested_name=requested_name,
        resolved_path=None,
        searched_locations=tuple((*searched_locations, *setup.searched_locations)),
        discovered_setup_script=setup.script_path,
        setup_explanation=setup.explanation,
        status="helper_not_found",
        setup_status=setup_status,
    )


def _resolve_candidate(candidate: str) -> str | None:
    if not candidate:
        return None
    if os.sep not in candidate:
        which = shutil.which(candidate)
        if which:
            return which
    path = Path(candidate).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def _candidate_workspace_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in (repo_root, *repo_root.parents, Path("/tmp/ws_overlay"), Path("/ws_aic"), Path.home() / "ws_aic"):
        if candidate not in roots:
            roots.append(candidate)
    return roots
