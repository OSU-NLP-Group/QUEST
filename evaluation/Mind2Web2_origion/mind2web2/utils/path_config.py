"""
Centralised project-relative path management for Mind2Web2.

Typical usage
-------------
from pathlib import Path
from mind2web2.utils.path_config import PathConfig

project_root = Path(__file__).resolve().parents[2]   # adapt as needed
paths = PathConfig(project_root)

# Override anything you like:
paths.apply_overrides(cache_root=Path("/tmp/my_cache"))

print(paths.eval_scripts_root)
print(paths.default_script_for("task_001"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """
    Holds every project-relative directory in one place.

    All attributes are absolute `Path` objects and never contain `~`.
    """
    project_root: Path

    # Dataset subtree
    dataset_root: Path = field(init=False)
    answers_root: Path = field(init=False)
    eval_scripts_root: Path = field(init=False)
    tasks_root: Path = field(init=False)
    eval_results_root: Path = field(init=False)
    cache_root: Path = field(init=False)
    
    # Evaluation version
    eval_version: str = field(init=False)

    # Scripts
    run_eval_script: Path = field(init=False)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve()

        # Dataset
        self.dataset_root = self.project_root / "dataset"
        self.answers_root = self.project_root / "answers"

        self.eval_scripts_root = self.project_root / "eval_scripts"
        self.tasks_root = self.project_root / "tasks"
        self.eval_results_root = self.project_root / "eval_results"

        self.cache_root = self.project_root / "cache"
        
        # Default eval version
        self.eval_version = "2025_07_14"

        # Scripts
        self.run_eval_script = self.project_root / "run_eval.py"


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def default_script_for(self, task_id: str) -> Path:
        """Return `<eval_scripts_root>/<eval_version>/<task_id>.py`."""
        return self.eval_scripts_root / self.eval_version / f"{task_id}.py"

    def apply_overrides(
        self,
        *,
        dataset_root: Optional[Path] = None,
        answers_root: Optional[Path] = None,
        eval_scripts_root: Optional[Path] = None,
        tasks_root: Optional[Path] = None,
        eval_results_root: Optional[Path] = None,
        cache_root: Optional[Path] = None,
        run_eval_script: Optional[Path] = None,
        eval_version: Optional[str] = None,
    ) -> None:
        """
        Overwrite selected directories in-place.
        All arguments are absolute or will be resolved/expanded.
        """
        if dataset_root is not None:
            self.dataset_root = dataset_root.expanduser().resolve()
        if answers_root is not None:
            self.answers_root = answers_root.expanduser().resolve()
        if eval_scripts_root is not None:
            self.eval_scripts_root = eval_scripts_root.expanduser().resolve()
        if tasks_root is not None:
            self.tasks_root = tasks_root.expanduser().resolve()
        if eval_results_root is not None:
            self.eval_results_root = eval_results_root.expanduser().resolve()
        if cache_root is not None:
            self.cache_root = cache_root.expanduser().resolve()
        if run_eval_script is not None:
            self.run_eval_script = run_eval_script.expanduser().resolve()
        if eval_version is not None:
            self.eval_version = eval_version

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"
