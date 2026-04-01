"""
Utilities for dynamically loading an evaluation script and returning its
`evaluate_answer` coroutine function.

Usage
-----
from mind2web2.utils.load_eval_script import load_eval_script

eval_fn = load_eval_script("/path/to/my_eval_script.py")
result  = await eval_fn(...)
"""

import importlib.util
import sys
import uuid
import inspect
import asyncio
from pathlib import Path
from types import ModuleType


def load_eval_script(path: str):
    """
    Load an external evaluation script and return its `evaluate_answer`
    coroutine function.

    Parameters
    ----------
    path : str
        Filesystem path to the Python script that defines `async def evaluate_answer(...)`.

    Returns
    -------
    Callable
        A reference to the `evaluate_answer` coroutine function.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If the module spec cannot be created.
    AttributeError
        If `evaluate_answer` is missing.
    TypeError
        If `evaluate_answer` is not an async function or has an invalid signature.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # Generate a unique module name to avoid namespace collisions.
    module_name = f"mind2web2_dynamic_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {path}")

    module: ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Register the module so that any relative imports inside the script work.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # --------------------------------------------------------------------- #
    # Validate the presence and signature of `evaluate_answer`.             #
    # --------------------------------------------------------------------- #
    if not hasattr(module, "evaluate_answer"):
        raise AttributeError(f"{path} does not define `evaluate_answer`")

    evaluate_answer = module.evaluate_answer  # type: ignore[attr-defined]

    if not asyncio.iscoroutinefunction(evaluate_answer):
        raise TypeError("`evaluate_answer` must be defined with `async def`")

    required_params = {
        "client",
        "answer",
        "agent_name",
        "answer_name",
        "cache",
        "semaphore",
        "logger",
    }
    sig = inspect.signature(evaluate_answer)
    missing = required_params - set(sig.parameters)
    if missing:
        raise TypeError(
            f"`evaluate_answer` is missing required parameters: {', '.join(sorted(missing))}"
        )

    return evaluate_answer
