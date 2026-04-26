from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Union
from uuid import uuid4

# When this module is imported as top-level `obj_task_eval`, ensure sibling `tools` is importable.
_DEEPRESEARCH_DIR = Path(__file__).resolve().parents[2]
if str(_DEEPRESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPRESEARCH_DIR))

from tools.visit_tool import DeepResearchVisitTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_visit_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="visit",
            description="Visit webpage(s) and return the summary of the content.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "url": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The URL(s) of the webpage(s) to visit.",
                    ),
                    "goal": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The goal of the visit for webpage(s).",
                    ),
                },
                required=["url", "goal"],
            ),
        ),
    )


class Visit:
    """Compatibility wrapper for obj_task_eval evaluator.

    Keeps the historical sync API (`newcall` / `call`) while delegating
    implementation to `tools.visit_tool.DeepResearchVisitTool`.
    """

    def __init__(self, config: Dict[str, Any] | None = None, *args, **kwargs):
        tool_config: Dict[str, Any] = {}
        if isinstance(config, dict):
            tool_config.update(config)
        kw_config = kwargs.get("config")
        if isinstance(kw_config, dict):
            tool_config.update(kw_config)

        self._tool = DeepResearchVisitTool(
            config=tool_config,
            tool_schema=_build_visit_schema(),
        )

    def _run_sync(self, coro: Any) -> Any:
        _timeout = float(os.environ.get("EVAL_VISIT_TIMEOUT_SECONDS", "300"))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(asyncio.wait_for(coro, timeout=_timeout))
            except asyncio.TimeoutError:
                logger.warning("visit._run_sync: timed out after %.0fs; giving up", _timeout)
                return ""

        # If an event loop is already running in this thread, execute coroutine in a helper thread.
        result: Dict[str, Any] = {}
        error: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(asyncio.wait_for(coro, timeout=_timeout))
            except Exception as exc:
                error["exc"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=_timeout)
        if t.is_alive():
            # Thread did not finish in time; the daemon thread will be reaped when the
            # process exits.  Return empty string so the caller treats it as a miss.
            logger.warning("visit._run_sync: helper thread still alive after %.0fs; giving up", _timeout)
            return ""
        if isinstance(error.get("exc"), asyncio.TimeoutError):
            logger.warning("visit._run_sync: helper thread timed out after %.0fs; giving up", _timeout)
            return ""
        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    async def _execute_async(self, parameters: Dict[str, Any]) -> str:
        instance_id, _ = await self._tool.create(instance_id=str(uuid4()))
        try:
            tool_response, _, _ = await self._tool.execute(instance_id, parameters)
            text = getattr(tool_response, "text", None)
            return (text or "").strip()
        finally:
            try:
                await self._tool.release(instance_id)
            except Exception as exc:
                logger.debug("Visit tool release failed for %s: %s", instance_id, exc)

    def newcall(self, url: str) -> str:
        return self.call({"url": url, "goal": ""})

    def call(self, params: Union[str, Dict[str, Any]]) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return (
                    "[Visit] Invalid request format: Input must be a JSON object "
                    "containing 'url' and 'goal' fields"
                )

        if not isinstance(params, dict):
            return (
                "[Visit] Invalid request format: Input must be a JSON object "
                "containing 'url' and 'goal' fields"
            )

        url = params.get("url")
        goal = params.get("goal", "")
        if not url:
            return "[Visit Error] URL cannot be empty."

        return self._run_sync(self._execute_async({"url": url, "goal": goal}))
