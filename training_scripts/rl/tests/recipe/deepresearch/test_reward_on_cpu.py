import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


RECIPE_DIR = Path(__file__).resolve().parents[3] / "recipe" / "deepresearch"
sys.path.insert(0, str(RECIPE_DIR))
spec = importlib.util.spec_from_file_location("deepresearch_reward", RECIPE_DIR / "reward.py")
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


class TestComputeScoreEvalFallbacks(unittest.IsolatedAsyncioTestCase):
    async def test_objective_config_is_the_default_for_openended_evaluation(self):
        base_score = AsyncMock(return_value={"score": 1.0})

        with (
            patch.dict(os.environ, {"OPENENDED_EVAL_LLM_MODEL_NAME": ""}),
            patch.object(reward, "_compute_base_score", base_score),
            patch.object(reward, "setup_llm_client_env"),
        ):
            result = await reward.compute_score(
                data_source="quest_obj",
                solution_str="<answer>Paris</answer>",
                ground_truth={"answer": "Paris"},
                eval_llm_address="http://objective:8000",
                eval_llm_model="objective-model",
                enable_inline_citation_score=False,
            )

        call_kwargs = base_score.await_args.kwargs
        self.assertEqual(call_kwargs["eval_llm_addresses"], "http://objective:8000")
        self.assertEqual(call_kwargs["openended_eval_llm_addresses"], "http://objective:8000")
        self.assertEqual(call_kwargs["openended_eval_llm_model"], "objective-model")
        self.assertEqual(result["base_score"], 1.0)

    async def test_explicit_openended_config_overrides_objective_config(self):
        openended_score = AsyncMock(return_value={"score": 1.0})

        with (
            patch.dict(os.environ, {"OPENENDED_EVAL_LLM_MODEL_NAME": "environment-model"}),
            patch.object(reward, "_has_eval_llm_backend", return_value=True) as has_backend,
            patch.object(reward, "drb_compute_score_openended", openended_score),
            patch.object(reward, "setup_llm_client_env"),
        ):
            result = await reward.compute_score(
                data_source="quest_openended",
                solution_str="<answer>Paris</answer>",
                ground_truth={"type": "openended"},
                eval_llm_address="http://objective:8000",
                eval_llm_model="objective-model",
                openended_eval_llm_address="http://openended:9000",
                openended_eval_llm_model="openended-model",
                enable_inline_citation_score=False,
            )

        call_kwargs = openended_score.await_args.kwargs
        self.assertEqual(call_kwargs["eval_llm_addresses"], "http://openended:9000")
        self.assertEqual(call_kwargs["eval_llm_model"], "openended-model")
        has_backend.assert_called_once_with(
            "http://openended:9000", model_name="openended-model", profile="openended"
        )
        self.assertEqual(result["base_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
