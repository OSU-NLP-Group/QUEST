import argparse
import asyncio
import glob
import json
import logging
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from litellm import completion
import litellm

DEFAULT_MODEL_ID = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# litellm.set_verbose = True
# try:
#     litellm._turn_on_debug()
# except AttributeError:
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     litellm.set_verbose = True

@dataclass
class RubricEvaluationResult:
    task_file: str
    prompt: str
    llm_raw_response: str
    parsed_response: Optional[Dict[str, Any]]
    question: str
    constraints: List[str]
    rubric_tree: Dict[str, Any]
    rubric_analysis: Dict[str, Any]
    rubric_summary: str
    judge_sections: Dict[str, str]
    error: Optional[str]
    # Normalized decision label extracted from the LLM's response.
    # "YES" means the rubric tree is accepted / correct, "NO" means rejected.
    decision: Optional[str] = None


class ClaudeVerifier:
    def __init__(self, model_id: str, region: str = None, max_tokens: int = 15000, temperature: float = 0.2):
        if not model_id.startswith(("bedrock/", "azure/", "openai/")):
            if region:
                model_id = f"bedrock/{model_id}"
            else:
                model_id = f"bedrock/{model_id}"
        
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def evaluate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        call_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 32768,
            "temperature": 1,
            "num_retries": 3,
            "thinking":{"type": "enabled", "budget_tokens": 16384},
        }
        if self.model_id.startswith("azure/"):
            api_key = os.environ.get("RUBRIC_VERIFIER_AZURE_API_KEY")
            api_base = os.environ.get("RUBRIC_VERIFIER_AZURE_API_BASE")
            api_version = os.environ.get("RUBRIC_VERIFIER_AZURE_API_VERSION")
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
            if api_version:
                call_kwargs["api_version"] = api_version
        elif self.model_id.startswith("bedrock/"):
            aws_access_key_id = os.environ.get("RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.environ.get("RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY")
            aws_region_name = os.environ.get("RUBRIC_VERIFIER_AWS_REGION_NAME")
            if aws_access_key_id:
                call_kwargs["aws_access_key_id"] = aws_access_key_id
            if aws_secret_access_key:
                call_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_region_name:
                call_kwargs["aws_region_name"] = aws_region_name
        else:
            api_key = os.environ.get("RUBRIC_VERIFIER_OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
            api_base = os.environ.get("RUBRIC_VERIFIER_API_BASE")
            if api_base:
                call_kwargs["api_base"] = api_base
        def _call_llm():
            response = completion(**call_kwargs)
            return response
        
        response = await asyncio.to_thread(_call_llm)
        text = response.choices[0].message.content

        for stop in ("\n",):
            if text.endswith(stop):
                text = text[: -len(stop)]

        return text.strip()


def flatten_rubric_tree(tree: Dict[str, Any]) -> str:
    lines: List[str] = []

    def format_label(node: Dict[str, Any]) -> str:
        name = node.get("node_name") or node.get("name") or "unnamed"
        description = textwrap.shorten(
            node.get("description", "").replace("\n", " "), width=120, placeholder="…"
        )
        critical = "critical" if node.get("critical") else "non-critical"
        aggregation = node.get("aggregation_strategy")
        aggregation_info = f" [{aggregation}]" if aggregation else ""
        return f"{name} ({critical}){aggregation_info}: {description}"

    def walk(node: Dict[str, Any], prefix: str = "", is_last: bool = True, depth: int = 0) -> None:
        label = format_label(node)
        if depth == 0:
            lines.append(label)
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{label}")

        children = node.get("children") or {}
        if isinstance(children, dict):
            children_list = list(children.values())
        else:
            children_list = list(children)

        for idx, child in enumerate(children_list):
            child_prefix = prefix + ("    " if is_last else "│   ")
            walk(child, child_prefix, idx == len(children_list) - 1, depth + 1)

    if "name" in tree or "node_name" in tree:
        roots = [tree]
    else:
        roots = list(tree.values())
    for idx, root in enumerate(roots):
        walk(root, "", idx == len(roots) - 1, depth=0)

    return "\n".join(lines)


def extract_judge_sections(response_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_section = stripped[4:].strip().lower()
            if current_section in ("analysis", "reasons", "decision"):
                sections[current_section] = []
            else:
                current_section = None
        elif current_section:
            sections[current_section].append(line)

    return {
        key: "\n".join(lines).strip()
        for key, lines in sections.items()
        if lines
    }


async def write_prompt_response_log(
    output_dir: str, basename: str, prompt: str, response: str
) -> None:
    def _write_log():
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, f"{basename}.txt")
        with open(target_path, "w", encoding="utf-8") as fout:
            fout.write("=== PROMPT ===\n")
            fout.write(prompt.strip() + "\n\n")
            fout.write("=== LLM RESPONSE ===\n")
            fout.write(response.strip() + "\n")
    
    await asyncio.to_thread(_write_log)

def build_prompt(
    task_name: str,
    question: str,
    constraints: List[str],
    rubric_summary: str,
    rubric_stats: Dict[str, Any],
) -> str:
    stats = []
    if "max_depth" in rubric_stats:
        stats.append(f"Max depth: {rubric_stats['max_depth']}")
    if "max_width" in rubric_stats:
        stats.append(f"Max width: {rubric_stats['max_width']}")
    if "width_by_level" in rubric_stats:
        level_summary = ", ".join(
            f"Level {level}: {count}" for level, count in rubric_stats["width_by_level"].items()
        )
        stats.append(f"Width by level: {level_summary}")

    constraints_block = "\n".join(f"- {c}" for c in constraints) if constraints else "No explicit constraints."

    prompt = f"""You are a **Rubric Verification Expert**. Your goal is to judge whether a proposed **Rubric Tree**:
1. Correctly aligns with the **Question** and its **Constraints**, and  
2. Fully obeys the **Rubric Tree Structural Requirements**.

Your evaluation must be rigorous and rule-based.

## Context
**Task Name:** {task_name}

**Proposed Question:**
{question}

**Constraints:**
{constraints_block}

**Rubric Tree Attributes:**
{"; ".join(stats) if stats else "No additional stats."}

**Rubric Tree Summary:**
{rubric_summary}

## Rubric Tree Structural Rules
When you evaluate, you MUST also check whether the Rubric Tree itself satisfies the following structural and scoring rules:

Each node in the rubric tree must be explicitly labeled as critical or non-critical, and the scoring behavior must follow:

1. Critical Node
    •   Represents an essential / mandatory criterion, including:
		•	  Explicitly stated constraints in the question
		•	  Professionally inferred requirements that domain experts universally consider necessary for producing a valid or meaningful solution.
    •   If a critical node fails, then its parent node automatically fails.
    •   No partial credit is allowed for failing a critical node.

2. Non-Critical Node
    •   Represents a less-critical / partial-credit criterion, such as:
		    •	  parallel evaluation, independent to other nodes in the same level
		    •	  additional required information that is not used as constraints of an item
		    •	  subnodes representing steps under a sequential node to allow partial scores
    •  Failing an individual non-critical node does not necessarily mandate a complete failure of the parent node. 
    •  Example: When the task requires finding multiple independent items (e.g. 4 hotels), each item node should be a non-critical parallel child of the root, so that partial credit can be awarded for partially correct item sets.
    

All intermediate nodes (i.e., nodes with children) must be designated as either sequential or parallel to determine how their children’s scores are aggregated:
1. Sequential Node
    •   Its children follow an explicit logical order. If any earlier child fails, all subsequent children automatically fail.
    •   Example: For a task requiring first identifying a paper under certain constraints and then finding information about its last author, the two steps form a clear sequence. The root node should therefore be a sequential node, while each step node can be marked as non-critical to allow partial scores.
    
2. Parallel Node
    •   All its children can be evaluated independently and do not have any logical order dependency
    •   Example: When asking for the author and the publication year of a book, these pieces of information can be evaluated independently. Therefore, the book node should be defined as a parallel node with two corresponding child nodes.

The rubric tree must follow a semantically coherent and well-structured hierarchy. Specifically:
1. Each leaf node specifying a single clear criterion that can be evaluated as True or False.
2. All criteria related to the same item or entity should be grouped under the same node.
3. The rubric tree should mirror the conceptual structure of the task.
4. The rubric tree is *not* required to include a dedicated node for validating URLs; the mere absence of a URL-validation node should **not** be treated as a structural error.

## Additional Rubric Validity Rules (beyond the structural rules above)
In addition to the structural rules, you must also enforce these rubric-quality constraints:

1. **No duplicate or redundant nodes**
    - Do **not** repeat the same factual requirement in multiple nodes.
    - Do **not** split a single logical requirement into multiple unnecessary sub-requirements (micro-nodes) when they are really one condition.
    - Example of what **not** to do: if the user requires "The director must have directed at least two films from 2017–2019.", you must not create four separate nodes such as "at least one film in 2017", "at least one film in 2018", "at least one film in 2019", and "minimum two films total". Instead, use a single node like "Directed ≥2 films between 2017–2019.".

2. **No hard-coded specific answers**
    - Rubric nodes should check **properties/conditions**, not encode specific answers that the solution is supposed to *discover* (e.g., do not hard-code a price, date, or name that only appears in the solution).
    - It **is allowed** to restate facts that are explicitly given in the question or constraints (e.g., a named company or an explicitly stated number).
    - It is also acceptable to include inherently unique facts that are uniquely determined by the question itself (e.g., "What is the capital of Ohio?", or "CEO of Company A").
    - It is **not allowed** to introduce facts that come only from the solution or from external knowledge that the question did not specify, nor to introduce specific item names, numbers, or outcomes when the question expects the model to **discover** them via reasoning or external search.

3. **Item-count limitation**
    - The rubric tree must not require evaluation of **more than 5 separate items**.
    - "Find all" style tasks must be rejected unless the domain guarantees that the total number of correct items is at most 5.

4. **Structure for "find all" tasks with deep trees**
    - For "find all"–type tasks where the rubric tree depth exceeds 2, all second-level nodes should correspond to "the k-th item" (e.g., the 1st item, 2nd item, etc.).
    - Each of these second-level "item" nodes must be marked as **non-critical**, so that partial credit can be given when only some items are correct.

5. **Avoid over-nesting independent attributes for the same entity**
    - When constraints involve extracting multiple independent attributes about the same entity, avoid organizing them as a long sequential dependency chain that exceeds three levels whenever possible.
    - Instead, organize such attribute checks as parallel sibling nodes; deep nesting of independent attributes is usually unnecessary.

## Evaluation Instructions
Please follow this three-step process to evaluate the rubric:

1.  **Step 1: Analysis (Deep Dive)**
    - Compare the Rubric Tree against the Question and Constraints item-by-item.
    - Check for **Completeness**: Does the rubric cover *all* required constraints and question parts?
    - Check for **Relevance**: Does the rubric contain extra/irrelevant nodes that were not asked for in the question? (Avoid "hallucinated" criteria).
    - Check for **Accuracy**: Do the specific checks in the rubric match the intent of the constraints?
    - Check for **Structural Validity**: Does the rubric obey all rubric-tree rules (critical vs non-critical usage, sensible sequential/parallel logic, atomic leaf criteria, coherent grouping)?

2.  **Step 2: Reasons (Synthesis)**
    - Summarize *why* the rubric is a match or a mismatch based on your analysis.
    - If there are errors, explicitly state what is missing, incorrect, superfluous, or structurally inconsistent with the rubric rules.

3.  **Step 3: Decision**
    - Output a strict binary decision: "YES" if and only if the rubric is both:
        - Well-aligned with the Question and Constraints, **and**
        - Structurally sound with respect to the Rubric Tree Structural Rules above.
    - Otherwise, output "NO".

## Output Format
Strictly follow this format for your response:

### Analysis
[Your detailed step-by-step comparison and reasoning goes here...]

### Reasons
[A concise summary of the key factors leading to your decision. E.g., "The rubric misses the constraint about X..." or "The rubric accurately covers all constraints and follows the rubric tree rules..."]

### Decision
[YES or NO]
"""
    return prompt.strip()


def parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        if start != -1:
            text = text[start:]

    if not text.endswith("}"):
        end = text.rfind("}")
        if end != -1:
            text = text[: end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Unable to parse JSON from LLM response.")
        return None


async def process_single_file(
    filepath: str,
    verifier: ClaudeVerifier,
    log_dir: Optional[str],
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> Optional[RubricEvaluationResult]:
    """Documentation omitted."""
    async with semaphore:
        logger.info("Evaluating %d/%d: %s", idx, total, filepath)
        def _read_json():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        
        data = await asyncio.to_thread(_read_json)

        metadata = data.get("metadata", {})
        question = data.get("proposed_question") or metadata.get("question") or ""
        constraints = data.get("constraints", [])
        # The data format:
        rubric_analysis_refined = data.get("rubric_tree_analysis_refined")
        if not rubric_analysis_refined:
            logger.warning("Skipping %s because no refined rubric tree was detected.", filepath)
            return None

        version = "refined"
        rubric_analysis = rubric_analysis_refined

        # If a log already exists for this file, skip re-evaluation
        # to avoid redundant Bedrock calls.
        if log_dir:
            basename = f"{Path(filepath).stem}__{version}"
            existing_log_path = os.path.join(log_dir, f"{basename}.txt")
            if os.path.exists(existing_log_path):
                logger.info(
                    "Skipping %s (%s) because log already exists at %s",
                    filepath,
                    version,
                    existing_log_path,
                )
                return None

        formatted_tree = rubric_analysis.get("formatted_tree", {})
        if not formatted_tree:
            logger.warning(
                "Skipping %s (%s) because no rubric tree was detected.",
                filepath,
                version,
            )
            return None

        rubric_summary = flatten_rubric_tree(formatted_tree)
        prompt = build_prompt(
            task_name=f"{os.path.basename(filepath)} [{version}]",
            question=question,
            constraints=constraints,
            rubric_summary=rubric_summary,
            rubric_stats=rubric_analysis,
        )

        try:
            response_text = await verifier.evaluate(prompt)
            parsed = parse_llm_response(response_text)
            error = None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Claude call failed for %s (%s): %s", filepath, version, exc
            )
            response_text = ""
            parsed = None
            error = str(exc)

        judge_sections = extract_judge_sections(response_text)

        # Try to extract a normalized binary decision ("YES" / "NO")
        decision_label: Optional[str] = None
        decision_block = judge_sections.get("decision", "").strip()
        if decision_block:
            # Take the last non-empty line, strip possible prefixes like "Decision:"
            last_line = ""
            for line in reversed(decision_block.splitlines()):
                stripped = line.strip()
                if stripped:
                    last_line = stripped
                    break
            if last_line:
                normalized = last_line.upper()
                # Remove a leading "DECISION:" if present
                if normalized.startswith("DECISION:"):
                    normalized = normalized[len("DECISION:") :].strip()
                # Remove markdown formatting (bold **text**, italic *text*, etc.)
                normalized = normalized.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
                # Remove any remaining whitespace
                normalized = normalized.strip()
                # Finally, only accept pure YES/NO
                if normalized in ("YES", "NO"):
                    decision_label = normalized

        if log_dir:
            # Use a version-specific basename so logs for different trees
            # of the same task file do not overwrite each other.
            basename = f"{Path(filepath).stem}__{version}"
            await write_prompt_response_log(log_dir, basename, prompt, response_text)

        return RubricEvaluationResult(
            task_file=os.path.abspath(filepath),
            prompt=prompt,
            llm_raw_response=response_text,
            parsed_response=parsed,
            question=question,
            constraints=constraints,
            rubric_tree=formatted_tree,
            rubric_analysis=rubric_analysis,
            rubric_summary=rubric_summary,
            judge_sections=judge_sections,
            error=error,
            decision=decision_label,
        )


async def evaluate_folder(
    folder: str,
    verifier: ClaudeVerifier,
    output_path: str,
    max_tasks: Optional[int] = None,
    log_dir: Optional[str] = None,
    workers: int = 5,
) -> None:
    files = sorted(
        glob.glob(os.path.join(folder, "**", "*_formatted.json"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"No formatted JSON files found in {folder}")

    if max_tasks:
        files = files[:max_tasks]

    logger.info("Found %d files to process", len(files))
    logger.info("Using %d concurrent workers", workers)
    print("=" * 80)
    semaphore = asyncio.Semaphore(workers)
    tasks = [
        process_single_file(filepath, verifier, log_dir, semaphore, idx, len(files))
        for idx, filepath in enumerate(files, start=1)
    ]
    results_raw = await asyncio.gather(*tasks)
    results: List[RubricEvaluationResult] = [r for r in results_raw if r is not None]

    # After collecting all results, copy accepted (YES) original JSONs
    # and prepare simple statistics for a filter report.
    accepted_dir = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), "accepted_trajectories"
    )
    os.makedirs(accepted_dir, exist_ok=True)
    def _copy_files():
        accepted_ids: List[str] = []
        filtered_ids: List[str] = []
        accepted_count = 0
        filtered_count = 0
        
        for result in results:
            # Derive the original (non-formatted) trajectory path from the formatted one.
            # If the formatted file lives under a directory named "formatted", the source
            # file is in the parent of that directory (e.g. .../valid/formatted/foo.json
            # -> .../valid/foo.json). Otherwise, same directory as the formatted file.
            task_path = Path(result.task_file)
            stem = task_path.stem  # e.g., traj_123_..._formatted
            orig_stem = stem[: -len("_formatted")] if stem.endswith("_formatted") else stem
            traj_id = orig_stem  # Use filename stem (without _formatted) as ID.
            orig_filename = orig_stem + ".json"
            # Walk up the path through any "formatted" or "refined" directories
            # to find the base directory containing original trajectory files.
            orig_dir = task_path.parent
            while orig_dir.name in ("formatted", "refined"):
                orig_dir = orig_dir.parent
            orig_path = orig_dir / orig_filename

            if result.decision == "YES":
                if orig_path.exists():
                    target_path = Path(accepted_dir) / orig_filename
                    try:
                        shutil.copy2(orig_path, target_path)
                        accepted_count += 1
                        accepted_ids.append(traj_id)
                    except OSError as copy_exc:  # noqa: PERF203
                        logger.error(
                            "Failed to copy accepted trajectory %s to %s: %s",
                            orig_path,
                            target_path,
                            copy_exc,
                        )
                    except Exception as exc:
                        print(e)
                else:
                    logger.warning(
                        "Original trajectory file for %s not found at %s",
                        result.task_file,
                        orig_path,
                    )
            elif result.decision == "NO":
                filtered_count += 1
                filtered_ids.append(traj_id)
        
        return accepted_count, filtered_count, accepted_ids, filtered_ids

    accepted_count, filtered_count, accepted_ids, filtered_ids = await asyncio.to_thread(_copy_files)

    serialized = [
        {
            "task_file": result.task_file,
            "question": result.question,
            "constraints": result.constraints,
            "rubric_tree_summary": result.rubric_summary,
            "rubric_tree": result.rubric_tree,
            "rubric_analysis": result.rubric_analysis,
            "prompt": result.prompt,
            "parsed_response": result.parsed_response,
            "llm_raw_response": result.llm_raw_response,
            "judge_sections": result.judge_sections,
            "decision": result.decision,
            "error": result.error,
        }
        for result in results
    ]
    def _write_output():
        with open(output_path, "w", encoding="utf-8") as outf:
            json.dump(serialized, outf, ensure_ascii=False, indent=2)
    
    await asyncio.to_thread(_write_output)
    logger.info("Wrote %d verification records to %s", len(serialized), output_path)

    # Write a compact filter report with basic statistics and IDs.
    report = {
        "total_evaluated": len(results),
        "accepted_count": accepted_count,
        "filtered_count": filtered_count,
        "accepted_ids": accepted_ids,
        "filtered_ids": filtered_ids,
    }
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), "rubric_filter_report.json"
    )
    def _write_report():
        with open(report_path, "w", encoding="utf-8") as report_out:
            json.dump(report, report_out, ensure_ascii=False, indent=2)
    
    await asyncio.to_thread(_write_report)
    logger.info(
        "Wrote rubric filter report to %s (accepted=%d, filtered=%d)",
        report_path,
        accepted_count,
        filtered_count,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Verify rubric trees via Claude 4.5 (Bedrock).")
    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        help="Absolute path to the directory containing *_formatted.json trajectory files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.getcwd(), "rubric_verification_results.json"),
        help="Path to write verification summaries.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("RUBRIC_VERIFIER_MODEL_NAME", DEFAULT_MODEL_ID),
        help="Model name in litellm format (e.g., bedrock/..., azure/..., gpt-4). Defaults to Claude 4.5.",
    )
    parser.add_argument(
        "--region",
        "-r",
        default=os.environ.get("RUBRIC_VERIFIER_AWS_REGION_NAME", "us-east-2"),
        help="AWS region for Bedrock runtime.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of files to verify (useful for dry runs).",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory to write prompt + response text logs.",
        default=None,
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=20,
        help="Number of concurrent workers for processing files.",
    )

    args = parser.parse_args()
    verifier = ClaudeVerifier(model_id=args.model, region=args.region)

    await evaluate_folder(
        folder=args.folder,
        verifier=verifier,
        output_path=args.output,
        max_tasks=args.max_tasks,
        log_dir=args.log_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    asyncio.run(main())

