import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "quantum_systems_2024_2025"
TASK_DESCRIPTION = (
    "Identify three distinct quantum computing systems that were announced or developed between January 1, 2024 and "
    "December 31, 2025. Each system must have at least 50 qubits and use a clearly specified quantum computing "
    "technology type (such as superconducting, trapped-ion, or neutral atom). The three systems must be from three "
    "different organizations or collaborative groups. At least two different quantum computing technology types must "
    "be represented among the three systems. For each system, provide the system name, organization/developer, exact "
    "qubit count, technology type, specific announcement date, and at least one reference URL from an official source "
    "or credible technology news outlet."
)

DATE_RANGE_START = datetime(2024, 1, 1)
DATE_RANGE_END = datetime(2025, 12, 31)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SystemItem(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None
    qubit_count: Optional[str] = None
    technology_type: Optional[str] = None
    announcement_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SystemsExtraction(BaseModel):
    systems: List[SystemItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_systems() -> str:
    return """
    Extract up to all quantum computing systems mentioned in the answer.
    For each system, return an object with the following fields exactly:
    - name: The system/device/processor name as stated in the answer (string).
    - organization: The organization or developer (or collaborative group) credited (string).
    - qubit_count: The exact qubit count as a single specific number (string). If the answer gives a range or approximation, still extract the text, but it's not considered exact.
    - technology_type: The stated technology type (e.g., superconducting, trapped-ion, neutral atom, photonic, spin qubit, etc.) (string).
    - announcement_date: The specific announcement or development date as written in the answer (string). If multiple dates are mentioned, choose the primary announcement date.
    - reference_urls: An array of at least one URL pointing to official sources or credible technology news outlets. Extract only URLs explicitly present in the answer text.

    Return a JSON object:
    {
      "systems": [
        { ... }, { ... }, ...
      ]
    }

    Rules:
    - Do not invent information not present in the answer.
    - If a field is missing for a system, set it to null (except reference_urls which should be an empty array if none are present).
    - For URLs, include full URLs. If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_integers(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\b(\d{1,6})\b", text)]


def parse_exact_qubit_count(qtext: Optional[str]) -> Optional[int]:
    """
    Return a single integer if the text clearly contains exactly one integer (e.g., '127', '127 qubits').
    If multiple integers or no integers, return None (not exact).
    """
    if not qtext:
        return None
    nums = extract_integers(qtext)
    if len(nums) == 1:
        return nums[0]
    return None


def parse_any_qubit_number(qtext: Optional[str]) -> Optional[int]:
    """
    Return a best-effort qubit number for comparisons (e.g., >= 50).
    If multiple integers present (e.g., in a range '50-60'), return the max to favor >= checks.
    """
    if not qtext:
        return None
    nums = extract_integers(qtext)
    if not nums:
        return None
    return max(nums)


def try_parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            # Normalize month-year or year-only to first day of that month/year for range checks
            if fmt in ("%B %Y", "%b %Y"):
                dt = datetime(dt.year, dt.month, 1)
            if fmt == "%Y":
                dt = datetime(dt.year, 1, 1)
            return dt
        except Exception:
            continue
    return None


def date_in_required_range(dt: Optional[datetime]) -> bool:
    if not dt:
        return False
    return DATE_RANGE_START <= dt <= DATE_RANGE_END


def take_first_three_systems(systems: List[SystemItem]) -> List[SystemItem]:
    first_three = systems[:3]
    # Pad with empty placeholders to ensure 3 entries for per-system checks tree, but global checks will use non-empty counts
    while len(first_three) < 3:
        first_three.append(SystemItem())
    return first_three


def count_identified_systems(systems: List[SystemItem]) -> int:
    return sum(1 for s in systems if s.name and s.name.strip())


def distinct_names_ok(systems: List[SystemItem]) -> bool:
    names = [normalize_text(s.name) for s in systems if s.name and s.name.strip()]
    return len(names) == 3 and len(set(names)) == 3


def distinct_orgs_ok(systems: List[SystemItem]) -> bool:
    orgs = [normalize_text(s.organization) for s in systems if s.organization and s.organization.strip()]
    return len(orgs) == 3 and len(set(orgs)) == 3


def technology_diversity_ok(systems: List[SystemItem]) -> bool:
    techs = [normalize_text(s.technology_type) for s in systems if s.technology_type and s.technology_type.strip()]
    return len(set([t for t in techs if t])) >= 2


# --------------------------------------------------------------------------- #
# Per-system verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_system(
    evaluator: Evaluator,
    parent_node,
    system: SystemItem,
    index: int,
) -> None:
    """
    Build the sub-tree for one system and perform checks/verification.
    """
    sys_node = evaluator.add_parallel(
        id=f"system_{index+1}",
        desc=f"System {index+1} satisfies all per-system requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Name provided (critical)
    evaluator.add_custom_node(
        result=bool(system.name and system.name.strip()),
        id=f"s{index+1}_name_provided",
        desc=f"Provides a system name for System {index+1}",
        parent=sys_node,
        critical=True,
    )

    # 2) Organization provided (critical)
    evaluator.add_custom_node(
        result=bool(system.organization and system.organization.strip()),
        id=f"s{index+1}_organization_provided",
        desc=f"Provides an organization/developer (or collaborative group) for System {index+1}",
        parent=sys_node,
        critical=True,
    )

    # 3) Exact qubit count provided (critical)
    exact_q = parse_exact_qubit_count(system.qubit_count)
    evaluator.add_custom_node(
        result=exact_q is not None,
        id=f"s{index+1}_qubit_count_provided_exact",
        desc=f"Provides an exact (specific) qubit count for System {index+1}",
        parent=sys_node,
        critical=True,
    )

    # 4) Qubits at least 50 (critical) – verify against sources if available
    q50_leaf = evaluator.add_leaf(
        id=f"s{index+1}_qubits_at_least_50",
        desc=f"System {index+1} qubit count is at least 50",
        parent=sys_node,
        critical=True,
    )
    q_claim_name = system.name or f"System {index+1}"
    q_claim = f"The system named '{q_claim_name}' has at least 50 qubits."
    await evaluator.verify(
        claim=q_claim,
        node=q50_leaf,
        sources=system.reference_urls if system.reference_urls else None,
        additional_instruction=(
            "Verify from the provided source(s) whether the device/system's qubit count is 50 or more. "
            "Allow equivalent phrasing like 'n-qubit', and device nicknames (e.g., processor names). "
            "If multiple numbers appear, focus on the stated system's qubit count."
        ),
    )

    # 5) Technology type specified (critical) – verify against sources
    tech_specified = bool(system.technology_type and system.technology_type.strip())
    if tech_specified:
        tech_leaf = evaluator.add_leaf(
            id=f"s{index+1}_technology_type_specified",
            desc=f"Provides a clearly specified quantum technology type for System {index+1} (e.g., superconducting, trapped-ion, neutral atom, etc.)",
            parent=sys_node,
            critical=True,
        )
        tech_claim = (
            f"The system '{q_claim_name}' uses the '{system.technology_type}' quantum technology type "
            f"(e.g., superconducting/transmon, trapped-ion, neutral atom/Rydberg, photonic, spin qubit, etc.)."
        )
        await evaluator.verify(
            claim=tech_claim,
            node=tech_leaf,
            sources=system.reference_urls if system.reference_urls else None,
            additional_instruction=(
                "Confirm the stated technology type (e.g., superconducting/transmon/fluxonium; trapped-ion; neutral atom/Rydberg; photonic; spin qubit). "
                "Allow common synonyms or subtypes. The claim should be clearly supported by the source."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"s{index+1}_technology_type_specified",
            desc=f"Provides a clearly specified quantum technology type for System {index+1} (e.g., superconducting, trapped-ion, neutral atom, etc.)",
            parent=sys_node,
            critical=True,
        )

    # 6) Announcement date provided (critical)
    evaluator.add_custom_node(
        result=bool(system.announcement_date and system.announcement_date.strip()),
        id=f"s{index+1}_announcement_date_provided",
        desc=f"Provides a specific announcement date for System {index+1}",
        parent=sys_node,
        critical=True,
    )

    # 7) Date in range 2024–2025 (critical) – verify against sources
    date_leaf = evaluator.add_leaf(
        id=f"s{index+1}_date_in_range_2024_2025",
        desc=f"System {index+1} announcement/development date is between January 1, 2024 and December 31, 2025 (inclusive)",
        parent=sys_node,
        critical=True,
    )
    date_str = system.announcement_date or ""
    date_claim = (
        f"The announcement or development date for '{q_claim_name}' is '{date_str}', and it falls between "
        f"January 1, 2024 and December 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=system.reference_urls if system.reference_urls else None,
        additional_instruction=(
            "Confirm the stated announcement or development date from the source(s). "
            "If the page date or press release date indicates 2024 or 2025, consider it in range."
        ),
    )

    # 8) Reference URL provided (critical)
    evaluator.add_custom_node(
        result=bool(system.reference_urls and len(system.reference_urls) > 0),
        id=f"s{index+1}_reference_url_provided",
        desc=f"Provides at least one reference URL for System {index+1} from an official source or credible technology news outlet",
        parent=sys_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an answer for the quantum systems (2024–2025) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract systems from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_systems(),
        template_class=SystemsExtraction,
        extraction_name="systems_extraction",
    )

    # Select first three systems (padding placeholders to build uniform per-system subtrees)
    first_three = take_first_three_systems(extracted.systems)

    # Build per-system verification subtree
    for idx, sys in enumerate(first_three):
        await verify_system(evaluator, root, sys, idx)

    # Global constraints checks
    # A) Exactly three systems identified (no fewer, no more) – count only those with names
    exactly_three_leaf = evaluator.add_custom_node(
        result=(count_identified_systems(first_three) == 3),
        id="exactly_three_systems",
        desc="Identifies exactly three quantum computing systems (no fewer, no more)",
        parent=root,
        critical=True,
    )

    # B) Systems are distinct (by name)
    systems_distinct_leaf = evaluator.add_custom_node(
        result=distinct_names_ok(first_three),
        id="systems_are_distinct",
        desc="The three identified systems are distinct (not the same system repeated)",
        parent=root,
        critical=True,
    )

    # C) Organization diversity (three distinct organizations)
    org_diversity_leaf = evaluator.add_custom_node(
        result=distinct_orgs_ok(first_three),
        id="organization_diversity",
        desc="The three systems are from three different organizations or collaborative groups",
        parent=root,
        critical=True,
    )

    # D) Technology diversity (at least two different technology types)
    tech_diversity_leaf = evaluator.add_custom_node(
        result=technology_diversity_ok(first_three),
        id="technology_diversity",
        desc="At least two different quantum computing technology types are represented among the three systems",
        parent=root,
        critical=True,
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "identified_systems_count": count_identified_systems(first_three),
            "names": [s.name for s in first_three],
            "organizations": [s.organization for s in first_three],
            "technology_types": [s.technology_type for s in first_three],
            "qubit_counts": [s.qubit_count for s in first_three],
            "announcement_dates": [s.announcement_date for s in first_three],
            "reference_url_counts": [len(s.reference_urls) for s in first_three],
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    return evaluator.get_summary()