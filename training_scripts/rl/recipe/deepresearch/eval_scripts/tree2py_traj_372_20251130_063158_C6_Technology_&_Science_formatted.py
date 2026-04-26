import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from urllib.parse import urlsplit
from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "laptop_requirements_comparison_4_universities"
TASK_DESCRIPTION = """For a student considering multiple engineering programs, identify the minimum laptop requirements specified by each of the following four universities for their engineering students:

1. University of Texas Austin - Mechanical Engineering program
2. University of Washington - College of Engineering
3. Penn State - College of Engineering
4. Michigan State University - Bagley College of Engineering

For each university, provide the following specifications with supporting reference URLs:
- Processor: Minimum processor requirement (include specific model/generation if specified)
- RAM: Minimum RAM capacity requirement
- Storage: Minimum storage capacity and type (SSD/HDD)
- Display: Minimum display size and/or resolution
- Operating System/Graphics: Operating system preference or graphics card requirements (if specified)

Format your answer as a structured comparison with clearly labeled sections for each university and component category. Each specification must include a reference URL from the university's official requirements page.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SpecItem(BaseModel):
    spec: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversitySpec(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    processor: Optional[SpecItem] = None
    ram: Optional[SpecItem] = None
    storage: Optional[SpecItem] = None
    display: Optional[SpecItem] = None
    os_or_graphics: Optional[SpecItem] = None


class SpecsExtraction(BaseModel):
    ut_austin_mechanical_engineering: Optional[UniversitySpec] = None
    uw_college_engineering: Optional[UniversitySpec] = None
    penn_state_engineering: Optional[UniversitySpec] = None
    msu_bagley_engineering: Optional[UniversitySpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
Extract the minimum laptop requirement details exactly as stated in the answer for each of the following sections. For each university/program, extract the minimum requirement (not recommendations) for each component, along with the reference URLs explicitly cited in the answer for that component.

Universities/programs to extract (use these exact keys in the JSON output):
- ut_austin_mechanical_engineering: "University of Texas Austin – Mechanical Engineering"
- uw_college_engineering: "University of Washington – College of Engineering"
- penn_state_engineering: "Penn State – College of Engineering"
- msu_bagley_engineering: "Michigan State University – Bagley College of Engineering"

For each university, extract:
- university_name: the university name as written in the answer
- program_name: the program/college name as written in the answer
- processor: { spec: the minimum processor requirement string; urls: an array of URLs cited for processor }
- ram: { spec: the minimum RAM requirement string; urls: an array of URLs cited for RAM }
- storage: { spec: the minimum storage requirement string, including type (SSD/HDD) if present; urls: an array of URLs cited for storage }
- display: { spec: the minimum display requirement string (size or resolution), or "not specified" if the answer explicitly says it's not specified; urls: an array of URLs cited for display }
- os_or_graphics: { spec: the operating system and/or graphics requirements string, or "not specified" if the answer explicitly says it's not specified; urls: an array of URLs cited for OS/graphics }

Rules:
- Only extract information that appears in the answer. Do not infer or invent values.
- Prefer "minimum requirement" phrasing. If the answer includes both recommended and minimum, extract the minimum.
- For URLs: extract only URLs explicitly present in the answer text for that specific component (accept plain URLs or markdown links).
- If the answer omits a component entirely for a university, set that component's 'spec' to null and 'urls' to [].
- If the answer explicitly states a component is "not specified", set 'spec' exactly to "not specified" and include the cited official page URLs that support this.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def any_official_url(urls: List[str], allowed_domain_substrings: List[str]) -> bool:
    """Return True if any url's domain contains one of the allowed substrings."""
    for u in urls:
        try:
            host = urlsplit(u).netloc.lower()
            if any(substr in host for substr in allowed_domain_substrings):
                return True
        except Exception:
            continue
    return False


def safe_spec(item: Optional[SpecItem]) -> Tuple[str, List[str]]:
    if item is None:
        return "", []
    return (item.spec or "").strip(), item.urls or []


def is_not_specified_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"", "n/a", "na", "none", "not specified", "not specified by the university", "not specified in the source"}


def comp_phrase(component_key: str) -> str:
    mapping = {
        "processor": "processor",
        "ram": "RAM",
        "storage": "storage (capacity and type if available)",
        "display": "display (size and/or resolution)",
        "os_or_graphics": "operating system and/or graphics"
    }
    return mapping.get(component_key, component_key)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_formatting_and_structure(evaluator: Evaluator, root) -> None:
    node = evaluator.add_leaf(
        id="formatting_and_structure",
        desc="Answer is formatted as a structured comparison with clearly labeled sections for each university and each required component category",
        parent=root,
        critical=True,
    )
    claim = (
        "The answer is organized as a structured comparison, with clearly labeled sections for all four specified universities "
        "(University of Texas Austin – Mechanical Engineering, University of Washington – College of Engineering, "
        "Penn State – College of Engineering, and Michigan State University – Bagley College of Engineering). "
        "Within each section, the five component categories (Processor, RAM, Storage, Display, and Operating System/Graphics) are clearly labeled."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge strictly based on the provided answer: look for clear headings/labels per university and per component category."
    )


async def verify_component(
    evaluator: Evaluator,
    parent_node,
    *,
    uni_key: str,
    uni_label: str,
    component_key: str,
    spec_item: Optional[SpecItem],
    allowed_domains: List[str],
    require_present: bool,
    allow_not_specified: bool,
    university_slug_for_desc: str
) -> None:
    """
    Build verification for a single component under a university:
    - spec_present_(and matches) OR spec_present_or_noted_absent_per_constraints
    - official_reference_url_present
    """
    # Build component container (parallel)
    comp_id = f"{uni_key}_{component_key}"
    if component_key == "processor":
        comp_desc = f"{university_slug_for_desc}: Processor requirement is provided with an official source URL"
    elif component_key == "ram":
        comp_desc = f"{university_slug_for_desc}: RAM requirement is provided with an official source URL"
    elif component_key == "storage":
        comp_desc = f"{university_slug_for_desc}: Storage minimum capacity/type is provided with an official source URL"
    elif component_key == "display":
        comp_desc = f"{university_slug_for_desc}: Display minimum size/resolution is provided (or explicitly not specified) with an official source URL"
    else:
        comp_desc = f"{university_slug_for_desc}: Operating system/graphics requirement is provided (or explicitly not specified) with an official source URL"

    comp_node = evaluator.add_parallel(
        id=comp_id,
        desc=comp_desc,
        parent=parent_node,
        critical=True
    )

    # Extract safe spec and urls
    spec_str, urls = safe_spec(spec_item)

    # 1) Spec presence / match (with constraints about minimum vs recommendation)
    # Decide which description to use
    if allow_not_specified:
        spec_leaf_desc = f"{university_slug_for_desc}: {component_key} requirement information matches constraints (or explicitly noted as not specified)"
        leaf_id = f"{comp_id}_spec_present_or_noted_absent_per_constraints"
    else:
        spec_leaf_desc = f"{university_slug_for_desc}: {component_key} requirement information matches constraints"
        leaf_id = f"{comp_id}_spec_present_and_matches_constraints"

    spec_leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=spec_leaf_desc,
        parent=comp_node,
        critical=True
    )

    # Build claim for verification
    component_nice = comp_phrase(component_key)

    if allow_not_specified and is_not_specified_text(spec_str):
        claim = (
            f"The provided official source(s) for {uni_label} do not specify any explicit minimum {component_nice} requirement for engineering students."
        )
        add_ins = (
            f"Support this claim if the cited official page(s) do not state a minimum {component_nice} requirement. "
            f"If the page(s) clearly specify a minimum, then the claim should be incorrect."
        )
    else:
        if require_present and not spec_str:
            # If presence is required but missing, construct a claim that will fail under answer check
            claim = f"The answer provides a minimum {component_nice} requirement for {uni_label}."
            add_ins = (
                "This check should be marked incorrect if the answer does not clearly state a minimum requirement "
                "value for this component."
            )
        else:
            claim = (
                f"According to the cited official source(s), the minimum {component_nice} requirement for {uni_label} is '{spec_str}'."
            )
            add_ins = (
                "Verify that the cited page(s) explicitly state a minimum requirement consistent with the quoted value. "
                "Focus on minimum (not recommended) requirements; allow minor wording differences but the threshold must match."
            )

    await evaluator.verify(
        claim=claim,
        node=spec_leaf,
        sources=urls,
        additional_instruction=add_ins
    )

    # 2) Official reference URL present
    official_leaf_desc = "At least one reference URL is provided and it is from an official university/college source"
    official_id = f"{comp_id}_official_reference_url_present"
    official_ok = (len(urls) > 0) and any_official_url(urls, allowed_domains)

    evaluator.add_custom_node(
        result=official_ok,
        id=official_id,
        desc=official_leaf_desc,
        parent=comp_node,
        critical=True
    )


async def verify_university_block(
    evaluator: Evaluator,
    root,
    *,
    uni_key: str,
    uni_label: str,
    uni_spec: Optional[UniversitySpec],
    allowed_domains: List[str],
    display_allow_absent: bool,
    os_allow_absent: bool,
    university_slug_for_desc: str
) -> None:
    """
    Build the university-level verification subtree.
    """
    uni_node = evaluator.add_parallel(
        id=uni_key,
        desc=f"{uni_label}: required laptop specifications provided with official-source citations",
        parent=root,
        critical=True
    )

    # Resolve component spec items safely
    processor_item = uni_spec.processor if uni_spec and uni_spec.processor else SpecItem()
    ram_item = uni_spec.ram if uni_spec and uni_spec.ram else SpecItem()
    storage_item = uni_spec.storage if uni_spec and uni_spec.storage else SpecItem()
    display_item = uni_spec.display if uni_spec and uni_spec.display else SpecItem()
    os_item = uni_spec.os_or_graphics if uni_spec and uni_spec.os_or_graphics else SpecItem()

    # Each component per rubric
    await verify_component(
        evaluator, uni_node,
        uni_key=uni_key,
        uni_label=uni_label,
        component_key="processor",
        spec_item=processor_item,
        allowed_domains=allowed_domains,
        require_present=True,
        allow_not_specified=False,
        university_slug_for_desc=university_slug_for_desc
    )

    await verify_component(
        evaluator, uni_node,
        uni_key=uni_key,
        uni_label=uni_label,
        component_key="ram",
        spec_item=ram_item,
        allowed_domains=allowed_domains,
        require_present=True,
        allow_not_specified=False,
        university_slug_for_desc=university_slug_for_desc
    )

    await verify_component(
        evaluator, uni_node,
        uni_key=uni_key,
        uni_label=uni_label,
        component_key="storage",
        spec_item=storage_item,
        allowed_domains=allowed_domains,
        require_present=True,
        allow_not_specified=False,
        university_slug_for_desc=university_slug_for_desc
    )

    await verify_component(
        evaluator, uni_node,
        uni_key=uni_key,
        uni_label=uni_label,
        component_key="display",
        spec_item=display_item,
        allowed_domains=allowed_domains,
        require_present=not display_allow_absent,
        allow_not_specified=display_allow_absent,
        university_slug_for_desc=university_slug_for_desc
    )

    await verify_component(
        evaluator, uni_node,
        uni_key=uni_key,
        uni_label=uni_label,
        component_key="os_or_graphics",
        spec_item=os_item,
        allowed_domains=allowed_domains,
        require_present=not os_allow_absent,
        allow_not_specified=os_allow_absent,
        university_slug_for_desc=university_slug_for_desc
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point used by the evaluation harness.
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
        default_model=model
    )

    # Update root per rubric (critical root with explicit description)
    root.critical = True
    root.desc = "Provide a structured comparison of laptop requirements for all four specified universities/programs, including the required component categories and official-source reference URLs"

    # 1) Extraction
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=SpecsExtraction,
        extraction_name="laptop_requirements_extraction"
    )

    # Record allowed official domain patterns (for transparency)
    official_domain_map = {
        "ut_austin_mechanical_engineering": ["utexas.edu"],
        "uw_college_engineering": ["washington.edu"],
        "penn_state_engineering": ["psu.edu"],
        # Accept both Michigan State University (msu.edu) and Mississippi State University (msstate.edu)
        # to accommodate the "Bagley College of Engineering" nomenclature in the rubric text.
        "msu_bagley_engineering": ["msu.edu", "msstate.edu"],
    }
    evaluator.add_custom_info(
        info={"official_domain_patterns": official_domain_map},
        info_type="config",
        info_name="domain_policy"
    )

    # 2) Formatting/structure check
    await verify_formatting_and_structure(evaluator, root)

    # 3) University blocks
    # UT Austin – Mechanical Engineering
    await verify_university_block(
        evaluator, root,
        uni_key="ut_austin_mechanical_engineering",
        uni_label="University of Texas Austin – Mechanical Engineering",
        uni_spec=extracted_specs.ut_austin_mechanical_engineering,
        allowed_domains=official_domain_map["ut_austin_mechanical_engineering"],
        display_allow_absent=False,
        os_allow_absent=True,  # per rubric: OS/graphics may be not specified
        university_slug_for_desc="UT Austin – Mechanical Engineering"
    )

    # University of Washington – College of Engineering
    await verify_university_block(
        evaluator, root,
        uni_key="uw_college_engineering",
        uni_label="University of Washington – College of Engineering",
        uni_spec=extracted_specs.uw_college_engineering,
        allowed_domains=official_domain_map["uw_college_engineering"],
        display_allow_absent=False,
        os_allow_absent=False,
        university_slug_for_desc="UW – College of Engineering"
    )

    # Penn State – College of Engineering
    await verify_university_block(
        evaluator, root,
        uni_key="penn_state_engineering",
        uni_label="Penn State – College of Engineering",
        uni_spec=extracted_specs.penn_state_engineering,
        allowed_domains=official_domain_map["penn_state_engineering"],
        display_allow_absent=True,   # per rubric
        os_allow_absent=False,
        university_slug_for_desc="Penn State – College of Engineering"
    )

    # MSU – Bagley College of Engineering (rubric text)
    await verify_university_block(
        evaluator, root,
        uni_key="msu_bagley_engineering",
        uni_label="Michigan State University – Bagley College of Engineering",
        uni_spec=extracted_specs.msu_bagley_engineering,
        allowed_domains=official_domain_map["msu_bagley_engineering"],
        display_allow_absent=True,   # per rubric
        os_allow_absent=True,        # per rubric
        university_slug_for_desc="MSU – Bagley College of Engineering"
    )

    return evaluator.get_summary()