import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hand_plane_prep"
TASK_DESCRIPTION = (
    "I have acquired a vintage Stanley Bailey hand plane that I plan to use for smoothing figured maple boards for a furniture project. "
    "The plane has the following features: it is marked \"No 5\" with a corrugated bottom, has a kidney-shaped hole in the lever cap, "
    "and shows two patent dates cast behind the frog (including \"PAT'D APR-19-10\" along with one earlier date). "
    "Please help me prepare this plane for use by providing the following information: "
    "(1) Identify the Stanley Bailey type of this plane based on the described features. "
    "(2) Confirm that the identified type is correct by verifying that the frog patent date configuration and lever cap design match the type's documented characteristics. "
    "(3) Specify the recommended blade bevel angle I should use for planing figured maple, considering that figured wood has reversing grain that can cause tearout. "
    "(4) State the acceptable moisture content range (in percentage) that the figured maple boards should have before I begin planing them for furniture-grade work. "
    "(5) Provide a reference URL to a Stanley plane type study or woodworking resource that documents the identification features of the plane type you identified."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlanePrepExtraction(BaseModel):
    # (1) Type identification
    plane_type: Optional[str] = None  # e.g., "Type 11"
    type_sources: List[str] = Field(default_factory=list)  # URLs cited for the type identification/features

    # (3) Bevel angle recommendation for figured maple
    bevel_angle: Optional[str] = None  # keep as string, e.g., "50°", "50-55°", "45° with 10° back bevel"
    bevel_sources: List[str] = Field(default_factory=list)

    # (4) Moisture content range
    moisture_range: Optional[str] = None  # keep as string, e.g., "6-8%", "8–10 %"
    moisture_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plane_prep() -> str:
    return """
    Extract the following information from the answer text. Do not invent anything; extract only what is explicitly present.

    1) plane_type: The Stanley Bailey type (e.g., "Type 11", "Type 16"). Return just the type label as a short string.
    2) type_sources: All URLs the answer cites as references to identify or confirm the plane type's distinguishing features (e.g., frog patent dates, lever cap shape). Include Stanley plane type study pages or woodworking resources. Return as an array of URLs.

    3) bevel_angle: The recommended blade bevel angle for planing figured maple to minimize tearout. Keep the exact phrasing/numbering from the answer (e.g., "50°", "45° with 10° back bevel", "50–55°"). Return as a string.
    4) bevel_sources: All URLs the answer cites as references for the bevel angle recommendation. Return as an array of URLs.

    5) moisture_range: The acceptable moisture content range (in %) for furniture-grade planing of figured maple before planing, as stated in the answer (e.g., "6–8%", "8–10%"). Return as a string.
    6) moisture_sources: All URLs the answer cites as references for the moisture content recommendation. Return as an array of URLs.

    URL extraction rules:
    - Extract only valid URLs explicitly present in the answer (plain or markdown links).
    - If a URL is missing a protocol, prepend "http://".
    - If any item above is not mentioned, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _prefer_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Use primary if present; otherwise fallback (deduplicated)."""
    if primary:
        return _unique_urls(primary)
    return _unique_urls(fallback)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_type_identification_stage(
    evaluator: Evaluator,
    parent,
    extracted: PlanePrepExtraction,
) -> None:
    """
    Step 1 (sequential): Type identification existence.
    """
    node = evaluator.add_parallel(
        id="plane_type_identification",
        desc="Identify the Stanley Bailey plane type based on the described features",
        parent=parent,
        critical=False,
    )

    # Leaf: plane type provided (critical for proceeding)
    evaluator.add_custom_node(
        result=bool(extracted.plane_type and extracted.plane_type.strip()),
        id="plane_type_provided",
        desc="Plane type is provided in the answer",
        parent=node,
        critical=True,
    )


async def build_reference_url_stage(
    evaluator: Evaluator,
    parent,
    extracted: PlanePrepExtraction,
) -> None:
    """
    Step 2 (sequential): Provide and validate a reference URL that documents identification features.
    """
    node = evaluator.add_parallel(
        id="reference_url",
        desc="Provide a reference URL that documents the Stanley Bailey plane type identification features",
        parent=parent,
        critical=True,  # This whole step is critical per rubric
    )

    # Leaf: at least one type reference URL present (critical)
    has_ref = bool(extracted.type_sources and len(extracted.type_sources) > 0)
    evaluator.add_custom_node(
        result=has_ref,
        id="reference_url_present",
        desc="At least one reference URL for plane type identification is provided",
        parent=node,
        critical=True,
    )

    # Leaf: verify that at least one provided URL is a relevant type study or woodworking resource
    ref_is_study = evaluator.add_leaf(
        id="reference_url_is_type_study",
        desc="Provided reference is a Stanley plane type study or woodworking resource documenting identification features",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This page is a Stanley plane type study or a woodworking resource that documents identification features "
            "of Stanley Bailey plane types (e.g., frog patent dates, lever cap designs) and/or the specific type identified."
        ),
        node=ref_is_study,
        sources=_unique_urls(extracted.type_sources),
        additional_instruction=(
            "Pass if any of the provided URLs clearly function as a type study or a woodworking resource that "
            "lists/demonstrates identification features for Stanley bench planes or the cited type. "
            "Pages like Stanley type studies (e.g., Hyperkitten, RexMill, George's Basement, Blood & Gore) or "
            "equivalent woodworking resources qualify."
        ),
    )


async def build_type_features_verification_stage(
    evaluator: Evaluator,
    parent,
    extracted: PlanePrepExtraction,
) -> None:
    """
    Step 3 (sequential): Confirm the identified type matches frog patent date configuration and lever cap design.
    """
    node = evaluator.add_parallel(
        id="type_features_verification",
        desc=(
            "Verify that the identified type matches the described frog patent date configuration "
            "('APR-19-10' and one earlier date; two dates cast) and lever cap kidney-shaped hole"
        ),
        parent=parent,
        critical=False,
    )

    plane_type = extracted.plane_type or ""
    sources = _unique_urls(extracted.type_sources)

    # Leaf: frog patent date configuration supported (critical)
    frog_dates_leaf = evaluator.add_leaf(
        id="frog_patent_dates_match_type",
        desc="Type's documented features include two frog-boss patent dates including 'APR-19-10'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The documented features of Stanley Bailey {plane_type} include two patent dates cast behind the frog "
            f"(including 'APR-19-10' and one earlier date)."
        ),
        node=frog_dates_leaf,
        sources=sources,
        additional_instruction=(
            "Check whether the cited source's description for the specified Stanley Bailey type explicitly mentions "
            "two patent dates cast behind the frog, one of which is 'APR-19-10' (the other is earlier). "
            "Allow minor wording/punctuation variants. If any provided page asserts this clearly for the given type, pass."
        ),
    )

    # Leaf: lever cap kidney-shaped hole supported (critical)
    lever_cap_leaf = evaluator.add_leaf(
        id="lever_cap_kidney_shape_match_type",
        desc="Type's documented features include a lever cap with a kidney-shaped hole",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The documented features of Stanley Bailey {plane_type} include a lever cap with a kidney-shaped hole."
        ),
        node=lever_cap_leaf,
        sources=sources,
        additional_instruction=(
            "Look for direct statements that the lever cap for this type has a 'kidney-shaped' hole "
            "(sometimes phrased as 'kidney hole' or similar). Allow minor phrasing variations."
        ),
    )


async def build_blade_angle_stage(
    evaluator: Evaluator,
    parent,
    extracted: PlanePrepExtraction,
) -> None:
    """
    Step 4 (sequential): Bevel angle recommendation for figured maple.
    """
    node = evaluator.add_parallel(
        id="blade_angle_specification",
        desc="Specify and support the recommended blade bevel angle for planing figured maple (to reduce tearout)",
        parent=parent,
        critical=False,
    )

    # Leaf: bevel angle provided (non-critical)
    evaluator.add_custom_node(
        result=bool(extracted.bevel_angle and extracted.bevel_angle.strip()),
        id="bevel_angle_provided",
        desc="Bevel angle recommendation is provided",
        parent=node,
        critical=False,
    )

    # Leaf: bevel angle is supported by cited sources (non-critical)
    bevel_supported_leaf = evaluator.add_leaf(
        id="bevel_angle_supported",
        desc="Bevel angle recommendation is supported by cited sources",
        parent=node,
        critical=False,
    )
    bevel_sources = _prefer_sources(extracted.bevel_sources, extracted.type_sources)
    await evaluator.verify(
        claim=(
            f"A blade bevel angle of {extracted.bevel_angle or ''} is recommended for planing figured maple to "
            f"reduce tearout from reversing grain."
        ),
        node=bevel_supported_leaf,
        sources=bevel_sources,
        additional_instruction=(
            "Accept if the cited page recommends an equivalent or very close angle/range for figured or difficult/reversing-grain hardwoods. "
            "Allow reasonable variations (e.g., if the source discusses effective cutting angle or back bevels that achieve the stated effect). "
            "Minor numeric differences (±2–3°) are acceptable when clearly equivalent guidance."
        ),
    )


async def build_moisture_stage(
    evaluator: Evaluator,
    parent,
    extracted: PlanePrepExtraction,
) -> None:
    """
    Step 5 (sequential): Moisture content range for furniture-grade planing.
    """
    node = evaluator.add_parallel(
        id="wood_moisture_requirement",
        desc="State and support acceptable moisture content range (%) before planing figured maple for furniture work",
        parent=parent,
        critical=False,
    )

    # Leaf: moisture content range provided (non-critical)
    evaluator.add_custom_node(
        result=bool(extracted.moisture_range and extracted.moisture_range.strip()),
        id="moisture_range_provided",
        desc="Moisture content range is provided",
        parent=node,
        critical=False,
    )

    # Leaf: moisture range supported by sources (non-critical)
    moisture_supported_leaf = evaluator.add_leaf(
        id="moisture_range_supported",
        desc="Moisture content range is supported by cited sources",
        parent=node,
        critical=False,
    )
    moisture_sources = _prefer_sources(extracted.moisture_sources, extracted.type_sources)
    await evaluator.verify(
        claim=(
            f"Figured maple boards for furniture-grade work should have a moisture content in the range of "
            f"{extracted.moisture_range or ''} before planing."
        ),
        node=moisture_supported_leaf,
        sources=moisture_sources,
        additional_instruction=(
            "Pass if the source recommends a matching or effectively equivalent moisture range for furniture woodworking. "
            "Allowance for minor formatting differences (e.g., 'around 8%' vs '6–8%'); the core range advice should align."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the vintage Stanley Bailey hand plane preparation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow the logical order of the task
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

    # 1) Extract structured information from the answer
    extracted: PlanePrepExtraction = await evaluator.extract(
        prompt=prompt_extract_plane_prep(),
        template_class=PlanePrepExtraction,
        extraction_name="plane_prep_extraction",
    )

    # 2) Build verification stages (sequential under root)
    await build_type_identification_stage(evaluator, root, extracted)
    await build_reference_url_stage(evaluator, root, extracted)
    await build_type_features_verification_stage(evaluator, root, extracted)
    await build_blade_angle_stage(evaluator, root, extracted)
    await build_moisture_stage(evaluator, root, extracted)

    # 3) Return structured evaluation summary
    return evaluator.get_summary()