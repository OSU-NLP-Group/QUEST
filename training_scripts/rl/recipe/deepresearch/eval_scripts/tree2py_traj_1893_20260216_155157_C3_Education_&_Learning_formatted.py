import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ray_jayawardhana_lineage"
TASK_DESCRIPTION = """Trace the academic lineage of Dr. Ray Jayawardhana, who received his PhD in Astronomy from Harvard University in 2000 and was recently appointed as the 10th President of the California Institute of Technology (Caltech). Starting with Ray Jayawardhana, identify his doctoral advisor Lee Hartmann, and then continue tracing upward through the advisor-advisee relationships for at least three additional generations of doctoral advisors (i.e., Hartmann's advisor, Hartmann's advisor's advisor, etc.). For each person in the lineage (excluding Ray Jayawardhana himself), provide the following information: (1) Full name of the advisor, (2) PhD-granting institution (the university where they earned their doctorate), (3) Year of PhD completion (when available), (4) Current or most recent professional position (when available), (5) At least one authoritative URL reference that verifies the advisor-advisee relationship or the person's academic credentials (such as university records, department alumni pages, Mathematics Genealogy Project, Astronomy Genealogy Project, official CVs, or academic databases). The lineage should specifically trace through Lee Hartmann as Ray Jayawardhana's advisor, and continue upward from there. You must trace at least four generations of advisors (Lee Hartmann plus three generations above him)."""

EXPECTED_G1_NAME = "Lee Hartmann"
EXPECTED_G2_NAME = "Carl M. Anderson"
EXPECTED_G1_PHD_INST = "University of Wisconsin"
EXPECTED_G1_PHD_YEAR = "1976"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonEntry(BaseModel):
    """Represents one advisor in the lineage."""
    name: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    position: Optional[str] = None
    relationship_urls: List[str] = Field(default_factory=list, description="Authoritative URLs specifically supporting the advisor-advisee relationship")
    credentials_urls: List[str] = Field(default_factory=list, description="URLs supporting credentials such as PhD institution/year or positions")


class LineageExtraction(BaseModel):
    """Structured information extracted from the answer."""
    # Starting point (Ray Jayawardhana)
    ray_phd_institution: Optional[str] = None
    ray_phd_field: Optional[str] = None
    ray_phd_year: Optional[str] = None
    ray_phd_urls: List[str] = Field(default_factory=list)
    ray_advisors: List[str] = Field(default_factory=list, description="Names of advisors explicitly mentioned for Ray")
    ray_advisor_urls: List[str] = Field(default_factory=list, description="URLs supporting Ray↔advisor(s) relationship")

    # Generations
    g1: Optional[PersonEntry] = None  # Lee Hartmann
    g2: Optional[PersonEntry] = None  # Carl M. Anderson
    g3: Optional[PersonEntry] = None  # Advisor of g2
    g4: Optional[PersonEntry] = None  # Advisor of g3


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lineage() -> str:
    return """
Extract the academic lineage information presented in the answer for Ray Jayawardhana (starting point) and at least four advisor generations upward through Lee Hartmann and beyond.

Return a JSON with these fields:
- ray_phd_institution: The institution for Ray Jayawardhana’s PhD as stated in the answer (e.g., Harvard University). If absent, null.
- ray_phd_field: The discipline/field for Ray's PhD (e.g., Astronomy). If absent, null.
- ray_phd_year: The year of Ray's PhD (e.g., 2000). If absent, null.
- ray_phd_urls: Array of URLs in the answer that support Ray’s PhD details.
- ray_advisors: Array of advisor names explicitly mentioned for Ray in the answer (e.g., ["Lee Hartmann", "Giovanni Fazio"]).
- ray_advisor_urls: Array of URLs that support Ray↔advisor(s) relationship.

For each generation g1 (Lee Hartmann), g2 (advisor of Hartmann; expected: Carl M. Anderson), g3 (advisor of g2), and g4 (advisor of g3), extract a PersonEntry object with:
- name: Full name as presented in the answer. For g1 expect "Lee Hartmann"; for g2 expect "Carl M. Anderson". For g3 and g4, extract whatever name the answer claims.
- phd_institution: The PhD-granting institution for this person (when provided).
- phd_year: The PhD completion year (when provided).
- position: The current or most recent professional position (when provided).
- relationship_urls: Array of URLs specifically verifying the advisor-advisee relationship to the person immediately below in the lineage (e.g., Hartmann↔Ray; g2↔Hartmann; etc.).
- credentials_urls: Array of URLs verifying the person’s credentials (PhD institution/year, positions), if provided in the answer.

GENERAL RULES:
- Only extract what is explicitly stated in the answer.
- If a field is missing, set it to null (or [] for arrays).
- Extract actual URLs (plain or markdown link targets). Ignore vague mentions without a URL.
- Do not invent any names or URLs.
"""


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, deduplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_starting_point(evaluator: Evaluator, parent) -> None:
    """
    Verify that the answer correctly establishes the starting point for Ray Jayawardhana.
    """
    sp_node = evaluator.add_parallel(
        id="starting_point_ray",
        desc="Establish Ray Jayawardhana as the starting point using the stated PhD details and co-advisorship information.",
        parent=parent,
        critical=True  # Both child checks must pass for this step
    )

    # Leaf: The answer states PhD in Astronomy from Harvard University in 2000
    ray_phd_leaf = evaluator.add_leaf(
        id="ray_phd_harvard_2000",
        desc="State that Ray Jayawardhana received a PhD in Astronomy from Harvard University in 2000.",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Ray Jayawardhana received a PhD in Astronomy from Harvard University in 2000.",
        node=ray_phd_leaf,
        additional_instruction="Judge only whether the statement appears in the answer text (allow minor wording variations such as 'Ph.D.' vs 'PhD'). Do not fact-check with external sources."
    )

    # Leaf: The answer acknowledges co-advisors Hartmann and Fazio
    ray_coadv_leaf = evaluator.add_leaf(
        id="ray_coadvised_hartmann_fazio",
        desc="Acknowledge that Ray Jayawardhana was co-advised by Lee Hartmann and Giovanni Fazio at Harvard.",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer acknowledges that Ray Jayawardhana was co-advised by Lee Hartmann and Giovanni Fazio at Harvard University.",
        node=ray_coadv_leaf,
        additional_instruction="Judge only whether the answer lists both 'Lee Hartmann' and 'Giovanni Fazio' as his advisors/co-advisors (minor naming variations are acceptable)."
    )


async def verify_generation_1(evaluator: Evaluator, parent, data: LineageExtraction) -> None:
    """
    Generation 1: Lee Hartmann (advisor of Ray)
    """
    g1 = data.g1 or PersonEntry()
    g1_node = evaluator.add_parallel(
        id="generation_1_lee_hartmann",
        desc="Generation 1 (advisor of Ray): Lee Hartmann — provide required fields and URL evidence.",
        parent=parent,
        critical=False  # Allow optional field to be non-critical
    )

    # Identity in answer: 'Lee Hartmann' used as the doctoral advisor for Ray
    g1_id_leaf = evaluator.add_leaf(
        id="g1_identity_relationship",
        desc="Identify Lee Hartmann (full name) as the doctoral advisor used for the lineage path from Ray Jayawardhana.",
        parent=g1_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies 'Lee Hartmann' as Ray Jayawardhana's doctoral advisor for the lineage path (advisor/co-advisor acceptable).",
        node=g1_id_leaf,
        additional_instruction="Check the answer text for this identification; allow 'advisor', 'doctoral advisor', 'PhD supervisor', or 'co-advisor'."
    )

    # URL support for Jayawardhana ↔ Hartmann relationship
    g1_rel_leaf = evaluator.add_leaf(
        id="g1_relationship_url_support",
        desc="Provide ≥1 publicly accessible authoritative URL verifying the Jayawardhana↔Hartmann doctoral advisor-advisee relationship.",
        parent=g1_node,
        critical=True
    )
    rel_urls = _merge_urls(g1.relationship_urls, data.ray_advisor_urls)
    await evaluator.verify(
        claim="Lee Hartmann is a doctoral advisor or co-advisor of Ray Jayawardhana at Harvard University.",
        node=g1_rel_leaf,
        sources=rel_urls,
        additional_instruction="Accept synonyms such as 'doctoral advisor', 'PhD supervisor', or 'thesis advisor'. A clear statement on the cited page is required."
    )

    # PhD institution (University of Wisconsin) — verify with URLs if provided
    g1_inst_leaf = evaluator.add_leaf(
        id="g1_phd_institution",
        desc="Provide Lee Hartmann's PhD-granting institution (University of Wisconsin).",
        parent=g1_node,
        critical=True
    )
    inst_urls = _merge_urls(g1.credentials_urls, g1.relationship_urls)
    await evaluator.verify(
        claim="Lee Hartmann earned his PhD from the University of Wisconsin (e.g., University of Wisconsin–Madison).",
        node=g1_inst_leaf,
        sources=inst_urls,
        additional_instruction="Treat 'University of Wisconsin–Madison', 'UW–Madison', or 'University of Wisconsin' as equivalent acceptable variants if evident in the source."
    )

    # PhD year (1976) — verify with URLs if provided
    g1_year_leaf = evaluator.add_leaf(
        id="g1_phd_year",
        desc="Provide Lee Hartmann's PhD year (1976).",
        parent=g1_node,
        critical=True
    )
    await evaluator.verify(
        claim="Lee Hartmann's PhD year is 1976.",
        node=g1_year_leaf,
        sources=inst_urls,
        additional_instruction="Confirm from the cited page. Minor differences such as context (e.g., dissertation year vs conferral) should still indicate the year 1976."
    )

    # Position when available — existence check (non-critical)
    pos_exists = g1.position is not None and str(g1.position).strip() != ""
    evaluator.add_custom_node(
        result=pos_exists,
        id="g1_position_when_available",
        desc="Provide Lee Hartmann's current or most recent professional position (when available).",
        parent=g1_node,
        critical=False
    )


async def verify_generation_2(evaluator: Evaluator, parent, data: LineageExtraction) -> None:
    """
    Generation 2: Carl M. Anderson (advisor of Hartmann)
    """
    g2 = data.g2 or PersonEntry()
    g2_node = evaluator.add_parallel(
        id="generation_2_carl_m_anderson",
        desc="Generation 2 (advisor of Hartmann): Carl M. Anderson — provide required fields and URL evidence.",
        parent=parent,
        critical=False
    )

    # Identity in answer: 'Carl M. Anderson' as Hartmann's doctoral advisor
    g2_id_leaf = evaluator.add_leaf(
        id="g2_identity_relationship",
        desc="Identify Carl M. Anderson (full name) as Lee Hartmann's doctoral advisor.",
        parent=g2_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies 'Carl M. Anderson' as Lee Hartmann's doctoral advisor.",
        node=g2_id_leaf,
        additional_instruction="Check only the answer text for this identification. Allow minor name variations (e.g., with/without middle initial)."
    )

    # URL support for Hartmann ↔ Anderson relationship
    g2_rel_leaf = evaluator.add_leaf(
        id="g2_relationship_url_support",
        desc="Provide ≥1 publicly accessible authoritative URL verifying the Hartmann↔Anderson doctoral advisor-advisee relationship.",
        parent=g2_node,
        critical=True
    )
    await evaluator.verify(
        claim="Carl M. Anderson is the doctoral advisor of Lee Hartmann.",
        node=g2_rel_leaf,
        sources=g2.relationship_urls,
        additional_instruction="Accept synonyms such as 'doctoral advisor', 'PhD supervisor', or 'thesis advisor'. The page should clearly support this relationship."
    )

    # Optional fields: existence checks
    evaluator.add_custom_node(
        result=(g2.phd_institution is not None and g2.phd_institution.strip() != ""),
        id="g2_phd_institution_when_available",
        desc="Provide Carl M. Anderson's PhD-granting institution (when available).",
        parent=g2_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(g2.phd_year is not None and g2.phd_year.strip() != ""),
        id="g2_phd_year_when_available",
        desc="Provide Carl M. Anderson's PhD year (when available).",
        parent=g2_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(g2.position is not None and g2.position.strip() != ""),
        id="g2_position_when_available",
        desc="Provide Carl M. Anderson's current or most recent professional position (when available).",
        parent=g2_node,
        critical=False
    )


async def verify_generation_dynamic(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    node_desc: str,
    subject_name: str,  # The advisee name (e.g., 'Carl M. Anderson' for g3)
    advisor_entry: Optional[PersonEntry],
) -> None:
    """
    Verify a dynamic generation (g3 or g4) where the expected advisor name is not pre-specified.
    """
    entry = advisor_entry or PersonEntry()
    gen_node = evaluator.add_parallel(
        id=node_id_prefix,
        desc=node_desc,
        parent=parent,
        critical=False
    )

    # Identity existence (critical): must provide a full name for this advisor
    id_leaf = evaluator.add_custom_node(
        result=(entry.name is not None and entry.name.strip() != ""),
        id=f"{node_id_prefix}_identity_relationship",
        desc="Provide the full name of the doctoral advisor (next advisor up the chain).",
        parent=gen_node,
        critical=True
    )

    # Relationship URL support (critical)
    rel_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_relationship_url_support",
        desc="Provide ≥1 publicly accessible authoritative URL verifying the advisor-advisee relationship for this generation.",
        parent=gen_node,
        critical=True
    )

    claim = f"{entry.name or 'UNKNOWN'} is the doctoral advisor (or PhD supervisor) of {subject_name}."
    await evaluator.verify(
        claim=claim,
        node=rel_leaf,
        sources=entry.relationship_urls,
        additional_instruction="Accept synonyms such as 'doctoral advisor', 'PhD supervisor', or 'thesis advisor'. The source must clearly support this relationship."
    )

    # Optional fields: existence only
    evaluator.add_custom_node(
        result=(entry.phd_institution is not None and entry.phd_institution.strip() != ""),
        id=f"{node_id_prefix}_phd_institution_when_available",
        desc="Provide the advisor's PhD-granting institution (when available).",
        parent=gen_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(entry.phd_year is not None and entry.phd_year.strip() != ""),
        id=f"{node_id_prefix}_phd_year_when_available",
        desc="Provide the advisor's PhD year (when available).",
        parent=gen_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(entry.position is not None and entry.position.strip() != ""),
        id=f"{node_id_prefix}_position_when_available",
        desc="Provide the advisor's current or most recent professional position (when available).",
        parent=gen_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating the provided answer against the lineage rubric.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce upward-tracing order
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

    # Extract structured information from the answer
    extraction: LineageExtraction = await evaluator.extract(
        prompt=prompt_extract_lineage(),
        template_class=LineageExtraction,
        extraction_name="lineage_extraction"
    )

    # Add ground truth hints (used only for transparency in summary)
    evaluator.add_ground_truth(
        {
            "required_generations": 4,
            "expected_generation_1": EXPECTED_G1_NAME,
            "expected_generation_2": EXPECTED_G2_NAME,
            "notes": "Generations 3 and 4 depend on the answer; must be supported by authoritative URLs."
        },
        gt_type="expected_lineage_requirements"
    )

    # Build and verify according to rubric (sequential)
    await verify_starting_point(evaluator, root)
    await verify_generation_1(evaluator, root, extraction)
    await verify_generation_2(evaluator, root, extraction)

    # Determine subject names for dynamic generations
    g2_name = (extraction.g2.name if extraction.g2 and extraction.g2.name else EXPECTED_G2_NAME)

    # Generation 3 (advisor of Anderson)
    await verify_generation_dynamic(
        evaluator=evaluator,
        parent=root,
        node_id_prefix="generation_3_advisor",
        node_desc="Generation 3 (advisor of Anderson): identify and provide required fields and URL evidence.",
        subject_name=g2_name,
        advisor_entry=extraction.g3
    )

    # Generation 4 (advisor of generation 3)
    g3_name = (extraction.g3.name if extraction.g3 and extraction.g3.name else "the generation-3 advisor")
    await verify_generation_dynamic(
        evaluator=evaluator,
        parent=root,
        node_id_prefix="generation_4_advisor",
        node_desc="Generation 4 (advisor of generation-3 advisor): identify and provide required fields and URL evidence.",
        subject_name=g3_name,
        advisor_entry=extraction.g4
    )

    # Return the structured evaluation summary
    return evaluator.get_summary()