import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usda_33rd_sec_2025"
TASK_DESCRIPTION = """
Identify the individual who was sworn in as the 33rd United States Secretary of Agriculture in February 2025. Then, determine which university this person attended for their undergraduate degree. Finally, identify the federal judge for whom this Secretary clerked after completing law school. For each piece of information, provide supporting URL references from reliable sources.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SecretarySwearInInfo(BaseModel):
    """Swearing-in details for the Secretary of Agriculture."""
    name: Optional[str] = None
    position_number: Optional[str] = None  # e.g., "33rd"
    sworn_in_month_year: Optional[str] = None  # e.g., "February 2025"
    oath_administered_by: Optional[str] = None  # Name of the administrator
    oath_administered_by_role: Optional[str] = None  # e.g., "U.S. Supreme Court Justice"
    sworn_in_sources: List[str] = Field(default_factory=list)  # URLs cited for identification & swearing-in


class EducationInfo(BaseModel):
    """Undergraduate and law school education details."""
    undergrad_university: Optional[str] = None
    undergrad_field: Optional[str] = None  # e.g., "Agricultural Economics", "Public Policy", etc.
    law_school_name: Optional[str] = None  # e.g., "Harvard Law School"
    law_school_after_undergrad: Optional[str] = None  # e.g., "Yes", "No", "after undergrad", etc.
    education_sources: List[str] = Field(default_factory=list)  # URLs supporting undergrad & law school timeline


class ClerkshipInfo(BaseModel):
    """Clerkship details."""
    judge_name: Optional[str] = None  # e.g., "Judge Jane Doe"
    district_court_name: Optional[str] = None  # e.g., "U.S. District Court for the District of Columbia"
    clerkship_after_law_school: Optional[str] = None  # e.g., "Yes", "after law school"
    clerkship_sources: List[str] = Field(default_factory=list)  # URLs supporting judge, court, and timing


class ResearchExtraction(BaseModel):
    """Complete extraction for the task."""
    secretary: Optional[SecretarySwearInInfo] = None
    education: Optional[EducationInfo] = None
    clerkship: Optional[ClerkshipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
    Extract the following structured information exactly as stated in the provided answer text. Do not invent or infer missing details.
    1) Secretary swearing-in:
       - name: Full name of the individual identified as the United States Secretary of Agriculture
       - position_number: The ordinal number stated for the Secretary of Agriculture (e.g., "33rd")
       - sworn_in_month_year: The stated month and year of the swearing-in (e.g., "February 2025")
       - oath_administered_by: The name of the person who administered the oath (if stated)
       - oath_administered_by_role: The role/title of the person who administered the oath (e.g., "U.S. Supreme Court Justice")
       - sworn_in_sources: An array of all URLs provided in the answer that support the identification and swearing-in details.
         Extract only explicit URLs (plain or markdown), ignore non-URL mentions.

    2) Education:
       - undergrad_university: The university where the Secretary earned an undergraduate degree
       - undergrad_field: The undergraduate degree field/major (e.g., "Agricultural Economics", "Public Policy", etc.)
       - law_school_name: The law school attended after undergraduate studies (if stated)
       - law_school_after_undergrad: A brief statement indicating that law school attendance occurred after undergrad (e.g., "Yes", "after undergrad", etc.)
       - education_sources: An array of all URLs provided in the answer that support the undergraduate details and the law school attendance timeline.
         Extract only explicit URLs (plain or markdown).

    3) Clerkship:
       - judge_name: The name of the federal judge the Secretary clerked for
       - district_court_name: The specific U.S. federal district court associated with that judge
       - clerkship_after_law_school: A brief statement indicating the clerkship occurred after completing law school (e.g., "Yes", "after law school")
       - clerkship_sources: An array of all URLs provided in the answer that support the clerkship details (judge, court, timing).
         Extract only explicit URLs (plain or markdown).

    Rules:
    - If any field is not explicitly stated in the answer, return null for that field.
    - For each 'sources' array, include only valid URLs explicitly present in the answer.
    - Do not deduplicate; include all URLs as they appear.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_step_1_secretary(
    evaluator: Evaluator,
    parent_node,
    sec: Optional[SecretarySwearInInfo],
) -> None:
    """
    Build and verify Step 1: Identify the individual and sworn-in constraints for the 33rd U.S. Secretary of Agriculture in February 2025.
    """
    step_node = evaluator.add_parallel(
        id="step_1_secretary_identification",
        desc="Identify the individual and verify the sworn-in constraints for the 33rd U.S. Secretary of Agriculture in February 2025.",
        parent=parent_node,
        critical=False,
    )

    # Prepare extracted values safely
    name = (sec.name or "").strip()
    position = (sec.position_number or "").strip()
    month_year = (sec.sworn_in_month_year or "").strip()
    admin_name = (sec.oath_administered_by or "").strip()
    admin_role = (sec.oath_administered_by_role or "").strip()
    sources = sec.sworn_in_sources if sec and sec.sworn_in_sources else []

    # Leaf 1: States the correct individual who was sworn in as the 33rd U.S. Secretary of Agriculture in February 2025.
    leaf_ident = evaluator.add_leaf(
        id="secretary_is_33rd_and_sworn_in_feb_2025",
        desc="States the correct individual who was sworn in as the 33rd U.S. Secretary of Agriculture in February 2025.",
        parent=step_node,
        critical=True,
    )
    claim_ident = f"{name} was sworn in as the 33rd United States Secretary of Agriculture in February 2025."
    await evaluator.verify(
        claim=claim_ident,
        node=leaf_ident,
        sources=sources,
        additional_instruction="Check that the source explicitly identifies the individual as the 33rd U.S. Secretary of Agriculture and that the swearing-in occurred in February 2025. Allow minor variations like 'Feb. 2025'.",
    )

    # Leaf 2: Oath administered by a U.S. Supreme Court Justice (and identifies the administrator).
    leaf_oath = evaluator.add_leaf(
        id="oath_administered_by_scotus_justice",
        desc="States that the oath of office was administered by a U.S. Supreme Court Justice (and identifies the administrator).",
        parent=step_node,
        critical=True,
    )
    # Craft a claim focusing on both role and person
    # If role is missing, still assert the person is a U.S. Supreme Court Justice in the source.
    claim_oath = f"The oath of office was administered by a U.S. Supreme Court Justice named {admin_name}."
    await evaluator.verify(
        claim=claim_oath,
        node=leaf_oath,
        sources=sources,
        additional_instruction="Verify that the source explicitly states the oath was administered by a U.S. Supreme Court Justice and names the administrator. Allow reasonable name formatting variations.",
    )

    # Leaf 3: Provides at least one reliable URL citation supporting identification and swearing-in details.
    # Implemented as a custom existence check for URLs (reliability implicitly ensured by prior evidence checks).
    evaluator.add_custom_node(
        result=bool(sources),
        id="secretary_sworn_in_citation_url",
        desc="Provides at least one reliable URL citation supporting the secretary identification and swearing-in details.",
        parent=step_node,
        critical=True,
    )


async def verify_step_2_education(
    evaluator: Evaluator,
    parent_node,
    edu: Optional[EducationInfo],
) -> None:
    """
    Build and verify Step 2: Undergraduate education (university + field) and law school attendance after undergrad, with citations.
    """
    step_node = evaluator.add_parallel(
        id="step_2_education",
        desc="Provide and verify the Secretary's undergraduate education (university + field) and law school attendance after undergraduate studies, with citations.",
        parent=parent_node,
        critical=False,
    )

    undergrad_university = (edu.undergrad_university or "").strip()
    undergrad_field = (edu.undergrad_field or "").strip()
    law_school_name = (edu.law_school_name or "").strip()
    law_after = (edu.law_school_after_undergrad or "").strip()
    sources = edu.education_sources if edu and edu.education_sources else []

    # Leaf 1: Undergrad university
    leaf_ug_uni = evaluator.add_leaf(
        id="undergrad_university",
        desc="Correctly identifies the university where the Secretary earned their undergraduate degree.",
        parent=step_node,
        critical=True,
    )
    claim_ug_uni = f"The Secretary earned an undergraduate degree at {undergrad_university}."
    await evaluator.verify(
        claim=claim_ug_uni,
        node=leaf_ug_uni,
        sources=sources,
        additional_instruction="Verify that the source explicitly names the undergraduate institution. Accept reasonable name variants and abbreviations.",
    )

    # Leaf 2: Undergrad field is agriculture-related or policy-related
    leaf_ug_field = evaluator.add_leaf(
        id="undergrad_field_ag_or_policy_related",
        desc="The undergraduate degree field/major is explicitly agriculture-related or policy-related as stated in a cited source.",
        parent=step_node,
        critical=True,
    )
    claim_ug_field = f"The undergraduate major/field was {undergrad_field}, which is agriculture-related or policy-related."
    await evaluator.verify(
        claim=claim_ug_field,
        node=leaf_ug_field,
        sources=sources,
        additional_instruction="Decide using the source whether the stated field is agriculture-related (e.g., ag econ, agronomy) or policy-related (e.g., public policy, political science, government). If ambiguous, judge reasonably based on the source's wording.",
    )

    # Leaf 3: Law school after undergrad
    leaf_law_after = evaluator.add_leaf(
        id="law_school_after_undergrad",
        desc="States that the Secretary attended law school after completing undergraduate studies.",
        parent=step_node,
        critical=True,
    )
    if law_school_name:
        claim_law_after = f"After completing undergraduate studies, the Secretary attended {law_school_name} (law school)."
    else:
        claim_law_after = "After completing undergraduate studies, the Secretary attended law school."
    await evaluator.verify(
        claim=claim_law_after,
        node=leaf_law_after,
        sources=sources,
        additional_instruction="Verify chronology: the law school attendance is after undergraduate graduation. The law school name can be used if provided, but the key is the timing.",
    )

    # Leaf 4: At least one citation URL exists
    evaluator.add_custom_node(
        result=bool(sources),
        id="education_citation_url",
        desc="Provides at least one reliable URL citation supporting the undergraduate details and the law school attendance timeline.",
        parent=step_node,
        critical=True,
    )


async def verify_step_3_clerkship(
    evaluator: Evaluator,
    parent_node,
    clk: Optional[ClerkshipInfo],
) -> None:
    """
    Build and verify Step 3: Federal judge clerkship after law school, including judge name and district court, with citation.
    """
    step_node = evaluator.add_parallel(
        id="step_3_clerkship",
        desc="Identify the federal judge clerkship after law school, including judge name and associated federal district court, with citation.",
        parent=parent_node,
        critical=False,
    )

    judge_name = (clk.judge_name or "").strip()
    district_court_name = (clk.district_court_name or "").strip()
    clerkship_timing = (clk.clerkship_after_law_school or "").strip()
    sources = clk.clerkship_sources if clk and clk.clerkship_sources else []

    # Leaf 1: Judge name
    leaf_judge = evaluator.add_leaf(
        id="judge_name",
        desc="Correctly identifies the federal judge (by name) for whom the Secretary clerked.",
        parent=step_node,
        critical=True,
    )
    claim_judge = f"The Secretary clerked for U.S. federal judge {judge_name}."
    await evaluator.verify(
        claim=claim_judge,
        node=leaf_judge,
        sources=sources,
        additional_instruction="Verify the source explicitly states that the Secretary served as a law clerk for the named federal judge.",
    )

    # Leaf 2: District court name
    leaf_court = evaluator.add_leaf(
        id="district_court_name",
        desc="Correctly identifies the specific federal district court associated with that judge.",
        parent=step_node,
        critical=True,
    )
    claim_court = f"Judge {judge_name} is associated with the {district_court_name}."
    await evaluator.verify(
        claim=claim_court,
        node=leaf_court,
        sources=sources,
        additional_instruction="Verify that the judge serves on (or is a judge of) the stated U.S. District Court. Focus on district courts, not appellate courts.",
    )

    # Leaf 3: Clerkship occurred after law school
    leaf_clk_timing = evaluator.add_leaf(
        id="clerkship_after_law_school_claim",
        desc="States that the clerkship occurred after completing law school.",
        parent=step_node,
        critical=True,
    )
    claim_clk_timing = "The clerkship occurred after completing law school."
    await evaluator.verify(
        claim=claim_clk_timing,
        node=leaf_clk_timing,
        sources=sources,
        additional_instruction="Verify the source implies or explicitly states the clerkship timing was post-law school.",
    )

    # Leaf 4: At least one citation URL exists
    evaluator.add_custom_node(
        result=bool(sources),
        id="clerkship_citation_url",
        desc="Provides at least one reliable URL citation supporting the clerkship details (judge, court, and timing).",
        parent=step_node,
        critical=True,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the USDA Secretary (33rd, Feb 2025) research task.
    """
    # Initialize evaluator with sequential root strategy to enforce step dependencies
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured research info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="secretary_education_clerkship",
    )

    # Build and verify Step 1
    await verify_step_1_secretary(
        evaluator=evaluator,
        parent_node=root,
        sec=extracted.secretary,
    )

    # Build and verify Step 2 (will be skipped automatically if Step 1 fails under sequential root)
    await verify_step_2_education(
        evaluator=evaluator,
        parent_node=root,
        edu=extracted.education,
    )

    # Build and verify Step 3 (will be skipped automatically if prior steps fail under sequential root)
    await verify_step_3_clerkship(
        evaluator=evaluator,
        parent_node=root,
        clk=extracted.clerkship,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()