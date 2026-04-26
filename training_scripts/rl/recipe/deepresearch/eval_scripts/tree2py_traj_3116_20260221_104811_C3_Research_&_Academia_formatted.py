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
TASK_ID = "shannon_curry_lineage"
TASK_DESCRIPTION = (
    "Identify the academic lineage of Dr. Shannon Curry, the Principal Investigator of NASA's MAVEN mission to Mars, "
    "by tracing her doctoral advisor and continuing upward through advisor-advisee relationships for three generations "
    "(excluding Shannon Curry herself).\n\n"
    "For each of the three generations, provide:\n"
    "1. The full name of the advisor\n"
    "2. The institution where the advisor earned their PhD (or was affiliated when supervising their student)\n"
    "3. The year the advisor earned their PhD (when traceable) or the year they supervised their student\n"
    "4. A direct URL reference that verifies the advisor-advisee relationship\n"
    "5. For Generation 3, additionally provide the research specialization or field of the advisor"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GenerationBase(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Generation3(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    research_field: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LineageExtraction(BaseModel):
    gen1: Optional[GenerationBase] = None
    gen2: Optional[GenerationBase] = None
    gen3: Optional[Generation3] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lineage() -> str:
    return """
    Extract Shannon Curry's academic lineage for three generations from the provided answer text.

    Return a JSON object with fields:
    - gen1: Object for Shannon Curry's doctoral advisor
        * name: Full name of Shannon Curry's doctoral advisor
        * institution: Institution where the advisor earned their PhD OR the institution they were affiliated with when supervising Shannon Curry
        * year: Either the year the advisor earned their PhD OR the year they supervised Shannon Curry's PhD (Shannon Curry completed her PhD in 2013 at the University of Michigan)
        * sources: An array of URLs explicitly cited in the answer that directly confirm the advisor-advisee relationship between the advisor and Shannon Curry
    - gen2: Object for the doctoral advisor of Generation 1
        * name: Full name
        * institution: Institution where this advisor earned their PhD OR their affiliation when supervising Generation 1's PhD
        * year: Either the year this advisor earned their PhD OR the year they supervised Generation 1's PhD
        * sources: An array of URLs explicitly cited in the answer that confirm the advisor-advisee relationship between this advisor and Generation 1
    - gen3: Object for the doctoral advisor of Generation 2
        * name: Full name
        * institution: Institution where this advisor earned their PhD OR their affiliation when supervising Generation 2's PhD
        * year: Either the year this advisor earned their PhD OR the year they supervised Generation 2's PhD
        * research_field: The research specialization or field of this advisor (e.g., planetary science, space physics)
        * sources: An array of URLs explicitly cited in the answer that confirm the advisor-advisee relationship or biographical information

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - If any field is missing, set it to null (for strings) or [] (for arrays).
    - For sources, include only valid URLs mentioned in the answer (plain URLs or within markdown links). Do not infer or create URLs.
    - Do not mix sources across generations; keep sources specific to each generation if the answer indicates so.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_advisee_name_for_generation(gen_index: int, extraction: LineageExtraction) -> str:
    """
    Determine the advisee's name for a given generation index.
    Generation 1's advisee is Shannon Curry.
    Generation 2's advisee is Generation 1's advisor.
    Generation 3's advisee is Generation 2's advisor.
    """
    if gen_index == 1:
        return "Shannon Curry"
    if gen_index == 2:
        return extraction.gen1.name if (extraction.gen1 and extraction.gen1.name) else "Generation 1 advisor"
    if gen_index == 3:
        return extraction.gen2.name if (extraction.gen2 and extraction.gen2.name) else "Generation 2 advisor"
    return "Unknown advisee"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_generation_1(evaluator: Evaluator, parent_node, ext: LineageExtraction) -> None:
    """
    Build and verify Generation 1 subtree:
    - Advisor Identification
    - Institution Info
    - Year Info
    - Source Verification
    """
    gen_node = evaluator.add_parallel(
        id="Generation_1_Advisor",
        desc="Identify Shannon Curry's doctoral advisor (the person who supervised her PhD at University of Michigan, completed in 2013)",
        parent=parent_node,
        critical=True
    )

    g1 = ext.gen1 or GenerationBase()
    advisee = get_advisee_name_for_generation(1, ext)
    sources = g1.sources

    # G1_Advisor_Identification
    g1_id_leaf = evaluator.add_leaf(
        id="G1_Advisor_Identification",
        desc="Provide the full name of Shannon Curry's doctoral advisor",
        parent=gen_node,
        critical=True
    )
    claim_id = f"{g1.name} was the doctoral advisor/supervisor of {advisee} for her PhD at the University of Michigan (completed in 2013)."
    await evaluator.verify(
        claim=claim_id,
        node=g1_id_leaf,
        sources=sources,
        additional_instruction="Verify that the sources explicitly confirm the advisor-advisee relationship (doctoral advisor/supervisor). Accept equivalent phrasing."
    )

    # G1_Institution_Info
    g1_inst_leaf = evaluator.add_leaf(
        id="G1_Institution_Info",
        desc="Provide the institution where the advisor earned their PhD or was affiliated when supervising Shannon Curry",
        parent=gen_node,
        critical=True
    )
    claim_inst = f"When supervising {advisee}, {g1.name} was affiliated with {g1.institution}, or {g1.name} earned their PhD at {g1.institution}."
    await evaluator.verify(
        claim=claim_inst,
        node=g1_inst_leaf,
        sources=sources,
        additional_instruction="Accept either affiliation-at-supervision or advisor's PhD-awarding institution if clearly indicated."
        ,
        extra_prerequisites=[g1_id_leaf]
    )

    # G1_Year_Info
    g1_year_leaf = evaluator.add_leaf(
        id="G1_Year_Info",
        desc="Provide the year the advisor earned their PhD or the year they supervised Shannon Curry's PhD (2013)",
        parent=gen_node,
        critical=True
    )
    year_text = g1.year if g1.year else ""
    claim_year = f"The provided sources indicate that either {g1.name} earned their PhD in {year_text}, or supervised {advisee} in {year_text} (Shannon Curry completed her PhD in 2013)."
    await evaluator.verify(
        claim=claim_year,
        node=g1_year_leaf,
        sources=sources,
        additional_instruction="Accept either PhD award year for the advisor or the year of supervision of Shannon Curry's PhD. Reasonable phrasing and synonymy acceptable.",
        extra_prerequisites=[g1_id_leaf]
    )

    # G1_Source_Verification
    g1_src_leaf = evaluator.add_leaf(
        id="G1_Source_Verification",
        desc="Provide a verifiable URL reference that confirms this advisor-advisee relationship",
        parent=gen_node,
        critical=True
    )
    claim_src = f"At least one provided source explicitly confirms that {g1.name} was the doctoral advisor/supervisor of {advisee}."
    await evaluator.verify(
        claim=claim_src,
        node=g1_src_leaf,
        sources=sources,
        additional_instruction="If any single URL clearly states the advisor-advisee relationship, consider this supported.",
        extra_prerequisites=[g1_id_leaf]
    )


async def verify_generation_2(evaluator: Evaluator, parent_node, ext: LineageExtraction) -> None:
    """
    Build and verify Generation 2 subtree:
    - Advisor Identification
    - Institution Info
    - Year Info
    - Source Verification
    """
    gen_node = evaluator.add_parallel(
        id="Generation_2_Advisor",
        desc="Identify the doctoral advisor of Generation 1 (the person who supervised Generation 1's PhD)",
        parent=parent_node,
        critical=True
    )

    g2 = ext.gen2 or GenerationBase()
    advisee = get_advisee_name_for_generation(2, ext)
    sources = g2.sources

    # G2_Advisor_Identification
    g2_id_leaf = evaluator.add_leaf(
        id="G2_Advisor_Identification",
        desc="Provide the full name of Generation 1's doctoral advisor",
        parent=gen_node,
        critical=True
    )
    claim_id = f"{g2.name} was the doctoral advisor/supervisor of {advisee} for their PhD."
    await evaluator.verify(
        claim=claim_id,
        node=g2_id_leaf,
        sources=sources,
        additional_instruction="Verify that the sources explicitly confirm the advisor-advisee relationship (doctoral advisor/supervisor).",
    )

    # G2_Institution_Info
    g2_inst_leaf = evaluator.add_leaf(
        id="G2_Institution_Info",
        desc="Provide the institution where this advisor earned their PhD or was affiliated when supervising Generation 1's PhD",
        parent=gen_node,
        critical=True
    )
    claim_inst = f"When supervising {advisee}, {g2.name} was affiliated with {g2.institution}, or {g2.name} earned their PhD at {g2.institution}."
    await evaluator.verify(
        claim=claim_inst,
        node=g2_inst_leaf,
        sources=sources,
        additional_instruction="Accept either affiliation-at-supervision or advisor's PhD-awarding institution.",
        extra_prerequisites=[g2_id_leaf]
    )

    # G2_Year_Info
    g2_year_leaf = evaluator.add_leaf(
        id="G2_Year_Info",
        desc="Provide the year this advisor earned their PhD or the year they supervised Generation 1's PhD",
        parent=gen_node,
        critical=True
    )
    year_text = g2.year if g2.year else ""
    claim_year = f"The provided sources indicate that either {g2.name} earned their PhD in {year_text}, or supervised {advisee} in {year_text}."
    await evaluator.verify(
        claim=claim_year,
        node=g2_year_leaf,
        sources=sources,
        additional_instruction="Accept either PhD award year for the advisor or the year of supervising Generation 1's PhD.",
        extra_prerequisites=[g2_id_leaf]
    )

    # G2_Source_Verification
    g2_src_leaf = evaluator.add_leaf(
        id="G2_Source_Verification",
        desc="Provide a verifiable URL reference that confirms this advisor-advisee relationship",
        parent=gen_node,
        critical=True
    )
    claim_src = f"At least one provided source explicitly confirms that {g2.name} was the doctoral advisor/supervisor of {advisee}."
    await evaluator.verify(
        claim=claim_src,
        node=g2_src_leaf,
        sources=sources,
        additional_instruction="If any single URL clearly states the advisor-advisee relationship, consider this supported.",
        extra_prerequisites=[g2_id_leaf]
    )


async def verify_generation_3(evaluator: Evaluator, parent_node, ext: LineageExtraction) -> None:
    """
    Build and verify Generation 3 subtree:
    - Advisor Identification
    - Institution Info
    - Year Info
    - Research Field
    - Source Verification
    """
    gen_node = evaluator.add_parallel(
        id="Generation_3_Advisor",
        desc="Identify the doctoral advisor of Generation 2 (the person who supervised Generation 2's PhD)",
        parent=parent_node,
        critical=True
    )

    g3 = ext.gen3 or Generation3()
    advisee = get_advisee_name_for_generation(3, ext)
    sources = g3.sources

    # G3_Advisor_Identification
    g3_id_leaf = evaluator.add_leaf(
        id="G3_Advisor_Identification",
        desc="Provide the full name of Generation 2's doctoral advisor",
        parent=gen_node,
        critical=True
    )
    claim_id = f"{g3.name} was the doctoral advisor/supervisor of {advisee} for their PhD."
    await evaluator.verify(
        claim=claim_id,
        node=g3_id_leaf,
        sources=sources,
        additional_instruction="Verify that the sources explicitly confirm the advisor-advisee relationship (doctoral advisor/supervisor).",
    )

    # G3_Institution_Info
    g3_inst_leaf = evaluator.add_leaf(
        id="G3_Institution_Info",
        desc="Provide the institution where this advisor earned their PhD or was affiliated when supervising Generation 2's PhD",
        parent=gen_node,
        critical=True
    )
    claim_inst = f"When supervising {advisee}, {g3.name} was affiliated with {g3.institution}, or {g3.name} earned their PhD at {g3.institution}."
    await evaluator.verify(
        claim=claim_inst,
        node=g3_inst_leaf,
        sources=sources,
        additional_instruction="Accept either affiliation-at-supervision or advisor's PhD-awarding institution.",
        extra_prerequisites=[g3_id_leaf]
    )

    # G3_Year_Info
    g3_year_leaf = evaluator.add_leaf(
        id="G3_Year_Info",
        desc="Provide the year this advisor earned their PhD or the year they supervised Generation 2's PhD",
        parent=gen_node,
        critical=True
    )
    year_text = g3.year if g3.year else ""
    claim_year = f"The provided sources indicate that either {g3.name} earned their PhD in {year_text}, or supervised {advisee} in {year_text}."
    await evaluator.verify(
        claim=claim_year,
        node=g3_year_leaf,
        sources=sources,
        additional_instruction="Accept either PhD award year for the advisor or the year of supervising Generation 2's PhD.",
        extra_prerequisites=[g3_id_leaf]
    )

    # G3_Research_Field
    g3_field_leaf = evaluator.add_leaf(
        id="G3_Research_Field",
        desc="Provide the research specialization or field of Generation 2's advisor",
        parent=gen_node,
        critical=True
    )
    field_text = g3.research_field if g3.research_field else ""
    claim_field = f"The research specialization or field of {g3.name} is {field_text}."
    await evaluator.verify(
        claim=claim_field,
        node=g3_field_leaf,
        sources=sources,
        additional_instruction="Verify the advisor's research specialization/field (e.g., planetary science, space physics, etc.) using the provided sources; biographical pages are acceptable.",
        extra_prerequisites=[g3_id_leaf]
    )

    # G3_Source_Verification
    g3_src_leaf = evaluator.add_leaf(
        id="G3_Source_Verification",
        desc="Provide a verifiable URL reference that confirms this advisor-advisee relationship or provides biographical information",
        parent=gen_node,
        critical=True
    )
    claim_src = f"At least one provided source confirms the advisor-advisee relationship between {g3.name} and {advisee}, or provides authoritative biographical information about {g3.name}."
    await evaluator.verify(
        claim=claim_src,
        node=g3_src_leaf,
        sources=sources,
        additional_instruction="If any single URL clearly states the advisor-advisee relationship or provides authoritative biographical info (e.g., institutional bio), consider this supported.",
        extra_prerequisites=[g3_id_leaf]
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
    Evaluate an answer for Shannon Curry's academic lineage task.
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

    # Create a critical sequential node under root to reflect the rubric's root criticality
    lineage_root = evaluator.add_sequential(
        id="Academic_Lineage_Tracing",
        desc="Trace Shannon Curry's academic lineage through doctoral advisor relationships for three generations",
        parent=root,
        critical=True
    )

    # Extract structured lineage information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_lineage(),
        template_class=LineageExtraction,
        extraction_name="lineage_extraction"
    )

    # Verify generation 1, 2, 3 under the sequential lineage node
    await verify_generation_1(evaluator, lineage_root, extraction)
    await verify_generation_2(evaluator, lineage_root, extraction)
    await verify_generation_3(evaluator, lineage_root, extraction)

    return evaluator.get_summary()