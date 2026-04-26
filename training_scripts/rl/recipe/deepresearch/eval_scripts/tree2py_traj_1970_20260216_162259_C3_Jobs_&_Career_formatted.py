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
TASK_ID = "wa_cte_stem_cert"
TASK_DESCRIPTION = """
A software engineer with a bachelor's degree in Computer Science and 4 years of professional software development experience wants to become a Career and Technical Education (CTE) teacher in a STEM field in Washington state. Determine which CTE certification route they qualify for based on their experience, and provide a complete list of all requirements they must fulfill to obtain their Washington state teacher certification. Include the specific degree requirements, required teacher preparation program approval standards, all testing requirements with score thresholds, and all administrative procedures they must complete.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CTERouteThreshold(BaseModel):
    threshold_text: Optional[str] = None  # e.g., "3 years / 6,000 hours"
    determination_statement: Optional[str] = None  # e.g., "You qualify for Business & Industry route"
    sources: List[str] = Field(default_factory=list)


class SimpleTopic(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TestingWESTB(BaseModel):
    statement: Optional[str] = None  # Should include mention of WEST-B and threshold if given
    threshold_text: Optional[str] = None  # e.g., "240 on each subtest (Reading, Writing, Math)"
    sources: List[str] = Field(default_factory=list)


class TestingContent(BaseModel):
    statement: Optional[str] = None  # e.g., "Must pass WEST-E or NES for the endorsement area"
    sources: List[str] = Field(default_factory=list)


class AdminTopic(BaseModel):
    statement: Optional[str] = None  # e.g., "Fingerprinting and background check required"
    sources: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    route_bi: Optional[CTERouteThreshold] = None
    route_cu: Optional[CTERouteThreshold] = None
    alt1: Optional[SimpleTopic] = None  # availability condition statement
    alt1_applicability: Optional[str] = None  # whether the answer addresses applicability given info
    residency: Optional[SimpleTopic] = None
    degree: Optional[SimpleTopic] = None
    pesb_program: Optional[SimpleTopic] = None
    west_b: Optional[TestingWESTB] = None
    content_test: Optional[TestingContent] = None
    fingerprinting: Optional[AdminTopic] = None
    character_fitness: Optional[AdminTopic] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
Extract, from the answer text, the specific statements and any URLs (sources) related to Washington CTE certification in STEM. Capture each topic separately. If something is not stated in the answer, return null for the field and an empty list for sources.

You must extract the following structured fields:

1) route_bi:
   - threshold_text: The experience requirement for the WA CTE Business & Industry route as stated in the answer (e.g., "3 years / 6,000 hours of industry experience in the occupational area").
   - determination_statement: The exact sentence/claim where the answer states the candidate qualifies (or does not) for the Business & Industry route.
   - sources: All URLs provided in the answer that are intended to support the Business & Industry route requirement and/or determination.

2) route_cu:
   - threshold_text: The experience requirement for the WA CTE College & University route as stated in the answer (e.g., "1 year / 2,000 hours").
   - determination_statement: The exact sentence/claim where the answer states the candidate qualifies (or does not) for the College & University route.
   - sources: All URLs provided in the answer that support the College & University route requirement and/or determination.

3) alt1:
   - statement: The statement of Alternative Route 1 availability condition as presented (e.g., "available for district employees with at least an associate degree").
   - sources: All URLs supporting this condition if provided.

   Additionally extract:
   - alt1_applicability: The sentence (if any) where the answer addresses whether Alternative Route 1 applies to this candidate given the provided information (e.g., "only if they are a district employee" or "district-employee status not provided").

4) residency:
   - statement: The statement that Washington requires educators to obtain a Residency Teacher Certificate (if stated).
   - sources: URLs supporting this statement if provided.

5) degree:
   - statement: The statement that the minimum required degree is a bachelor's (and whether the answer aligns this with the candidate’s bachelor's degree).
   - sources: URLs supporting this statement if provided.

6) pesb_program:
   - statement: The statement that the teacher preparation program must be approved by the Professional Educator Standards Board (PESB).
   - sources: URLs supporting this statement if provided.

7) west_b:
   - statement: The statement that WEST-B must be passed and the passing thresholds if given.
   - threshold_text: The passing score requirement as stated (e.g., "240 on each subtest in Reading, Writing, Math").
   - sources: URLs supporting this if provided.

8) content_test:
   - statement: The statement that the WEST-E or NES content knowledge assessment must be passed for the endorsement area.
   - sources: URLs supporting this if provided.

9) fingerprinting:
   - statement: The statement that fingerprinting and a background check must be completed before a certificate is issued.
   - sources: URLs supporting this if provided.

10) character_fitness:
   - statement: The statement that a Character and Fitness supplement must be completed at the time of certificate application.
   - sources: URLs supporting this if provided.

11) all_urls:
   - A flat list of all URLs present anywhere in the answer (including those not obviously categorized).

Rules:
- Do not invent information. Only extract what appears in the answer.
- For each topic’s 'sources', include only URLs that the answer appears to use to support that topic; leave empty if none.
- Always fill all_urls with every URL in the answer (including those that are also in topic-specific sources).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str], limit: int = 6) -> List[str]:
    seen = set()
    unique = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            unique.append(u)
        if len(unique) >= limit:
            break
    return unique


def _choose_sources(topic_urls: Optional[List[str]], all_urls: List[str], limit: int = 6) -> List[str]:
    if topic_urls and len(topic_urls) > 0:
        return _dedupe_urls(topic_urls, limit=limit)
    return _dedupe_urls(all_urls, limit=limit)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_route_qualification_nodes(evaluator: Evaluator, parent_node, extr: RequirementsExtraction) -> None:
    """
    Build and verify the route qualification determinations.
    We keep Business & Industry and College & University as critical checks.
    Alternative Route 1 is handled separately (non-critical) due to framework constraints.
    """
    # Parent node for route qualification (critical, parallel)
    route_node = evaluator.add_parallel(
        id="CTE_Route_Qualification_Determination",
        desc="Determines the candidate’s qualification status for WA CTE routes using stated thresholds and background.",
        parent=parent_node,
        critical=True
    )

    # ------------------ Business & Industry route ------------------ #
    bi_node = evaluator.add_sequential(
        id="CTE_Business_and_Industry_Route_Qualification",
        desc="Business & Industry route: uses the stated requirement (e.g., 3 years / 6,000 hours) to determine qualification.",
        parent=route_node,
        critical=True
    )
    bi = extr.route_bi or CTERouteThreshold()

    # Leaf: threshold mentioned in the answer (existence)
    evaluator.add_custom_node(
        result=bi.threshold_text is not None and len(bi.threshold_text.strip()) > 0,
        id="bi_threshold_extracted",
        desc="Answer states the Business & Industry route experience requirement",
        parent=bi_node,
        critical=True
    )

    # Leaf: sources present for B&I
    bi_sources = _choose_sources(bi.sources, extr.all_urls)
    src_exist_bi = evaluator.add_custom_node(
        result=len(bi_sources) > 0,
        id="bi_sources_present",
        desc="Sources are provided for the Business & Industry route requirement",
        parent=bi_node,
        critical=True
    )

    # Leaf: requirement supported by sources
    bi_supported_leaf = evaluator.add_leaf(
        id="bi_requirement_supported",
        desc="The stated Business & Industry route experience requirement is correct per cited sources",
        parent=bi_node,
        critical=True
    )
    bi_claim = f"In Washington, the CTE Business & Industry route requires {bi.threshold_text} of paid occupational experience in the occupational/CTE area."
    await evaluator.verify(
        claim=bi_claim,
        node=bi_supported_leaf,
        sources=bi_sources,
        additional_instruction="Verify the stated experience threshold for the Business & Industry CTE route (e.g., 3 years / 6,000 hours). The source should explicitly match or clearly support this requirement."
    )

    # Leaf: answer states the candidate qualifies for B&I route
    evaluator.add_custom_node(
        result=bi.determination_statement is not None and len(bi.determination_statement.strip()) > 0,
        id="bi_qualifies_present",
        desc="Answer states that the candidate qualifies for the Business & Industry route",
        parent=bi_node,
        critical=True
    )

    # ------------------ College & University route ------------------ #
    cu_node = evaluator.add_sequential(
        id="CTE_College_and_University_Route_Qualification",
        desc="College & University route: uses the stated requirement (e.g., 1 year / 2,000 hours) to determine qualification.",
        parent=route_node,
        critical=True
    )
    cu = extr.route_cu or CTERouteThreshold()

    # Leaf: threshold mentioned in the answer (existence)
    evaluator.add_custom_node(
        result=cu.threshold_text is not None and len(cu.threshold_text.strip()) > 0,
        id="cu_threshold_extracted",
        desc="Answer states the College & University route experience requirement",
        parent=cu_node,
        critical=True
    )

    # Leaf: sources present for C&U
    cu_sources = _choose_sources(cu.sources, extr.all_urls)
    src_exist_cu = evaluator.add_custom_node(
        result=len(cu_sources) > 0,
        id="cu_sources_present",
        desc="Sources are provided for the College & University route requirement",
        parent=cu_node,
        critical=True
    )

    # Leaf: requirement supported by sources
    cu_supported_leaf = evaluator.add_leaf(
        id="cu_requirement_supported",
        desc="The stated College & University route experience requirement is correct per cited sources",
        parent=cu_node,
        critical=True
    )
    cu_claim = f"In Washington, the CTE College & University route requires {cu.threshold_text} of industry experience in the occupational/CTE area."
    await evaluator.verify(
        claim=cu_claim,
        node=cu_supported_leaf,
        sources=cu_sources,
        additional_instruction="Verify the stated experience threshold for the College & University CTE route (e.g., 1 year / 2,000 hours). The source should explicitly match or clearly support this requirement."
    )

    # Leaf: answer states the candidate qualifies for C&U route
    evaluator.add_custom_node(
        result=cu.determination_statement is not None and len(cu.determination_statement.strip()) > 0,
        id="cu_qualifies_present",
        desc="Answer states that the candidate qualifies for the College & University route",
        parent=cu_node,
        critical=True
    )


async def build_alternative_route_1_node(evaluator: Evaluator, parent_node, extr: RequirementsExtraction) -> None:
    """
    Build and verify Alternative Route 1 applicability as a non-critical node.
    We place this at the top-level (non-critical) to satisfy critical-node constraints.
    """
    alt1_node = evaluator.add_sequential(
        id="Alternative_Route_1_Applicability",
        desc="Alternative Route 1 availability condition (district employees with at least an associate degree) and applicability to the candidate.",
        parent=parent_node,
        critical=False
    )
    alt1 = extr.alt1 or SimpleTopic()
    alt1_sources = _choose_sources(alt1.sources, extr.all_urls)

    # Leaf: condition stated in the answer
    evaluator.add_custom_node(
        result=alt1.statement is not None and len(alt1.statement.strip()) > 0,
        id="alt1_condition_stated",
        desc="Answer states Alternative Route 1 availability condition",
        parent=alt1_node,
        critical=True
    )

    # Leaf: sources present for Alt1
    evaluator.add_custom_node(
        result=len(alt1_sources) > 0,
        id="alt1_sources_present",
        desc="Sources are provided for Alternative Route 1 condition",
        parent=alt1_node,
        critical=True
    )

    # Leaf: Alternative Route 1 condition supported by sources
    alt1_supported_leaf = evaluator.add_leaf(
        id="alt1_condition_supported",
        desc="Alternative Route 1 availability condition is correct per cited sources",
        parent=alt1_node,
        critical=True
    )
    alt1_claim = "In Washington, Alternative Route 1 is available for district employees (e.g., classified staff) who hold at least an associate degree."
    await evaluator.verify(
        claim=alt1_claim,
        node=alt1_supported_leaf,
        sources=alt1_sources,
        additional_instruction="Verify the eligibility condition for Alternative Route 1: availability for district employees (such as paraprofessionals or classified staff) with at least an associate degree."
    )

    # Leaf: answer addresses applicability given provided info
    evaluator.add_custom_node(
        result=extr.alt1_applicability is not None and len(extr.alt1_applicability.strip()) > 0,
        id="alt1_applicability_addressed",
        desc="Answer addresses whether Alternative Route 1 applies given district-employee status (notes if not established)",
        parent=alt1_node,
        critical=True
    )


async def build_residency_degree_pesb_nodes(evaluator: Evaluator, parent_node, extr: RequirementsExtraction) -> None:
    """
    Build nodes for residency certificate requirement, bachelor’s degree requirement, and PESB-approved program requirement.
    """
    # Residency certificate requirement (critical sequential)
    residency_node = evaluator.add_sequential(
        id="Residency_Teacher_Certificate_Requirement",
        desc="States that Washington requires educators to obtain a residency teacher certificate to teach in public schools.",
        parent=parent_node,
        critical=True
    )
    residency = extr.residency or SimpleTopic()
    residency_sources = _choose_sources(residency.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=residency.statement is not None and len(residency.statement.strip()) > 0,
        id="residency_statement_present",
        desc="Answer states the residency teacher certificate requirement",
        parent=residency_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(residency_sources) > 0,
        id="residency_sources_present",
        desc="Sources are provided for residency certificate requirement",
        parent=residency_node,
        critical=True
    )
    residency_supported = evaluator.add_leaf(
        id="residency_supported_by_sources",
        desc="Residency teacher certificate requirement is supported by cited sources",
        parent=residency_node,
        critical=True
    )
    residency_claim = "Washington requires educators to obtain a Residency Teacher Certificate to teach in public K–12 schools."
    await evaluator.verify(
        claim=residency_claim,
        node=residency_supported,
        sources=residency_sources,
        additional_instruction="Confirm that Washington requires a Residency Teacher Certificate for public-school teaching."
    )

    # Degree requirement (critical sequential)
    degree_node = evaluator.add_sequential(
        id="Degree_Requirement",
        desc="States that a minimum of a bachelor’s degree is required for Washington teacher certification.",
        parent=parent_node,
        critical=True
    )
    degree = extr.degree or SimpleTopic()
    degree_sources = _choose_sources(degree.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=degree.statement is not None and ("bachelor" in degree.statement.lower() if degree.statement else False),
        id="degree_statement_present",
        desc="Answer states that at least a bachelor's degree is required",
        parent=degree_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(degree_sources) > 0,
        id="degree_sources_present",
        desc="Sources are provided for minimum degree requirement",
        parent=degree_node,
        critical=True
    )
    degree_supported = evaluator.add_leaf(
        id="degree_supported_by_sources",
        desc="Minimum degree requirement (bachelor’s) is supported by cited sources",
        parent=degree_node,
        critical=True
    )
    degree_claim = "Washington teacher certification requires at minimum a bachelor's degree."
    await evaluator.verify(
        claim=degree_claim,
        node=degree_supported,
        sources=degree_sources,
        additional_instruction="Confirm that Washington requires at least a bachelor's degree for teacher certification."
    )

    # PESB-approved program requirement (critical sequential)
    pesb_node = evaluator.add_sequential(
        id="PESB_Approved_Preparation_Program",
        desc="States that the teacher preparation program must be approved by the Professional Educator Standards Board (PESB).",
        parent=parent_node,
        critical=True
    )
    pesb = extr.pesb_program or SimpleTopic()
    pesb_sources = _choose_sources(pesb.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=pesb.statement is not None and ("pesb" in pesb.statement.lower() if pesb.statement else False),
        id="pesb_statement_present",
        desc="Answer states that the teacher prep program must be PESB-approved",
        parent=pesb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(pesb_sources) > 0,
        id="pesb_sources_present",
        desc="Sources are provided for PESB-approval requirement",
        parent=pesb_node,
        critical=True
    )
    pesb_supported = evaluator.add_leaf(
        id="pesb_supported_by_sources",
        desc="PESB-approval requirement is supported by cited sources",
        parent=pesb_node,
        critical=True
    )
    pesb_claim = "In Washington, the teacher preparation program must be approved by the Professional Educator Standards Board (PESB)."
    await evaluator.verify(
        claim=pesb_claim,
        node=pesb_supported,
        sources=pesb_sources,
        additional_instruction="Confirm that Washington requires completion of a PESB-approved teacher preparation program."
    )


async def build_testing_nodes(evaluator: Evaluator, parent_node, extr: RequirementsExtraction) -> None:
    """
    Build nodes for testing requirements: WEST-B (with thresholds) and content knowledge tests (WEST-E or NES).
    """
    testing_node = evaluator.add_parallel(
        id="Testing_Requirements",
        desc="Includes all testing requirements with thresholds where specified.",
        parent=parent_node,
        critical=True
    )

    # WEST-B (critical sequential)
    westb_node = evaluator.add_sequential(
        id="WEST_B_Basic_Skills_Test_And_Thresholds",
        desc="States WEST-B requirement and that passing requires a scaled score of 240+ on each subtest (Reading, Writing, Math).",
        parent=testing_node,
        critical=True
    )
    west_b = extr.west_b or TestingWESTB()
    westb_sources = _choose_sources(west_b.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=west_b.threshold_text is not None and ("240" in west_b.threshold_text if west_b.threshold_text else False),
        id="westb_threshold_statement_present",
        desc="Answer states WEST-B thresholds (mentions 240 per subtest)",
        parent=westb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(westb_sources) > 0,
        id="westb_sources_present",
        desc="Sources are provided for WEST-B and thresholds",
        parent=westb_node,
        critical=True
    )
    westb_supported = evaluator.add_leaf(
        id="westb_supported_by_sources",
        desc="WEST-B passing thresholds are supported by cited sources",
        parent=westb_node,
        critical=True
    )
    westb_claim = "Passing WEST-B requires a scaled score of 240 or higher on each of the Reading, Writing, and Mathematics subtests."
    await evaluator.verify(
        claim=westb_claim,
        node=westb_supported,
        sources=westb_sources,
        additional_instruction="Verify from official sources (e.g., OSPI/Pearson WEST) that WEST-B requires a scaled score of 240+ on each subtest."
    )

    # Content Knowledge Test (critical sequential)
    content_node = evaluator.add_sequential(
        id="Content_Knowledge_Test",
        desc="States that the WEST-E or NES content knowledge assessment must be passed for the endorsement area.",
        parent=testing_node,
        critical=True
    )
    content = extr.content_test or TestingContent()
    content_sources = _choose_sources(content.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=content.statement is not None and any(k in content.statement.lower() for k in ["west-e", "nes"]),
        id="content_test_statement_present",
        desc="Answer states the content test (WEST-E or NES) requirement",
        parent=content_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(content_sources) > 0,
        id="content_test_sources_present",
        desc="Sources are provided for content test requirement",
        parent=content_node,
        critical=True
    )
    content_supported = evaluator.add_leaf(
        id="content_test_supported_by_sources",
        desc="Content test (WEST-E or NES) requirement is supported by cited sources",
        parent=content_node,
        critical=True
    )
    content_claim = "Washington requires passing a content knowledge assessment (either WEST-E or NES) for the endorsement area for teacher certification."
    await evaluator.verify(
        claim=content_claim,
        node=content_supported,
        sources=content_sources,
        additional_instruction="Confirm that WA requires candidates to pass a subject/content knowledge test (WEST-E or NES) for their endorsement area."
    )


async def build_admin_nodes(evaluator: Evaluator, parent_node, extr: RequirementsExtraction) -> None:
    """
    Build nodes for administrative procedures: fingerprinting/background check and Character & Fitness supplement.
    """
    admin_node = evaluator.add_parallel(
        id="Administrative_Procedures",
        desc="Includes all administrative procedures: fingerprinting/background and character & fitness.",
        parent=parent_node,
        critical=True
    )

    # Fingerprinting and Background Check (critical sequential)
    fp_node = evaluator.add_sequential(
        id="Fingerprinting_and_Background_Check",
        desc="States that fingerprinting and a background check must be completed before a certificate is issued.",
        parent=admin_node,
        critical=True
    )
    fp = extr.fingerprinting or AdminTopic()
    fp_sources = _choose_sources(fp.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=fp.statement is not None and any(k in fp.statement.lower() for k in ["fingerprint", "background"]),
        id="fingerprinting_statement_present",
        desc="Answer states fingerprinting and background check requirement",
        parent=fp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(fp_sources) > 0,
        id="fingerprinting_sources_present",
        desc="Sources are provided for fingerprinting/background requirement",
        parent=fp_node,
        critical=True
    )
    fp_supported = evaluator.add_leaf(
        id="fingerprinting_supported_by_sources",
        desc="Fingerprinting and background check requirement is supported by cited sources",
        parent=fp_node,
        critical=True
    )
    fp_claim = "Fingerprinting and a background check must be completed before a Washington teaching certificate is issued."
    await evaluator.verify(
        claim=fp_claim,
        node=fp_supported,
        sources=fp_sources,
        additional_instruction="Verify that WA requires fingerprinting and a background check prior to issuance of the teaching certificate."
    )

    # Character and Fitness Supplement (critical sequential)
    cf_node = evaluator.add_sequential(
        id="Character_and_Fitness_Supplement",
        desc="States that a character and fitness supplement must be completed at the time of certificate application.",
        parent=admin_node,
        critical=True
    )
    cf = extr.character_fitness or AdminTopic()
    cf_sources = _choose_sources(cf.sources, extr.all_urls)

    evaluator.add_custom_node(
        result=cf.statement is not None and all(k in cf.statement.lower() for k in ["character", "fitness"]),
        id="character_fitness_statement_present",
        desc="Answer states Character and Fitness supplement requirement",
        parent=cf_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(cf_sources) > 0,
        id="character_fitness_sources_present",
        desc="Sources are provided for Character and Fitness supplement requirement",
        parent=cf_node,
        critical=True
    )
    cf_supported = evaluator.add_leaf(
        id="character_fitness_supported_by_sources",
        desc="Character & Fitness supplement requirement is supported by cited sources",
        parent=cf_node,
        critical=True
    )
    cf_claim = "A Character and Fitness Supplement must be completed at the time of the certificate application in Washington."
    await evaluator.verify(
        claim=cf_claim,
        node=cf_supported,
        sources=cf_sources,
        additional_instruction="Verify that WA requires a Character and Fitness supplement as part of the certification application."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the WA STEM CTE certification requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall evaluation has independent components
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    # Top-level response node (kept non-critical to allow optional sub-criteria like Alternative Route 1)
    response_node = evaluator.add_parallel(
        id="CTE_STEM_WA_Certification_Response",
        desc="Evaluate route qualification and completeness of requirements with supported evidence.",
        parent=root,
        critical=False
    )

    # Build critical components
    await build_route_qualification_nodes(evaluator, response_node, extraction)
    await build_residency_degree_pesb_nodes(evaluator, response_node, extraction)
    await build_testing_nodes(evaluator, response_node, extraction)
    await build_admin_nodes(evaluator, response_node, extraction)

    # Handle Alternative Route 1 separately (non-critical)
    await build_alternative_route_1_node(evaluator, response_node, extraction)

    # Add small custom info for visibility
    evaluator.add_custom_info(
        info={
            "all_urls_count": len(extraction.all_urls if extraction and extraction.all_urls else []),
            "note": "Alternative Route 1 evaluated as non-critical due to framework constraints on critical parent/child."
        },
        info_type="meta",
        info_name="evaluation_notes"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()