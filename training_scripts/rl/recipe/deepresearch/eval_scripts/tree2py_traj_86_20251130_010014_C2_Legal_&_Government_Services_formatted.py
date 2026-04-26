import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "co_senate_vacancy_faith_winter_2025"
TASK_DESCRIPTION = (
    "Colorado State Senator Faith Winter, who represented District 25, died in a car crash on November 26, 2025. "
    "Based on Colorado's state legislative vacancy filling procedures, provide a comprehensive explanation of the legal requirements for filling her vacant seat. "
    "Your answer must include:\n\n"
    "1. Vacancy Committee Composition: Identify who must serve on the vacancy committee that will select her replacement, including both the party affiliation requirement and any geographic residency requirements for committee members.\n\n"
    "2. Election Timeline: Determine when the person appointed by the vacancy committee must stand for election, applying Colorado's timing rules based on when the vacancy occurred (November 26, 2025).\n\n"
    "3. Voter Eligibility: Identify which categories of voters are eligible to participate in the vacancy election for this seat.\n\n"
    "For each requirement, provide the relevant legal or procedural source (such as Colorado statutes or official government guidance) and include reference URLs to support your findings."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CommitteeComposition(BaseModel):
    party_affiliation: Optional[str] = None
    geographic_residency: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SelectionMethod(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ElectionTimeline(BaseModel):
    statement: Optional[str] = None
    conclusion: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VoterEligibility(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Authorities(BaseModel):
    names: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class VacancyProcedureExtraction(BaseModel):
    committee: Optional[CommitteeComposition] = None
    selection: Optional[SelectionMethod] = None
    timeline: Optional[ElectionTimeline] = None
    voter_eligibility: Optional[VoterEligibility] = None
    authorities: Optional[Authorities] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vacancy_procedure() -> str:
    return (
        "Extract from the answer the specific statements and source URLs relevant to Colorado's process for filling "
        "the vacant District 25 Colorado State Senate seat after a vacancy on November 26, 2025. Return a JSON object "
        "matching the following fields precisely:\n\n"
        "- committee:\n"
        "  - party_affiliation: The exact statement the answer makes about what party's members must compose the vacancy committee (e.g., members of the Democratic Party).\n"
        "  - geographic_residency: The exact statement the answer makes about any geographic residency requirements for vacancy committee members (e.g., Democratic county commissioners who live in Senate District 25 must be included).\n"
        "  - urls: Array of all URLs in the answer that the answer associates with or cites to support the committee composition info.\n"
        "- selection:\n"
        "  - statement: The exact statement the answer makes about how the replacement is selected (e.g., selected through the Democratic Party vacancy committee and NOT by governor or a districtwide special election for the initial appointment).\n"
        "  - urls: URLs cited for that selection method.\n"
        "- timeline:\n"
        "  - statement: The exact statement explaining the timing rule as applied to a vacancy occurring on November 26, 2025.\n"
        "  - conclusion: The answer’s explicit conclusion about when the appointee must run (e.g., next general election in November 2026).\n"
        "  - urls: URLs cited to support the timing rule and conclusion.\n"
        "- voter_eligibility:\n"
        "  - statement: The exact statement regarding which categories of voters are eligible to participate in the vacancy election for this seat (e.g., Democratic registered voters and unaffiliated voters within District 25).\n"
        "  - urls: URLs cited for this voter eligibility statement.\n"
        "- authorities:\n"
        "  - names: List of the formal authorities explicitly cited by name in the answer, if any, such as 'HB25-1315', 'House Bill 25-1315', 'C.R.S. 1-12-203', or 'Colorado Revised Statutes § 1-12-203'.\n"
        "  - urls: URLs specifically pointing to those authorities, if present in the answer.\n\n"
        "Rules:\n"
        "1) Only extract URLs that are explicitly present in the answer (including markdown links). Do not fabricate URLs.\n"
        "2) If a field is not present in the answer, set it to null (for strings/objects) or an empty array (for url lists).\n"
        "3) Do not paraphrase; capture the statements exactly as written in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


async def _verify_supported_by_urls(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    add_ins: str,
    critical: bool = True,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    cleaned_urls = _dedup_urls(urls)
    if not cleaned_urls:
        # No URLs were provided in the answer for this requirement; fail the node
        leaf.score = 0.0
        leaf.status = "failed"
        return
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=cleaned_urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #
async def _build_tree_and_verify(evaluator: Evaluator, ex: VacancyProcedureExtraction) -> None:
    # Top-level critical node mirroring the rubric's root (since Evaluator's real root is always non-critical)
    proc_node = evaluator.add_parallel(
        id="ColoradoSenateVacancyProcedure",
        desc="Complete and accurate explanation of the legal/procedural requirements for filling Faith Winter's vacant Colorado State Senate District 25 seat, matching all stated constraints",
        parent=evaluator.root,
        critical=True
    )

    # 1) Vacancy Committee Composition (parallel, critical)
    comp_node = evaluator.add_parallel(
        id="VacancyCommitteeComposition",
        desc="Correct identification of who comprises the vacancy committee",
        parent=proc_node,
        critical=True
    )

    # 1.a) Party Affiliation Requirement (presence in the answer)
    party_leaf = evaluator.add_leaf(
        id="PartyAffiliationRequirement",
        desc="States that the vacancy committee must consist of members of the Democratic Party (same party as Faith Winter)",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the vacancy committee must consist of members of the Democratic Party (i.e., of the same party as the vacating legislator).",
        node=party_leaf,
        additional_instruction="Judge only based on the answer text. Allow minor wording variations but the substance must clearly indicate Democratic Party members compose the vacancy committee."
    )

    # 1.b) Geographic Residency Requirement (presence in the answer)
    geo_leaf = evaluator.add_leaf(
        id="GeographicResidencyRequirement",
        desc="States that Democratic county commissioners who live in Colorado State Senate District 25 must be included in the vacancy committee",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that Democratic county commissioners who live in Senate District 25 must be included on the vacancy committee.",
        node=geo_leaf,
        additional_instruction="Judge only based on the answer text. Allow minor wording variations but the requirement about Democratic county commissioners residing in District 25 being included must be clearly conveyed."
    )

    # 2) Selection Method Requirement (presence in the answer)
    selection_leaf = evaluator.add_leaf(
        id="SelectionMethodRequirement",
        desc="States that the replacement is selected through the Democratic Party vacancy committee process (not gubernatorial appointment or a special election for the initial appointment)",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the initial replacement is selected through the Democratic Party's vacancy committee process, and explicitly not by the Governor and not by a districtwide special election for the initial appointment.",
        node=selection_leaf,
        additional_instruction="Judge strictly from the answer text. The statement must clearly rule out gubernatorial appointment and a districtwide special election for the initial appointment."
    )

    # 3) Election Timeline Requirement (presence in the answer)
    timeline_leaf = evaluator.add_leaf(
        id="ElectionTimelineRequirement",
        desc="Applies Colorado timing rules to a vacancy occurring on Nov 26, 2025 (odd-numbered year, after July 31) and concludes the appointee must run in the next general election (Nov 2026)",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer applies Colorado’s election-timing rules to a vacancy occurring on November 26, 2025 (an odd-numbered year and after July 31) and concludes the appointee must stand in the next general election in November 2026.",
        node=timeline_leaf,
        additional_instruction="Judge from the answer text only; it must both reference the vacancy date context (odd year after July 31) and the conclusion of November 2026 general election."
    )

    # 4) Voter Eligibility Requirement (presence in the answer)
    voter_leaf = evaluator.add_leaf(
        id="VoterEligibilityRequirement",
        desc="Identifies that eligible voters are Democratic Party registered voters and unaffiliated voters within District 25 for the vacancy election",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that, for the vacancy election for this seat, eligible voters are Democratic Party registered voters and unaffiliated voters within Senate District 25.",
        node=voter_leaf,
        additional_instruction="Judge from the answer text only. Allow small wording variations but the categories must match: Democratic registrants and unaffiliated voters within the district."
    )

    # 5) Sources and URLs (parallel, critical)
    sources_node = evaluator.add_parallel(
        id="SourcesAndURLs",
        desc="Provides required supporting legal/procedural sources and reference URLs",
        parent=proc_node,
        critical=True
    )

    # Retrieve URL buckets from extraction (safe)
    committee_urls = _dedup_urls(ex.committee.urls if ex and ex.committee else [])
    selection_urls = _dedup_urls(ex.selection.urls if ex and ex.selection else [])
    timeline_urls = _dedup_urls(ex.timeline.urls if ex and ex.timeline else [])
    voter_urls = _dedup_urls(ex.voter_eligibility.urls if ex and ex.voter_eligibility else [])
    authorities_names = (ex.authorities.names if ex and ex.authorities and ex.authorities.names else [])

    # 5.a) Committee composition source support
    await _verify_supported_by_urls(
        evaluator=evaluator,
        parent_node=sources_node,
        node_id="SourceAndURLForCommitteeComposition",
        desc="Provides at least one relevant legal/procedural source AND at least one reference URL supporting the vacancy committee composition claims",
        claim="At least one of these pages provides legal or procedural support for the answer's vacancy committee composition description, "
              "confirming either that the vacancy committee is composed of members of the vacating legislator’s political party and/or that "
              "Democratic county commissioners residing in the affected Senate district must be included.",
        urls=committee_urls,
        add_ins="Treat a page as supportive if it clearly corroborates at least one of the stated committee composition rules. If none of the pages support any of the rules, fail."
    )

    # 5.b) Selection method source support
    await _verify_supported_by_urls(
        evaluator=evaluator,
        parent_node=sources_node,
        node_id="SourceAndURLForSelectionMethod",
        desc="Provides at least one relevant legal/procedural source AND at least one reference URL supporting the selection-method claim (party vacancy committee process, not governor/special election for initial appointment)",
        claim="At least one of these pages confirms that for a Colorado state senate vacancy, the initial replacement is selected by the relevant political party’s vacancy committee, "
              "not by gubernatorial appointment and not by a districtwide special election for the initial appointment.",
        urls=selection_urls,
        add_ins="Look for explicit statements about party vacancy committees making the appointment and the absence of governor appointment or initial special election."
    )

    # 5.c) Election timeline source support
    await _verify_supported_by_urls(
        evaluator=evaluator,
        parent_node=sources_node,
        node_id="SourceAndURLForElectionTimeline",
        desc="Provides at least one relevant legal/procedural source AND at least one reference URL supporting the election-timeline claim",
        claim="At least one of these pages supports the timing analysis that a vacancy occurring on November 26, 2025 (odd-numbered year after July 31) requires the appointee to stand in the next general election in November 2026.",
        urls=timeline_urls,
        add_ins="Accept a page if it clearly describes the rule that a vacancy after July 31 of an odd year results in the seat appearing on the next general election (November 2026 for a vacancy on Nov 26, 2025)."
    )

    # 5.d) Voter eligibility source support
    await _verify_supported_by_urls(
        evaluator=evaluator,
        parent_node=sources_node,
        node_id="SourceAndURLForVoterEligibility",
        desc="Provides at least one relevant legal/procedural source AND at least one reference URL supporting the voter-eligibility claim",
        claim="At least one of these pages supports the answer’s description that eligible voters for the vacancy election are Democratic Party registered voters and unaffiliated voters within Senate District 25.",
        urls=voter_urls,
        add_ins="The page must address which voters can participate in the relevant vacancy-related election context for this seat; accept if it confirms Democratic registrants and unaffiliated voters are eligible."
    )

    # 5.e) Cites governing authorities by name/citation in the answer text
    cites_leaf = evaluator.add_leaf(
        id="CitesStatedGoverningAuthorities",
        desc="Cites the stated governing authorities: HB25-1315 and Colorado Revised Statutes Title 1, Elections § 1-12-203 (as sources relied upon or referenced)",
        parent=sources_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly cites both Colorado House Bill 25-1315 (which may be written as HB25-1315 or House Bill 25-1315) and Colorado Revised Statutes § 1-12-203 (which may be written as C.R.S. 1-12-203 or CRS 1-12-203).",
        node=cites_leaf,
        additional_instruction="Judge based on the answer text only. Allow common citation variants and spacing (e.g., 'HB 25-1315', 'C.R.S. 1-12-203'). Both authorities must be cited to pass."
    )

    # Record some helpful URL stats
    evaluator.add_custom_info(
        info={
            "committee_urls_count": len(committee_urls),
            "selection_urls_count": len(selection_urls),
            "timeline_urls_count": len(timeline_urls),
            "voter_urls_count": len(voter_urls),
            "authorities_names": authorities_names,
        },
        info_type="url_statistics",
        info_name="url_and_authority_extraction_stats"
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
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extraction: VacancyProcedureExtraction = await evaluator.extract(
        prompt=prompt_extract_vacancy_procedure(),
        template_class=VacancyProcedureExtraction,
        extraction_name="vacancy_procedure_extraction"
    )

    # Build verification tree and run verifications
    await _build_tree_and_verify(evaluator, extraction)

    # Return the standard summary
    return evaluator.get_summary()