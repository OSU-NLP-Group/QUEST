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
TASK_ID = "itcs2026_researcher_criteria"
TASK_DESCRIPTION = """
The 17th Innovations in Theoretical Computer Science (ITCS) conference was held at Bocconi University in Milan, Italy from January 27-30, 2026. Identify a researcher who satisfies ALL of the following criteria:

1. Serves on the ITCS 2026 program committee
2. Is affiliated with a university that ranks in the top 3 for Computer Science in the QS World University Rankings by Subject 2025 (Massachusetts Institute of Technology, Stanford University, or Carnegie Mellon University)
3. Has an h-index of at least 40 according to Google Scholar
4. Has received National Science Foundation (NSF) funding, specifically either an NSF CAREER Award or an NSF grant in the Algorithmic Foundations or Computer and Information Science and Engineering (CISE) programs
5. Has received at least one additional major research award or fellowship (such as Sloan Fellowship, Packard Fellowship, ONR Young Investigator Award, or similar prestigious recognition)

Provide the researcher's name, their current university affiliation, and reference URLs supporting each of the five criteria.
"""

TOP3_UNIVERSITIES_FULL = [
    "Massachusetts Institute of Technology",
    "Stanford University",
    "Carnegie Mellon University"
]
TOP3_UNIVERSITY_ALIASES = {
    "Massachusetts Institute of Technology": {"massachusetts institute of technology", "mit"},
    "Stanford University": {"stanford university", "stanford"},
    "Carnegie Mellon University": {"carnegie mellon university", "carnegie mellon", "cmu"},
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResearcherExtraction(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None

    # URLs explicitly provided in the answer to support each criterion
    itcs_pc_urls: List[str] = Field(default_factory=list)
    affiliation_urls: List[str] = Field(default_factory=list)
    google_scholar_urls: List[str] = Field(default_factory=list)
    nsf_urls: List[str] = Field(default_factory=list)
    major_award_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher() -> str:
    return """
    Extract the single primary researcher identified in the answer who is claimed to satisfy all five criteria.
    Return the following fields:
    - name: The researcher's full name as written in the answer.
    - affiliation: The current university affiliation for the researcher as stated in the answer.
    - itcs_pc_urls: A list of URLs that the answer cites to support that the researcher serves on the ITCS 2026 program committee.
    - affiliation_urls: A list of URLs that support the researcher's current university affiliation.
    - google_scholar_urls: A list of URLs to the researcher's Google Scholar profile (or pages that show h-index) supporting the h-index criterion.
    - nsf_urls: A list of URLs to NSF (or institutional) pages supporting the NSF funding criterion (e.g., NSF CAREER Award page, award/grant pages in Algorithmic Foundations/CISE).
    - major_award_urls: A list of URLs that support at least one additional major research award or fellowship (e.g., Sloan Research Fellowship, Packard Fellowship, ONR Young Investigator Award, etc.).

    Rules:
    - Only include URLs that are explicitly present in the answer text. Do not invent URLs.
    - If the answer mentions multiple researchers, choose the one that is ultimately presented as satisfying all criteria; if unclear, choose the first researcher described as meeting them.
    - If any field is missing in the answer, return null for the string field or an empty list for the URL lists.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_itcs_pc(
    evaluator: Evaluator,
    parent,
    data: ResearcherExtraction
) -> None:
    # Criterion node: Sequential to enforce source existence before verification
    crit_node = evaluator.add_sequential(
        id="itcs_2026_committee",
        desc="The researcher must be listed on the ITCS 2026 program committee",
        parent=parent,
        critical=True,
    )

    # Existence of supporting sources
    evaluator.add_custom_node(
        result=bool(data.name) and bool(data.itcs_pc_urls),
        id="itcs_sources_provided",
        desc="ITCS 2026 PC membership sources provided",
        parent=crit_node,
        critical=True
    )

    # Verification leaf
    pc_leaf = evaluator.add_leaf(
        id="itcs_pc_verified",
        desc="Researcher serves on the ITCS 2026 program committee (supported by cited URLs)",
        parent=crit_node,
        critical=True
    )

    name = data.name or ""
    claim = f"{name} serves on the ITCS 2026 (17th ITCS, January 27–30, 2026) program committee."
    await evaluator.verify(
        claim=claim,
        node=pc_leaf,
        sources=data.itcs_pc_urls,
        additional_instruction=(
            "Confirm that the provided page(s) specifically refer to the ITCS 2026 edition (17th ITCS, or explicitly 2026) "
            "and that the 'Program Committee' or equivalent list includes the researcher's name (allowing minor variants). "
            "If the pages are irrelevant, outdated, or do not list the name, do not support the claim."
        )
    )


async def verify_affiliation(
    evaluator: Evaluator,
    parent,
    data: ResearcherExtraction
) -> None:
    crit_node = evaluator.add_sequential(
        id="university_affiliation",
        desc="The researcher must be currently affiliated with MIT, Stanford University, or Carnegie Mellon University (QS 2025 top 3 CS)",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.name) and bool(data.affiliation) and bool(data.affiliation_urls),
        id="affiliation_sources_provided",
        desc="Affiliation and supporting sources are provided",
        parent=crit_node,
        critical=True
    )

    # Verify affiliation is supported by URLs
    aff_supported_leaf = evaluator.add_leaf(
        id="affiliation_supported_by_urls",
        desc="Researcher's stated affiliation is supported by the cited URLs",
        parent=crit_node,
        critical=True
    )
    name = data.name or ""
    affiliation = data.affiliation or ""
    claim_supported = f"{name} is currently affiliated with {affiliation} (as faculty or researcher)."
    await evaluator.verify(
        claim=claim_supported,
        node=aff_supported_leaf,
        sources=data.affiliation_urls,
        additional_instruction=(
            "Confirm from the page(s) that the individual is affiliated with the stated university (current affiliation). "
            "This can be a department profile page, personal faculty page hosted by the institution, or other authoritative affiliation page."
        )
    )

    # Verify affiliation is one of the QS top-3 list
    aff_top3_leaf = evaluator.add_leaf(
        id="affiliation_in_top3",
        desc="Affiliation is one of the QS 2025 top-3 CS universities (MIT, Stanford, or Carnegie Mellon University)",
        parent=crit_node,
        critical=True
    )
    claim_top3 = (
        f"The affiliation '{affiliation}' is one of the following (allowing common aliases): "
        "Massachusetts Institute of Technology (MIT), Stanford University (Stanford), or Carnegie Mellon University (CMU)."
    )
    await evaluator.verify(
        claim=claim_top3,
        node=aff_top3_leaf,
        additional_instruction=(
            "Treat 'MIT' as equivalent to 'Massachusetts Institute of Technology'; 'Stanford' as 'Stanford University'; "
            "'CMU' or 'Carnegie Mellon' as 'Carnegie Mellon University'. This check is a simple logical membership check."
        )
    )


async def verify_hindex(
    evaluator: Evaluator,
    parent,
    data: ResearcherExtraction
) -> None:
    crit_node = evaluator.add_sequential(
        id="research_impact_metrics",
        desc="The researcher must have an h-index of at least 40 according to Google Scholar",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.google_scholar_urls),
        id="scholar_sources_provided",
        desc="Google Scholar URL(s) provided",
        parent=crit_node,
        critical=True
    )

    hidx_leaf = evaluator.add_leaf(
        id="hindex_40_or_more",
        desc="Google Scholar shows h-index >= 40",
        parent=crit_node,
        critical=True
    )
    name = data.name or ""
    claim = f"The Google Scholar profile for {name} shows an h-index of at least 40."
    await evaluator.verify(
        claim=claim,
        node=hidx_leaf,
        sources=data.google_scholar_urls,
        additional_instruction=(
            "On the Google Scholar profile page(s), locate the 'h-index' in the metrics/indices section. "
            "If the h-index is 40 or higher, support the claim. If the page is not a Google Scholar profile or the h-index is below 40 or not shown, do not support."
        )
    )


async def verify_nsf_funding(
    evaluator: Evaluator,
    parent,
    data: ResearcherExtraction
) -> None:
    crit_node = evaluator.add_sequential(
        id="nsf_funding",
        desc="The researcher must have received NSF funding: either NSF CAREER Award or an NSF grant in Algorithmic Foundations or a CISE division",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.nsf_urls),
        id="nsf_sources_provided",
        desc="NSF funding source URL(s) provided",
        parent=crit_node,
        critical=True
    )

    nsf_leaf = evaluator.add_leaf(
        id="nsf_funding_verified",
        desc="NSF CAREER or NSF grant in AF/CISE verified by cited URLs",
        parent=crit_node,
        critical=True
    )
    name = data.name or ""
    claim = (
        f"{name} has received NSF funding, specifically either an NSF CAREER Award or an NSF grant under Algorithmic Foundations "
        "(AF) or a CISE division."
    )
    await evaluator.verify(
        claim=claim,
        node=nsf_leaf,
        sources=data.nsf_urls,
        additional_instruction=(
            "Accept if the page(s) show an NSF CAREER Award OR an NSF award/grant under Algorithmic Foundations (AF) or any CISE division "
            "(e.g., CCF, IIS, CNS, OAC). It's fine if the page abbreviates divisions (e.g., 'CCF' under CISE). "
            "Ensure the award is indeed from NSF. If unclear or unrelated, do not support."
        )
    )


async def verify_major_award(
    evaluator: Evaluator,
    parent,
    data: ResearcherExtraction
) -> None:
    crit_node = evaluator.add_sequential(
        id="additional_major_award",
        desc="The researcher must have received at least one additional major research award or fellowship beyond NSF funding",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.major_award_urls),
        id="major_award_sources_provided",
        desc="Major award/fellowship source URL(s) provided",
        parent=crit_node,
        critical=True
    )

    award_leaf = evaluator.add_leaf(
        id="major_award_verified",
        desc="At least one major research award/fellowship is verified by cited URLs",
        parent=crit_node,
        critical=True
    )
    name = data.name or ""
    claim = (
        f"{name} has received at least one major research award or fellowship (e.g., Sloan Research Fellowship, Packard Fellowship, "
        "ONR Young Investigator Award, AFOSR Young Investigator, DARPA Young Faculty Award, PECASE, Simons Investigator, or similar)."
    )
    await evaluator.verify(
        claim=claim,
        node=award_leaf,
        sources=data.major_award_urls,
        additional_instruction=(
            "Confirm the page(s) clearly indicate the researcher received a prestigious research award or fellowship beyond NSF funding. "
            "Examples: Sloan Research Fellowship, Packard Fellowship, ONR/AFOSR Young Investigator, DARPA YFA, PECASE, Simons Investigator, "
            "or comparable national-level early-career/major research honors. Departmental teaching awards, travel grants, or small internal "
            "awards should not count."
        )
    )


async def verify_researcher(
    evaluator: Evaluator,
    root,
    data: ResearcherExtraction
) -> None:
    # Top-level critical parallel node aggregating all five criteria
    top = evaluator.add_parallel(
        id="researcher_identification",
        desc="Identify a researcher who satisfies all five specified academic and professional criteria",
        parent=root,
        critical=True
    )

    # Build subtrees for each criterion
    await verify_itcs_pc(evaluator, top, data)
    await verify_affiliation(evaluator, top, data)
    await verify_hindex(evaluator, top, data)
    await verify_nsf_funding(evaluator, top, data)
    await verify_major_award(evaluator, top, data)


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
    Evaluate an answer for the ITCS 2026 researcher identification task.
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

    # Extract the researcher's structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_researcher(),
        template_class=ResearcherExtraction,
        extraction_name="researcher_extraction"
    )

    # Record constraint info
    evaluator.add_custom_info(
        info={
            "qs_2025_top3_cs_universities": TOP3_UNIVERSITIES_FULL,
            "accepted_aliases": {k: list(v) for k, v in TOP3_UNIVERSITY_ALIASES.items()}
        },
        info_type="constraints",
        info_name="top_university_criteria"
    )

    # Build verification tree
    await verify_researcher(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()