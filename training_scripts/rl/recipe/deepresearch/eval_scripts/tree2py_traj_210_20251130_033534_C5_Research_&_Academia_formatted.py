import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "seligman_3i_atlas_first_paper"
TASK_DESCRIPTION = (
    "In July 2025, the interstellar object 3I/ATLAS was discovered, and multiple research teams rushed to publish "
    "scientific papers about it. Identify the first scientific paper on 3I/ATLAS that was led by Darryl Seligman from "
    "Michigan State University, and provide the following information with URL references for verification: "
    "(1) The paper's arXiv identifier or published journal citation, "
    "(2) The name of the journal where the paper was published or accepted for publication, "
    "(3) Darryl Seligman's primary institutional affiliation at the time of publication, "
    "(4) Any fellowship or grant affiliation held by Darryl Seligman at his institution, "
    "(5) The names of at least three U.S.-based research institutions (other than the lead author's institution) that collaborated on this paper, "
    "(6) The names of at least two non-U.S.-based institutions that collaborated on this paper, "
    "(7) The date when 3I/ATLAS was first discovered, and "
    "(8) The name of the survey telescope system that discovered 3I/ATLAS."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PaperExtraction(BaseModel):
    # Identification (allow optional title for stronger verification context)
    paper_title: Optional[str] = None
    paper_identifier: Optional[str] = None  # either an arXiv ID (e.g., "arXiv:2507.xxxxx") or a full journal citation text
    paper_reference_urls: List[str] = Field(default_factory=list)  # URLs that point to this paper (arXiv, journal, ADS, etc.)

    # Publication venue
    publication_venue: Optional[str] = None  # journal name (e.g., "ApJL", "Nature"), or "arXiv" / "arXiv-only" if preprint only
    publication_venue_urls: List[str] = Field(default_factory=list)

    # Lead author affiliation and any fellowship/grant
    primary_institution: Optional[str] = None
    primary_institution_urls: List[str] = Field(default_factory=list)
    fellowship_or_grant: Optional[str] = None  # e.g., "MSU Presidential Fellow", "NSF grant PHY-XXXXX", etc.
    fellowship_or_grant_urls: List[str] = Field(default_factory=list)

    # Collaborating institutions
    us_institutions: List[str] = Field(default_factory=list)  # exclude the lead author's institution
    us_institutions_urls: List[str] = Field(default_factory=list)
    non_us_institutions: List[str] = Field(default_factory=list)
    non_us_institutions_urls: List[str] = Field(default_factory=list)

    # Discovery details
    discovery_date: Optional[str] = None        # e.g., "2025-07-XX" or "July XX, 2025"
    discovery_date_urls: List[str] = Field(default_factory=list)
    discovery_system: Optional[str] = None      # e.g., "ATLAS (Asteroid Terrestrial-impact Last Alert System)"
    discovery_system_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
You must extract information ONLY from the provided answer text. Do not invent anything.

Goal: Identify the FIRST scientific paper on 3I/ATLAS that is led (first author) by Darryl Seligman (Michigan State University) and extract the following for that single target paper only. If multiple papers are mentioned, choose the earliest (first) one specifically led by Darryl Seligman on 3I/ATLAS based on dates or explicit wording (e.g., "first", "earliest", "initial", or the earliest dated arXiv submission/publication).

For the chosen single paper, extract:
- paper_title: The paper’s title as written in the answer (if present), otherwise null.
- paper_identifier: Either the exact arXiv identifier (e.g., "arXiv:2507.xxxxx") or the full formal journal citation text as given in the answer. Prefer the arXiv ID if both are present.
- paper_reference_urls: All URLs in the answer that directly point to that exact paper (e.g., arXiv page, journal page, ADS page). Use only URLs explicitly present in the answer.

- publication_venue: The name of the journal where this paper was published or accepted for publication (e.g., "ApJ Letters", "Nature Astronomy"). If clearly arXiv-only in the answer, set to "arXiv" or "arXiv-only" exactly as stated.
- publication_venue_urls: URL(s) in the answer that support the venue info (journal page, arXiv page comments indicating acceptance, press release, ADS, etc.).

- primary_institution: Darryl Seligman's primary institutional affiliation for the paper (as listed in the paper metadata/author affiliations in the answer).
- primary_institution_urls: URL(s) in the answer that show this affiliation for the paper (paper page, journal page, ADS, etc.).

- fellowship_or_grant: Any fellowship or grant affiliation held by Darryl Seligman at his institution that is stated/acknowledged for the paper in the answer (e.g., fellowships, endowed chairs, specific grants). If none is mentioned in the answer, set null.
- fellowship_or_grant_urls: URL(s) in the answer that support the fellowship/grant (paper metadata, acknowledgments, institutional page). If none cited in the answer, return an empty array.

- us_institutions: Names of at least three U.S.-based collaborating institutions on the paper (NOT including the lead author’s institution). Extract exactly as written in the answer. If fewer than three are present, extract as many as are present.
- us_institutions_urls: URL(s) in the answer that list these institutions as co-author affiliations for the paper (paper/journal/ADS pages, official collaboration pages). If none cited in the answer, return an empty array.

- non_us_institutions: Names of at least two non-U.S.-based collaborating institutions on the paper. Extract exactly as written in the answer. If fewer than two are present, extract as many as are present.
- non_us_institutions_urls: URL(s) in the answer that list these institutions as co-author affiliations for the paper. If none cited in the answer, return an empty array.

- discovery_date: The date when 3I/ATLAS was first discovered, as stated in the answer.
- discovery_date_urls: URL(s) in the answer that support the discovery date (e.g., ATLAS/IAU/MPEC/press releases). If none cited in the answer, return an empty array.

- discovery_system: The name of the survey telescope system that discovered 3I/ATLAS (as stated in the answer).
- discovery_system_urls: URL(s) in the answer that support the discovering system (official survey pages, press releases, MPECs). If none cited, return an empty array.

Rules:
- Extract only what is explicitly present in the answer.
- Return null where a single-valued field is missing, and an empty array where URL lists are missing.
- For URLs, include the actual URL strings from the answer (plain URLs or within markdown).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_name(s: Optional[str]) -> str:
    return " ".join((s or "").strip().lower().split())


def _join_names(items: List[str]) -> str:
    clean = [i.strip() for i in items if i and i.strip()]
    return ", ".join(clean)


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: List[str],
    additional_instruction: str
) -> bool:
    """If no URLs are provided, proactively mark the node failed; else run verification."""
    if not urls:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Section builders                                                            #
# --------------------------------------------------------------------------- #
async def add_section_paper_identification(evaluator: Evaluator, root, ext: PaperExtraction):
    # Parent node (critical, parallel)
    section = evaluator.add_parallel(
        id="paper_identification",
        desc="Correctly identify the target paper and its publication metadata.",
        parent=root,
        critical=True
    )

    # 1) Paper identifier existence (arXiv ID or journal citation)
    evaluator.add_custom_node(
        result=bool(ext.paper_identifier and ext.paper_identifier.strip()),
        id="paper_reference",
        desc="Provide the paper's arXiv identifier or a published journal citation.",
        parent=section,
        critical=True
    )

    # 2) URL supports the identifier/citation
    node_ref_url = evaluator.add_leaf(
        id="paper_reference_url",
        desc="Provide a URL that supports the provided arXiv identifier or journal citation.",
        parent=section,
        critical=True
    )
    id_text = ext.paper_identifier or ""
    title_part = f" titled '{ext.paper_title}'" if ext.paper_title else ""
    claim_ref = (
        f"The provided page(s) correspond to the paper{title_part} identified as '{id_text}', "
        f"which is about the interstellar object 3I/ATLAS and is led (first author) by Darryl Seligman."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_ref_url,
        claim_ref,
        ext.paper_reference_urls,
        additional_instruction=(
            "Verify that the URL(s) clearly refer to the same paper (match by arXiv ID or the full journal citation). "
            "Confirm Darryl Seligman is the first/lead author and that the paper concerns 3I/ATLAS. "
            "Accept minor formatting variations or common journal abbreviations."
        )
    )

    # 3) Publication venue existence
    evaluator.add_custom_node(
        result=bool(ext.publication_venue and ext.publication_venue.strip()),
        id="publication_venue",
        desc="Provide the name of the journal where the paper was published or accepted for publication (or clearly state arXiv-only if applicable).",
        parent=section,
        critical=True
    )

    # 4) Publication venue supported by URL(s)
    node_pub_url = evaluator.add_leaf(
        id="publication_venue_url",
        desc="Provide a URL that supports the stated publication/published-or-accepted venue information.",
        parent=section,
        critical=True
    )
    venue_text = ext.publication_venue or ""
    claim_venue = (
        f"The paper{title_part} was published in or accepted for publication in the venue '{venue_text}'. "
        f"If the answer states 'arXiv' or 'arXiv-only', the page should clearly indicate it is only an arXiv preprint."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_pub_url,
        claim_venue,
        ext.publication_venue_urls,
        additional_instruction=(
            "Look for explicit journal name or acceptance notes on arXiv/ADS/journal pages. "
            "If arXiv-only, confirm lack of a journal citation and the presence of 'preprint' indication. "
            "Allow standard abbreviations (e.g., ApJL for Astrophysical Journal Letters)."
        )
    )


async def add_section_lead_author_affiliation(evaluator: Evaluator, root, ext: PaperExtraction):
    section = evaluator.add_parallel(
        id="lead_author_affiliation",
        desc="Provide Darryl Seligman's affiliation information as used for the paper, including any fellowship/grant affiliation at the institution.",
        parent=root,
        critical=True
    )

    # 1) Primary institution existence
    evaluator.add_custom_node(
        result=bool(ext.primary_institution and ext.primary_institution.strip()),
        id="primary_institution_name",
        desc="Provide Darryl Seligman's primary institutional affiliation at the time of publication.",
        parent=section,
        critical=True
    )

    # 2) Primary institution supported by URL(s)
    node_inst_url = evaluator.add_leaf(
        id="primary_institution_url",
        desc="Provide a URL confirming Darryl Seligman's primary institutional affiliation for the paper.",
        parent=section,
        critical=True
    )
    inst_text = ext.primary_institution or ""
    claim_inst = (
        f"For this paper{(' titled ' + ext.paper_title) if ext.paper_title else ''}, "
        f"Darryl Seligman's primary institutional affiliation is '{inst_text}'."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_inst_url,
        claim_inst,
        ext.primary_institution_urls,
        additional_instruction=(
            "Verify the author affiliation listing for Darryl Seligman on the paper/journal/ADS/arXiv page. "
            "Minor variations (e.g., department vs university name, acronym vs full name) should be treated as equivalent."
        )
    )

    # 3) Fellowship/grant existence
    evaluator.add_custom_node(
        result=bool(ext.fellowship_or_grant and ext.fellowship_or_grant.strip()),
        id="fellowship_or_grant_affiliation",
        desc="Provide any fellowship or grant affiliation held by Darryl Seligman at his institution (as stated/acknowledged for the paper).",
        parent=section,
        critical=True
    )

    # 4) Fellowship/grant supported by URL(s)
    node_fellow_url = evaluator.add_leaf(
        id="fellowship_or_grant_url",
        desc="Provide a URL supporting the stated fellowship or grant affiliation.",
        parent=section,
        critical=True
    )
    fellow_text = ext.fellowship_or_grant or ""
    claim_fellow = (
        f"For this paper, Darryl Seligman held the fellowship or grant affiliation: '{fellow_text}'."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_fellow_url,
        claim_fellow,
        ext.fellowship_or_grant_urls,
        additional_instruction=(
            "Look for acknowledgments, author footnotes, or institutional descriptions that explicitly mention the fellowship/grant. "
            "Confirm that it applies to Darryl Seligman."
        )
    )


async def add_section_collaboration_institutions(evaluator: Evaluator, root, ext: PaperExtraction):
    section = evaluator.add_parallel(
        id="collaboration_institutions",
        desc="Document collaborating institutions meeting the U.S. and non-U.S. count requirements (excluding the lead author's institution from the U.S. list).",
        parent=root,
        critical=True
    )

    # Normalize to compare with lead institution when excluding
    lead_norm = _norm_name(ext.primary_institution)

    # 1) U.S. institutions count and names (>=3 and exclude lead institution if present)
    us_list = [u for u in (ext.us_institutions or []) if u and u.strip()]
    us_valid_excluding_lead = [u for u in us_list if _norm_name(u) and _norm_name(u) != lead_norm] if lead_norm else us_list
    evaluator.add_custom_node(
        result=len(us_valid_excluding_lead) >= 3,
        id="us_institutions_count_and_names",
        desc="List at least three U.S.-based research institutions (other than the lead author's institution) that collaborated on the paper.",
        parent=section,
        critical=True
    )

    # 2) URL supporting U.S. institutions
    node_us_url = evaluator.add_leaf(
        id="us_institutions_url",
        desc="Provide a URL supporting the listed U.S.-based collaborating institutions.",
        parent=section,
        critical=True
    )
    claim_us = (
        "The paper lists the following collaborating U.S.-based institutions among the author affiliations: "
        f"{_join_names(us_list)}. Verify that these institutions appear on the provided page(s) as co-author affiliations."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_us_url,
        claim_us,
        ext.us_institutions_urls,
        additional_instruction=(
            "Check the author affiliations section on the paper/journal/ADS/arXiv pages. "
            "Minor name variations and standard acronyms should be accepted (e.g., 'Caltech' vs 'California Institute of Technology')."
        )
    )

    # 3) Non-U.S. institutions count and names (>=2)
    non_us_list = [n for n in (ext.non_us_institutions or []) if n and n.strip()]
    evaluator.add_custom_node(
        result=len(non_us_list) >= 2,
        id="non_us_institutions_count_and_names",
        desc="List at least two non-U.S.-based institutions that collaborated on the paper.",
        parent=section,
        critical=True
    )

    # 4) URL supporting non-U.S. institutions
    node_non_us_url = evaluator.add_leaf(
        id="non_us_institutions_url",
        desc="Provide a URL supporting the listed non-U.S.-based collaborating institutions.",
        parent=section,
        critical=True
    )
    claim_non_us = (
        "The paper lists the following collaborating non-U.S.-based institutions among the author affiliations: "
        f"{_join_names(non_us_list)}. Verify that these institutions appear on the provided page(s) as co-author affiliations."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        node_non_us_url,
        claim_non_us,
        ext.non_us_institutions_urls,
        additional_instruction=(
            "Check the author affiliations section on the paper/journal/ADS/arXiv pages. "
            "Minor name variations and local-language spellings should be accepted as equivalent."
        )
    )


async def add_section_discovery_details(evaluator: Evaluator, root, ext: PaperExtraction):
    section = evaluator.add_parallel(
        id="discovery_details",
        desc="Provide the discovery date and the discovering survey/telescope system for 3I/ATLAS with verification.",
        parent=root,
        critical=True
    )

    # 1) Discovery date existence
    evaluator.add_custom_node(
        result=bool(ext.discovery_date and ext.discovery_date.strip()),
        id="discovery_date",
        desc="Provide the date when 3I/ATLAS was first discovered.",
        parent=section,
        critical=True
    )

    # 2) Discovery date supported by URL(s)
    node_disc_date_url = evaluator.add_leaf(
        id="discovery_date_url",
        desc="Provide a URL supporting the stated discovery date.",
        parent=section,
        critical=True
    )
    date_text = ext.discovery_date or ""
    claim_disc_date = f"The interstellar object 3I/ATLAS was first discovered on {date_text}."
    await _verify_with_urls_or_fail(
        evaluator,
        node_disc_date_url,
        claim_disc_date,
        ext.discovery_date_urls,
        additional_instruction=(
            "Prefer official or authoritative sources (ATLAS site, IAU/MPC/MPEC notices, institutional press releases). "
            "Allow minor timezone or formatting differences for the date, but the day/month/year must match."
        )
    )

    # 3) Discovery system existence
    evaluator.add_custom_node(
        result=bool(ext.discovery_system and ext.discovery_system.strip()),
        id="discovery_system",
        desc="Provide the name of the survey telescope system that discovered 3I/ATLAS.",
        parent=section,
        critical=True
    )

    # 4) Discovery system supported by URL(s)
    node_disc_sys_url = evaluator.add_leaf(
        id="discovery_system_url",
        desc="Provide a URL supporting the stated discovering survey/telescope system.",
        parent=section,
        critical=True
    )
    sys_text = ext.discovery_system or ""
    claim_disc_sys = f"The interstellar object 3I/ATLAS was discovered by the survey/telescope system '{sys_text}'."
    await _verify_with_urls_or_fail(
        evaluator,
        node_disc_sys_url,
        claim_disc_sys,
        ext.discovery_system_urls,
        additional_instruction=(
            "Prefer official source pages (ATLAS site, IAU/MPC/MPEC, institutional pages). "
            "Accept common abbreviations if they clearly refer to the same named system (e.g., 'ATLAS' for 'Asteroid Terrestrial-impact Last Alert System')."
        )
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
    Evaluate an answer for the 'first 3I/ATLAS paper led by Darryl Seligman' task.
    Builds a verification tree that mirrors the rubric, extracts structured info,
    and verifies claims against the provided URLs.
    """
    # Initialize evaluator with a parallel root strategy
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

    # According to rubric, the root is critical; enforce critical consistency
    evaluator.root.critical = True

    # Extraction
    extracted: PaperExtraction = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction"
    )

    # Build rubric sections
    # paper_identification
    await add_section_paper_identification(evaluator, root, extracted)

    # lead_author_affiliation
    await add_section_lead_author_affiliation(evaluator, root, extracted)

    # collaboration_institutions
    await add_section_collaboration_institutions(evaluator, root, extracted)

    # discovery_details
    await add_section_discovery_details(evaluator, root, extracted)

    # Return consolidated summary
    return evaluator.get_summary()