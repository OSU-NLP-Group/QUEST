import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "wos_journal_quality_check"
TASK_DESCRIPTION = """
A postdoctoral researcher in materials science needs to identify an academic journal for publishing their research findings. Their institution's research office has established strict criteria for journals to qualify for their publication record, requiring that the journal must meet all of the following Web of Science quality standards:

1. Be indexed in the Web of Science Core Collection (ESCI, SCIE, SSCI, or AHCI)
2. Have a registered ISSN that is verifiable through the ISSN International Centre database (https://portal.issn.org/)
3. Have a publisher with a clearly stated, verifiable physical address (P.O. Box addresses are not acceptable)
4. Display identifiable Editorial Board Members with their names and institutional affiliations, including country or region
5. Have a readily accessible, clear statement of commitment to peer review for all primary research articles
6. Have transparent ethical requirements for authors, with proper attribution to recognized standards (such as COPE, WAME, or Declaration of Helsinki) or functioning links to such guidelines

Identify one specific journal that meets all of these requirements. For each criterion, provide the journal's name and verifiable URL references demonstrating compliance with the stated requirements.
"""


# -----------------------------------------------------------------------------
# Pydantic data models for extraction
# -----------------------------------------------------------------------------
class JournalExtraction(BaseModel):
    """
    Structured extraction of the journal and all relevant verification URLs
    explicitly cited in the agent's answer.
    """
    journal_name: Optional[str] = None

    # 1) Web of Science (WoS) indexing references (e.g., Clarivate Master Journal List, WoS profile)
    wos_indexing_urls: List[str] = Field(default_factory=list)

    # 2) ISSN and verification sources
    issn_value: Optional[str] = None
    issn_portal_url: Optional[str] = None  # should be a portal.issn.org URL
    issn_display_page_urls: List[str] = Field(default_factory=list)  # journal/about pages showing ISSN

    # 3) Publisher identification and physical address references (not P.O. Box)
    publisher_name: Optional[str] = None
    publisher_address_text: Optional[str] = None
    publisher_address_urls: List[str] = Field(default_factory=list)

    # 4) Editorial board page(s)
    editorial_board_urls: List[str] = Field(default_factory=list)

    # 5) Peer review policy page(s)
    peer_review_policy_urls: List[str] = Field(default_factory=list)

    # 6) Ethics policy page(s)
    ethics_policy_urls: List[str] = Field(default_factory=list)

    # 7) English abstract availability references (policy and/or sample articles)
    english_abstracts_policy_urls: List[str] = Field(default_factory=list)
    sample_article_urls: List[str] = Field(default_factory=list)

    # 8) Publication frequency statement references
    publication_frequency_text: Optional[str] = None
    publication_frequency_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_journal_sources() -> str:
    return """
Extract exactly one journal and all verification URLs explicitly provided in the answer.

Return a JSON object with these fields:
- journal_name: string | null
- wos_indexing_urls: string[]  // URLs that directly confirm Web of Science Core Collection indexing (e.g., Clarivate Master Journal List, Web of Science profile). Include only URLs that the answer actually lists.
- issn_value: string | null    // The ISSN (print or online) mentioned in the answer, as written (include hyphen if present).
- issn_portal_url: string | null // The ISSN record URL on https://portal.issn.org/ mentioned in the answer, or null if none is present.
- issn_display_page_urls: string[] // Journal website page(s) where the ISSN is shown, if provided in the answer.
- publisher_name: string | null
- publisher_address_text: string | null // If the answer quotes a postal address, include it verbatim; otherwise null.
- publisher_address_urls: string[] // URL(s) where a physical street address is shown (avoid contact forms only), as provided in the answer.
- editorial_board_urls: string[] // URL(s) to editorial board page(s) listed in the answer.
- peer_review_policy_urls: string[] // URL(s) to peer review policy statements listed in the answer.
- ethics_policy_urls: string[] // URL(s) to ethics/publication ethics statements listed in the answer.
- english_abstracts_policy_urls: string[] // URL(s) to policy pages stating abstracts are in English, as provided.
- sample_article_urls: string[] // URL(s) to specific articles referenced in the answer that demonstrate English abstracts.
- publication_frequency_text: string | null // Frequency description quoted in the answer (e.g., "monthly", "continuous").
- publication_frequency_urls: string[] // URL(s) to frequency/schedule pages listed in the answer.

Strict rules:
- Extract only URLs explicitly present in the answer. Do not invent or infer.
- For issn_portal_url, include only a URL on https://portal.issn.org if it is explicitly cited.
- If a field is missing in the answer, set it to null (for single values) or [] (for arrays).
- Preserve the exact text for journal_name, issn_value, publisher_name, and publication_frequency_text as written in the answer.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _jn(name: Optional[str]) -> str:
    return name.strip() if name else "the journal"

def _combine_sources(*args: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in args:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in out:
                    out.append(uu)
    return out

def _maybe_list(s: Optional[str]) -> List[str]:
    return [s] if s else []


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def add_wos_indexing_verification(evaluator: Evaluator,
                                        parent,
                                        info: JournalExtraction) -> None:
    """
    Build the "WoS_Indexing_Verified" sequential subtree:
      1) WoS_Reference_Provided (leaf)
      2) Technical_Requirements_Met (parallel, with multiple subgroups)
    """
    # Level 1 under Root: "WoS_Indexing_Verified" (sequential, critical)
    wos_seq = evaluator.add_sequential(
        id="WoS_Indexing_Verified",
        desc="Confirm the journal is indexed in Web of Science Core Collection with verifiable reference",
        parent=parent,
        critical=True,
    )

    # Child 1: WoS reference verification (leaf, critical)
    wos_leaf = evaluator.add_leaf(
        id="WoS_Reference_Provided",
        desc="Provide URL or reference confirming Web of Science indexing status",
        parent=wos_seq,
        critical=True,
    )

    claim = (
        f"This page confirms that {_jn(info.journal_name)} is indexed in the "
        "Web of Science Core Collection (ESCI, SCIE, SSCI, or AHCI)."
    )
    await evaluator.verify(
        claim=claim,
        node=wos_leaf,
        sources=info.wos_indexing_urls,
        additional_instruction=(
            "Accept authoritative Clarivate sources such as mjl.clarivate.com (Master Journal List) or "
            "official Web of Science profile pages showing Core Collection coverage (e.g., SCIE/SSCI/AHCI/ESCI). "
            "Look for explicit labels like 'Web of Science Core Collection' and target index names (SCIE, SSCI, AHCI, ESCI). "
            "If the page does not clearly state Core Collection coverage, mark as not supported."
        ),
    )

    # Child 2 (created now, children verified later): Technical requirements met (parallel, critical)
    technical_parallel = evaluator.add_parallel(
        id="Technical_Requirements_Met",
        desc="Verify the journal meets all technical identification and publisher requirements",
        parent=wos_seq,
        critical=True,
    )

    # Build three parallel groups under Technical_Requirements_Met
    await add_issn_publisher_group(evaluator, technical_parallel, info)
    await add_editorial_policy_group(evaluator, technical_parallel, info)
    await add_content_publication_group(evaluator, technical_parallel, info)


async def add_issn_publisher_group(evaluator: Evaluator, parent, info: JournalExtraction) -> None:
    """
    Group: ISSN_Publisher_Requirements (parallel, critical)
      - ISSN_Requirements (sequential, critical)
          - ISSN_Reference_URL (leaf, critical)
      - Publisher_Address_Verified (leaf, critical)
    """
    group = evaluator.add_parallel(
        id="ISSN_Publisher_Requirements",
        desc="Verify ISSN registration and publisher identification requirements",
        parent=parent,
        critical=True,
    )

    # ISSN requirements subtree
    issn_seq = evaluator.add_sequential(
        id="ISSN_Requirements",
        desc="Confirm ISSN is registered on portal.issn.org and clearly displayed with supporting reference",
        parent=group,
        critical=True,
    )

    # Leaf: ISSN reference on portal.issn.org (and optionally journal page showing ISSN)
    issn_leaf = evaluator.add_leaf(
        id="ISSN_Reference_URL",
        desc="Provide URL reference for ISSN verification",
        parent=issn_seq,
        critical=True,
    )
    issn_sources = _combine_sources(_maybe_list(info.issn_portal_url), info.issn_display_page_urls)

    issn_part = f" with ISSN '{info.issn_value}'" if info.issn_value else ""
    claim = (
        f"This page shows a registered ISSN record on portal.issn.org for {_jn(info.journal_name)}{issn_part}, "
        "and/or the journal website displays this ISSN."
    )
    await evaluator.verify(
        claim=claim,
        node=issn_leaf,
        sources=issn_sources,
        additional_instruction=(
            "At least one source should be a valid portal.issn.org record that matches the journal title (or former titles) "
            "and indicates a registered ISSN (e.g., ISSN or eISSN). "
            "If multiple ISSNs exist (print/electronic), accept either. "
            "If no portal.issn.org record is present, mark as not supported."
        ),
    )

    # Leaf: Publisher physical address verification (not P.O. Box)
    publisher_addr_leaf = evaluator.add_leaf(
        id="Publisher_Address_Verified",
        desc="Confirm publisher has verifiable physical address (not P.O. Box)",
        parent=group,
        critical=True,
    )

    claim = (
        f"This page provides a clear, verifiable physical street address (not a P.O. Box) for the publisher of {_jn(info.journal_name)}."
    )
    await evaluator.verify(
        claim=claim,
        node=publisher_addr_leaf,
        sources=info.publisher_address_urls,
        additional_instruction=(
            "Look for a street address with number/street/city/region/country details. "
            "If the address is only a P.O. Box or purely a contact form without a physical location, "
            "mark as not supported."
        ),
    )


async def add_editorial_policy_group(evaluator: Evaluator, parent, info: JournalExtraction) -> None:
    """
    Group: Editorial_Policy_Requirements (parallel, critical)
      - Editorial_Board_Standards (leaf, critical)
      - Peer_Review_Policy (leaf, critical)
      - Ethics_Standards (leaf, critical)
    """
    group = evaluator.add_parallel(
        id="Editorial_Policy_Requirements",
        desc="Verify editorial board composition and journal policies meet Web of Science standards",
        parent=parent,
        critical=True,
    )

    # Create leaves
    editorial_board_leaf = evaluator.add_leaf(
        id="Editorial_Board_Standards",
        desc="Confirm editorial board members are identifiable with institutional affiliations including country/region",
        parent=group,
        critical=True,
    )
    peer_review_leaf = evaluator.add_leaf(
        id="Peer_Review_Policy",
        desc="Verify journal has clear, accessible peer review policy statement",
        parent=group,
        critical=True,
    )
    ethics_leaf = evaluator.add_leaf(
        id="Ethics_Standards",
        desc="Confirm journal has transparent ethics requirements (COPE, WAME, etc.) with proper attribution",
        parent=group,
        critical=True,
    )

    # Prepare claims and run in parallel
    tasks = []

    claim_board = (
        "This page lists identifiable editorial board members with their names and institutional affiliations, "
        "including country or region."
    )
    tasks.append((claim_board, info.editorial_board_urls, editorial_board_leaf,
                  "Names and affiliations should be visible; country/region should be present explicitly or as part of the affiliation."))

    claim_peer = (
        "This page provides a clear statement that all primary research articles in the journal are subject to peer review "
        "(e.g., single-blind, double-blind, or open peer review)."
    )
    tasks.append((claim_peer, info.peer_review_policy_urls, peer_review_leaf,
                  "Look for phrases like 'peer reviewed', 'double-blind peer review', 'all research articles undergo peer review', etc."))

    claim_ethics = (
        "This page describes transparent ethical requirements for authors and references recognized standards "
        "such as COPE, WAME, ICMJE, or the Declaration of Helsinki, or includes working links to such guidelines."
    )
    tasks.append((claim_ethics, info.ethics_policy_urls, ethics_leaf,
                  "Accept explicit references or functioning links to COPE/WAME/ICMJE/Declaration of Helsinki (or equivalent recognized bodies)."))

    await evaluator.batch_verify(tasks)


async def add_content_publication_group(evaluator: Evaluator, parent, info: JournalExtraction) -> None:
    """
    Group: Content_Publication_Requirements (parallel, critical)
      - English_Abstracts_Available (leaf, critical)
      - Publication_Frequency_Stated (leaf, critical)
    """
    group = evaluator.add_parallel(
        id="Content_Publication_Requirements",
        desc="Verify article content and publication schedule requirements",
        parent=parent,
        critical=True,
    )

    english_abs_leaf = evaluator.add_leaf(
        id="English_Abstracts_Available",
        desc="Confirm all scholarly articles have abstracts available in English or translated to English",
        parent=group,
        critical=True,
    )
    pub_freq_leaf = evaluator.add_leaf(
        id="Publication_Frequency_Stated",
        desc="Verify the journal states and adheres to a publication frequency or schedule",
        parent=group,
        critical=True,
    )

    # Run in parallel
    tasks = []

    english_sources = _combine_sources(info.english_abstracts_policy_urls, info.sample_article_urls)
    claim_english = (
        "This page states or demonstrates that all scholarly research articles include abstracts available in English "
        "(either originally in English or with an English translation)."
    )
    tasks.append((claim_english, english_sources, english_abs_leaf,
                  "Policy pages stating English abstracts, or multiple sample articles consistently showing English abstracts, are acceptable evidence."))

    claim_freq = (
        "This page states the journal's publication frequency or schedule (e.g., monthly, quarterly, continuous/rolling publication), "
        "and indicates it is the current policy."
    )
    tasks.append((claim_freq, info.publication_frequency_urls, pub_freq_leaf,
                  "Accept frequency statements such as 'monthly', 'quarterly', 'bi-monthly', or 'continuous'/'rolling'. "
                  "If no explicit schedule is stated, mark as not supported."))

    await evaluator.batch_verify(tasks)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Web of Science journal quality criteria task.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extracted: JournalExtraction = await evaluator.extract(
        prompt=prompt_extract_journal_sources(),
        template_class=JournalExtraction,
        extraction_name="journal_extraction",
    )

    # Build the rubric tree according to the JSON: Use a critical sequential "Root" under the framework root
    root_task = evaluator.add_sequential(
        id="Root",
        desc="Evaluate whether the identified journal satisfies all Web of Science Core Collection quality criteria",
        parent=root,
        critical=True,
    )

    # WoS indexing subtree with all technical, editorial, and content checks beneath
    await add_wos_indexing_verification(evaluator, root_task, extracted)

    # Return the final structured evaluation summary
    return evaluator.get_summary()