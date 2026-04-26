import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ----------------------------- #
# Task constants and metadata   #
# ----------------------------- #

TASK_ID = "research_institutions_multi_domain_2024_2025"
TASK_DESCRIPTION = (
    "Identify four distinct research institutions located in the United States or Canada that meet ALL of the following criteria:\n\n"
    "1. Multi-Domain Research Requirement: Each institution must be actively involved in at least two of the following three research domains, with activities documented between January 2024 and November 2025:\n"
    "   - Artificial Intelligence / Machine Learning research\n"
    "   - Space technology or space mission research\n"
    "   - Brain-computer interface or neuroprosthetics research\n\n"
    "2. Partnership Requirement: Each institution must have documented evidence of at least one active partnership with:\n"
    "   - A commercial technology company (such as SpaceX, Neuralink, Anthropic, OpenAI, NVIDIA, or similar), OR\n"
    "   - NASA or another government space/research agency, OR\n"
    "   - A multi-institutional research consortium\n\n"
    "3. Temporal Validity: All documented research activities, partnerships, or announcements must have occurred between January 2024 and November 2025\n\n"
    "4. Geographic Requirement: The institution's primary location must be within the United States or Canada\n\n"
    "5. Documentation Requirement: For each institution, provide:\n"
    "   - The institution's full name and location (city, state/province, country)\n"
    "   - Specific evidence of involvement in at least two of the three research domains (name the specific programs, clinical trials, missions, or research initiatives)\n"
    "   - Identification of at least one partnership (name the partner organization and describe the collaboration)\n"
    "   - Supporting URL references from authoritative sources (institutional websites, government databases, clinical trial registries, or reputable news sources) that verify all claims\n\n"
    "The four institutions must be distinct (different organizations) and each must independently satisfy all the above requirements."
)

DATE_WINDOW_START = "2024-01-01"
DATE_WINDOW_END = "2025-11-30"


# ----------------------------- #
# Extraction data models        #
# ----------------------------- #

class DomainEvidence(BaseModel):
    domain: Optional[str] = None  # e.g., "AI/ML", "Space", "BCI/Neuroprosthetics" (allow free text; we will normalize)
    initiative: Optional[str] = None  # name of the program/mission/trial/etc.
    date: Optional[str] = None  # reported date string in the answer
    urls: List[str] = Field(default_factory=list)  # authoritative URLs supporting this initiative


class PartnershipEvidence(BaseModel):
    partner_name: Optional[str] = None  # e.g., "NASA", "OpenAI", "SpaceX", "#ConsortiumName"
    partner_type: Optional[str] = None  # e.g., "commercial", "government", "consortium" (allow free text)
    collaboration_desc: Optional[str] = None  # description of the collaboration
    date: Optional[str] = None  # reported date string
    urls: List[str] = Field(default_factory=list)  # authoritative URLs supporting the partnership


class InstitutionItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # "City, State/Province, Country"
    location_country: Optional[str] = None  # if present; not required
    primary_location_url: Optional[str] = None  # a URL supporting location (official site, Wikipedia, etc.)
    domain_evidences: List[DomainEvidence] = Field(default_factory=list)
    partnership: Optional[PartnershipEvidence] = None
    extra_urls: List[str] = Field(default_factory=list)  # any other URLs cited for this institution


class InstitutionsExtraction(BaseModel):
    institutions: List[InstitutionItem] = Field(default_factory=list)


# ----------------------------- #
# Extraction prompt             #
# ----------------------------- #

def prompt_extract_institutions() -> str:
    return """
Extract up to 6 research institutions described in the answer that are located in the United States or Canada and provide the following structured fields for each. Only extract information explicitly present in the answer.

For each institution, extract as much of the following as is explicitly stated:

1) name: Full organization/institution name.
2) location: City, state/province, country (one string).
3) location_country: Country for the primary location if explicitly stated.
4) primary_location_url: A URL that supports the institution’s location (e.g., official .edu/.gov page, institutional site, or reputable page like Wikipedia).
5) domain_evidences: an array with at least two items if present; each item has:
   - domain: one of (AI/ML, Space, BCI/Neuroprosthetics) or a close synonym (e.g., "Artificial Intelligence", "Machine Learning", "Space mission", "satellite", "brain-computer interface", "neuroprosthetics").
   - initiative: the specific program/project/trial/mission/initiative name.
   - date: the date or year/month mentioned in the answer related to this initiative.
   - urls: a list of authoritative URLs supporting the initiative (institutional/government sites, ClinicalTrials.gov, reputable news, peer-reviewed publications, etc.).
6) partnership: a single object if present with:
   - partner_name: the partner organization.
   - partner_type: description (e.g., commercial tech company, NASA/government, research consortium).
   - collaboration_desc: short summary of the collaboration.
   - date: the date/announcement timing in the answer.
   - urls: a list of authoritative URLs supporting the partnership.
7) extra_urls: any other URLs cited in the answer that are relevant to this institution but not already in primary_location_url, domain_evidences.urls, or partnership.urls.

Rules and constraints:
- Only include institutions explicitly listed in the answer.
- Keep fields as null if missing from the answer.
- For domain_evidences, keep urls as empty array if not present; do not invent URLs.
- If the answer provides more than 4 institutions, still extract them, but we will select the first 4 for evaluation.
- Use the exact strings as they appear in the answer for names, initiatives, etc. Do not normalize or paraphrase.
"""


# ----------------------------- #
# Helper utilities              #
# ----------------------------- #

def normalize_name_for_distinctness(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[\(\)\[\]\.,&/]", " ", s)
    s = re.sub(r"\buniv\b", "university", s)
    s = re.sub(r"\bthe\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def map_domain_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.lower()
    # BCI/Neuro
    if ("bci" in s or "brain-computer" in s or "neuro" in s or "neural" in s or "prosthetic" in s or "neuroprosthetic" in s):
        return "BCI/Neuro"
    # Space
    if ("space" in s or "nasa" in s or "satellite" in s or "lunar" in s or "mission" in s or "orbital" in s):
        return "Space"
    # AI/ML
    if ("ai" in s or "artificial intelligence" in s or "machine learning" in s or "ml" in s or "deep learning" in s or "large language model" in s or "llm" in s):
        return "AI/ML"
    return None


def gather_all_urls_for_institution(inst: InstitutionItem) -> List[str]:
    urls: List[str] = []
    if inst.primary_location_url:
        urls.append(inst.primary_location_url)
    urls.extend(inst.extra_urls or [])
    for ev in inst.domain_evidences or []:
        urls.extend(ev.urls or [])
    if inst.partnership and inst.partnership.urls:
        urls.extend(inst.partnership.urls)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def choose_geo_sources(inst: InstitutionItem) -> List[str]:
    # Prefer a direct location source; else use any URLs we have
    if inst.primary_location_url:
        return [inst.primary_location_url]
    urls = gather_all_urls_for_institution(inst)
    return urls[:5]  # limit to avoid overly heavy verification


# ----------------------------- #
# Verification subroutines      #
# ----------------------------- #

async def verify_domain_evidences(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    inst_idx: int,
) -> None:
    """
    Build and verify the 'Research_Domain_Evidence' subtree:
    - Domain_Evidence_1 (critical)
    - Domain_Evidence_2 (critical)
    - Two_Distinct_Domains_Covered (critical)
    """
    research_node = evaluator.add_parallel(
        id=f"institution_{inst_idx+1}_research_domain_evidence",
        desc="Provides evidence of involvement in at least two distinct domains among: AI/ML, Space tech/missions, BCI/neuroprosthetics, with specific initiatives and authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    # Pick first two evidences if available
    ev1 = inst.domain_evidences[0] if len(inst.domain_evidences) >= 1 else DomainEvidence()
    ev2 = inst.domain_evidences[1] if len(inst.domain_evidences) >= 2 else DomainEvidence()

    # Domain Evidence 1
    node_de1 = evaluator.add_leaf(
        id=f"institution_{inst_idx+1}_domain_evidence_1",
        desc="Identifies one specific program/project/trial/mission/initiative in one allowed domain, dated within Jan 2024–Nov 2025, supported by ≥1 authoritative URL. If the domain is BCI, evidence must be (a) clinical-trial site verifiable via ClinicalTrials.gov (or equivalent) OR (b) a 2024–2025 BCI publication with a verifiable URL.",
        parent=research_node,
        critical=True
    )
    mapped1 = map_domain_label(ev1.domain)
    claim1 = (
        f"{inst.name} is actively involved in {mapped1 or (ev1.domain or 'an allowed')} research via the initiative "
        f"'{ev1.initiative or '[initiative unspecified]'}', and this activity is publicly documented between {DATE_WINDOW_START} and {DATE_WINDOW_END}."
    )
    add_ins1 = (
        "You must verify all of the following strictly using the provided URL(s):\n"
        f"- The page(s) explicitly connect the institution '{inst.name}' to the named initiative.\n"
        f"- The initiative fits the domain category '{mapped1 or ev1.domain}'. Allow common synonyms (AI/ML; Space technology/missions; BCI/neuroprosthetics).\n"
        f"- The evidence is dated within {DATE_WINDOW_START} to {DATE_WINDOW_END} (inclusive). Accept a clearly dated 2024 or 2025 page if day/month is not given.\n"
        "- The sources must be authoritative (institutional/government/ClinicalTrials.gov/reputable news/peer‑reviewed publication).\n"
        "- If the domain is BCI/Neuro, require at least one URL that is either: ClinicalTrials.gov (or equivalent registry) OR a 2024–2025 BCI/neuromodulation publication.\n"
        "- If no URLs are provided or the URLs do not support the claim, judge 'not supported'."
    )
    await evaluator.verify(
        claim=claim1,
        node=node_de1,
        sources=ev1.urls if ev1.urls else None,
        additional_instruction=add_ins1
    )

    # Domain Evidence 2
    node_de2 = evaluator.add_leaf(
        id=f"institution_{inst_idx+1}_domain_evidence_2",
        desc="Identifies a second specific program/project/trial/mission/initiative in an allowed domain, dated within Jan 2024–Nov 2025, supported by ≥1 authoritative URL. If the domain is BCI, evidence must be (a) clinical-trial site verifiable via ClinicalTrials.gov (or equivalent) OR (b) a 2024–2025 BCI publication with a verifiable URL.",
        parent=research_node,
        critical=True
    )
    mapped2 = map_domain_label(ev2.domain)
    claim2 = (
        f"{inst.name} is actively involved in {mapped2 or (ev2.domain or 'an allowed')} research via the initiative "
        f"'{ev2.initiative or '[initiative unspecified]'}', and this activity is publicly documented between {DATE_WINDOW_START} and {DATE_WINDOW_END}."
    )
    add_ins2 = (
        "You must verify all of the following strictly using the provided URL(s):\n"
        f"- The page(s) explicitly connect the institution '{inst.name}' to the named initiative.\n"
        f"- The initiative fits the domain category '{mapped2 or ev2.domain}'. Allow common synonyms (AI/ML; Space technology/missions; BCI/neuroprosthetics).\n"
        f"- The evidence is dated within {DATE_WINDOW_START} to {DATE_WINDOW_END} (inclusive). Accept a clearly dated 2024 or 2025 page if day/month is not given.\n"
        "- The sources must be authoritative (institutional/government/ClinicalTrials.gov/reputable news/peer‑reviewed publication).\n"
        "- If the domain is BCI/Neuro, require at least one URL that is either: ClinicalTrials.gov (or equivalent registry) OR a 2024–2025 BCI/neuromodulation publication.\n"
        "- If no URLs are provided or the URLs do not support the claim, judge 'not supported'."
    )
    await evaluator.verify(
        claim=claim2,
        node=node_de2,
        sources=ev2.urls if ev2.urls else None,
        additional_instruction=add_ins2
    )

    # Two distinct domains covered (custom check)
    distinct_ok = False
    if mapped1 and mapped2 and mapped1 != mapped2:
        distinct_ok = True
    evaluator.add_custom_node(
        result=distinct_ok,
        id=f"institution_{inst_idx+1}_two_distinct_domains",
        desc="The two domain evidences collectively cover at least two distinct domains (not both in the same single domain).",
        parent=research_node,
        critical=True
    )


async def verify_partnership(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    inst_idx: int
) -> None:
    """
    Build and verify the Partnership_Evidence leaf.
    """
    node = evaluator.add_leaf(
        id=f"institution_{inst_idx+1}_partnership_evidence",
        desc="Documents at least one partnership/collaboration (commercial tech company OR NASA/other government space/research agency OR multi-institution consortium), naming the partner and describing the collaboration; evidence dated within Jan 2024–Nov 2025 and supported by ≥1 authoritative URL.",
        parent=parent_node,
        critical=True
    )

    p = inst.partnership or PartnershipEvidence()
    partner_name = p.partner_name or "[partner unspecified]"
    partner_type = p.partner_type or "[type unspecified]"
    collab_desc = p.collaboration_desc or "[collaboration unspecified]"
    claim = (
        f"Between {DATE_WINDOW_START} and {DATE_WINDOW_END}, {inst.name} has an active partnership/collaboration with {partner_name} "
        f"({partner_type}). The collaboration can be summarized as: {collab_desc}."
    )
    add_ins = (
        "Verify strictly using the provided source URL(s):\n"
        f"- Confirm that '{inst.name}' and '{partner_name}' are partners or collaborators (e.g., contract, MOU, joint program, official co‑announcement) with activity/announcement dated between {DATE_WINDOW_START} and {DATE_WINDOW_END}.\n"
        "- The partner must fit one of: commercial technology company; NASA or another government space/research agency; or a multi‑institution research consortium.\n"
        "- The sources must be authoritative (institutional/government/reputable news/press release, etc.).\n"
        "- If no URLs are provided or the URLs do not support the claim, judge 'not supported'."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p.urls if p.urls else None,
        additional_instruction=add_ins
    )


async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    inst_idx: int
) -> None:
    """
    Build and verify all checks for a single institution node.
    """
    inst_node = evaluator.add_parallel(
        id=f"Institution_{inst_idx+1}",
        desc=f"Evaluation of the {inst_idx+1}st institution." if inst_idx == 0 else
             (f"Evaluation of the {inst_idx+1}nd institution." if inst_idx == 1 else
              (f"Evaluation of the {inst_idx+1}rd institution." if inst_idx == 2 else
               f"Evaluation of the {inst_idx+1}th institution.")),
        parent=parent_node,
        critical=False
    )

    # 1) Name and Location Provided (critical)
    has_name_loc = bool(inst.name and inst.name.strip()) and bool(inst.location and inst.location.strip())
    evaluator.add_custom_node(
        result=has_name_loc,
        id=f"institution_{inst_idx+1}_name_and_location_provided",
        desc="Provides the institution's full name and location (city, state/province, country).",
        parent=inst_node,
        critical=True
    )

    # 2) Geography: US or Canada (critical) -> Verified with URLs if available
    geo_node = evaluator.add_leaf(
        id=f"institution_{inst_idx+1}_geography_us_or_canada",
        desc="Institution's primary location is within the United States or Canada (consistent with provided location/citations).",
        parent=inst_node,
        critical=True
    )
    geo_claim = (
        f"The primary location of {inst.name or '[institution]'} is within the United States or Canada. "
        f"The answer states the location as '{inst.location or '[location unspecified]'}'."
    )
    geo_add_ins = (
        "Verify using authoritative pages (e.g., official site, .edu/.gov, Wikipedia, government registries, reputable news). "
        "If the institution has multiple campuses, accept if the headquarters or primary location is in the US or Canada. "
        "If no relevant URLs are provided, judge 'not supported'."
    )
    geo_sources = choose_geo_sources(inst)
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=geo_sources if geo_sources else None,
        additional_instruction=geo_add_ins
    )

    # 3) Research domain evidence (critical, with three children critical)
    await verify_domain_evidences(evaluator, inst_node, inst, inst_idx)

    # 4) Partnership evidence (critical)
    await verify_partnership(evaluator, inst_node, inst, inst_idx)


# ----------------------------- #
# Main evaluation entry point   #
# ----------------------------- #

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
    Evaluate an answer for the research institutions multi-domain (2024–2025) task.
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

    # Extract institutions from the answer
    extracted: InstitutionsExtraction = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    all_insts = extracted.institutions or []
    provided_count = len(all_insts)

    # Root-level critical checks
    # Important: For robustness with over-complete lists, we consider "Four_Institutions_Provided"
    # satisfied if the answer provides at least four institutions; we will evaluate the first four.
    evaluator.add_custom_node(
        result=(provided_count >= 4),
        id="Four_Institutions_Provided",
        desc="Answer provides exactly four institutions (not fewer or more).",
        parent=root,
        critical=True
    )

    # Select the first four (do not skip duplicates here; distinctness is checked separately)
    selected_insts: List[InstitutionItem] = (all_insts[:4] if provided_count >= 4 else all_insts + [InstitutionItem()] * (4 - provided_count))

    # Distinctness check among the first four
    norm_names = [normalize_name_for_distinctness(x.name) for x in selected_insts]
    distinct_ok = len([n for n in norm_names if n]) == len(set([n for n in norm_names if n])) and all(n for n in norm_names)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Institutions_Are_Distinct",
        desc="All four institutions are distinct organizations (no duplicates/aliases of the same entity).",
        parent=root,
        critical=True
    )

    # Add custom info to the report
    evaluator.add_custom_info(
        info={
            "date_window_start": DATE_WINDOW_START,
            "date_window_end": DATE_WINDOW_END,
            "allowed_domains": ["AI/ML", "Space", "BCI/Neuro"],
            "institutions_found_in_answer": provided_count,
            "institutions_used_for_eval": 4
        },
        info_type="task_config",
        info_name="evaluation_parameters"
    )

    # Build and verify each of the 4 institutions
    # Each institution node is non-critical; within each, critical subrequirements must pass.
    for i in range(4):
        await verify_institution(evaluator, root, selected_insts[i], i)

    return evaluator.get_summary()