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
TASK_ID = "early_career_ai_nsf_careers"
TASK_DESCRIPTION = (
    "Identify four early-career computer science researchers currently employed at US universities who meet the following criteria:\n\n"
    "1. NSF CAREER Eligibility: Ph.D. in CS/closely related field; tenure-track or equivalent faculty at a US NSF-eligible institution; early-career (≈ within 5 years of first academic appointment).\n"
    "2. Research Specialization: Primary focus in AI/ML or closely related subfield.\n"
    "3. Publication Record: ≥3 papers in the last 3 years (2023–2025) and ≥1 publication at a top-tier CS venue (NeurIPS, ICML, CVPR, ICCV, ACL, EMNLP, AAAI, IJCAI) or a prestigious journal.\n"
    "4. Research Impact: h-index ≥ 5 or ≥ 100 total citations.\n"
    "5. Verifiability: Has an institutional faculty webpage usable to verify affiliation, position, and research area.\n\n"
    "For each researcher, provide: Full name; current institutional affiliation and department; brief research focus; link to institutional faculty webpage; link to publication profile (Scholar/DBLP/etc.); summary of publication record (recent pubs and example top-tier venue); h-index and/or total citations."
)

CURRENT_YEAR = 2026
LAST_3_YEARS = [2023, 2024, 2025]
TOP_TIER_VENUES = [
    "NeurIPS", "ICML", "CVPR", "ICCV", "ACL", "EMNLP", "AAAI", "IJCAI"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResearcherEntry(BaseModel):
    """A single researcher entry extracted from the answer."""
    full_name: Optional[str] = None
    affiliation_department: Optional[str] = None  # combined string
    research_focus: Optional[str] = None

    institutional_webpage: Optional[str] = None
    publication_profile: Optional[str] = None

    # Publication record summary
    publication_record_summary: Optional[str] = None
    recent_publications_count: Optional[str] = None
    top_tier_example_venue: Optional[str] = None

    # Impact metrics
    h_index: Optional[str] = None
    total_citations: Optional[str] = None

    # Eligibility details (supporting verification)
    phd_field: Optional[str] = None
    phd_institution: Optional[str] = None
    position_title: Optional[str] = None
    first_appointment_year: Optional[str] = None

    # Any other URLs provided (e.g., CV page, lab page, DBLP in addition to publication_profile)
    additional_sources: List[str] = Field(default_factory=list)


class ResearchersExtraction(BaseModel):
    """Top-level extraction of researcher entries."""
    researchers: List[ResearcherEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return """
    Extract up to four researcher entries as presented in the answer. Return a JSON object:
    {
      "researchers": [
        {
          "full_name": str or null,
          "affiliation_department": str or null,   // combined, e.g., "University of X, Department of Y"
          "research_focus": str or null,           // brief description of AI/ML-related focus
          "institutional_webpage": url or null,    // faculty profile page at the institution
          "publication_profile": url or null,      // Google Scholar, DBLP, or institutional research publications page
          "publication_record_summary": str or null,
          "recent_publications_count": str or null, // a string like "3", "3+", "at least 3"
          "top_tier_example_venue": str or null,   // e.g., "NeurIPS", "ICML", or a prestigious journal name
          "h_index": str or null,
          "total_citations": str or null,
          "phd_field": str or null,                // e.g., "Computer Science" or closely related
          "phd_institution": str or null,
          "position_title": str or null,           // e.g., "Assistant Professor"
          "first_appointment_year": str or null,   // e.g., "2022" if explicitly stated
          "additional_sources": [url, ...]         // any other URLs provided (CV, lab page, DBLP, etc.)
        },
        ... up to 4 entries ...
      ]
    }

    Rules:
    - Extract exactly what is explicitly present in the answer. Do not invent.
    - If some field is missing for an entry, use null for that field.
    - For URLs, include the full URL (prepend http:// if protocol missing).
    - If the answer includes more than four entries, return only the first four.
    - If fewer than four entries are present, return as many as available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return f"{n}{'tsnrhtdd'[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]}"  # 1st, 2nd, 3rd, 4th


def collect_inst_sources(entry: ResearcherEntry) -> List[str]:
    """Sources for institutional/affiliation-related checks."""
    urls = []
    if entry.institutional_webpage:
        urls.append(entry.institutional_webpage)
    urls.extend([u for u in entry.additional_sources if u])
    # Deduplicate, preserve order
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_pub_sources(entry: ResearcherEntry) -> List[str]:
    """Sources for publication/impact-related checks."""
    urls = []
    if entry.publication_profile:
        urls.append(entry.publication_profile)
    urls.extend([u for u in entry.additional_sources if u])
    # Deduplicate, preserve order
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification sub-tree for one researcher                                    #
# --------------------------------------------------------------------------- #
async def verify_researcher_entry(
    evaluator: Evaluator,
    parent_node,
    entry: ResearcherEntry,
    idx: int,
) -> None:
    """
    Build the verification tree for a single researcher and run necessary checks.
    """
    ord_str = ordinal(idx + 1)
    rnode = evaluator.add_parallel(
        id=f"researcher_{idx + 1}",
        desc=f"{ord_str} researcher entry satisfies constraints and includes requested fields.",
        parent=parent_node,
        critical=False,
    )

    # 1) Requested fields (critical group)
    req_fields = evaluator.add_parallel(
        id=f"researcher_{idx + 1}_requested_fields",
        desc="Provides all requested fields for this researcher.",
        parent=rnode,
        critical=True,
    )

    # 1.a Full name
    evaluator.add_custom_node(
        result=bool(entry.full_name and entry.full_name.strip()),
        id=f"researcher_{idx + 1}_full_name",
        desc="Provides full name.",
        parent=req_fields,
        critical=True,
    )

    # 1.b Affiliation + Department (combined string)
    evaluator.add_custom_node(
        result=bool(entry.affiliation_department and entry.affiliation_department.strip()),
        id=f"researcher_{idx + 1}_affiliation_department",
        desc="Provides current institutional affiliation and department.",
        parent=req_fields,
        critical=True,
    )

    # 1.c Research focus
    evaluator.add_custom_node(
        result=bool(entry.research_focus and entry.research_focus.strip()),
        id=f"researcher_{idx + 1}_research_focus_description",
        desc="Provides a brief description of research focus.",
        parent=req_fields,
        critical=True,
    )

    # 1.d Institutional webpage link
    evaluator.add_custom_node(
        result=bool(entry.institutional_webpage and entry.institutional_webpage.strip()),
        id=f"researcher_{idx + 1}_institutional_webpage_link",
        desc="Provides an institutional faculty webpage/profile link.",
        parent=req_fields,
        critical=True,
    )

    # 1.e Publication profile link
    evaluator.add_custom_node(
        result=bool(entry.publication_profile and entry.publication_profile.strip()),
        id=f"researcher_{idx + 1}_publication_profile_link",
        desc="Provides a publication profile link (e.g., Google Scholar/DBLP/institutional research page).",
        parent=req_fields,
        critical=True,
    )

    # 1.f Publication record summary
    evaluator.add_custom_node(
        result=bool(
            (entry.publication_record_summary and entry.publication_record_summary.strip())
            or (entry.recent_publications_count and entry.recent_publications_count.strip())
            or (entry.top_tier_example_venue and entry.top_tier_example_venue.strip())
        ),
        id=f"researcher_{idx + 1}_publication_record_summary",
        desc="Summarizes publication record including number of recent publications and an example top-tier venue.",
        parent=req_fields,
        critical=True,
    )

    # 1.g Impact metrics reported
    evaluator.add_custom_node(
        result=bool(
            (entry.h_index and entry.h_index.strip()) or (entry.total_citations and entry.total_citations.strip())
        ),
        id=f"researcher_{idx + 1}_impact_metrics_reported",
        desc="Reports h-index and/or total citation count.",
        parent=req_fields,
        critical=True,
    )

    # 2) Eligibility constraints (critical group)
    elig = evaluator.add_parallel(
        id=f"researcher_{idx + 1}_eligibility_constraints",
        desc="Meets NSF CAREER-style eligibility constraints stated in the question.",
        parent=rnode,
        critical=True,
    )

    # 2.a PhD field: CS or closely related
    phd_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_phd_field",
        desc="Holds a Ph.D. in computer science or closely related field.",
        parent=elig,
        critical=True,
    )
    phd_field_text = entry.phd_field or "unspecified field"
    phd_inst_text = entry.phd_institution or ""
    claim_phd = (
        f"The researcher holds a Ph.D. in computer science or a closely related field "
        f"(provided field: '{phd_field_text}', institution: '{phd_inst_text}')."
    )
    await evaluator.verify(
        claim=claim_phd,
        node=phd_node,
        sources=collect_inst_sources(entry),
        additional_instruction="Check the institutional page or CV for degree information. Closely related fields like ECE, Statistics, or Data Science are acceptable.",
    )

    # 2.b Tenure-track or equivalent position
    tt_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_tenure_track_position",
        desc="Currently holds a tenure-track or tenure-track equivalent faculty position.",
        parent=elig,
        critical=True,
    )
    pos_title = entry.position_title or "unspecified title"
    claim_tt = (
        f"The researcher currently holds a tenure-track or tenure-track equivalent faculty position "
        f"(e.g., Assistant/Associate Professor; provided title: '{pos_title}')."
    )
    await evaluator.verify(
        claim=claim_tt,
        node=tt_node,
        sources=collect_inst_sources(entry),
        additional_instruction="From the institutional webpage/profile, verify the position title. Consider Assistant/Associate/Full Professor as tenure-track; do not count adjunct/lecturer unless explicitly tenure-track equivalent.",
    )

    # 2.c US NSF-eligible institution (US-based)
    nsf_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_us_nsf_eligible_institution",
        desc="Affiliated with a US institution eligible for NSF funding.",
        parent=elig,
        critical=True,
    )
    affil_text = entry.affiliation_department or "unspecified affiliation"
    claim_nsf = (
        f"The researcher is affiliated with a US institution eligible for NSF funding "
        f"(affiliation/department: '{affil_text}')."
    )
    await evaluator.verify(
        claim=claim_nsf,
        node=nsf_node,
        sources=collect_inst_sources(entry),
        additional_instruction="Verify the institution is US-based via the institutional page contents (location/address/branding). Typical US universities are NSF-eligible.",
    )

    # 2.d Early-career (≈ within 5 years of first academic appointment)
    ec_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_early_career",
        desc="Qualifies as early-career (typically within 5 years of first academic appointment).",
        parent=elig,
        critical=True,
    )
    first_year = entry.first_appointment_year or "unspecified"
    claim_ec = (
        f"The researcher qualifies as early-career (≈ within 5 years of first academic appointment). "
        f"First appointment year provided: '{first_year}'."
    )
    await evaluator.verify(
        claim=claim_ec,
        node=ec_node,
        sources=collect_inst_sources(entry),
        additional_instruction=(
            f"Use any hire/join date, CV timeline, or bio statement. If first appointment year is 2021 or later, "
            f"consider early-career as satisfied for the {CURRENT_YEAR} context. If explicitly described as 'early-career', accept."
        ),
    )

    # 3) Research specialization (critical leaf)
    spec_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_research_specialization",
        desc="Primary research focus is in AI/ML (or closely related subfield within CS).",
        parent=rnode,
        critical=True,
    )
    focus_text = entry.research_focus or "unspecified"
    claim_spec = (
        f"The researcher's primary research focus is in AI/ML or a closely related subfield within CS "
        f"(described as: '{focus_text}')."
    )
    inst_sources = collect_inst_sources(entry)
    pub_sources = collect_pub_sources(entry)
    spec_sources = inst_sources if inst_sources else pub_sources
    await evaluator.verify(
        claim=claim_spec,
        node=spec_node,
        sources=spec_sources,
        additional_instruction="Check research summary keywords and topics; allow related areas (e.g., NLP, CV, robotics ML, data mining, AI systems).",
    )

    # 4) Publication constraints (critical group)
    pubc_node = evaluator.add_parallel(
        id=f"researcher_{idx + 1}_publication_constraints",
        desc="Meets the publication record constraints stated in the question.",
        parent=rnode,
        critical=True,
    )

    # 4.a Recent publications: ≥3 in 2023–2025
    recent_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_recent_publications",
        desc="Has published at least 3 papers in the last 3 years (2023–2025).",
        parent=pubc_node,
        critical=True,
    )
    claim_recent = "The researcher has at least 3 publications dated 2023–2025 (inclusive)."
    await evaluator.verify(
        claim=claim_recent,
        node=recent_node,
        sources=collect_pub_sources(entry),
        additional_instruction=(
            "Inspect the publication profile (Google Scholar/DBLP/etc.). Count items in years 2023, 2024, and 2025. "
            "Preprints/accepted papers listed with those years can be counted."
        ),
    )

    # 4.b Top-tier or prestigious venue: ≥1 at listed venues or a prestigious journal
    top_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_top_tier_or_prestigious",
        desc="Has ≥1 publication at a listed top-tier conference or a prestigious journal in the field.",
        parent=pubc_node,
        critical=True,
    )
    venue_hint = entry.top_tier_example_venue or "unspecified"
    claim_top = (
        "The researcher has at least one publication at a top-tier CS venue "
        f"(e.g., {', '.join(TOP_TIER_VENUES)}) or a prestigious journal. "
        f"Example provided: '{venue_hint}'."
    )
    await evaluator.verify(
        claim=claim_top,
        node=top_node,
        sources=collect_pub_sources(entry),
        additional_instruction=(
            "Look for venue names or journal titles in the profile. Allow reasonable abbreviations (e.g., NIPS=NeurIPS, AAAI, IJCAI, CVPR, ICCV, ACL, EMNLP). "
            "Prestigious journals in AI/ML/CS are acceptable."
        ),
    )

    # 5) Impact constraint (critical leaf): h-index ≥ 5 OR citations ≥ 100
    impact_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_impact_constraint",
        desc="Meets the impact threshold: h-index ≥ 5 OR total citations ≥ 100.",
        parent=rnode,
        critical=True,
    )
    h_txt = entry.h_index or "unspecified"
    c_txt = entry.total_citations or "unspecified"
    claim_impact = (
        f"The researcher's metrics satisfy: h-index ≥ 5 OR total citations ≥ 100 "
        f"(provided h-index: '{h_txt}', citations: '{c_txt}')."
    )
    await evaluator.verify(
        claim=claim_impact,
        node=impact_node,
        sources=collect_pub_sources(entry),
        additional_instruction="Use Google Scholar or similar metrics page; allow approximate values and rounding.",
    )

    # 6) Verifiability constraint (critical leaf): institutional page usable to verify affiliation/position/area
    verif_node = evaluator.add_leaf(
        id=f"researcher_{idx + 1}_verifiability_constraint",
        desc="Institutional webpage link is usable to verify affiliation/position/research area.",
        parent=rnode,
        critical=True,
    )
    claim_verif = "The institutional faculty webpage provides enough information to verify the affiliation, position title, and research area."
    await evaluator.verify(
        claim=claim_verif,
        node=verif_node,
        sources=collect_inst_sources(entry),
        additional_instruction="Check that the page contains affiliation/institution name, faculty position title, and a research area/keywords/summary.",
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
    Evaluate an answer for the early-career AI/ML researchers eligibility/publication/impact/verifiability task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Gate with four_entries_present first
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

    # 1) Extract researcher entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )
    raw_count = len(extracted.researchers)

    # Record custom info for transparency
    evaluator.add_custom_info(
        {"raw_researcher_entries_count": raw_count, "top_tier_list": TOP_TIER_VENUES, "years_considered": LAST_3_YEARS},
        info_type="extraction_stats",
        info_name="extraction_stats",
    )

    # 2) Root-level critical existence check: provide four entries
    # Use raw_count prior to any padding
    evaluator.add_custom_node(
        result=(raw_count >= 4),
        id="four_entries_present",
        desc="Provides four researcher entries (1st–4th).",
        parent=root,
        critical=True,
    )

    # 3) Build researcher entries evaluation (parallel, partial credit allowed)
    entries_node = evaluator.add_parallel(
        id="researcher_entries",
        desc="Evaluate each researcher entry for constraint satisfaction and presence of requested fields (partial credit per entry).",
        parent=root,
        critical=False,
    )

    # Pad to exactly 4 entries for downstream structure (placeholders for missing)
    researchers: List[ResearcherEntry] = list(extracted.researchers[:4])
    while len(researchers) < 4:
        researchers.append(ResearcherEntry())

    # 4) Verify each of the four entries
    for idx, entry in enumerate(researchers):
        await verify_researcher_entry(evaluator, entries_node, entry, idx)

    # 5) Return the summary with the verification tree and aggregated score
    return evaluator.get_summary()