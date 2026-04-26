import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "early_ai_ml_researchers_2024"
TASK_DESCRIPTION = """
Identify exactly 4 distinct early-career researchers in artificial intelligence and machine learning who meet ALL of the following criteria:

1. Education and Career Stage: Received their PhD degree between 2020 and 2024 (inclusive).

2. Current Position: Currently hold a position as an assistant professor, postdoctoral fellow, or research scientist at a U.S.-based university that is classified as an R1 research university according to the Carnegie Classification of Institutions of Higher Education.

3. 2024 Publication Record: Have at least one paper accepted in the main conference track (not workshop) of one of the following top-tier conferences in 2024: NeurIPS 2024, ICML 2024, or CVPR 2024.

4. Recognition and Awards: Received at least one of the following forms of significant recognition between 2023 and 2024:
   - A best paper award or honorable mention at a major computer science or AI conference
   - A doctoral dissertation award from ACM, AAAI, or another major professional organization
   - A named postdoctoral fellowship or junior faculty research award

5. Professional Engagement: Are a member of at least one major professional computing or artificial intelligence society (ACM, IEEE, or AAAI).

6. Research Focus: Have their primary research area in machine learning, artificial intelligence, computer vision, or a closely related field, as evidenced by their publications and institutional affiliation.

7. Verifiable Information: Have a publicly accessible research profile (either on their university's official website or a verified Google Scholar profile) that includes their current position, institutional affiliation, and publication list.

For each of the 4 researchers you identify, provide:
- Full name
- Current position title
- Current institution
- PhD completion year
- At least one 2024 conference paper title (from NeurIPS, ICML, or CVPR)
- Recognition/award received (2023-2024)
- Professional society membership
- URL to their research profile
- URL confirming their 2024 publication
- URL confirming their recognition/award
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherEntry(BaseModel):
    name: Optional[str] = None
    position_title: Optional[str] = None
    institution: Optional[str] = None
    phd_year: Optional[str] = None
    research_area: Optional[str] = None
    profile_url: Optional[str] = None
    paper_title_2024: Optional[str] = None
    publication_url: Optional[str] = None
    recognition: Optional[str] = None
    recognition_url: Optional[str] = None
    society_memberships: List[str] = Field(default_factory=list)


class ResearchersExtraction(BaseModel):
    researchers: List[ResearcherEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return """
    Extract the list of researchers described in the answer.

    For each researcher, extract the following fields exactly as stated in the answer:
    - name: Full name (string)
    - position_title: Current position title (string), e.g., "Assistant Professor", "Postdoctoral Fellow", "Research Scientist"
    - institution: Current institution (string), which should be a U.S.-based university
    - phd_year: PhD completion year (string as it appears, e.g., "2021")
    - research_area: Primary research area (string, e.g., "Machine Learning", "Artificial Intelligence", "Computer Vision")
    - profile_url: URL to their research profile (must be either an official university page or a Google Scholar profile URL)
    - paper_title_2024: Title of at least one 2024 main-track conference paper (NeurIPS 2024, ICML 2024, or CVPR 2024)
    - publication_url: A URL that confirms the 2024 qualifying publication (e.g., official proceedings page, openaccess page, or the conference site)
    - recognition: A description of a recognition/award received in 2023 or 2024 that qualifies (e.g., "Best Paper Award at XYZ 2024", "ACM Doctoral Dissertation Award")
    - recognition_url: A URL that confirms the recognition/award
    - society_memberships: An array of membership names, including any of: "ACM", "IEEE", "AAAI". Include variants like "ACM member", "IEEE Senior Member", "AAAI member" if present.

    IMPORTANT:
    - Extract URLs exactly as provided in the answer. If the answer uses markdown links, extract the underlying URL.
    - If any field is missing for a researcher, set it to null (or an empty array for society_memberships).
    - If the answer lists more than 4 researchers, extract all of them; the evaluator will later consider only the first 4.
    - Do not invent any information; only extract what is explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    import re
    s = name.lower().strip()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value.strip())
    except Exception:
        # Try to extract digits
        import re
        m = re.search(r"\d{4}", value)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
    return None


def _year_in_range(year: Optional[str], lo: int = 2020, hi: int = 2024) -> bool:
    y = _safe_int(year)
    if y is None:
        return False
    return lo <= y <= hi


def _has_allowed_position(position_title: Optional[str]) -> bool:
    if not position_title:
        return False
    s = position_title.lower()
    allowed_keywords = [
        "assistant professor",
        "postdoctoral fellow",
        "postdoc",
        "postdoctoral researcher",
        "research scientist",
        "ai research scientist",
    ]
    return any(k in s for k in allowed_keywords)


def _non_empty_url(url: Optional[str]) -> bool:
    return bool(url and url.strip())


def _non_empty_text(text: Optional[str]) -> bool:
    return bool(text and text.strip())


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def _add_set_level_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: ResearchersExtraction,
) -> None:
    # Gate node: Set-level requirements (critical)
    set_node = evaluator.add_parallel(
        id="Set_Level_Requirements",
        desc="Requirements about the set of researchers as a whole",
        parent=parent_node,
        critical=True,
    )

    # Consider only the first 4 researchers (filtering to satisfy exact count requirement)
    total_listed = len(extracted.researchers)
    first_four = extracted.researchers[:4]

    # Exactly 4 researchers listed
    evaluator.add_custom_node(
        result=(total_listed >= 4),
        id="Exactly_4_Researchers_Listed",
        desc="Response identifies exactly 4 researchers (evaluating only the first 4 if more are listed)",
        parent=set_node,
        critical=True,
    )

    # All 4 researchers distinct
    normalized_names = [_normalize_name(r.name) for r in first_four]
    all_named = all(n for n in normalized_names)
    distinct_count = len(set(normalized_names))
    evaluator.add_custom_node(
        result=(all_named and distinct_count == 4),
        id="All_Researchers_Distinct",
        desc="All 4 researchers are distinct individuals (based on normalized full names)",
        parent=set_node,
        critical=True,
    )


async def _verify_researcher(
    evaluator: Evaluator,
    parent_node,
    researcher: ResearcherEntry,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single researcher (partial credit allowed).
    """
    rnode = evaluator.add_parallel(
        id=f"Researcher_{idx+1}",
        desc=f"Researcher {idx+1} satisfies all constraints and required output fields",
        parent=parent_node,
        critical=False,
    )

    # 1. Name provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_text(researcher.name),
        id=f"Researcher_{idx+1}_Name_Provided",
        desc="Full name is provided",
        parent=rnode,
        critical=True,
    )

    # 2. Position Title provided + allowed
    # 2.1 Provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_text(researcher.position_title),
        id=f"Researcher_{idx+1}_Position_Title_Provided",
        desc="Current position title is provided",
        parent=rnode,
        critical=True,
    )
    # 2.2 Allowed (critical)
    pos_allowed_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Current_Position_Title_Provided_And_Allowed",
        desc="Current position title is provided and is one of: assistant professor, postdoctoral fellow, or research scientist",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The position title '{researcher.position_title}' corresponds to an assistant professor OR postdoctoral fellow OR research scientist role.",
        node=pos_allowed_leaf,
        additional_instruction="Allow reasonable variants like 'Postdoc', 'Postdoctoral Researcher', 'AI Research Scientist'. Case-insensitive.",
    )

    # 3. Institution provided + qualifies (U.S.-based R1)
    evaluator.add_custom_node(
        result=_non_empty_text(researcher.institution),
        id=f"Researcher_{idx+1}_Institution_Provided",
        desc="Current institution is provided",
        parent=rnode,
        critical=True,
    )
    inst_qual_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Current_Institution_Provided_And_Qualifies",
        desc="Current institution is provided and is a U.S.-based Carnegie-classified R1 university",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The institution '{researcher.institution}' is a U.S.-based university classified as R1 (Very High Research Activity) by the Carnegie Classification.",
        node=inst_qual_leaf,
        sources=researcher.profile_url,  # Try to support via the researcher's profile page when possible
        additional_instruction="Prefer evidence on the official university page if available. If not explicitly stated, judge based on well-known classification when reasonable.",
    )

    # 4. PhD year provided and in range (2020–2024)
    evaluator.add_custom_node(
        result=_year_in_range(researcher.phd_year, 2020, 2024),
        id=f"Researcher_{idx+1}_PhD_Year_Provided_And_In_Range",
        desc="PhD completion year is provided and is between 2020 and 2024 inclusive",
        parent=rnode,
        critical=True,
    )

    # 5. Research area in scope (ML/AI/CV or closely related)
    ra_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Research_Area_In_Scope",
        desc="Primary research area is ML/AI/CV or a closely related field (as evidenced by profile/publications/institutional affiliation)",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The research area '{researcher.research_area}' is within machine learning, artificial intelligence, computer vision, or a closely related field.",
        node=ra_leaf,
        sources=researcher.profile_url,
        additional_instruction="Use the profile page to corroborate that the area is ML/AI/CV (e.g., look at publications/topics). Allow closely related areas like representation learning, deep learning, probabilistic modeling.",
    )

    # 6. Profile URL provided and qualifies
    evaluator.add_custom_node(
        result=_non_empty_url(researcher.profile_url),
        id=f"Researcher_{idx+1}_Profile_URL_Provided",
        desc="Research profile URL is provided",
        parent=rnode,
        critical=True,
    )
    profile_qual_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Profile_URL_Provided_And_Qualifies",
        desc="Provides a publicly accessible research profile URL that is either an official university page or a verified Google Scholar profile",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim="This profile URL is either an official university page or a verified Google Scholar profile.",
        node=profile_qual_leaf,
        sources=researcher.profile_url,
        additional_instruction=(
            "For university pages, check that the domain is the official institution domain (often .edu) and the page is a faculty/postdoc profile. "
            "For Google Scholar, confirm it's on scholar.google.com and ideally shows 'Verified email at <institution>' or is otherwise clearly the correct profile."
        ),
    )

    # 7. Profile includes position, affiliation, and publications
    profile_contents_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Profile_Includes_Position_Affiliation_And_Publications",
        desc="The provided research profile includes current position, institutional affiliation, and publication list",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim="The profile page includes the current position title, the institutional affiliation, and a list of publications.",
        node=profile_contents_leaf,
        sources=researcher.profile_url,
        additional_instruction="Check the page for role labels (e.g., Assistant Professor, Postdoc), the institution name, and a publications section or a list of selected papers.",
    )

    # 8. 2024 main-track paper provided and qualifies
    evaluator.add_custom_node(
        result=_non_empty_text(researcher.paper_title_2024),
        id=f"Researcher_{idx+1}_Paper_Title_2024_Provided",
        desc="A 2024 paper title is provided",
        parent=rnode,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_url(researcher.publication_url),
        id=f"Researcher_{idx+1}_Publication_URL_Provided",
        desc="A publication confirmation URL is provided",
        parent=rnode,
        critical=True,
    )
    paper_qual_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_2024_Main_Track_Paper_Provided_And_Qualifies",
        desc="Provides at least one 2024 main-track (not workshop) paper title accepted at NeurIPS 2024, ICML 2024, or CVPR 2024",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This URL corresponds to a 2024 main-track paper (not workshop) at NeurIPS 2024, ICML 2024, or CVPR 2024 "
            f"with a title matching or equivalent to '{researcher.paper_title_2024}'."
        ),
        node=paper_qual_leaf,
        sources=researcher.publication_url,
        additional_instruction=(
            "Confirm the page is for the main conference track (e.g., official proceedings, openaccess page, or conference site) rather than a workshop. "
            "For ICML, PMLR proceedings volumes are acceptable; for CVPR, thecvf.com openaccess pages are acceptable; for NeurIPS, neurips.cc or official proceedings pages are acceptable. "
            "Accept categories like Oral/Poster/Spotlight as main track. Reject workshop-only entries."
        ),
    )

    # 9. Publication confirmation URL provided and matches
    pub_match_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Publication_Confirmation_URL_Provided_And_Matches",
        desc="Provides a URL confirming the claimed 2024 qualifying publication (e.g., proceedings/paper listing) and it matches the stated paper/conference/year",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The publication page shows the same paper title '{researcher.paper_title_2024}' and clearly indicates "
            "it is a 2024 main conference paper at NeurIPS/ICML/CVPR."
        ),
        node=pub_match_leaf,
        sources=researcher.publication_url,
        additional_instruction="Match the title (allow minor variants and case differences) and verify the year (2024) and the conference (NeurIPS, ICML, or CVPR).",
    )

    # 10. Recognition provided and qualifies (2023–2024)
    evaluator.add_custom_node(
        result=_non_empty_text(researcher.recognition),
        id=f"Researcher_{idx+1}_Recognition_Provided",
        desc="Recognition description is provided",
        parent=rnode,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_url(researcher.recognition_url),
        id=f"Researcher_{idx+1}_Recognition_URL_Provided",
        desc="Recognition confirmation URL is provided",
        parent=rnode,
        critical=True,
    )
    recog_qual_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Recognition_Provided_And_Qualifies",
        desc="Provides at least one qualifying recognition received in 2023–2024 (best paper/honorable mention at major CS/AI conference OR doctoral dissertation award from ACM/AAAI/major org OR named postdoc fellowship/junior faculty research award)",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The recognition '{researcher.recognition}' occurred in 2023 or 2024 and qualifies as one of: "
            "best paper/honorable mention at a major CS/AI conference; doctoral dissertation award from ACM/AAAI/major org; "
            "named postdoctoral fellowship or junior faculty research award."
        ),
        node=recog_qual_leaf,
        sources=researcher.recognition_url,
        additional_instruction="Use the recognition page to confirm both the type and timeframe (2023–2024).",
    )

    recog_match_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Recognition_Confirmation_URL_Provided_And_Matches",
        desc="Provides a URL confirming the claimed recognition/award and it matches the stated recognition and timeframe",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The recognition page confirms the stated recognition '{researcher.recognition}' and indicates it occurred in 2023 or 2024.",
        node=recog_match_leaf,
        sources=researcher.recognition_url,
        additional_instruction="Confirm that the description and year on the page align with the claim.",
    )

    # 11. Professional society membership provided and qualifies
    # Existence
    evaluator.add_custom_node(
        result=bool(researcher.society_memberships),
        id=f"Researcher_{idx+1}_Professional_Society_Membership_Provided",
        desc="Professional society membership is provided",
        parent=rnode,
        critical=True,
    )
    # Qualifies
    membership_leaf = evaluator.add_leaf(
        id=f"Researcher_{idx+1}_Professional_Society_Membership_Provided_And_Qualifies",
        desc="Professional society membership is stated and includes at least one of: ACM, IEEE, or AAAI",
        parent=rnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated memberships {researcher.society_memberships} include at least one of: ACM, IEEE, or AAAI.",
        node=membership_leaf,
        additional_instruction="Allow variants like 'ACM member', 'IEEE Senior Member', 'Association for the Advancement of Artificial Intelligence (AAAI)'. Case-insensitive.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the early-career AI/ML researchers task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Gate researchers on set-level critical checks
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

    # Extract researchers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={"total_extracted": len(extracted.researchers)},
        info_type="extraction_stats",
        info_name="extraction_overview",
    )

    # Build set-level requirements (critical) under root
    await _add_set_level_requirements(evaluator, root, extracted)

    # Build per-researcher verification subtree (non-critical aggregate)
    researchers_parent = evaluator.add_parallel(
        id="Researchers",
        desc="Evaluate each of the 4 researcher entries independently (partial credit allowed across entries)",
        parent=root,
        critical=False,
    )

    # Consider only the first 4 researchers per instructions
    selected: List[ResearcherEntry] = extracted.researchers[:4]
    # Pad to ensure exactly 4 entries are checked, even if fewer are provided
    while len(selected) < 4:
        selected.append(ResearcherEntry())

    # Verify each researcher
    for i, researcher in enumerate(selected):
        await _verify_researcher(evaluator, researchers_parent, researcher, i)

    # Return standard summary
    return evaluator.get_summary()