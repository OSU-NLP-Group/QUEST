import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "southern_universities_multi_criteria"
TASK_DESCRIPTION = """Identify three public universities that meet all of the following criteria:

Geographic and Institutional Requirements:
- Located in one of these three states: Mississippi, Florida, or South Carolina
- Publicly operated institution

Accreditation Requirements:
- Accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)
- Business school or college holds AACSB (Association to Advance Collegiate Schools of Business) accreditation

Academic Program Requirements:
- Graduate school minimum GPA requirement for standard admission is 3.0 or lower
- Offers doctoral programs in at least three different academic fields
- Has at least two interdisciplinary research centers or institutes

Research and Resources Requirements:
- Has documented research and development expenditures reported in the NSF HERD Survey
- University endowment exceeds $100 million

Athletic Requirements:
- Has NCAA Division I athletic programs

Library and Faculty Requirements:
- Library system includes a Special Collections and/or University Archives department
- Has an established Distinguished University Professor award or equivalent distinguished professorship recognition program for faculty

International Engagement Requirements:
- Has documented international partnership agreements or study abroad programs

For each university, provide: (1) the official name of the university, (2) the state where it is located, (3) verification that it meets each criterion with supporting URL references."""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    homepage_url: Optional[str] = None

    sacscoc_urls: List[str] = Field(default_factory=list)
    aacsb_urls: List[str] = Field(default_factory=list)

    location_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)

    grad_gpa: Optional[str] = None
    grad_gpa_urls: List[str] = Field(default_factory=list)

    herd_urls: List[str] = Field(default_factory=list)

    ncaa_division: Optional[str] = None
    conference: Optional[str] = None
    ncaa_urls: List[str] = Field(default_factory=list)

    special_collections_urls: List[str] = Field(default_factory=list)
    distinguished_professorship_urls: List[str] = Field(default_factory=list)

    doctoral_fields: List[str] = Field(default_factory=list)
    doctoral_urls: List[str] = Field(default_factory=list)

    centers: List[str] = Field(default_factory=list)
    centers_urls: List[str] = Field(default_factory=list)

    international_urls: List[str] = Field(default_factory=list)

    endowment: Optional[str] = None
    endowment_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract universities and all supporting details explicitly mentioned in the answer. Return a JSON with a `universities` array, each item having the following fields. Use null for missing scalar fields and [] for missing URL arrays. Extract as many universities as present; do not fabricate.

For each university, extract:
- name: Official university name as written
- state: The U.S. state (e.g., "Mississippi", "Florida", or "South Carolina"); allow postal abbreviations if that is how it appears
- homepage_url: Official university homepage URL

Accreditation URLs:
- sacscoc_urls: URLs that confirm SACSCOC accreditation
- aacsb_urls: URLs that confirm AACSB accreditation of the business school/college

Geography & Public Status URLs:
- location_urls: URLs that state the university's location (state)
- public_status_urls: URLs that confirm the university is public

Graduate GPA:
- grad_gpa: The minimum GPA for standard graduate admission (string as shown, e.g., "3.0", "2.75")
- grad_gpa_urls: URLs supporting the stated graduate GPA requirement

Research Expenditures:
- herd_urls: URLs supporting that the university has R&D expenditures in the NSF HERD survey (NSF pages, institutional research pages, or credible summaries)

Athletics:
- ncaa_division: Division string as written (e.g., "NCAA Division I") if present
- conference: Athletics conference name if present (e.g., "SEC", "ACC")
- ncaa_urls: URLs supporting NCAA Division I status and/or conference

Library & Faculty:
- special_collections_urls: URLs showing Special Collections and/or University Archives
- distinguished_professorship_urls: URLs showing a distinguished professorship award program (e.g., Distinguished University Professor)

Doctoral Programs:
- doctoral_fields: A list of at least three distinct academic fields in which the university offers doctoral programs (e.g., ["Chemistry", "Mechanical Engineering", "History"])
- doctoral_urls: URLs supporting availability of doctoral programs/fields

Interdisciplinary Centers:
- centers: Names of at least two interdisciplinary research centers/institutes
- centers_urls: URLs supporting the existence of these centers/institutes

International:
- international_urls: URLs demonstrating international partnership agreements or study abroad programs

Endowment:
- endowment: The endowment value as a string exactly as shown (e.g., "$1.2 billion", "$250 million")
- endowment_urls: URLs supporting the endowment figure

SPECIAL RULES:
- Only extract URLs that appear in the answer. Do not infer or invent URLs.
- If a field isn’t mentioned, set it to null (for scalar) or [] (for lists).
- Do not normalize numbers; record them exactly as written in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third"}
    return mapping.get(n, f"#{n}")


def merge_urls(*url_lists: Optional[List[str]], fallbacks: Optional[List[Optional[str]]] = None) -> List[str]:
    """Merge and de-duplicate URL lists, dropping empties; also include fallbacks if provided."""
    urls: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    if fallbacks:
        for fb in fallbacks:
            if isinstance(fb, str) and fb and fb.strip():
                urls.append(fb.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification logic for a single university                                  #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, index: int) -> None:
    uni_short = f"U{index+1}"
    uni_name = uni.name or "the university"

    # 0) University official name + homepage (sequential)
    name_seq = evaluator.add_sequential(
        id=f"{uni_short}_University_Name",
        desc="Official name of the university provided",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=non_empty_str(uni.name),
        id=f"{uni_short}_Name_Stated",
        desc="University name clearly stated",
        parent=name_seq,
        critical=True,
    )
    name_hp_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Homepage_Reference",
        desc="URL reference to university's official homepage",
        parent=name_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is the official homepage of {uni_name}.",
        node=name_hp_leaf,
        sources=uni.homepage_url,
        additional_instruction="Confirm the page is the top-level institutional site (e.g., branding/header/title matches the university's official name). Minor naming variants are acceptable.",
    )

    # 1) Regional accreditation (SACSCOC) - sequential
    sac_seq = evaluator.add_sequential(
        id=f"{uni_short}_Regional_Accreditation",
        desc="University is accredited by SACSCOC",
        parent=parent_node,
        critical=True,
    )
    sac_status_leaf = evaluator.add_leaf(
        id=f"{uni_short}_SACSCOC_Status",
        desc="Confirmed SACSCOC accreditation status",
        parent=sac_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).",
        node=sac_status_leaf,
        sources=uni.sacscoc_urls,
        additional_instruction="Prefer SACSCOC's own directory or institutional page showing SACSCOC accreditation. The page must clearly indicate SACSCOC accreditation.",
    )
    evaluator.add_custom_node(
        result=len(uni.sacscoc_urls) > 0,
        id=f"{uni_short}_SACSCOC_Reference",
        desc="URL reference confirming SACSCOC accreditation",
        parent=sac_seq,
        critical=True,
    )

    # 2) Business accreditation (AACSB) - sequential
    aacsb_seq = evaluator.add_sequential(
        id=f"{uni_short}_Business_Accreditation",
        desc="University has AACSB-accredited business programs",
        parent=parent_node,
        critical=True,
    )
    aacsb_status_leaf = evaluator.add_leaf(
        id=f"{uni_short}_AACSB_Status",
        desc="Confirmed AACSB accreditation for business school",
        parent=aacsb_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The business school/college at {uni_name} holds AACSB accreditation.",
        node=aacsb_status_leaf,
        sources=uni.aacsb_urls,
        additional_instruction="Accept either the AACSB accredited school directory or an official page stating AACSB accreditation. Campus-specific business school page is acceptable.",
    )
    evaluator.add_custom_node(
        result=len(uni.aacsb_urls) > 0,
        id=f"{uni_short}_AACSB_Reference",
        desc="URL reference confirming AACSB accreditation",
        parent=aacsb_seq,
        critical=True,
    )

    # 3) State location (sequential)
    state_seq = evaluator.add_sequential(
        id=f"{uni_short}_State_Location",
        desc="University is located in Mississippi, Florida, or South Carolina",
        parent=parent_node,
        critical=True,
    )
    state_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Specific_State",
        desc="Identification of the specific state (MS, FL, or SC)",
        parent=state_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is located in the U.S. state of {uni.state or '[state not provided]'}, which is one of Mississippi, Florida, or South Carolina.",
        node=state_leaf,
        sources=merge_urls(uni.location_urls, fallbacks=[uni.homepage_url]),
        additional_instruction="Verify the state location on the provided official or credible page. Allow variants like postal abbreviations (e.g., FL for Florida).",
    )
    evaluator.add_custom_node(
        result=len(merge_urls(uni.location_urls, fallbacks=[uni.homepage_url])) > 0,
        id=f"{uni_short}_State_Reference",
        desc="URL reference confirming state location",
        parent=state_seq,
        critical=True,
    )

    # 4) Public institution (sequential)
    public_seq = evaluator.add_sequential(
        id=f"{uni_short}_Public_Institution",
        desc="University is a public institution",
        parent=parent_node,
        critical=True,
    )
    public_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Public_Status",
        desc="Confirmed public university status",
        parent=public_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public university (publicly operated).",
        node=public_leaf,
        sources=merge_urls(uni.public_status_urls, fallbacks=[uni.homepage_url]),
        additional_instruction="Use official or authoritative sources (e.g., About pages, governance pages, state higher education listings) that explicitly state public status.",
    )
    evaluator.add_custom_node(
        result=len(merge_urls(uni.public_status_urls, fallbacks=[uni.homepage_url])) > 0,
        id=f"{uni_short}_Public_Reference",
        desc="URL reference confirming public institution status",
        parent=public_seq,
        critical=True,
    )

    # 5) Graduate GPA requirement (sequential)
    gpa_seq = evaluator.add_sequential(
        id=f"{uni_short}_Graduate_GPA",
        desc="Graduate school minimum GPA requirement is 3.0 or lower",
        parent=parent_node,
        critical=True,
    )
    gpa_leaf = evaluator.add_leaf(
        id=f"{uni_short}_GPA_Value",
        desc="Specific minimum GPA requirement stated",
        parent=gpa_seq,
        critical=True,
    )
    gpa_value_shown = uni.grad_gpa or "[GPA not provided]"
    await evaluator.verify(
        claim=f"The standard minimum GPA for graduate admission at {uni_name} is 3.0 or lower (e.g., listed as {gpa_value_shown}).",
        node=gpa_leaf,
        sources=uni.grad_gpa_urls,
        additional_instruction="Confirm the baseline/minimum GPA for general/standard graduate admission (not a higher departmental/program-specific GPA). If multiple values, the university-wide minimum should be 3.0 or below.",
    )
    evaluator.add_custom_node(
        result=len(uni.grad_gpa_urls) > 0,
        id=f"{uni_short}_GPA_Reference",
        desc="URL reference for graduate admission GPA requirement",
        parent=gpa_seq,
        critical=True,
    )

    # 6) Research expenditures (HERD) (sequential)
    herd_seq = evaluator.add_sequential(
        id=f"{uni_short}_Research_Expenditures",
        desc="University has documented R&D expenditures in HERD Survey data",
        parent=parent_node,
        critical=True,
    )
    herd_leaf = evaluator.add_leaf(
        id=f"{uni_short}_HERD_Data",
        desc="Confirmed presence in HERD Survey or documented research expenditures",
        parent=herd_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} has research and development expenditures reported in the NSF HERD Survey.",
        node=herd_leaf,
        sources=uni.herd_urls,
        additional_instruction="Accept NSF HERD data pages or official institutional research pages that explicitly reference HERD reporting.",
    )
    evaluator.add_custom_node(
        result=len(uni.herd_urls) > 0,
        id=f"{uni_short}_Research_Reference",
        desc="URL reference for research expenditure data",
        parent=herd_seq,
        critical=True,
    )

    # 7) NCAA Division I athletics (parallel)
    ncaa_par = evaluator.add_parallel(
        id=f"{uni_short}_NCAA_Program",
        desc="University has NCAA Division I athletic programs",
        parent=parent_node,
        critical=True,
    )
    ncaa_div_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Division_Status",
        desc="Confirmed NCAA Division I status",
        parent=ncaa_par,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} competes in NCAA Division I athletics.",
        node=ncaa_div_leaf,
        sources=uni.ncaa_urls,
        additional_instruction="Verify via NCAA profiles, official athletics pages, or credible conference/league pages that explicitly indicate NCAA Division I.",
    )

    # To satisfy framework critical-child constraint, mark this as critical;
    # craft a generic claim that the conference affiliation is identified.
    conference_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Conference",
        desc="Athletic conference affiliation identified",
        parent=ncaa_par,
        critical=True,
    )
    conf_claim = (
        f"The athletic conference affiliation for {uni_name} is identified on the provided athletics-related page(s)."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conference_leaf,
        sources=uni.ncaa_urls,
        additional_instruction="Pass if the page indicates a specific NCAA Division I conference (e.g., SEC, ACC, Sun Belt, AAC, etc.), even if the extracted conference name varies slightly.",
    )

    evaluator.add_custom_node(
        result=len(uni.ncaa_urls) > 0,
        id=f"{uni_short}_NCAA_Reference",
        desc="URL reference for NCAA Division I status",
        parent=ncaa_par,
        critical=True,
    )

    # 8) Library Special Collections / Archives (sequential)
    lib_seq = evaluator.add_sequential(
        id=f"{uni_short}_Special_Collections",
        desc="University library has Special Collections and/or University Archives",
        parent=parent_node,
        critical=True,
    )
    lib_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Collections_Exist",
        desc="Confirmed existence of Special Collections or Archives department",
        parent=lib_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name}'s library system includes a Special Collections and/or University Archives department.",
        node=lib_leaf,
        sources=uni.special_collections_urls,
        additional_instruction="Verify via library pages that clearly indicate 'Special Collections' and/or 'University Archives'.",
    )
    evaluator.add_custom_node(
        result=len(uni.special_collections_urls) > 0,
        id=f"{uni_short}_Collections_Reference",
        desc="URL reference for Special Collections/Archives",
        parent=lib_seq,
        critical=True,
    )

    # 9) Distinguished Professorship program (sequential)
    fac_seq = evaluator.add_sequential(
        id=f"{uni_short}_Distinguished_Faculty",
        desc="University has Distinguished University Professor or equivalent award program",
        parent=parent_node,
        critical=True,
    )
    fac_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Faculty_Award",
        desc="Confirmed existence of distinguished professorship program",
        parent=fac_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} has an established program recognizing faculty with a 'Distinguished University Professor' (or equivalent) title.",
        node=fac_leaf,
        sources=uni.distinguished_professorship_urls,
        additional_instruction="Accept equivalent distinguished professorship titles (e.g., 'University Distinguished Professor') if clearly an institution-level honor.",
    )
    evaluator.add_custom_node(
        result=len(uni.distinguished_professorship_urls) > 0,
        id=f"{uni_short}_Faculty_Reference",
        desc="URL reference for faculty award program",
        parent=fac_seq,
        critical=True,
    )

    # 10) Doctoral programs in at least three different fields (sequential)
    doc_seq = evaluator.add_sequential(
        id=f"{uni_short}_Doctoral_Programs",
        desc="University offers doctoral programs in at least three different fields",
        parent=parent_node,
        critical=True,
    )
    doc_count = evaluator.add_parallel(
        id=f"{uni_short}_Doctoral_Count",
        desc="At least three doctoral programs identified",
        parent=doc_seq,
        critical=True,
    )
    # Create three field checks (all critical to satisfy framework constraint)
    fields = uni.doctoral_fields[:3] if uni.doctoral_fields else []
    while len(fields) < 3:
        fields.append(None)  # Pad with missing to enforce requirement

    for j, field_name in enumerate(fields, start=1):
        fld_leaf = evaluator.add_leaf(
            id=f"{uni_short}_Doctoral_Field_{j}",
            desc=f"{'First' if j==1 else ('Second' if j==2 else 'Third')} doctoral program field identified",
            parent=doc_count,
            critical=True,
        )
        if field_name and field_name.strip():
            claim = f"{uni_name} offers a doctoral program in {field_name}."
        else:
            claim = f"{uni_name} offers a doctoral program in a specific, explicitly named field (field {j} must be identifiable)."
        await evaluator.verify(
            claim=claim,
            node=fld_leaf,
            sources=uni.doctoral_urls,
            additional_instruction="Verify that the program is at the doctoral level (e.g., PhD, EdD, DBA, etc.). The field must be discernible from the source(s).",
        )

    doc_ref_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Doctoral_Reference",
        desc="URL reference for doctoral programs",
        parent=doc_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} offers doctoral programs in at least three distinct academic fields.",
        node=doc_ref_leaf,
        sources=uni.doctoral_urls,
        additional_instruction="The page(s) should list or imply at least three different doctoral fields (not three tracks within one field).",
    )

    # 11) Interdisciplinary research centers (sequential)
    cen_seq = evaluator.add_sequential(
        id=f"{uni_short}_Interdisciplinary_Centers",
        desc="University has at least two interdisciplinary research centers",
        parent=parent_node,
        critical=True,
    )
    cen_count = evaluator.add_parallel(
        id=f"{uni_short}_Centers_Count",
        desc="At least two interdisciplinary centers identified",
        parent=cen_seq,
        critical=True,
    )
    centers = uni.centers[:2] if uni.centers else []
    while len(centers) < 2:
        centers.append(None)

    for j, center_name in enumerate(centers, start=1):
        cen_leaf = evaluator.add_leaf(
            id=f"{uni_short}_Center_{j}",
            desc=f"{'First' if j==1 else 'Second'} interdisciplinary research center identified",
            parent=cen_count,
            critical=True,
        )
        if center_name and center_name.strip():
            claim = f"{uni_name} has an interdisciplinary research center or institute named '{center_name}'."
        else:
            claim = f"{uni_name} has a named interdisciplinary research center or institute (center {j} must be identifiable)."
        await evaluator.verify(
            claim=claim,
            node=cen_leaf,
            sources=uni.centers_urls,
            additional_instruction="The center/institute should clearly span multiple disciplines or be labeled 'interdisciplinary.'",
        )

    cen_ref_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Centers_Reference",
        desc="URL reference for research centers",
        parent=cen_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} has at least two interdisciplinary research centers or institutes.",
        node=cen_ref_leaf,
        sources=uni.centers_urls,
        additional_instruction="The sources should collectively support the existence of two or more interdisciplinary centers/institutes.",
    )

    # 12) International partnerships or study abroad (sequential)
    intl_seq = evaluator.add_sequential(
        id=f"{uni_short}_International_Partnerships",
        desc="University has documented international partnerships or study abroad programs",
        parent=parent_node,
        critical=True,
    )
    intl_leaf = evaluator.add_leaf(
        id=f"{uni_short}_International_Programs",
        desc="Confirmed existence of international partnerships or study abroad",
        parent=intl_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} has documented international partnership agreements or operates study abroad programs.",
        node=intl_leaf,
        sources=uni.international_urls,
        additional_instruction="Accept International Office/Global Engagement pages showing formal partnerships, MOUs, or study abroad program listings.",
    )
    evaluator.add_custom_node(
        result=len(uni.international_urls) > 0,
        id=f"{uni_short}_International_Reference",
        desc="URL reference for international programs",
        parent=intl_seq,
        critical=True,
    )

    # 13) Endowment exceeds $100 million (sequential)
    end_seq = evaluator.add_sequential(
        id=f"{uni_short}_Endowment",
        desc="University has endowment exceeding $100 million",
        parent=parent_node,
        critical=True,
    )
    end_leaf = evaluator.add_leaf(
        id=f"{uni_short}_Endowment_Value",
        desc="Specific endowment value stated and exceeds $100 million",
        parent=end_seq,
        critical=True,
    )
    end_str = uni.endowment or "[endowment not provided]"
    await evaluator.verify(
        claim=f"{uni_name}'s endowment exceeds $100 million (e.g., reported as {end_str}).",
        node=end_leaf,
        sources=uni.endowment_urls,
        additional_instruction="The page should clearly indicate an endowment amount and that it is above $100 million (simple numerical comparison is acceptable).",
    )
    evaluator.add_custom_node(
        result=len(uni.endowment_urls) > 0,
        id=f"{uni_short}_Endowment_Reference",
        desc="URL reference for endowment data",
        parent=end_seq,
        critical=True,
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Southern universities multi-criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification per university
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

    # Extract structured universities info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    universities: List[UniversityItem] = list(extracted.universities) if extracted and extracted.universities else []

    # Select first three; pad with placeholders if fewer
    while len(universities) < 3:
        universities.append(UniversityItem())

    selected = universities[:3]

    # Build per-university verification trees
    for i in range(3):
        uni = selected[i]
        uni_node = evaluator.add_parallel(
            id=f"University_{i+1}",
            desc=f"{ordinal(i+1)} university meeting all specified criteria",
            parent=root,
            critical=False,  # Allow partial credit across different universities
        )
        await verify_university(evaluator, uni_node, uni, i)

    return evaluator.get_summary()