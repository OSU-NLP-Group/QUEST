import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hyperscale_dc_2023_2025"
TASK_DESCRIPTION = """
Identify three hyperscale data center development projects in the United States where construction was announced or started between January 2023 and December 2025. Each project must meet the following requirements:

Critical Requirements (must all be satisfied):
1. The project must be located in one of these metropolitan areas: Northern Virginia, Phoenix (Arizona), Dallas (Texas), Chicago (Illinois), or Atlanta (Georgia)
2. The project must be operated by or built for one of these companies: Google, Amazon/AWS, Microsoft Azure, or Meta
3. The facility must have a planned capacity of at least 50 megawatts (MW) of critical IT load, or be part of a campus development where the aggregate planned capacity meets this threshold
4. The project must involve a multi-building campus development (not a single standalone building)
5. The total project investment must be at least $500 million
6. The operator must have a publicly documented commitment to power the facility with renewable energy sources (solar, wind, or other renewables)

Additional Requirements (preferred but not mandatory):
7. The facility should be projected to create at least 100 permanent operational jobs (not including construction jobs)
8. The facility should incorporate water-efficient cooling systems such as air cooling, closed-loop cooling systems, or reclaimed water usage
9. The site should have documented plans for future expansion phases beyond the initial construction
10. The site should have confirmed access to at least two of the three major US wireless carriers' 5G networks (AT&T, T-Mobile, or Verizon)
11. The project should have received state or local economic development incentives or tax benefits

For each of the three projects, provide:
- The specific location (city/county and state)
- The hyperscale operator
- The project capacity or campus aggregate capacity
- A description of the campus development structure
- The construction announcement or start date
- The total investment amount
- Details of the renewable energy commitment
- A reference URL that confirms the project's specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectInfo(BaseModel):
    name: Optional[str] = None
    city_or_county: Optional[str] = None
    state: Optional[str] = None
    metro_area: Optional[str] = None
    operator: Optional[str] = None
    announcement_or_start_date: Optional[str] = None
    capacity: Optional[str] = None  # keep string to allow ranges and units like "150 MW", "0.2 GW"
    campus_structure: Optional[str] = None  # description text
    total_investment: Optional[str] = None  # e.g., "$1.2 billion"
    renewables_details: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Preferred (non-critical) fields
    jobs_projected: Optional[str] = None
    water_cooling: Optional[str] = None
    future_expansion: Optional[str] = None
    wireless_5g_access: Optional[str] = None
    incentives: Optional[str] = None


class ProjectsExtraction(BaseModel):
    projects: List[ProjectInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
Extract up to three hyperscale data center projects explicitly mentioned in the answer, preserving the original order of appearance. For each project, extract the following fields exactly as stated (do not invent or infer):

- name: The project or campus name, if given (otherwise null)
- city_or_county: The specific city or county, if stated (otherwise null)
- state: The US state, if stated (otherwise null)
- metro_area: The metropolitan area if explicitly stated (otherwise null)
- operator: The hyperscale operator (company operating or the project is being built for)
- announcement_or_start_date: The announcement date or construction start date as given (any format)
- capacity: The planned capacity or campus aggregate capacity string (e.g., "72 MW", "0.2 GW", "≥ 100 MW"), as written
- campus_structure: The description of the campus development structure, as written
- total_investment: The total investment amount, as written
- renewables_details: Details about the renewable energy commitment related to powering the facility
- reference_urls: A list of all explicit URLs provided for this specific project in the answer; include only valid URLs explicitly shown in the answer text (do not infer)

Preferred (non-critical) fields (if explicitly stated):
- jobs_projected: The number of permanent operational jobs projected (exclude construction jobs)
- water_cooling: Any mention of water-efficient cooling (air cooling, closed-loop, reclaimed water, etc.)
- future_expansion: Any mention of plans for future expansion phases beyond initial construction
- wireless_5g_access: Any mention confirming access to AT&T, T-Mobile, or Verizon 5G
- incentives: Any mention of state/local incentives or tax benefits

Rules:
- Do not fabricate any data. If a field is not present in the answer, set it to null (or an empty array for reference_urls).
- If the answer lists more than three projects, extract only the first three mentioned.
- If the answer lists fewer than three projects, extract as many as are present.
Return a JSON object with a single field:
{
  "projects": [ { ... up to 3 objects ... } ]
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_METROS = [
    "Northern Virginia",
    "Phoenix (AZ)",
    "Phoenix, AZ",
    "Phoenix, Arizona",
    "Dallas (TX)",
    "Dallas, TX",
    "Dallas, Texas",
    "Chicago (IL)",
    "Chicago, IL",
    "Chicago, Illinois",
    "Atlanta (GA)",
    "Atlanta, GA",
    "Atlanta, Georgia",
]

ALLOWED_OPERATORS_STR = "Google; Amazon/AWS; Microsoft Azure; Meta (Facebook)"


def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def build_location_string(p: ProjectInfo) -> str:
    parts = []
    if nonempty(p.city_or_county):
        parts.append(p.city_or_county.strip())
    if nonempty(p.state):
        parts.append(p.state.strip())
    loc = ", ".join(parts) if parts else ""
    if nonempty(p.metro_area):
        if loc:
            loc += f" (metro: {p.metro_area.strip()})"
        else:
            loc = p.metro_area.strip()
    return loc


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not nonempty(u):
            continue
        key = u.strip()
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def is_project_non_empty(p: Optional[ProjectInfo]) -> bool:
    if p is None:
        return False
    return any([
        nonempty(p.operator),
        nonempty(p.city_or_county),
        nonempty(p.state),
        nonempty(p.capacity),
        nonempty(p.total_investment),
        nonempty(p.campus_structure),
        nonempty(p.announcement_or_start_date),
        len(p.reference_urls) > 0
    ])


def distinct_key(p: ProjectInfo) -> str:
    op = normalize_text(p.operator)
    loc = normalize_text(build_location_string(p))
    cap = normalize_text(p.capacity)
    name = normalize_text(p.name)
    return "|".join([op, loc, name or cap])


# --------------------------------------------------------------------------- #
# Verification for a single project                                           #
# --------------------------------------------------------------------------- #
async def verify_project(evaluator: Evaluator, parent_node, project: ProjectInfo, idx: int) -> None:
    pid = idx + 1
    proj_node = evaluator.add_parallel(
        id=f"Project_{pid}",
        desc=f"Project #{pid} (scored independently for partial credit).",
        parent=parent_node,
        critical=False
    )

    # ---------------- Location group (sequential) ----------------
    loc_group = evaluator.add_sequential(
        id=f"P{pid}_Location_Group",
        desc=f"P{pid}: Location checks",
        parent=proj_node,
        critical=True
    )
    # P{pid}_Location_Provided
    loc_provided = evaluator.add_custom_node(
        result=(nonempty(project.state) and (nonempty(project.city_or_county) or nonempty(project.metro_area))),
        id=f"P{pid}_Location_Provided",
        desc=f"P{pid}: Provides the specific location (city/county and state).",
        parent=loc_group,
        critical=True
    )
    # P{pid}_Location_In_Allowed_Metro
    loc_allowed_node = evaluator.add_leaf(
        id=f"P{pid}_Location_In_Allowed_Metro",
        desc=f"P{pid}: Project location is within allowed metros (Northern Virginia, Phoenix, Dallas, Chicago, or Atlanta).",
        parent=loc_group,
        critical=True
    )
    loc_str = build_location_string(project)
    await evaluator.verify(
        claim=(
            f"The project located at '{loc_str}' is within one of these metropolitan areas: "
            f"Northern Virginia; Phoenix (AZ); Dallas (TX); Chicago (IL); or Atlanta (GA)."
        ),
        node=loc_allowed_node,
        additional_instruction="Allow county-level or suburb locations that are commonly considered part of the listed metros. For example, Loudoun/Prince William/Fairfax/Arlington counties are part of Northern Virginia (NoVA). Treat common suburbs as part of their metropolitan areas."
    )

    # ---------------- Operator group (sequential) ----------------
    op_group = evaluator.add_sequential(
        id=f"P{pid}_Operator_Group",
        desc=f"P{pid}: Operator checks",
        parent=proj_node,
        critical=True
    )
    op_provided = evaluator.add_custom_node(
        result=nonempty(project.operator),
        id=f"P{pid}_Operator_Provided",
        desc=f"P{pid}: States the hyperscale operator / company the project is for.",
        parent=op_group,
        critical=True
    )
    op_allowed_node = evaluator.add_leaf(
        id=f"P{pid}_Operator_In_Allowed_List",
        desc=f"P{pid}: Operator is one of Google, Amazon/AWS, Microsoft Azure, or Meta.",
        parent=op_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The operator '{project.operator or ''}' is one of: Google, Amazon/AWS, Microsoft Azure, or Meta (Facebook).",
        node=op_allowed_node,
        additional_instruction="Consider common synonyms or branding: 'AWS' for Amazon, 'Microsoft' or 'Azure' for Microsoft Azure, 'Meta' or 'Facebook' for Meta."
    )

    # ---------------- Date group (sequential) ----------------
    date_group = evaluator.add_sequential(
        id=f"P{pid}_Date_Group",
        desc=f"P{pid}: Announcement/start date checks",
        parent=proj_node,
        critical=True
    )
    date_provided = evaluator.add_custom_node(
        result=nonempty(project.announcement_or_start_date),
        id=f"P{pid}_Date_Provided",
        desc=f"P{pid}: States an announcement or construction start date.",
        parent=date_group,
        critical=True
    )
    date_window_node = evaluator.add_leaf(
        id=f"P{pid}_Date_In_2023_2025_Window",
        desc=f"P{pid}: Date falls between Jan 2023 and Dec 2025 (inclusive).",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{project.announcement_or_start_date or ''}' occurs between 2023-01-01 and 2025-12-31, inclusive.",
        node=date_window_node,
        additional_instruction="Treat formats like 'Q2 2024', 'Summer 2025', or 'March 2023' as valid and interpret their calendar-year placement. Accept if the textual date clearly falls within 2023–2025 inclusive."
    )

    # ---------------- Capacity group (sequential) ----------------
    cap_group = evaluator.add_sequential(
        id=f"P{pid}_Capacity_Group",
        desc=f"P{pid}: Capacity checks",
        parent=proj_node,
        critical=True
    )
    cap_provided = evaluator.add_custom_node(
        result=nonempty(project.capacity),
        id=f"P{pid}_Capacity_Provided",
        desc=f"P{pid}: Provides planned capacity (or campus aggregate planned capacity).",
        parent=cap_group,
        critical=True
    )
    cap_threshold_node = evaluator.add_leaf(
        id=f"P{pid}_Capacity_At_Least_50MW",
        desc=f"P{pid}: Planned (or aggregate campus) capacity is ≥ 50 MW critical IT load.",
        parent=cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated capacity '{project.capacity or ''}' is at least 50 MW of critical IT load (or the aggregate campus capacity is ≥ 50 MW).",
        node=cap_threshold_node,
        additional_instruction="Interpret units robustly: '0.05 GW' equals 50 MW; '100+ MW', '≥50 MW', 'approximately 60MW' all qualify if ≥ 50 MW. If a range is given, accept if the upper or lower bound meets or exceeds 50 MW."
    )

    # ---------------- Campus structure group (sequential) ----------------
    campus_group = evaluator.add_sequential(
        id=f"P{pid}_Campus_Group",
        desc=f"P{pid}: Campus structure checks",
        parent=proj_node,
        critical=True
    )
    campus_desc_provided = evaluator.add_custom_node(
        result=nonempty(project.campus_structure),
        id=f"P{pid}_Campus_Structure_Description_Provided",
        desc=f"P{pid}: Provides a description of the campus development structure.",
        parent=campus_group,
        critical=True
    )
    campus_multi_node = evaluator.add_leaf(
        id=f"P{pid}_Multi_Building_Campus",
        desc=f"P{pid}: Project involves a multi-building campus (not a standalone building).",
        parent=campus_group,
        critical=True
    )
    await evaluator.verify(
        claim="This project is a multi-building data center campus (multiple buildings, phases, or blocks), not a single standalone building.",
        node=campus_multi_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Look for phrases like 'campus', 'multi-building', 'multiple data centers', 'phased campus', 'several buildings', or descriptions of separate buildings/phases on the provided source(s)."
    )

    # ---------------- Investment group (sequential) ----------------
    inv_group = evaluator.add_sequential(
        id=f"P{pid}_Investment_Group",
        desc=f"P{pid}: Investment checks",
        parent=proj_node,
        critical=True
    )
    inv_provided = evaluator.add_custom_node(
        result=nonempty(project.total_investment),
        id=f"P{pid}_Investment_Amount_Provided",
        desc=f"P{pid}: States the total investment amount.",
        parent=inv_group,
        critical=True
    )
    inv_threshold_node = evaluator.add_leaf(
        id=f"P{pid}_Investment_At_Least_500M",
        desc=f"P{pid}: Total project investment is ≥ $500 million.",
        parent=inv_group,
        critical=True
    )
    await evaluator.verify(
        claim="The total project investment is at least $500 million (USD).",
        node=inv_threshold_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Accept equivalent expressions like 'over $0.5 billion', '$700M', '$1.2 billion', or '>$500 million'."
    )

    # ---------------- Renewables group (sequential) ----------------
    ren_group = evaluator.add_sequential(
        id=f"P{pid}_Renewables_Group",
        desc=f"P{pid}: Renewable energy commitment checks",
        parent=proj_node,
        critical=True
    )
    ren_provided = evaluator.add_custom_node(
        result=nonempty(project.renewables_details),
        id=f"P{pid}_Renewables_Details_Provided",
        desc=f"P{pid}: Provides details of renewable energy commitment relevant to powering the facility.",
        parent=ren_group,
        critical=True
    )
    ren_commit_node = evaluator.add_leaf(
        id=f"P{pid}_Renewables_Public_Commitment",
        desc=f"P{pid}: Publicly documented commitment to power the facility with renewable energy.",
        parent=ren_group,
        critical=True
    )
    await evaluator.verify(
        claim="There is a publicly documented commitment to power this facility with renewable energy (e.g., solar, wind, or other renewables).",
        node=ren_commit_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Evidence could include PPAs, green tariffs, renewable energy goals explicitly tied to the site/campus/region, utility agreements, or operator statements that the facility will be powered by renewables."
    )

    # ---------------- Reference corroboration group (sequential) ----------------
    ref_group = evaluator.add_sequential(
        id=f"P{pid}_Reference_Group",
        desc=f"P{pid}: Reference corroboration",
        parent=proj_node,
        critical=True
    )
    refs_provided_node = evaluator.add_custom_node(
        result=(len(unique_urls(project.reference_urls)) > 0),
        id=f"P{pid}_Reference_URLs_Provided",
        desc=f"P{pid}: Provides at least one reference URL.",
        parent=ref_group,
        critical=True
    )
    ref_specs_node = evaluator.add_leaf(
        id=f"P{pid}_Reference_URL_Corroborates_Specs",
        desc=f"P{pid}: Reference URL corroborates key project specifications.",
        parent=ref_group,
        critical=True
    )
    loc_for_claim = build_location_string(project)
    await evaluator.verify(
        claim=(
            f"The provided source(s) corroborate that there is a hyperscale data center campus project in '{loc_for_claim}' "
            f"for operator '{project.operator or ''}' that was announced or started between 2023 and 2025, "
            f"with planned (or aggregate campus) capacity of at least 50 MW and total investment of at least $500 million, "
            f"and that there is a renewable energy commitment for powering the facility."
        ),
        node=ref_specs_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Any one strong source is sufficient. Look for explicit mentions of operator, location, timeframe (2023–2025), capacity (≥50 MW), investment (≥$500M), and a renewable energy commitment applicable to the facility."
    )

    # ---------------- Preferred (non-critical, parallel) ----------------
    preferred_group = evaluator.add_parallel(
        id=f"P{pid}_Preferred_Group",
        desc=f"P{pid}: Preferred (non-critical) attributes",
        parent=proj_node,
        critical=False
    )

    # PREFERRED: Jobs >= 100
    pref_jobs_node = evaluator.add_leaf(
        id=f"P{pid}_PREFERRED_Jobs_100plus",
        desc=f"P{pid}: Preferred – projected to create ≥ 100 permanent operational jobs.",
        parent=preferred_group,
        critical=False
    )
    await evaluator.verify(
        claim="The project is projected to create at least 100 permanent (non-construction) operational jobs.",
        node=pref_jobs_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Accept phrasing like '100+ jobs', 'over a hundred permanent jobs', or explicit counts ≥ 100. Exclude construction-only job counts."
    )

    # PREFERRED: Water-efficient cooling
    pref_water_node = evaluator.add_leaf(
        id=f"P{pid}_PREFERRED_Water_Efficient_Cooling",
        desc=f"P{pid}: Preferred – incorporates water-efficient cooling (air cooling, closed-loop, reclaimed water, etc.).",
        parent=preferred_group,
        critical=False
    )
    await evaluator.verify(
        claim="The project incorporates water-efficient cooling methods such as air cooling, closed-loop systems, or reclaimed/non-potable water usage.",
        node=pref_water_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Look for mentions of 'air-cooled', 'closed-loop', 'reclaimed water', 'greywater', 'non-potable water', or similar efficiency-focused cooling approaches."
    )

    # PREFERRED: Future expansion
    pref_expansion_node = evaluator.add_leaf(
        id=f"P{pid}_PREFERRED_Future_Expansion_Plans",
        desc=f"P{pid}: Preferred – documented future expansion phases beyond initial construction.",
        parent=preferred_group,
        critical=False
    )
    await evaluator.verify(
        claim="There are documented plans for future expansion phases beyond the initial construction.",
        node=pref_expansion_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Evidence includes multi-phase site plans, referenced future buildings/blocks, or expansion capacity/land reserved for later phases."
    )

    # PREFERRED: 5G carrier access
    pref_5g_node = evaluator.add_leaf(
        id=f"P{pid}_PREFERRED_5G_Carrier_Access",
        desc=f"P{pid}: Preferred – confirmed access to at least two of AT&T, T-Mobile, or Verizon 5G networks.",
        parent=preferred_group,
        critical=False
    )
    await evaluator.verify(
        claim="The site has confirmed access to at least two of AT&T, T-Mobile, or Verizon 5G networks.",
        node=pref_5g_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Only accept explicit confirmations or credible statements by the operator, carrier, utility, or authoritative authority; otherwise mark as not supported."
    )

    # PREFERRED: Incentives
    pref_incentives_node = evaluator.add_leaf(
        id=f"P{pid}_PREFERRED_Incentives",
        desc=f"P{pid}: Preferred – received state or local incentives or tax benefits.",
        parent=preferred_group,
        critical=False
    )
    await evaluator.verify(
        claim="The project received state or local economic development incentives or tax benefits.",
        node=pref_incentives_node,
        sources=unique_urls(project.reference_urls),
        additional_instruction="Evidence includes approvals or news articles describing tax abatements, grants, economic development packages, or similar incentives for the project."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root is non-critical to allow mixed critical/ non-critical children)
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

    # Extract projects from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction"
    )

    projects = extraction.projects if extraction and extraction.projects else []
    # Keep only first three items as required
    projects = projects[:3]

    # Global checks (critical)
    global_node = evaluator.add_parallel(
        id="Root_Task",
        desc="Identify three distinct US hyperscale data center campus development projects that meet all critical constraints and provide required fields and reference URL(s).",
        parent=root,
        critical=False
    )

    # Exactly three non-empty projects
    num_present = sum(1 for p in projects if is_project_non_empty(p))
    evaluator.add_custom_node(
        result=(num_present == 3),
        id="Global_Exactly_Three_Projects",
        desc="Response provides exactly three projects.",
        parent=global_node,
        critical=True
    )

    # Projects are distinct
    keys = [distinct_key(p) for p in projects if is_project_non_empty(p)]
    evaluator.add_custom_node(
        result=(len(keys) == len(set(keys)) and len(keys) == 3),
        id="Global_Projects_Are_Distinct",
        desc="The three projects are distinct development projects (not duplicates).",
        parent=global_node,
        critical=True
    )

    # Build three project subtrees
    # If fewer than 3 extracted, create empty placeholders so the tree is consistent
    while len(projects) < 3:
        projects.append(ProjectInfo())

    # Verify each project
    for i in range(3):
        await verify_project(evaluator, global_node, projects[i], i)

    return evaluator.get_summary()