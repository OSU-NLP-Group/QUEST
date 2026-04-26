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
TASK_ID = "regional_public_research_universities"
TASK_DESCRIPTION = """
Identify three U.S. public research universities, one from each of the following regions, that meet the specified criteria for each region:

University 1 (Mountain West Region):
- Located in one of these states: Arizona, Colorado, Idaho, Montana, Nevada, New Mexico, Utah, or Wyoming
- Ranks in the top 50 U.S. institutions by total research and development (R&D) expenditures according to the National Science Foundation's Higher Education Research and Development (HERD) survey
- Received at least 5 NSF CAREER awards in 2024

University 2 (Southern Region):
- Located in one of these states: Alabama, Arkansas, Delaware, Florida, Georgia, Kentucky, Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, West Virginia, or the District of Columbia
- Appears in the Times Higher Education Interdisciplinary Science Rankings 2025 (announced in November 2024)
- Appears in the Top 100 U.S. Universities Granted Utility Patents list for either 2023 or 2024
- Has a ranked computer science graduate program according to U.S. News, QS, or similar established rankings

University 3 (Midwest Region):
- Located in one of these states: Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin
- Research expenditures increased between 2023 and 2024 according to publicly reported data
- Among the top institutions receiving National Science Foundation funding as of 2024
- Has substantial graduate research program enrollment (over 5,000 graduate students)

For each university, provide: (1) the official name, (2) the state where it is located, (3) confirmation of how it meets each specified criterion for its region, and (4) at least one reference URL supporting its qualification.
"""

MOUNTAIN_WEST_STATES = {
    "arizona", "az",
    "colorado", "co",
    "idaho", "id",
    "montana", "mt",
    "nevada", "nv",
    "new mexico", "nm",
    "utah", "ut",
    "wyoming", "wy"
}
SOUTHERN_STATES = {
    "alabama", "al",
    "arkansas", "ar",
    "delaware", "de",
    "florida", "fl",
    "georgia", "ga",
    "kentucky", "ky",
    "louisiana", "la",
    "maryland", "md",
    "mississippi", "ms",
    "north carolina", "nc",
    "oklahoma", "ok",
    "south carolina", "sc",
    "tennessee", "tn",
    "texas", "tx",
    "virginia", "va",
    "west virginia", "wv",
    "district of columbia", "dc", "d.c.", "washington, dc", "washington dc"
}
MIDWEST_STATES = {
    "illinois", "il",
    "indiana", "in",
    "iowa", "ia",
    "kansas", "ks",
    "michigan", "mi",
    "minnesota", "mn",
    "missouri", "mo",
    "nebraska", "ne",
    "north dakota", "nd",
    "ohio", "oh",
    "south dakota", "sd",
    "wisconsin", "wi"
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class U1Info(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    public_research_claim: Optional[str] = None
    public_research_sources: List[str] = Field(default_factory=list)
    herd_top50_claim: Optional[str] = None
    herd_sources: List[str] = Field(default_factory=list)
    nsf_career_2024_claim: Optional[str] = None
    career_sources: List[str] = Field(default_factory=list)


class U2Info(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    public_research_claim: Optional[str] = None
    public_research_sources: List[str] = Field(default_factory=list)
    the_interdisciplinary_2025_claim: Optional[str] = None
    the_interdisciplinary_sources: List[str] = Field(default_factory=list)
    top100_patents_claim: Optional[str] = None
    patents_sources: List[str] = Field(default_factory=list)
    cs_ranked_claim: Optional[str] = None
    cs_rank_sources: List[str] = Field(default_factory=list)


class U3Info(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    public_research_claim: Optional[str] = None
    public_research_sources: List[str] = Field(default_factory=list)
    research_expenditures_increase_claim: Optional[str] = None
    research_exp_sources: List[str] = Field(default_factory=list)
    nsf_top_funding_claim: Optional[str] = None
    nsf_top_sources: List[str] = Field(default_factory=list)
    grad_enrollment_claim: Optional[str] = None
    grad_enroll_sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    university1: Optional[U1Info] = None
    university2: Optional[U2Info] = None
    university3: Optional[U3Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract exactly three universities presented in the answer, corresponding to:
    - University 1 (Mountain West Region)
    - University 2 (Southern Region)
    - University 3 (Midwest Region)

    For each university, extract the following fields exactly as presented in the answer text (do not invent or infer values):
    Common fields:
    - name: Official university name (string)
    - state: The state (or DC) where the university is located (string)
    - general_sources: All URLs provided for this university in general (list of strings)
    - public_research_claim: Text snippet or statement in the answer indicating it is a U.S. public research university (string or null)
    - public_research_sources: URLs explicitly supporting the public research status (list, can be empty)

    University 1 specific fields:
    - herd_top50_claim: Statement confirming it ranks in Top 50 by total R&D expenditures per NSF HERD (string or null)
    - herd_sources: URLs cited for HERD Top 50 (list, can be empty)
    - nsf_career_2024_claim: Statement confirming at least 5 NSF CAREER awards in 2024 (string or null)
    - career_sources: URLs cited for CAREER awards claim (list, can be empty)

    University 2 specific fields:
    - the_interdisciplinary_2025_claim: Statement confirming it appears in Times Higher Education Interdisciplinary Science Rankings 2025 (string or null)
    - the_interdisciplinary_sources: URLs cited for the THE ranking (list, can be empty)
    - top100_patents_claim: Statement confirming it appears in the Top 100 U.S. Universities Granted Utility Patents list for 2023 or 2024 (string or null)
    - patents_sources: URLs cited for the Top 100 Utility Patents list (list, can be empty)
    - cs_ranked_claim: Statement confirming its CS graduate program is ranked (U.S. News, QS, etc.) (string or null)
    - cs_rank_sources: URLs cited for CS program ranking (list, can be empty)

    University 3 specific fields:
    - research_expenditures_increase_claim: Statement confirming research expenditures increased from 2023 to 2024 (string or null)
    - research_exp_sources: URLs cited for expenditures increase (list, can be empty)
    - nsf_top_funding_claim: Statement confirming it is among top NSF funding recipients as of 2024 (string or null)
    - nsf_top_sources: URLs cited for top NSF funding recipient status (list, can be empty)
    - grad_enrollment_claim: Statement confirming graduate enrollment over 5,000 (string or null)
    - grad_enroll_sources: URLs cited for graduate enrollment (list, can be empty)

    Rules:
    - Only extract URLs explicitly present in the answer (raw URLs or markdown links). If none, return an empty list.
    - If a field is not mentioned, return null (for strings) or [] (for lists).
    - Do not attempt to infer missing values. Keep the extraction faithful to the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    # Normalize DC variants
    if s in {"dc", "d.c.", "washington, dc", "washington dc"}:
        return "district of columbia"
    return s


def state_in_region(state: Optional[str], region_states: set) -> bool:
    s = normalize_state_name(state)
    if not s:
        return False
    # Direct match on full name
    if s in region_states:
        return True
    # Try matching by abbreviations if provided separately (already included in sets)
    return s in region_states


def unique_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university_1(
    evaluator: Evaluator,
    parent_node,
    u: Optional[U1Info]
) -> None:
    uni_node = evaluator.add_parallel(
        id="University_1",
        desc="Mountain West region university meeting all stated U1 criteria with required reporting fields",
        parent=parent_node,
        critical=False
    )

    name = u.name if u else None
    state = u.state if u else None

    # Official name provided
    evaluator.add_custom_node(
        result=(name is not None and name.strip() != ""),
        id="U1_Official_Name_Provided",
        desc="Response provides the university’s official name",
        parent=uni_node,
        critical=True
    )

    # U.S. public research university (verify with any available sources)
    u1_pub_node = evaluator.add_leaf(
        id="U1_Is_US_Public_Research_University",
        desc="University is a U.S. public research university (public institution in the United States with substantive research activity consistent with the query’s intent)",
        parent=uni_node,
        critical=True
    )
    pub_claim = f"{name} is a U.S. public research university."
    pub_sources = unique_sources(u.general_sources if u else [], u.public_research_sources if u else [])
    await evaluator.verify(
        claim=pub_claim,
        node=u1_pub_node,
        sources=pub_sources if pub_sources else None,
        additional_instruction="Verify that the institution is public and engages in substantial research (e.g., R1 status or similar official classification). Prefer authoritative sources such as the university’s official site or Carnegie classifications."
    )

    # State provided and within Mountain West list (custom check)
    evaluator.add_custom_node(
        result=state_in_region(state, MOUNTAIN_WEST_STATES),
        id="U1_State_Provided_And_In_Mountain_West_List",
        desc="Response provides the state and the state is one of: Arizona, Colorado, Idaho, Montana, Nevada, New Mexico, Utah, or Wyoming",
        parent=uni_node,
        critical=True
    )

    # Top50 HERD R&D
    herd_node = evaluator.add_leaf(
        id="U1_Top50_HERD_RnD_Expenditures",
        desc="Response confirms the university ranks in the top 50 U.S. institutions by total R&D expenditures per NSF HERD survey (and the claim is consistent with the cited evidence/source)",
        parent=uni_node,
        critical=True
    )
    herd_claim = f"{name} ranks in the top 50 U.S. institutions by total R&D expenditures per the NSF HERD survey."
    herd_sources = unique_sources(u.general_sources if u else [], u.herd_sources if u else [])
    await evaluator.verify(
        claim=herd_claim,
        node=herd_node,
        sources=herd_sources if herd_sources else None,
        additional_instruction="Confirm that the NSF HERD data or an authoritative summary shows this university in the top 50 by total R&D expenditures. The source must explicitly indicate the ranking or position."
    )

    # NSF CAREER awards in 2024 >= 5
    career_node = evaluator.add_leaf(
        id="U1_AtLeast5_NSF_CAREER_2024",
        desc="Response confirms the university received at least 5 NSF CAREER awards in 2024 (and the claim is consistent with the cited evidence/source)",
        parent=uni_node,
        critical=True
    )
    career_claim = f"{name} received at least 5 NSF CAREER awards in 2024."
    career_sources = unique_sources(u.general_sources if u else [], u.career_sources if u else [])
    await evaluator.verify(
        claim=career_claim,
        node=career_node,
        sources=career_sources if career_sources else None,
        additional_instruction="Verify via credible sources (e.g., NSF award database or official university announcements) that the number of NSF CAREER awards in calendar year 2024 is at least 5."
    )

    # At least one reference URL provided
    evaluator.add_custom_node(
        result=(len(unique_sources(u.general_sources if u else [])) > 0),
        id="U1_Reference_URLs",
        desc="Response provides at least one reference URL that supports the university’s qualification (i.e., substantiates one or more required criteria)",
        parent=uni_node,
        critical=True
    )


async def verify_university_2(
    evaluator: Evaluator,
    parent_node,
    u: Optional[U2Info]
) -> None:
    uni_node = evaluator.add_parallel(
        id="University_2",
        desc="Southern region university meeting all stated U2 criteria with required reporting fields",
        parent=parent_node,
        critical=False
    )

    name = u.name if u else None
    state = u.state if u else None

    # Official name provided
    evaluator.add_custom_node(
        result=(name is not None and name.strip() != ""),
        id="U2_Official_Name_Provided",
        desc="Response provides the university’s official name",
        parent=uni_node,
        critical=True
    )

    # U.S. public research university
    pub_node = evaluator.add_leaf(
        id="U2_Is_US_Public_Research_University",
        desc="University is a U.S. public research university (public institution in the United States with substantive research activity consistent with the query’s intent)",
        parent=uni_node,
        critical=True
    )
    pub_claim = f"{name} is a U.S. public research university."
    pub_sources = unique_sources(u.general_sources if u else [], u.public_research_sources if u else [])
    await evaluator.verify(
        claim=pub_claim,
        node=pub_node,
        sources=pub_sources if pub_sources else None,
        additional_instruction="Verify that it is a public U.S. institution and recognized for substantive research activity (e.g., Carnegie R1)."
    )

    # State provided and within Southern list
    evaluator.add_custom_node(
        result=state_in_region(state, SOUTHERN_STATES),
        id="U2_State_Provided_And_In_Southern_List",
        desc="Response provides the state (or DC) and it is one of: Alabama, Arkansas, Delaware, Florida, Georgia, Kentucky, Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, West Virginia, or the District of Columbia",
        parent=uni_node,
        critical=True
    )

    # THE Interdisciplinary Science Rankings 2025
    the_node = evaluator.add_leaf(
        id="U2_In_THE_Interdisciplinary_Science_2025",
        desc="Response confirms the university appears in the Times Higher Education Interdisciplinary Science Rankings 2025 (announced Nov 2024) and the claim is supported by cited evidence/source",
        parent=uni_node,
        critical=True
    )
    the_claim = f"{name} appears in the Times Higher Education Interdisciplinary Science Rankings 2025."
    the_sources = unique_sources(u.general_sources if u else [], u.the_interdisciplinary_sources if u else [])
    await evaluator.verify(
        claim=the_claim,
        node=the_node,
        sources=the_sources if the_sources else None,
        additional_instruction="Confirm that the Times Higher Education Interdisciplinary Science Rankings 2025 includes this institution by name."
    )

    # Top 100 Utility Patents 2023 or 2024
    patents_node = evaluator.add_leaf(
        id="U2_In_Top100_Utility_Patents_2023_or_2024",
        desc="Response confirms the university appears in the Top 100 U.S. Universities Granted Utility Patents list for 2023 or 2024 and the claim is supported by cited evidence/source",
        parent=uni_node,
        critical=True
    )
    patents_claim = f"{name} appears in the Top 100 U.S. Universities Granted Utility Patents list for 2023 or 2024."
    patents_sources = unique_sources(u.general_sources if u else [], u.patents_sources if u else [])
    await evaluator.verify(
        claim=patents_claim,
        node=patents_node,
        sources=patents_sources if patents_sources else None,
        additional_instruction="Check the official Top 100 U.S. Universities Granted Utility Patents list (typically from NAI) for either 2023 or 2024 to verify inclusion."
    )

    # Ranked CS graduate program
    cs_node = evaluator.add_leaf(
        id="U2_Ranked_CS_Graduate_Program",
        desc="Response confirms the university has a ranked computer science graduate program per U.S. News, QS, or similar established rankings and the claim is supported by cited evidence/source",
        parent=uni_node,
        critical=True
    )
    cs_claim = f"{name} has a ranked computer science graduate program according to major ranking organizations such as U.S. News or QS."
    cs_sources = unique_sources(u.general_sources if u else [], u.cs_rank_sources if u else [])
    await evaluator.verify(
        claim=cs_claim,
        node=cs_node,
        sources=cs_sources if cs_sources else None,
        additional_instruction="Confirm that at least one authoritative ranking (U.S. News, QS, THE, etc.) lists a ranked CS graduate program for this institution."
    )

    # At least one reference URL provided
    evaluator.add_custom_node(
        result=(len(unique_sources(u.general_sources if u else [])) > 0),
        id="U2_Reference_URLs",
        desc="Response provides at least one reference URL that supports the university’s qualification (i.e., substantiates one or more required criteria)",
        parent=uni_node,
        critical=True
    )


async def verify_university_3(
    evaluator: Evaluator,
    parent_node,
    u: Optional[U3Info]
) -> None:
    uni_node = evaluator.add_parallel(
        id="University_3",
        desc="Midwest region university meeting all stated U3 criteria with required reporting fields",
        parent=parent_node,
        critical=False
    )

    name = u.name if u else None
    state = u.state if u else None

    # Official name provided
    evaluator.add_custom_node(
        result=(name is not None and name.strip() != ""),
        id="U3_Official_Name_Provided",
        desc="Response provides the university’s official name",
        parent=uni_node,
        critical=True
    )

    # U.S. public research university
    pub_node = evaluator.add_leaf(
        id="U3_Is_US_Public_Research_University",
        desc="University is a U.S. public research university (public institution in the United States with substantive research activity consistent with the query’s intent)",
        parent=uni_node,
        critical=True
    )
    pub_claim = f"{name} is a U.S. public research university."
    pub_sources = unique_sources(u.general_sources if u else [], u.public_research_sources if u else [])
    await evaluator.verify(
        claim=pub_claim,
        node=pub_node,
        sources=pub_sources if pub_sources else None,
        additional_instruction="Verify that the institution is a public U.S. university with substantial research activity (e.g., Carnegie R1 or similar classification)."
    )

    # State provided and within Midwest list
    evaluator.add_custom_node(
        result=state_in_region(state, MIDWEST_STATES),
        id="U3_State_Provided_And_In_Midwest_List",
        desc="Response provides the state and the state is one of: Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin",
        parent=uni_node,
        critical=True
    )

    # Research expenditures increased 2023 -> 2024
    exp_node = evaluator.add_leaf(
        id="U3_Research_Expenditures_Increased_2023_to_2024",
        desc="Response confirms research expenditures increased between 2023 and 2024 per publicly reported data and the claim is supported by cited evidence/source",
        parent=uni_node,
        critical=True
    )
    exp_claim = f"{name} reported an increase in total research expenditures from 2023 to 2024."
    exp_sources = unique_sources(u.general_sources if u else [], u.research_exp_sources if u else [])
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=exp_sources if exp_sources else None,
        additional_instruction="Verify with authoritative data (e.g., university fact book, NSF HERD) that the total research expenditures increased between 2023 and 2024. If numbers are shown, ensure 2024 > 2023."
    )

    # Top NSF funding recipient (as of 2024)
    nsf_node = evaluator.add_leaf(
        id="U3_Top_NSF_Funding_Recipient_AsOf_2024",
        desc="Response confirms the university is identified by an authoritative 2024 source as being among the top NSF funding recipients (i.e., the source explicitly places the institution within a 'top' NSF funding recipient set/list/ranking) and provides supporting citation",
        parent=uni_node,
        critical=True
    )
    nsf_claim = f"As of 2024, {name} is among the top recipients of NSF funding."
    nsf_sources = unique_sources(u.general_sources if u else [], u.nsf_top_sources if u else [])
    await evaluator.verify(
        claim=nsf_claim,
        node=nsf_node,
        sources=nsf_sources if nsf_sources else None,
        additional_instruction="Confirm via an authoritative 2024 source (e.g., NSF data or reputable ranking/list) that the institution is explicitly included among top NSF funding recipients."
    )

    # Graduate enrollment over 5,000
    grad_node = evaluator.add_leaf(
        id="U3_Graduate_Enrollment_Over_5000",
        desc="Response confirms the university has over 5,000 graduate students and the claim is supported by cited evidence/source",
        parent=uni_node,
        critical=True
    )
    grad_claim = f"{name} has over 5,000 graduate students."
    grad_sources = unique_sources(u.general_sources if u else [], u.grad_enroll_sources if u else [])
    await evaluator.verify(
        claim=grad_claim,
        node=grad_node,
        sources=grad_sources if grad_sources else None,
        additional_instruction="Verify using institutional data (e.g., enrollment statistics, fact books) that total graduate student enrollment exceeds 5,000."
    )

    # At least one reference URL provided
    evaluator.add_custom_node(
        result=(len(unique_sources(u.general_sources if u else [])) > 0),
        id="U3_Reference_URLs",
        desc="Response provides at least one reference URL that supports the university’s qualification (i.e., substantiates one or more required criteria)",
        parent=uni_node,
        critical=True
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
    Evaluate an answer for identifying three U.S. public research universities across specified regions and criteria.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract structured information for the three universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Build verification subtrees for each university
    await verify_university_1(evaluator, root, extracted.university1)
    await verify_university_2(evaluator, root, extracted.university2)
    await verify_university_3(evaluator, root, extracted.university3)

    # Return summary
    return evaluator.get_summary()