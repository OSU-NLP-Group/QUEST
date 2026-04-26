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
TASK_ID = "identify_journalist_2009_awards_profile"
TASK_DESCRIPTION = (
    "Identify the journalist who meets all of the following criteria: Was born in 1961 during the month of May; "
    "Won a Pulitzer Prize in 2009 as part of a team for coverage of Pakistan and Afghanistan; "
    "Won the National Book Critics Circle Award for nonfiction in 2009 for a book about war; "
    "Has won the George Polk Award at least twice during their career; "
    "Began their major newspaper career at the Miami Herald; "
    "Worked as a correspondent for The New York Times, including serving as Baghdad correspondent from 2003-2006; "
    "Currently works as a staff writer for The New Yorker, having joined in 2011; "
    "Was a Nieman Fellow at Harvard University during the 2006-2007 academic year; "
    "Holds a Bachelor of Arts degree in political science from the University of Florida (awarded in 1983); "
    "Holds a Master of Philosophy degree in international relations from St Antony's College, Oxford. "
    "Provide the journalist's full name and supporting reference URLs for each major criterion."
)

# Common verification instruction when URLs are required
BASE_URL_VERIFY_INSTRUCTION = (
    "IMPORTANT: If no valid source URL is provided or the URLs are irrelevant/inaccessible, "
    "you MUST conclude the claim is NOT SUPPORTED. Base your decision strictly on the provided webpage content "
    "and screenshots. Allow reasonable phrasing variants (e.g., 'staff writer' vs 'Staff Writer'), "
    "but the key facts must be explicitly supported by the source."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BioInfo(BaseModel):
    full_name: Optional[str] = None
    birth_year: Optional[str] = None
    birth_month: Optional[str] = None
    birth_day: Optional[str] = None
    bio_sources: List[str] = Field(default_factory=list)


class PulitzerInfo(BaseModel):
    year: Optional[str] = None
    team_award: Optional[str] = None
    coverage_regions: Optional[str] = None
    organization: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NBCCInfo(BaseModel):
    year: Optional[str] = None
    category: Optional[str] = None
    book_title: Optional[str] = None
    book_about: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PolkInfo(BaseModel):
    times: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OPCInfo(BaseModel):
    times: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MiamiHeraldInfo(BaseModel):
    first_major_newspaper: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NYTInfo(BaseModel):
    role: Optional[str] = None
    start_month_year: Optional[str] = None
    baghdad_years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NewYorkerInfo(BaseModel):
    role: Optional[str] = None
    joined_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NiemanInfo(BaseModel):
    held: Optional[str] = None
    institution: Optional[str] = None
    academic_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CarrCenterInfo(BaseModel):
    held: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BAInfo(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MPhilInfo(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    college: Optional[str] = None
    university: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class JournalistExtraction(BaseModel):
    bio: Optional[BioInfo] = None
    pulitzer: Optional[PulitzerInfo] = None
    nbcc: Optional[NBCCInfo] = None
    polk: Optional[PolkInfo] = None
    opc: Optional[OPCInfo] = None
    miami_herald: Optional[MiamiHeraldInfo] = None
    nyt: Optional[NYTInfo] = None
    new_yorker: Optional[NewYorkerInfo] = None
    nieman: Optional[NiemanInfo] = None
    carr_center: Optional[CarrCenterInfo] = None
    ba: Optional[BAInfo] = None
    mphil: Optional[MPhilInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journalist_profile() -> str:
    return """
Extract the journalist's identity and all requested criteria explicitly from the provided answer text. Return the data in the following JSON schema. Extract only what is explicitly present in the answer; do not infer or add information. For each criterion, also extract the exact source URL(s) cited in the answer text that support it. If a required value is missing, set it to null; if no URL is cited for a criterion, return an empty array for its sources.

Return fields:
- bio:
  - full_name: Journalist's full name
  - birth_year
  - birth_month
  - birth_day
  - bio_sources: array of URLs that support date of birth or biography details
- pulitzer:
  - year
  - team_award: textual indicator that it was a team award (e.g., "team", "staff")
  - coverage_regions: e.g., "Pakistan and Afghanistan"
  - organization: e.g., "The New York Times"
  - sources: array of URLs supporting the Pulitzer details
- nbcc:
  - year
  - category: e.g., "nonfiction"
  - book_title: e.g., "The Forever War"
  - book_about: short phrase (e.g., "war")
  - sources: URLs supporting the NBCC details
- polk:
  - times: textual number or phrase indicating count (e.g., "twice", "2", "three")
  - sources: URLs supporting the George Polk Awards
- opc:
  - times: textual number for Overseas Press Club awards (if mentioned)
  - sources: URLs supporting OPC claim (if any)
- miami_herald:
  - first_major_newspaper: textual indicator that this was first major newspaper job
  - sources: URLs supporting Miami Herald early career
- nyt:
  - role: textual indicator that they worked as a correspondent
  - start_month_year: e.g., "September 2000"
  - baghdad_years: e.g., "2003-2006"
  - sources: URLs supporting New York Times employment
- new_yorker:
  - role: textual indicator that they are a staff writer
  - joined_year: e.g., "2011"
  - sources: URLs supporting The New Yorker details
- nieman:
  - held: textual indicator of holding a Nieman Fellowship
  - institution: e.g., "Harvard University"
  - academic_year: e.g., "2006-2007"
  - sources: URLs supporting Nieman details
- carr_center:
  - held: textual indicator of Carr Center fellowship (if mentioned)
  - institution: e.g., "Harvard University"
  - year: e.g., "2007-2008"
  - sources: URLs supporting Carr Center details (if any)
- ba:
  - degree: e.g., "Bachelor of Arts"
  - field: e.g., "political science"
  - institution: e.g., "University of Florida"
  - year: e.g., "1983"
  - sources: URLs supporting undergraduate degree
- mphil:
  - degree: e.g., "Master of Philosophy"
  - field: e.g., "international relations"
  - college: e.g., "St Antony's College"
  - university: e.g., "University of Oxford" or "Oxford"
  - sources: URLs supporting graduate degree

Special rules for URL extraction:
- Extract only URLs that are explicitly present in the answer text (including markdown links). Do not invent URLs.
- Include full URLs; if protocol is missing, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def name_or_placeholder(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the journalist"


def safe_sources(sources: Optional[List[str]]) -> List[str]:
    return sources or []


async def add_leaf_and_verify(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: List[str],
    critical: bool = True,
    add_ins_extra: Optional[str] = None,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    add_ins = BASE_URL_VERIFY_INSTRUCTION
    if add_ins_extra:
        add_ins = add_ins + " " + add_ins_extra
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
    )
    return node


# --------------------------------------------------------------------------- #
# Verification builders (subtrees)                                            #
# --------------------------------------------------------------------------- #
async def build_biographical_information(evaluator: Evaluator, root, data: JournalistExtraction):
    bio = data.bio or BioInfo()
    nm = name_or_placeholder(bio.full_name)
    sources = safe_sources(bio.bio_sources)

    node = evaluator.add_parallel(
        id="biographical_information",
        desc="The journalist's biographical details match the specified criteria",
        parent=root,
        critical=True,  # All children under this are critical
    )

    # Birth year 1961
    await add_leaf_and_verify(
        evaluator,
        node_id="birth_year_1961",
        desc="The journalist was born in 1961",
        parent=node,
        claim=f"{nm} was born in 1961.",
        sources=sources,
        critical=True,
        add_ins_extra="Accept if the page states 'May 24, 1961' or otherwise clearly shows the year 1961.",
    )

    # Birth month May
    await add_leaf_and_verify(
        evaluator,
        node_id="birth_month_may",
        desc="The journalist was born in May",
        parent=node,
        claim=f"{nm} was born in May.",
        sources=sources,
        critical=True,
        add_ins_extra="Accept if the page states 'May 24, 1961' or otherwise clearly shows the month May.",
    )

    # Birth date May 24
    await add_leaf_and_verify(
        evaluator,
        node_id="birth_date_may_24",
        desc="The journalist was born on May 24",
        parent=node,
        claim=f"{nm} was born on May 24.",
        sources=sources,
        critical=True,
        add_ins_extra="Accept if the source explicitly shows the day 24 (e.g., 'May 24, 1961').",
    )

    # Biographical source support
    await add_leaf_and_verify(
        evaluator,
        node_id="biographical_source",
        desc="Biographical information is supported by a verifiable source URL",
        parent=node,
        claim=f"The provided sources include at least one reputable page that states the date of birth for {nm}.",
        sources=sources,
        critical=True,
    )

    return node


async def build_career_achievements(evaluator: Evaluator, root, data: JournalistExtraction):
    nm = name_or_placeholder(data.bio.full_name if data.bio else None)

    node = evaluator.add_parallel(
        id="career_achievements_2009",
        desc="The journalist won multiple major awards in 2009",
        parent=root,
        critical=False  # Contains optional sub-criteria; will enforce critical children within
    )

    # Pulitzer subtree (critical)
    pul_node = evaluator.add_parallel(
        id="pulitzer_prize_2009",
        desc="Won a Pulitzer Prize in 2009 as part of a New York Times team for coverage of Pakistan and Afghanistan",
        parent=node,
        critical=True
    )
    pul = data.pulitzer or PulitzerInfo()
    pul_src = safe_sources(pul.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="pulitzer_year_2009",
        desc="The Pulitzer Prize was won in 2009",
        parent=pul_node,
        claim=f"{nm} won a Pulitzer Prize in 2009.",
        sources=pul_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="pulitzer_team_award",
        desc="The Pulitzer was a team award (part of a team)",
        parent=pul_node,
        claim=f"{nm} won the Pulitzer Prize as part of a team (e.g., staff award or team award).",
        sources=pul_src,
        critical=True,
        add_ins_extra="Accept if the page shows a team/staff Pulitzer recognition that includes the journalist.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="pulitzer_coverage_pakistan_afghanistan",
        desc="The Pulitzer was for coverage of Pakistan and Afghanistan",
        parent=pul_node,
        claim="The Pulitzer Prize recognized coverage of Pakistan and Afghanistan.",
        sources=pul_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="pulitzer_nyt_team",
        desc="The team was from The New York Times",
        parent=pul_node,
        claim="The Pulitzer-recognized team was from The New York Times.",
        sources=pul_src,
        critical=True,
        add_ins_extra="Accept 'The New York Times' or 'NYT' as equivalent.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="pulitzer_source",
        desc="Pulitzer Prize information is supported by a verifiable source URL",
        parent=pul_node,
        claim=f"At least one source clearly states {nm}'s 2009 Pulitzer team award for coverage of Pakistan and Afghanistan at The New York Times.",
        sources=pul_src,
        critical=True,
    )

    # NBCC subtree (critical)
    nbcc_node = evaluator.add_parallel(
        id="nbcc_award_2009",
        desc="Won the National Book Critics Circle Award for nonfiction in 2009",
        parent=node,
        critical=True
    )
    nbcc = data.nbcc or NBCCInfo()
    nbcc_src = safe_sources(nbcc.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="nbcc_year_2009",
        desc="The NBCC Award was won in 2009",
        parent=nbcc_node,
        claim=f"{nm} won the National Book Critics Circle Award in 2009.",
        sources=nbcc_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nbcc_category_nonfiction",
        desc="The award was in the nonfiction category",
        parent=nbcc_node,
        claim="The NBCC award was in the nonfiction category.",
        sources=nbcc_src,
        critical=True,
        add_ins_extra="Accept 'nonfiction' or 'non-fiction' as equivalent.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nbcc_book_forever_war",
        desc="The winning book was 'The Forever War'",
        parent=nbcc_node,
        claim="The winning book was 'The Forever War'.",
        sources=nbcc_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nbcc_book_about_war",
        desc="The book was about war",
        parent=nbcc_node,
        claim="The book 'The Forever War' is about war (specifically the wars in Iraq and/or Afghanistan).",
        sources=nbcc_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nbcc_source",
        desc="NBCC Award information is supported by a verifiable source URL",
        parent=nbcc_node,
        claim=f"At least one source clearly states {nm}'s 2009 NBCC nonfiction award for 'The Forever War'.",
        sources=nbcc_src,
        critical=True,
    )

    # George Polk Award multiple times (critical)
    polk_node = evaluator.add_parallel(
        id="george_polk_award_multiple",
        desc="Won the George Polk Award at least twice during career",
        parent=node,
        critical=True
    )
    polk = data.polk or PolkInfo()
    polk_src = safe_sources(polk.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="polk_award_minimum_two",
        desc="Won the George Polk Award at least two times",
        parent=polk_node,
        claim=f"{nm} has won the George Polk Award at least two times.",
        sources=polk_src,
        critical=True,
        add_ins_extra="If the page lists multiple Polk Awards totaling two or more, accept.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="polk_award_source",
        desc="George Polk Award information is supported by a verifiable source URL",
        parent=polk_node,
        claim=f"At least one source confirms {nm}'s multiple (>=2) George Polk Awards.",
        sources=polk_src,
        critical=True,
    )

    # Overseas Press Club (optional, non-critical)
    opc_node = evaluator.add_parallel(
        id="overseas_press_club_award",
        desc="Won Overseas Press Club Award multiple times",
        parent=node,
        critical=False
    )
    opc = data.opc or OPCInfo()
    opc_src = safe_sources(opc.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="opc_award_three_times",
        desc="Won the Overseas Press Club Award three times",
        parent=opc_node,
        claim=f"{nm} has won the Overseas Press Club Award three times.",
        sources=opc_src,
        critical=False,
        add_ins_extra="If exactly three awards are listed for the journalist, accept. If fewer or unclear, reject.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="opc_award_source",
        desc="Overseas Press Club Award information is supported by a verifiable source URL",
        parent=opc_node,
        claim=f"At least one source confirms {nm}'s Overseas Press Club Awards.",
        sources=opc_src,
        critical=False,
    )

    return node


async def build_institutional_affiliations(evaluator: Evaluator, root, data: JournalistExtraction):
    nm = name_or_placeholder(data.bio.full_name if data.bio else None)

    node = evaluator.add_parallel(
        id="institutional_affiliations",
        desc="The journalist has worked at specific major news organizations in a particular sequence",
        parent=root,
        critical=True
    )

    # Miami Herald early career (critical)
    mh_node = evaluator.add_parallel(
        id="miami_herald_early_career",
        desc="Began major newspaper career at the Miami Herald",
        parent=node,
        critical=True
    )
    mh = data.miami_herald or MiamiHeraldInfo()
    mh_src = safe_sources(mh.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="miami_herald_first_major",
        desc="Miami Herald was the first major newspaper position",
        parent=mh_node,
        claim=f"{nm} began their major newspaper career at the Miami Herald.",
        sources=mh_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="miami_herald_source",
        desc="Miami Herald employment is supported by a verifiable source URL",
        parent=mh_node,
        claim=f"At least one source confirms {nm}'s early major newspaper role at the Miami Herald.",
        sources=mh_src,
        critical=True,
    )

    # New York Times employment (critical)
    nyt_node = evaluator.add_parallel(
        id="new_york_times_employment",
        desc="Worked as a correspondent for The New York Times",
        parent=node,
        critical=True
    )
    nyt = data.nyt or NYTInfo()
    nyt_src = safe_sources(nyt.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="nyt_correspondent",
        desc="Worked as a correspondent for The New York Times",
        parent=nyt_node,
        claim=f"{nm} worked as a correspondent for The New York Times.",
        sources=nyt_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nyt_start_september_2000",
        desc="Joined The New York Times in September 2000",
        parent=nyt_node,
        claim=f"{nm} joined The New York Times in September 2000.",
        sources=nyt_src,
        critical=True,
        add_ins_extra="Accept reasonable date formats that clearly indicate September 2000.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nyt_baghdad_correspondent_2003_2006",
        desc="Served as Baghdad correspondent from 2003 to 2006",
        parent=nyt_node,
        claim=f"{nm} served as Baghdad correspondent from 2003 to 2006.",
        sources=nyt_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nyt_source",
        desc="New York Times employment is supported by a verifiable source URL",
        parent=nyt_node,
        claim=f"At least one source confirms {nm}'s New York Times employment (correspondent, start date, Baghdad 2003–2006).",
        sources=nyt_src,
        critical=True,
    )

    # The New Yorker (critical)
    newy_node = evaluator.add_parallel(
        id="new_yorker_current",
        desc="Currently works as a staff writer for The New Yorker",
        parent=node,
        critical=True
    )
    newy = data.new_yorker or NewYorkerInfo()
    newy_src = safe_sources(newy.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="new_yorker_staff_writer",
        desc="Works as a staff writer for The New Yorker",
        parent=newy_node,
        claim=f"{nm} currently works as a staff writer for The New Yorker.",
        sources=newy_src,
        critical=True,
        add_ins_extra="If the page indicates 'staff writer' status at The New Yorker, accept.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="new_yorker_joined_2011",
        desc="Joined The New Yorker in 2011",
        parent=newy_node,
        claim=f"{nm} joined The New Yorker in 2011.",
        sources=newy_src,
        critical=True,
        add_ins_extra="Accept if the source clearly says 'joined in 2011'.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="new_yorker_source",
        desc="New Yorker employment is supported by a verifiable source URL",
        parent=newy_node,
        claim=f"At least one source confirms {nm}'s New Yorker employment and joining year.",
        sources=newy_src,
        critical=True,
    )

    return node


async def build_fellowships(evaluator: Evaluator, root, data: JournalistExtraction):
    nm = name_or_placeholder(data.bio.full_name if data.bio else None)

    node = evaluator.add_parallel(
        id="fellowship_and_academic_affiliation",
        desc="The journalist held prestigious fellowships at Harvard University",
        parent=root,
        critical=False  # contains optional Carr Center block
    )

    # Nieman Fellowship (critical)
    niem_node = evaluator.add_parallel(
        id="nieman_fellowship",
        desc="Held a Nieman Fellowship at Harvard University during 2006-2007",
        parent=node,
        critical=True
    )
    niem = data.nieman or NiemanInfo()
    niem_src = safe_sources(niem.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="nieman_fellowship_held",
        desc="Held a Nieman Fellowship",
        parent=niem_node,
        claim=f"{nm} held a Nieman Fellowship.",
        sources=niem_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nieman_harvard_university",
        desc="The fellowship was at Harvard University",
        parent=niem_node,
        claim="The Nieman Fellowship was at Harvard University.",
        sources=niem_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nieman_year_2006_2007",
        desc="The Nieman Fellowship was during the 2006-2007 academic year",
        parent=niem_node,
        claim="The Nieman Fellowship took place during the 2006–2007 academic year.",
        sources=niem_src,
        critical=True,
        add_ins_extra="Accept reasonable date presentation like '2006-07'.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="nieman_source",
        desc="Nieman Fellowship information is supported by a verifiable source URL",
        parent=niem_node,
        claim=f"At least one source confirms {nm}'s Nieman Fellowship at Harvard during 2006–2007.",
        sources=niem_src,
        critical=True,
    )

    # Carr Center (optional)
    carr_node = evaluator.add_parallel(
        id="carr_center_fellowship",
        desc="Was a fellow at the Carr Center for Human Rights Policy at Harvard",
        parent=node,
        critical=False
    )
    carr = data.carr_center or CarrCenterInfo()
    carr_src = safe_sources(carr.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="carr_center_fellow",
        desc="Was a fellow at the Carr Center for Human Rights Policy",
        parent=carr_node,
        claim=f"{nm} was a fellow at the Carr Center for Human Rights Policy.",
        sources=carr_src,
        critical=False,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="carr_center_harvard",
        desc="The Carr Center is at Harvard",
        parent=carr_node,
        claim="The Carr Center for Human Rights Policy is at Harvard.",
        sources=carr_src,
        critical=False,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="carr_center_year_2007_2008",
        desc="The fellowship was in 2007-2008",
        parent=carr_node,
        claim="The Carr Center fellowship occurred in 2007–2008.",
        sources=carr_src,
        critical=False,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="carr_center_source",
        desc="Carr Center fellowship is supported by a verifiable source URL",
        parent=carr_node,
        claim=f"At least one source confirms {nm}'s Carr Center fellowship (2007–2008).",
        sources=carr_src,
        critical=False,
    )

    return node


async def build_education(evaluator: Evaluator, root, data: JournalistExtraction):
    nm = name_or_placeholder(data.bio.full_name if data.bio else None)

    node = evaluator.add_parallel(
        id="educational_background",
        desc="The journalist has specific undergraduate and graduate degrees",
        parent=root,
        critical=True
    )

    # Undergraduate degree (critical)
    ug_node = evaluator.add_parallel(
        id="undergraduate_degree",
        desc="Holds a Bachelor of Arts degree in political science from the University of Florida (1983)",
        parent=node,
        critical=True
    )
    ba = data.ba or BAInfo()
    ba_src = safe_sources(ba.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="ba_degree",
        desc="Holds a Bachelor of Arts (B.A.) degree",
        parent=ug_node,
        claim=f"{nm} holds a Bachelor of Arts (B.A.) degree.",
        sources=ba_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="ba_political_science",
        desc="The B.A. is in political science",
        parent=ug_node,
        claim="The B.A. degree is in political science.",
        sources=ba_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="ba_university_florida",
        desc="The B.A. was from the University of Florida",
        parent=ug_node,
        claim="The B.A. degree was from the University of Florida.",
        sources=ba_src,
        critical=True,
        add_ins_extra="Accept 'University of Florida' or 'UF' as equivalent if clearly referring to the university.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="ba_year_1983",
        desc="The B.A. was awarded in 1983",
        parent=ug_node,
        claim="The B.A. degree was awarded in 1983.",
        sources=ba_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="ba_source",
        desc="Undergraduate education is supported by a verifiable source URL",
        parent=ug_node,
        claim=f"At least one source confirms {nm}'s B.A. in political science from the University of Florida in 1983.",
        sources=ba_src,
        critical=True,
    )

    # Graduate degree (critical)
    grad_node = evaluator.add_parallel(
        id="graduate_degree",
        desc="Holds a Master of Philosophy degree in international relations from St Antony's College, Oxford",
        parent=node,
        critical=True
    )
    mph = data.mphil or MPhilInfo()
    mph_src = safe_sources(mph.sources)

    await add_leaf_and_verify(
        evaluator,
        node_id="mphil_degree",
        desc="Holds a Master of Philosophy (M.Phil.) degree",
        parent=grad_node,
        claim=f"{nm} holds a Master of Philosophy (M.Phil.) degree.",
        sources=mph_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="mphil_international_relations",
        desc="The M.Phil. is in international relations",
        parent=grad_node,
        claim="The M.Phil. is in international relations.",
        sources=mph_src,
        critical=True,
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="mphil_st_antonys_oxford",
        desc="The M.Phil. was from St Antony's College, Oxford",
        parent=grad_node,
        claim="The M.Phil. degree was from St Antony's College, Oxford (University of Oxford).",
        sources=mph_src,
        critical=True,
        add_ins_extra="Accept if the source clearly states St Antony's College, part of the University of Oxford.",
    )
    await add_leaf_and_verify(
        evaluator,
        node_id="mphil_source",
        desc="Graduate education is supported by a verifiable source URL",
        parent=grad_node,
        claim=f"At least one source confirms {nm}'s M.Phil. in international relations from St Antony's College, Oxford.",
        sources=mph_src,
        critical=True,
    )

    return node


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
    Build the verification tree and evaluate the answer according to the rubric.
    Notes on criticality:
    - Root is set to non-critical to allow inclusion of optional sub-criteria; a dedicated critical gate
      node ('mandatory_criteria_satisfied') is added to enforce that all mandatory categories pass.
    - Categories containing optional children are non-critical; their internal critical children are enforced
      within the category.
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
        default_model=model,
    )

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_journalist_profile(),
        template_class=JournalistExtraction,
        extraction_name="journalist_profile_extraction",
    )

    # 2) Build verification subtrees
    bio_node = await build_biographical_information(evaluator, root, extracted)
    achievements_node = await build_career_achievements(evaluator, root, extracted)
    inst_node = await build_institutional_affiliations(evaluator, root, extracted)
    fellows_node = await build_fellowships(evaluator, root, extracted)
    edu_node = await build_education(evaluator, root, extracted)

    # 3) Add a final mandatory gate node to enforce that all required categories pass
    # Mandatory categories to all pass:
    # - biographical_information (critical)
    # - career_achievements_2009 (non-critical parent, but has critical children; aggregated_score==1 means all required achievements passed)
    # - institutional_affiliations (critical)
    # - fellowship_and_academic_affiliation (non-critical parent with critical Nieman; aggregated_score==1 means Nieman passed)
    # - educational_background (critical)
    must_pass = [
        bio_node.aggregated_score == 1.0,
        achievements_node.aggregated_score == 1.0,
        inst_node.aggregated_score == 1.0,
        fellows_node.aggregated_score == 1.0,
        edu_node.aggregated_score == 1.0,
    ]
    all_mandatory_ok = all(must_pass)

    evaluator.add_custom_node(
        result=all_mandatory_ok,
        id="mandatory_criteria_satisfied",
        desc="All mandatory categories (bio, achievements, institutional affiliations, Nieman fellowship, education) are fully satisfied",
        parent=root,
        critical=True  # This enforces overall failure if mandatory categories are not all satisfied
    )

    # 4) Return the evaluation summary
    return evaluator.get_summary()