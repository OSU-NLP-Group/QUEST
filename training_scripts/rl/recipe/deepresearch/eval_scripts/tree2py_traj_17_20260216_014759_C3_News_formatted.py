import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fed_chair_nomination_2026"
TASK_DESCRIPTION = """
On January 30, 2026, President Donald Trump announced the nomination of an individual to serve as Chairman of the Board of Governors of the Federal Reserve System, replacing Jerome Powell whose term as Chair expires on May 15, 2026. Identify this nominee and provide the following verified information about their background: (1) Their complete name; (2) The exact dates of their service as a member of the Federal Reserve Board of Governors (start and end dates); (3) Their age at the time of their Federal Reserve appointment in 2006, and confirmation that they were the youngest-ever appointee to the Federal Reserve Board of Governors in the institution's history; (4) Their employment history at Morgan Stanley, including the time period and their final position in the mergers and acquisitions department; (5) Their service in the George W. Bush White House, including the time period and their specific titles; (6) Their educational credentials, including both their undergraduate degree (institution and graduation year) and law degree (institution, graduation year, and any honors); (7) Their birthdate and birthplace. All information must be verifiable through official government sources, reputable news organizations, or the individual's professional profiles.
"""

# Optional Ground Truth (for reference only; not used in scoring)
GROUND_TRUTH = {
    "nominee_name": "Kevin A. Warsh",
    "fed_board_service": {"start": "February 24, 2006", "end": "March 31, 2011"},
    "age_at_appointment": "35",
    "youngest_ever": True,
    "morgan_stanley": {"start_year": "1995", "end_year": "2002", "location": "New York", "final_position": "Executive Director", "department": "Mergers and Acquisitions"},
    "white_house": {"start_year": "2002", "end_year": "2006", "titles": ["Special Assistant to the President for Economic Policy", "Executive Secretary of the National Economic Council"]},
    "undergrad": {"institution": "Stanford University", "degree": "Bachelor of Arts", "year": "1992"},
    "law": {"institution": "Harvard Law School", "degree": "Juris Doctor", "year": "1995", "honors": "cum laude"},
    "birth": {"date": "April 13, 1970", "place": "Albany, New York"},
    "crisis_roles": {"liaison_wall_street": True, "g20_rep": True},
    "post_fed": {"hoover_fellow": True, "gsb_lecturer": True},
    "nomination_context": {
        "date": "January 30, 2026",
        "announcer": "President Donald Trump",
        "role": "Chair (Chairman) of the Board of Governors of the Federal Reserve System",
        "replacement": "Jerome Powell",
        "powell_term_expiry": "May 15, 2026"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NomineeInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NominationContext(BaseModel):
    announcement_date: Optional[str] = None
    announcer: Optional[str] = None
    role: Optional[str] = None
    replacement: Optional[str] = None
    powell_term_expiry: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FedBoardService(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AppointmentClaims(BaseModel):
    appointment_year: Optional[str] = None
    age_at_appointment: Optional[str] = None
    youngest_ever_claim: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MorganStanleyEmployment(BaseModel):
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    location: Optional[str] = None
    final_position: Optional[str] = None
    department: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WhiteHouseService(BaseModel):
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    titles: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class UndergraduateCredentials(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    graduation_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LawSchoolCredentials(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    graduation_year: Optional[str] = None
    honors: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BirthInformation(BaseModel):
    birthdate: Optional[str] = None
    birthplace: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CrisisRoles(BaseModel):
    liaison_to_wall_street: Optional[str] = None
    g20_representative: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PostFedRoles(BaseModel):
    hoover_distinguished_visiting_fellow: Optional[str] = None
    stanford_gsb_lecturer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NomineeProfileExtraction(BaseModel):
    nominee: Optional[NomineeInfo] = None
    nomination_context: Optional[NominationContext] = None
    fed_board_service: Optional[FedBoardService] = None
    appointment_claims: Optional[AppointmentClaims] = None
    morgan_stanley: Optional[MorganStanleyEmployment] = None
    white_house: Optional[WhiteHouseService] = None
    undergraduate: Optional[UndergraduateCredentials] = None
    law_school: Optional[LawSchoolCredentials] = None
    birth_info: Optional[BirthInformation] = None
    crisis_roles: Optional[CrisisRoles] = None
    post_fed_roles: Optional[PostFedRoles] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_nominee_profile() -> str:
    return """
    Extract the nominee's identity and all requested background facts exactly as presented in the answer. For each section, also extract all cited source URLs mentioned in the answer text (plain URLs or markdown links). If a section has multiple relevant URLs, include all of them. If information is missing, use null; if no sources are cited for that section, return an empty array.

    Return a JSON object with the following structure and fields:

    {
      "nominee": {
        "name": string|null,
        "sources": string[]    // URLs supporting nomination/name identification
      },
      "nomination_context": {
        "announcement_date": string|null,     // e.g., "January 30, 2026"
        "announcer": string|null,             // e.g., "President Donald Trump"
        "role": string|null,                  // e.g., "Chairman of the Board of Governors of the Federal Reserve System"
        "replacement": string|null,           // e.g., "Jerome Powell"
        "powell_term_expiry": string|null,    // e.g., "May 15, 2026"
        "sources": string[]                   // URLs supporting nomination context
      },
      "fed_board_service": {
        "start_date": string|null,            // e.g., "February 24, 2006"
        "end_date": string|null,              // e.g., "March 31, 2011"
        "sources": string[]                   // URLs confirming FR Board service dates
      },
      "appointment_claims": {
        "appointment_year": string|null,      // usually "2006"
        "age_at_appointment": string|null,    // e.g., "35"
        "youngest_ever_claim": string|null,   // e.g., "youngest-ever appointee"
        "sources": string[]                   // URLs supporting age & youngest-ever claims
      },
      "morgan_stanley": {
        "start_year": string|null,            // e.g., "1995"
        "end_year": string|null,              // e.g., "2002"
        "location": string|null,              // e.g., "New York"
        "final_position": string|null,        // e.g., "Executive Director"
        "department": string|null,            // e.g., "mergers and acquisitions" or "M&A"
        "sources": string[]                   // URLs confirming MS employment details
      },
      "white_house": {
        "start_year": string|null,            // e.g., "2002"
        "end_year": string|null,              // e.g., "2006"
        "titles": string[],                   // e.g., ["Special Assistant to the President for Economic Policy", "Executive Secretary of the National Economic Council"]
        "sources": string[]                   // URLs confirming White House roles
      },
      "undergraduate": {
        "institution": string|null,           // e.g., "Stanford University"
        "degree": string|null,                // e.g., "Bachelor of Arts"
        "graduation_year": string|null,       // e.g., "1992"
        "sources": string[]                   // URLs confirming undergrad credentials
      },
      "law_school": {
        "institution": string|null,           // e.g., "Harvard Law School"
        "degree": string|null,                // e.g., "Juris Doctor"
        "graduation_year": string|null,       // e.g., "1995"
        "honors": string|null,                // e.g., "cum laude"
        "sources": string[]                   // URLs confirming law credentials
      },
      "birth_info": {
        "birthdate": string|null,             // e.g., "April 13, 1970"
        "birthplace": string|null,            // e.g., "Albany, New York"
        "sources": string[]                   // URLs confirming birth info
      },
      "crisis_roles": {
        "liaison_to_wall_street": string|null,// e.g., "primary liaison to Wall Street"
        "g20_representative": string|null,    // e.g., "Board’s representative to the G-20"
        "sources": string[]                   // URLs confirming crisis roles
      },
      "post_fed_roles": {
        "hoover_distinguished_visiting_fellow": string|null, // confirmation text if present
        "stanford_gsb_lecturer": string|null,                 // confirmation text if present
        "sources": string[]                   // URLs confirming post-Fed roles
      }
    }

    Special rules:
    - Extract sources as actual URLs only (if a markdown link is present, extract the destination URL).
    - Do not invent or infer any values; copy exactly from the answer.
    - If a URL is missing protocol, prepend "http://".
    - Preserve exact phrasing and dates as they appear in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if lst:
            for s in lst:
                if isinstance(s, str) and s.strip():
                    merged.append(s.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in merged:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


ACCEPTABLE_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com", "apnews.com", "ft.com",
    "washingtonpost.com", "cnbc.com", "cnn.com", "abcnews.go.com", "foxnews.com",
    "npr.org", "economist.com", "politico.com"
}
ACCEPTABLE_PROFILE_DOMAINS = {
    "stanford.edu", "gsb.stanford.edu", "hoover.org", "linkedin.com", "harvard.edu",
    "whitehouse.gov", "archives.gov", "georgewbush-whitehouse.archives.gov"
}


def _is_acceptable_source(url: str) -> bool:
    domain = _extract_domain(url)
    if not domain:
        return False
    # Government sites (.gov) and Federal Reserve official site
    if domain.endswith(".gov") or "federalreserve.gov" in domain:
        return True
    # Reputable news organizations
    if any(domain.endswith(d) or domain == d for d in ACCEPTABLE_NEWS_DOMAINS):
        return True
    # Professional or academic profiles
    if any(d in domain for d in ACCEPTABLE_PROFILE_DOMAINS):
        return True
    return False


def _section_has_acceptable_source(sources: List[str]) -> bool:
    return any(_is_acceptable_source(u) for u in sources)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_answer_quality_tree(evaluator: Evaluator, root, ex: NomineeProfileExtraction) -> None:
    """
    Build the rubric tree and perform verifications according to the provided extraction.
    """
    # Create the critical Answer Quality node
    answer_quality = evaluator.add_parallel(
        id="Answer_Quality_Rubric",
        desc="Identify the nominee and provide all required background facts with verifiable sourcing.",
        parent=root,
        critical=True
    )

    # Helper vars
    nominee_name = (ex.nominee.name if ex.nominee and ex.nominee.name else "").strip()
    # Combine core nomination-related sources for name and context
    name_sources = _merge_sources(
        ex.nominee.sources if ex.nominee else [],
        ex.nomination_context.sources if ex.nomination_context else []
    )

    # 1) Nominee Complete Name
    leaf = evaluator.add_leaf(
        id="Nominee_Complete_Name",
        desc="Provide the nominee’s complete name (sufficient to uniquely identify the nominated individual).",
        parent=answer_quality,
        critical=True
    )
    claim = f"The nominee announced on January 30, 2026 is {nominee_name}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=name_sources,
        additional_instruction="Verify the identity based on the cited sources. Allow minor variants in formatting (e.g., inclusion or omission of middle initial) to be considered equivalent."
    )

    # 2) Nomination Context
    nomination_context_node = evaluator.add_parallel(
        id="Nomination_Context",
        desc="Provide the required nomination context details.",
        parent=answer_quality,
        critical=True
    )
    nc_sources = ex.nomination_context.sources if ex.nomination_context else []

    # 2.1 Announcement Date and Announcer
    nc_leaf1 = evaluator.add_leaf(
        id="Nomination_Announcement_Date_and_Announcer",
        desc="State that the nomination was announced on January 30, 2026 by President Donald Trump.",
        parent=nomination_context_node,
        critical=True
    )
    nc_claim1 = "The nomination was announced on January 30, 2026 by President Donald Trump."
    await evaluator.verify(
        claim=nc_claim1,
        node=nc_leaf1,
        sources=nc_sources,
        additional_instruction="Confirm both the specific date (January 30, 2026) and the announcer (President Donald Trump)."
    )

    # 2.2 Nomination Role
    nc_leaf2 = evaluator.add_leaf(
        id="Nomination_Role",
        desc="State that the nomination was to serve as Chairman/Chair of the Board of Governors of the Federal Reserve System.",
        parent=nomination_context_node,
        critical=True
    )
    nc_claim2 = "The nomination was to serve as Chair (Chairman) of the Board of Governors of the Federal Reserve System."
    await evaluator.verify(
        claim=nc_claim2,
        node=nc_leaf2,
        sources=nc_sources,
        additional_instruction="Accept synonyms such as 'Chair' and 'Chairman'. Ensure the role is specifically for the Board of Governors of the Federal Reserve System."
    )

    # 2.3 Replacement Context
    nc_leaf3 = evaluator.add_leaf(
        id="Replacement_Context",
        desc="State that the nomination was to replace Jerome Powell, whose term as Chair expires on May 15, 2026.",
        parent=nomination_context_node,
        critical=True
    )
    nc_claim3 = "The nomination was to replace Jerome Powell, whose term as Chair expires on May 15, 2026."
    await evaluator.verify(
        claim=nc_claim3,
        node=nc_leaf3,
        sources=nc_sources,
        additional_instruction="Verify both that the nominee would replace Jerome Powell and that Powell’s term as Chair expires on May 15, 2026."
    )

    # 3) Fed Board Service
    fbs_node = evaluator.add_parallel(
        id="Fed_Board_Service",
        desc="Provide the nominee’s service dates as a member of the Federal Reserve Board of Governors.",
        parent=answer_quality,
        critical=True
    )
    fbs_leaf = evaluator.add_leaf(
        id="Fed_Board_Service_Dates",
        desc="State the service start and end dates: February 24, 2006 to March 31, 2011.",
        parent=fbs_node,
        critical=True
    )
    start_date = ex.fed_board_service.start_date if ex.fed_board_service else ""
    end_date = ex.fed_board_service.end_date if ex.fed_board_service else ""
    fbs_claim = f"{nominee_name} served as a member of the Federal Reserve Board of Governors from {start_date} to {end_date}."
    await evaluator.verify(
        claim=fbs_claim,
        node=fbs_leaf,
        sources=ex.fed_board_service.sources if ex.fed_board_service else [],
        additional_instruction="Verify the exact dates of service as a Governor on the Federal Reserve Board."
    )

    # 4) Fed Appointment Claims
    fac_node = evaluator.add_parallel(
        id="Fed_Appointment_Claims",
        desc="Provide the required appointment-age and historical-record claims.",
        parent=answer_quality,
        critical=True
    )
    # 4.1 Age at Appointment
    fac_leaf1 = evaluator.add_leaf(
        id="Fed_Appointment_Age",
        desc="State the nominee’s age at the time of their Federal Reserve appointment in 2006: 35.",
        parent=fac_node,
        critical=True
    )
    age_str = ex.appointment_claims.age_at_appointment if ex.appointment_claims else ""
    fac_claim1 = f"At the time of appointment in 2006, {nominee_name} was {age_str} years old."
    await evaluator.verify(
        claim=fac_claim1,
        node=fac_leaf1,
        sources=ex.appointment_claims.sources if ex.appointment_claims else [],
        additional_instruction="Confirm the age stated at the time of the 2006 Federal Reserve appointment. Allow sources that compute age from birthdate and appointment date."
    )

    # 4.2 Youngest-ever Confirmation
    fac_leaf2 = evaluator.add_leaf(
        id="Youngest_Ever_Confirmation",
        desc="Confirm the nominee was the youngest-ever appointee to the Federal Reserve Board of Governors in the institution’s history.",
        parent=fac_node,
        critical=True
    )
    fac_claim2 = f"{nominee_name} was the youngest-ever appointee to the Federal Reserve Board of Governors."
    await evaluator.verify(
        claim=fac_claim2,
        node=fac_leaf2,
        sources=ex.appointment_claims.sources if ex.appointment_claims else [],
        additional_instruction="Accept equivalent statements (e.g., youngest person ever appointed as a Governor). Verify the superlative claim via cited sources."
    )

    # 5) Morgan Stanley Employment
    ms_node = evaluator.add_parallel(
        id="Morgan_Stanley_Employment",
        desc="Provide the nominee’s Morgan Stanley employment history details.",
        parent=answer_quality,
        critical=True
    )
    ms_sources = ex.morgan_stanley.sources if ex.morgan_stanley else []

    # 5.1 Time Period
    ms_leaf1 = evaluator.add_leaf(
        id="Morgan_Stanley_Time_Period",
        desc="State the time period: 1995 to 2002.",
        parent=ms_node,
        critical=True
    )
    ms_claim1 = f"{nominee_name} worked at Morgan Stanley from {ex.morgan_stanley.start_year if ex.morgan_stanley else ''} to {ex.morgan_stanley.end_year if ex.morgan_stanley else ''}."
    await evaluator.verify(
        claim=ms_claim1,
        node=ms_leaf1,
        sources=ms_sources,
        additional_instruction="Verify the start and end years of employment at Morgan Stanley."
    )

    # 5.2 Location
    ms_leaf2 = evaluator.add_leaf(
        id="Morgan_Stanley_Location",
        desc="State the location: New York.",
        parent=ms_node,
        critical=True
    )
    ms_claim2 = f"This employment took place in {ex.morgan_stanley.location if ex.morgan_stanley else ''}."
    await evaluator.verify(
        claim=ms_claim2,
        node=ms_leaf2,
        sources=ms_sources,
        additional_instruction="Location may be stated as 'New York' or 'New York City'; treat these as equivalent."
    )

    # 5.3 Final Position and Group
    ms_leaf3 = evaluator.add_leaf(
        id="Morgan_Stanley_Final_Position_and_Group",
        desc="State the final position and department: Executive Director in mergers and acquisitions.",
        parent=ms_node,
        critical=True
    )
    ms_claim3 = f"The final position was {ex.morgan_stanley.final_position if ex.morgan_stanley else ''} in the {ex.morgan_stanley.department if ex.morgan_stanley else ''} department."
    await evaluator.verify(
        claim=ms_claim3,
        node=ms_leaf3,
        sources=ms_sources,
        additional_instruction="Accept synonyms for mergers and acquisitions such as 'M&A'. Verify seniority 'Executive Director'."
    )

    # 6) Bush White House Service
    wh_node = evaluator.add_parallel(
        id="Bush_White_House_Service",
        desc="Provide the nominee’s George W. Bush White House service details.",
        parent=answer_quality,
        critical=True
    )
    wh_sources = ex.white_house.sources if ex.white_house else []

    # 6.1 Time Period
    wh_leaf1 = evaluator.add_leaf(
        id="White_House_Time_Period",
        desc="State the time period: 2002 to 2006.",
        parent=wh_node,
        critical=True
    )
    wh_claim1 = f"{nominee_name} served in the George W. Bush White House from {ex.white_house.start_year if ex.white_house else ''} to {ex.white_house.end_year if ex.white_house else ''}."
    await evaluator.verify(
        claim=wh_claim1,
        node=wh_leaf1,
        sources=wh_sources,
        additional_instruction="Verify service span in the Bush administration."
    )

    # 6.2 Title: Special Assistant
    wh_leaf2 = evaluator.add_leaf(
        id="White_House_Title_Special_Assistant",
        desc="State the title: Special Assistant to the President for Economic Policy.",
        parent=wh_node,
        critical=True
    )
    wh_claim2 = "The nominee held the title 'Special Assistant to the President for Economic Policy' at the White House."
    await evaluator.verify(
        claim=wh_claim2,
        node=wh_leaf2,
        sources=wh_sources,
        additional_instruction="Verify specific title accuracy; minor variants of phrasing are acceptable."
    )

    # 6.3 Title: Exec Secretary NEC
    wh_leaf3 = evaluator.add_leaf(
        id="White_House_Title_Exec_Secretary_NEC",
        desc="State the title: Executive Secretary of the National Economic Council.",
        parent=wh_node,
        critical=True
    )
    wh_claim3 = "The nominee served as the Executive Secretary of the National Economic Council."
    await evaluator.verify(
        claim=wh_claim3,
        node=wh_leaf3,
        sources=wh_sources,
        additional_instruction="Verify title accuracy; synonyms or minor phrasing differences should be accepted."
    )

    # 7) Education
    edu_node = evaluator.add_parallel(
        id="Education",
        desc="Provide the nominee’s educational credentials.",
        parent=answer_quality,
        critical=True
    )

    # 7.1 Undergraduate Credentials
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Credentials",
        desc="Provide the undergraduate degree institution and graduation year.",
        parent=edu_node,
        critical=True
    )
    ug_sources = ex.undergraduate.sources if ex.undergraduate else []

    ug_leaf1 = evaluator.add_leaf(
        id="Undergrad_Institution_and_Degree",
        desc="State the undergraduate degree and institution: Bachelor of Arts from Stanford University.",
        parent=ug_node,
        critical=True
    )
    ug_claim1 = f"{nominee_name} earned a {ex.undergraduate.degree if ex.undergraduate else ''} from {ex.undergraduate.institution if ex.undergraduate else ''}."
    await evaluator.verify(
        claim=ug_claim1,
        node=ug_leaf1,
        sources=ug_sources,
        additional_instruction="Verify the exact degree name and the institution. Minor naming variants are acceptable."
    )

    ug_leaf2 = evaluator.add_leaf(
        id="Undergrad_Graduation_Year",
        desc="State the undergraduate graduation year: 1992.",
        parent=ug_node,
        critical=True
    )
    ug_claim2 = f"The undergraduate graduation year was {ex.undergraduate.graduation_year if ex.undergraduate else ''}."
    await evaluator.verify(
        claim=ug_claim2,
        node=ug_leaf2,
        sources=ug_sources,
        additional_instruction="Verify the specific year of undergraduate graduation."
    )

    # 7.2 Law School Credentials
    law_node = evaluator.add_parallel(
        id="Law_School_Credentials",
        desc="Provide the law degree institution, graduation year, and honors (if any).",
        parent=edu_node,
        critical=True
    )
    law_sources = ex.law_school.sources if ex.law_school else []

    law_leaf1 = evaluator.add_leaf(
        id="Law_Degree_and_Institution",
        desc="State the law degree and institution: Juris Doctor from Harvard Law School.",
        parent=law_node,
        critical=True
    )
    law_claim1 = f"{nominee_name} earned a {ex.law_school.degree if ex.law_school else ''} from {ex.law_school.institution if ex.law_school else ''}."
    await evaluator.verify(
        claim=law_claim1,
        node=law_leaf1,
        sources=law_sources,
        additional_instruction="Verify the exact law degree name and the institution."
    )

    law_leaf2 = evaluator.add_leaf(
        id="Law_Graduation_Year",
        desc="State the law school graduation year: 1995.",
        parent=law_node,
        critical=True
    )
    law_claim2 = f"The law school graduation year was {ex.law_school.graduation_year if ex.law_school else ''}."
    await evaluator.verify(
        claim=law_claim2,
        node=law_leaf2,
        sources=law_sources,
        additional_instruction="Verify the specific year of law school graduation."
    )

    law_leaf3 = evaluator.add_leaf(
        id="Law_Honors",
        desc="State the honors: cum laude.",
        parent=law_node,
        critical=True
    )
    law_claim3 = f"The law degree included honors: {ex.law_school.honors if ex.law_school else ''}."
    await evaluator.verify(
        claim=law_claim3,
        node=law_leaf3,
        sources=law_sources,
        additional_instruction="Verify any stated honors (e.g., 'cum laude')."
    )

    # 8) Birth Information
    birth_node = evaluator.add_parallel(
        id="Birth_Information",
        desc="Provide the nominee’s birthdate and birthplace.",
        parent=answer_quality,
        critical=True
    )
    birth_sources = ex.birth_info.sources if ex.birth_info else []

    birth_leaf1 = evaluator.add_leaf(
        id="Birthdate",
        desc="State the birthdate: April 13, 1970.",
        parent=birth_node,
        critical=True
    )
    birth_claim1 = f"{nominee_name} was born on {ex.birth_info.birthdate if ex.birth_info else ''}."
    await evaluator.verify(
        claim=birth_claim1,
        node=birth_leaf1,
        sources=birth_sources,
        additional_instruction="Verify the exact birthdate."
    )

    birth_leaf2 = evaluator.add_leaf(
        id="Birthplace",
        desc="State the birthplace: Albany, New York.",
        parent=birth_node,
        critical=True
    )
    birth_claim2 = f"The place of birth was {ex.birth_info.birthplace if ex.birth_info else ''}."
    await evaluator.verify(
        claim=birth_claim2,
        node=birth_leaf2,
        sources=birth_sources,
        additional_instruction="Verify the exact birthplace; minor variants (e.g., abbreviations like 'N.Y.') are acceptable."
    )

    # 9) 2008 Crisis Roles
    crisis_node = evaluator.add_parallel(
        id="2008_Crisis_Roles",
        desc="Provide the nominee’s roles during the 2008 financial crisis (as required).",
        parent=answer_quality,
        critical=True
    )
    crisis_sources = ex.crisis_roles.sources if ex.crisis_roles else []

    crisis_leaf1 = evaluator.add_leaf(
        id="Liaison_to_Wall_Street",
        desc="State that the nominee served as the Federal Reserve’s primary liaison to Wall Street.",
        parent=crisis_node,
        critical=True
    )
    crisis_claim1 = f"{nominee_name} served as the Federal Reserve’s primary liaison to Wall Street during the 2008 financial crisis."
    await evaluator.verify(
        claim=crisis_claim1,
        node=crisis_leaf1,
        sources=crisis_sources,
        additional_instruction="Verify the liaison role; allow phrasing variants (e.g., 'principal liaison')."
    )

    crisis_leaf2 = evaluator.add_leaf(
        id="G20_Representative",
        desc="State that the nominee served as the Board’s representative to the Group of Twenty (G-20).",
        parent=crisis_node,
        critical=True
    )
    crisis_claim2 = f"{nominee_name} served as the Board’s representative to the Group of Twenty (G-20)."
    await evaluator.verify(
        claim=crisis_claim2,
        node=crisis_leaf2,
        sources=crisis_sources,
        additional_instruction="Verify representation to the G-20; minor variants are acceptable."
    )

    # 10) Post-Fed Roles After 2011
    post_node = evaluator.add_parallel(
        id="Post_Fed_Roles_After_2011",
        desc="Provide the nominee’s roles after departing the Fed in 2011 (as required).",
        parent=answer_quality,
        critical=True
    )
    post_sources = ex.post_fed_roles.sources if ex.post_fed_roles else []

    post_leaf1 = evaluator.add_leaf(
        id="Hoover_Distinguished_Visiting_Fellow",
        desc="State that the nominee became a Distinguished Visiting Fellow at Stanford University’s Hoover Institution.",
        parent=post_node,
        critical=True
    )
    post_claim1 = f"After departing the Fed in 2011, {nominee_name} became a Distinguished Visiting Fellow at Stanford University’s Hoover Institution."
    await evaluator.verify(
        claim=post_claim1,
        node=post_leaf1,
        sources=post_sources,
        additional_instruction="Verify the Hoover Institution fellowship status and timing after departing the Fed."
    )

    post_leaf2 = evaluator.add_leaf(
        id="Stanford_GSB_Lecturer",
        desc="State that the nominee became a lecturer at the Stanford Graduate School of Business.",
        parent=post_node,
        critical=True
    )
    post_claim2 = f"After departing the Fed in 2011, {nominee_name} became a lecturer at the Stanford Graduate School of Business."
    await evaluator.verify(
        claim=post_claim2,
        node=post_leaf2,
        sources=post_sources,
        additional_instruction="Verify lecturer status at Stanford GSB; timing after departing the Fed."
    )

    # 11) Source Verifiability Compliance (Custom binary node)
    # Check that every section has at least one acceptable source type.
    sections_and_sources: List[Tuple[str, List[str]]] = [
        ("nominee", ex.nominee.sources if ex.nominee else []),
        ("nomination_context", ex.nomination_context.sources if ex.nomination_context else []),
        ("fed_board_service", ex.fed_board_service.sources if ex.fed_board_service else []),
        ("appointment_claims", ex.appointment_claims.sources if ex.appointment_claims else []),
        ("morgan_stanley", ex.morgan_stanley.sources if ex.morgan_stanley else []),
        ("white_house", ex.white_house.sources if ex.white_house else []),
        ("undergraduate", ex.undergraduate.sources if ex.undergraduate else []),
        ("law_school", ex.law_school.sources if ex.law_school else []),
        ("birth_info", ex.birth_info.sources if ex.birth_info else []),
        ("crisis_roles", ex.crisis_roles.sources if ex.crisis_roles else []),
        ("post_fed_roles", ex.post_fed_roles.sources if ex.post_fed_roles else []),
    ]

    compliance_per_section: Dict[str, bool] = {
        sec: _section_has_acceptable_source(srcs) for sec, srcs in sections_and_sources
    }
    compliance_ok = all(compliance_per_section.values()) and sum(len(srcs) for _, srcs in sections_and_sources) > 0

    # Record compliance details for transparency
    evaluator.add_custom_info(
        info={"compliance_per_section": compliance_per_section},
        info_type="compliance_details",
        info_name="source_type_compliance"
    )

    evaluator.add_custom_node(
        result=compliance_ok,
        id="Source_Verifiability_Compliance",
        desc="All required facts are supported by citations from at least one acceptable source type (official government sources, reputable news organizations, or the individual’s professional profiles).",
        parent=answer_quality,
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
    Evaluate an answer for the 2026 Federal Reserve Chair nomination task.
    """
    # Initialize evaluator
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

    # Extract structured nominee profile and sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_nominee_profile(),
        template_class=NomineeProfileExtraction,
        extraction_name="nominee_profile",
    )

    # Add optional ground truth for reference
    evaluator.add_ground_truth({"expected": GROUND_TRUTH}, gt_type="ground_truth")

    # Build rubric tree and run verifications
    await build_answer_quality_tree(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()