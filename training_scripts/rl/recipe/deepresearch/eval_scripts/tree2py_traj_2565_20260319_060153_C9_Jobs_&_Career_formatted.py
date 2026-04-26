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
TASK_ID = "career_transitions_2025_2026"
TASK_DESCRIPTION = """
Identify three individuals who made significant career transitions in higher education administration or athletics during the 2025-2026 period, based on the following specific criteria:

Individual 1:
- Born in 1967
- Holds a BA from Harvard University (earned in 1988), a JD from Yale Law School (earned in 1995), and a PhD from MIT (earned in 1999, specifically in history and sociology of science and technology)
- Served as dean of UCLA School of Law from August 2015 to June 2022
- Became chancellor of University of Wisconsin-Madison on August 4, 2022
- Was announced as the next president of Columbia University on January 25, 2026, with the presidency becoming effective on July 1, 2026
- Will serve as the 21st president of Columbia University

Individual 2:
- Born on December 2, 1978
- Played college football as a quarterback at Springfield College from 1997-2000
- Served as an assistant football coach at Yale University from 2012-2022
- Was hired as head football coach at Lehigh University on December 19, 2022
- Posted records of 2-9 (2023), 9-4 (2024), and 12-1 (2025) at Lehigh
- Won the Eddie Robinson Award as FCS Coach of the Year in 2025
- Was hired as head football coach at Yale on February 23, 2026, six days after the previous head coach Tony Reno resigned on February 17, 2026 due to health reasons

Individual 3:
- Born in 1970
- Played football at University of Georgia from 1988-1991
- Earned a BBA in finance from Georgia's Terry College of Business in 1992 and an M.Ed. in sports management from Georgia in 1994
- Served as University of Georgia athletic director from 2004-2010, resigning in 2010
- Served as University of Maryland athletic director from 2018-2025
- Was announced as SMU's director of athletics on March 21, 2025, and began the role on March 31, 2025
- Succeeded Rick Hart, who stepped down after the 2024-25 academic year

For each individual, provide:
1. Their full name (first and last name)
2. A brief description (2-3 sentences) of their career trajectory
3. At least two reference URLs that verify the key facts about their career transitions and credentials
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class I1Education(BaseModel):
    ba_institution: Optional[str] = None
    ba_year: Optional[str] = None
    jd_institution: Optional[str] = None
    jd_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    phd_field: Optional[str] = None
    birth_year: Optional[str] = None


class I1Deanship(BaseModel):
    ucla_dean_start: Optional[str] = None  # e.g., "August 2015"
    ucla_dean_end: Optional[str] = None    # e.g., "June 2022"


class I1Wisconsin(BaseModel):
    chancellor_appointment_date: Optional[str] = None  # e.g., "August 4, 2022"
    chancellor_position_number: Optional[str] = None   # e.g., "30th"


class I1Columbia(BaseModel):
    announcement_date: Optional[str] = None  # e.g., "January 25, 2026"
    effective_date: Optional[str] = None     # e.g., "July 1, 2026"
    president_position_number: Optional[str] = None  # e.g., "21st"


class Individual1(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    education: Optional[I1Education] = None
    deanship: Optional[I1Deanship] = None
    wisconsin: Optional[I1Wisconsin] = None
    columbia: Optional[I1Columbia] = None


class I2PlayingEdu(BaseModel):
    playing_school: Optional[str] = None     # "Springfield College"
    playing_position: Optional[str] = None   # "quarterback"
    playing_years: Optional[str] = None      # "1997-2000"
    birth_date: Optional[str] = None         # "December 2, 1978"


class I2YaleAssistant(BaseModel):
    start_year: Optional[str] = None  # "2012"
    end_year: Optional[str] = None    # "2022"


class I2LehighHead(BaseModel):
    hire_date: Optional[str] = None       # "December 19, 2022"
    record_2023: Optional[str] = None     # "2-9"
    record_2024: Optional[str] = None     # "9-4"
    record_2025: Optional[str] = None     # "12-1"


class I2Awards(BaseModel):
    eddie_robinson_award_year: Optional[str] = None  # "2025"


class I2YaleHead(BaseModel):
    reno_resignation_date: Optional[str] = None  # "February 17, 2026"
    hire_date: Optional[str] = None              # "February 23, 2026"


class Individual2(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    playing_edu: Optional[I2PlayingEdu] = None
    yale_assistant: Optional[I2YaleAssistant] = None
    lehigh_head: Optional[I2LehighHead] = None
    awards: Optional[I2Awards] = None
    yale_head: Optional[I2YaleHead] = None


class I3EducationPlay(BaseModel):
    birth_year: Optional[str] = None               # "1970"
    georgia_playing_years: Optional[str] = None    # "1988-1991"
    bba_year: Optional[str] = None                 # "1992"
    med_year: Optional[str] = None                 # "1994"


class I3GeorgiaAD(BaseModel):
    start_year: Optional[str] = None      # "2004"
    resignation_year: Optional[str] = None  # "2010"


class I3MarylandAD(BaseModel):
    start_year: Optional[str] = None  # "2018"
    end_year: Optional[str] = None    # "2025"


class I3SMUAD(BaseModel):
    announcement_date: Optional[str] = None  # "March 21, 2025"
    start_date: Optional[str] = None         # "March 31, 2025"
    predecessor_name: Optional[str] = None   # "Rick Hart"


class Individual3(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    education_play: Optional[I3EducationPlay] = None
    georgia_ad: Optional[I3GeorgiaAD] = None
    maryland_ad: Optional[I3MarylandAD] = None
    smu_ad: Optional[I3SMUAD] = None


class IndividualsExtraction(BaseModel):
    individual_1: Optional[Individual1] = None
    individual_2: Optional[Individual2] = None
    individual_3: Optional[Individual3] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_individuals() -> str:
    return """
Extract structured information about exactly three individuals described in the answer. The answer should include a name, a brief summary, and reference URLs for each individual. Additionally, extract the specific facts required for verification according to the schema below. Extract only what is explicitly present in the answer text.

Return a JSON object that strictly conforms to this schema (use null for any missing fields):

{
  "individual_1": {
    "name": string | null,
    "summary": string | null,
    "sources": string[] (URLs, can be empty),
    "education": {
      "ba_institution": string | null,
      "ba_year": string | null,
      "jd_institution": string | null,
      "jd_year": string | null,
      "phd_institution": string | null,
      "phd_year": string | null,
      "phd_field": string | null,
      "birth_year": string | null
    },
    "deanship": {
      "ucla_dean_start": string | null,   // e.g., "August 2015"
      "ucla_dean_end": string | null      // e.g., "June 2022"
    },
    "wisconsin": {
      "chancellor_appointment_date": string | null,  // e.g., "August 4, 2022"
      "chancellor_position_number": string | null    // e.g., "30th"
    },
    "columbia": {
      "announcement_date": string | null,            // e.g., "January 25, 2026"
      "effective_date": string | null,               // e.g., "July 1, 2026"
      "president_position_number": string | null     // e.g., "21st"
    }
  },
  "individual_2": {
    "name": string | null,
    "summary": string | null,
    "sources": string[] (URLs, can be empty),
    "playing_edu": {
      "playing_school": string | null,
      "playing_position": string | null,
      "playing_years": string | null,
      "birth_date": string | null
    },
    "yale_assistant": {
      "start_year": string | null,
      "end_year": string | null
    },
    "lehigh_head": {
      "hire_date": string | null,
      "record_2023": string | null,
      "record_2024": string | null,
      "record_2025": string | null
    },
    "awards": {
      "eddie_robinson_award_year": string | null
    },
    "yale_head": {
      "reno_resignation_date": string | null,
      "hire_date": string | null
    }
  },
  "individual_3": {
    "name": string | null,
    "summary": string | null,
    "sources": string[] (URLs, can be empty),
    "education_play": {
      "birth_year": string | null,
      "georgia_playing_years": string | null,
      "bba_year": string | null,
      "med_year": string | null
    },
    "georgia_ad": {
      "start_year": string | null,
      "resignation_year": string | null
    },
    "maryland_ad": {
      "start_year": string | null,
      "end_year": string | null
    },
    "smu_ad": {
      "announcement_date": string | null,
      "start_date": string | null,
      "predecessor_name": string | null
    }
  }
}

SPECIAL RULES:
- Extract only URLs explicitly present in the answer text (plain or markdown); do not invent any.
- If any field is not present in the answer, set it to null (or empty list for sources).
- Keep dates and years as strings exactly as written in the answer, even if formatted differently.
- For names, extract only the person’s name, not titles.

Return just the JSON.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def filter_urls(urls: List[str], keywords: List[str]) -> List[str]:
    """Filter URLs that contain any of the given keyword substrings (case-insensitive)."""
    if not urls:
        return []
    lowered = [u for u in urls if isinstance(u, str)]
    return [u for u in lowered if any(k.lower() in u.lower() for k in keywords)]


def has_min_sources(urls: List[str], min_count: int = 2) -> bool:
    return isinstance(urls, list) and len([u for u in urls if isinstance(u, str) and u.strip()]) >= min_count


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_individual_1(evaluator: Evaluator, parent_node, i1: Individual1):
    # Create main node for Individual 1 (non-critical to allow partial credit within this branch)
    i1_node = evaluator.add_parallel(
        id="Individual_1",
        desc="Identify the individual who transitioned from a university chancellor role to a university president role in 2026, with specific educational credentials and prior dean experience",
        parent=parent_node,
        critical=False
    )

    name = i1.name or ""
    srcs = i1.sources if i1 and i1.sources else []

    # Gating nodes (existence & minimum sources) - critical siblings to gate downstream verification
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="i1_name_present",
        desc="Individual 1 name is provided",
        parent=i1_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_min_sources(srcs, 2),
        id="i1_min_two_sources",
        desc="Individual 1 has at least two reference URLs",
        parent=i1_node,
        critical=True
    )

    # Education background (critical group; children leaves are critical)
    edu_node = evaluator.add_parallel(
        id="Education_Background_I1",
        desc="Verify the educational credentials including undergraduate, law, and doctoral degrees from specific institutions",
        parent=i1_node,
        critical=False  # Parent non-critical to allow mix of critical leaves
    )

    # Undergraduate degree
    leaf = evaluator.add_leaf(
        id="Undergraduate_Degree_I1",
        desc="Bachelor of Arts degree earned from Harvard University in 1988",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} earned a Bachelor of Arts (BA) degree from Harvard University in 1988."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow reasonable wording variants such as 'A.B.' or 'Bachelor's degree' that clearly indicate a BA from Harvard in 1988."
    )

    # Law degree
    leaf = evaluator.add_leaf(
        id="Law_Degree_I1",
        desc="Juris Doctor degree earned from Yale Law School in 1995",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} earned a Juris Doctor (JD) degree from Yale Law School in 1995."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Accept 'Yale Law School' and 'JD' or 'Juris Doctor' wording. Minor format differences are fine."
    )

    # Doctoral degree
    leaf = evaluator.add_leaf(
        id="Doctoral_Degree_I1",
        desc="PhD earned from Massachusetts Institute of Technology in 1999, specifically in history and sociology of science and technology",
        parent=edu_node,
        critical=True
    )
    phd_field = (i1.education.phd_field if (i1 and i1.education) else None) or "history and sociology of science and technology"
    claim = f"{name} earned a PhD from the Massachusetts Institute of Technology in 1999 in a field described as '{phd_field}' or equivalent wording."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow equivalent field phrasing such as 'history and social study of science and technology' if clearly equivalent."
    )

    # Birth year
    leaf = evaluator.add_leaf(
        id="Birth_Year_I1",
        desc="Individual was born in 1967",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} was born in 1967."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Minor discrepancies in exact birth date format are acceptable as long as the birth year is 1967."
    )

    # UCLA Law Deanship
    deanship_node = evaluator.add_parallel(
        id="UCLA_Law_Deanship_I1",
        desc="Served as dean of UCLA School of Law with specific start and end dates",
        parent=i1_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="UCLA_Dean_Start_Date_I1",
        desc="Appointment as dean became effective in August 2015",
        parent=deanship_node,
        critical=True
    )
    claim = f"{name}'s appointment as dean of the UCLA School of Law became effective in August 2015."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Accept equivalent statements indicating the deanship began in August 2015."
    )

    leaf = evaluator.add_leaf(
        id="UCLA_Dean_End_Date_I1",
        desc="Deanship concluded in June 2022",
        parent=deanship_node,
        critical=True
    )
    claim = f"{name}'s deanship at the UCLA School of Law concluded in June 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Accept 'stepped down', 'ended', or similar phrasing indicating June 2022 as end."
    )

    leaf = evaluator.add_leaf(
        id="UCLA_Dean_Duration_I1",
        desc="Served approximately 7 years as dean (2015-2022)",
        parent=deanship_node,
        critical=False
    )
    claim = f"{name} served approximately seven years as dean of UCLA Law from 2015 to 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Approximation language is acceptable if 2015–2022 tenure is clearly indicated."
    )

    # Wisconsin Chancellorship
    wisc_node = evaluator.add_parallel(
        id="Wisconsin_Chancellorship_I1",
        desc="Served as chancellor of University of Wisconsin-Madison with specific appointment date",
        parent=i1_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Wisconsin_Appointment_Date_I1",
        desc="Began role as UW-Madison chancellor on August 4, 2022",
        parent=wisc_node,
        critical=True
    )
    claim = f"{name} began as chancellor of the University of Wisconsin–Madison on August 4, 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow 'UW–Madison' and hyphen/emdash variations; the date should match Aug 4, 2022."
    )

    leaf = evaluator.add_leaf(
        id="Wisconsin_Position_Type_I1",
        desc="Held the position of 30th chancellor of UW-Madison",
        parent=wisc_node,
        critical=True
    )
    claim = f"{name} served as the 30th chancellor of the University of Wisconsin–Madison."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow '30th' spelled out or numeric; must clearly indicate the ordinal number."
    )

    # Columbia Presidency (sequential for announcement -> effective -> position number)
    col_node = evaluator.add_sequential(
        id="Columbia_Presidency_I1",
        desc="Appointed as president of Columbia University in 2026 with specific announcement and effective dates",
        parent=i1_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Columbia_Announcement_Date_I1",
        desc="Appointment was announced on January 25, 2026",
        parent=col_node,
        critical=True
    )
    claim = f"On January 25, 2026, it was announced that {name} would become the next president of Columbia University."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Prefer checking an official Columbia University page or widely reported announcement with the date Jan 25, 2026."
    )

    leaf = evaluator.add_leaf(
        id="Columbia_Effective_Date_I1",
        desc="Presidency becomes effective on July 1, 2026",
        parent=col_node,
        critical=True
    )
    claim = f"{name}'s Columbia University presidency becomes effective on July 1, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Check that the effective date is July 1, 2026 on the provided sources."
    )

    leaf = evaluator.add_leaf(
        id="Columbia_Position_Number_I1",
        desc="Will serve as the 21st president of Columbia University",
        parent=col_node,
        critical=True
    )
    claim = f"{name} will serve as the 21st president of Columbia University."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow '21st' in words or numerals; it must clearly indicate the ordinal number."
    )

    # Reference URLs presence checks (non-critical; ensure diversity of sources)
    ref_node = evaluator.add_parallel(
        id="Reference_URLs_I1",
        desc="Provide verifiable reference URLs for Individual 1's career information",
        parent=i1_node,
        critical=False
    )

    # Presence of a Wikipedia-like reference
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["wikipedia.org"])) >= 1,
        id="Wikipedia_Reference_I1",
        desc="Wikipedia page containing biographical and career information (presence check)",
        parent=ref_node,
        critical=False
    )
    # Presence of an official Columbia page
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["columbia.edu"])) >= 1,
        id="Columbia_Announcement_Reference_I1",
        desc="Official Columbia University announcement of the appointment (presence check)",
        parent=ref_node,
        critical=False
    )
    # Presence of a Wisconsin/uwmadison page (optional)
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["wisc.edu", "wisconsin.edu", "news.wisc.edu", "wisc"])) >= 1,
        id="Wisconsin_Reference_I1",
        desc="University of Wisconsin–Madison official page or announcement (presence check)",
        parent=ref_node,
        critical=False
    )


async def verify_individual_2(evaluator: Evaluator, parent_node, i2: Individual2):
    i2_node = evaluator.add_parallel(
        id="Individual_2",
        desc="Identify the individual who returned to Yale as head football coach in February 2026 after serving as Lehigh's head coach",
        parent=parent_node,
        critical=False
    )

    name = i2.name or ""
    srcs = i2.sources if i2 and i2.sources else []

    # Gating nodes
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="i2_name_present",
        desc="Individual 2 name is provided",
        parent=i2_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_min_sources(srcs, 2),
        id="i2_min_two_sources",
        desc="Individual 2 has at least two reference URLs",
        parent=i2_node,
        critical=True
    )

    # Education/Playing background
    edu_node = evaluator.add_parallel(
        id="Education_Background_I2",
        desc="Verify playing career and educational institution",
        parent=i2_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="College_Playing_Career_I2",
        desc="Played college football as quarterback at Springfield College from 1997-2000",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} played college football as a quarterback at Springfield College from 1997 to 2000."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Check for Springfield College football bio pages or Wikipedia/coaching bios confirming position and years."
    )

    leaf = evaluator.add_leaf(
        id="Birth_Date_I2",
        desc="Individual was born on December 2, 1978",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} was born on December 2, 1978."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Birth date verification should match or clearly support Dec 2, 1978."
    )

    # Yale assistant coaching
    yale_asst_node = evaluator.add_parallel(
        id="Yale_Assistant_Coaching_I2",
        desc="Served as assistant coach at Yale University from 2012-2022",
        parent=i2_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Yale_Assistant_Start_Year_I2",
        desc="Joined Yale coaching staff in 2012",
        parent=yale_asst_node,
        critical=True
    )
    claim = f"{name} joined the Yale University football coaching staff in 2012."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Look for coaching bios or news pages indicating 2012 as start year at Yale."
    )

    leaf = evaluator.add_leaf(
        id="Yale_Assistant_End_Year_I2",
        desc="Left Yale coaching staff in 2022",
        parent=yale_asst_node,
        critical=True
    )
    claim = f"{name} left the Yale football coaching staff in 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Look for transitions indicating departure in 2022."
    )

    leaf = evaluator.add_leaf(
        id="Yale_Assistant_Duration_I2",
        desc="Served 10 seasons (2012-2022) on Yale staff",
        parent=yale_asst_node,
        critical=False
    )
    claim = f"{name} served about 10 seasons on Yale's football staff from 2012 to 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Approximate duration is acceptable if years 2012–2022 are stated."
    )

    # Lehigh head coaching
    lehigh_node = evaluator.add_parallel(
        id="Lehigh_Head_Coaching_I2",
        desc="Served as Lehigh University head football coach from 2023-2025 with specific records",
        parent=i2_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Lehigh_Hire_Date_I2",
        desc="Hired as Lehigh head coach on December 19, 2022",
        parent=lehigh_node,
        critical=True
    )
    claim = f"{name} was hired as Lehigh University's head football coach on December 19, 2022."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Prefer an official Lehigh athletics release for the Dec 19, 2022 hire date."
    )

    leaf = evaluator.add_leaf(
        id="Lehigh_2023_Record_I2",
        desc="Posted a 2-9 record in first season (2023)",
        parent=lehigh_node,
        critical=True
    )
    claim = f"In the 2023 season, {name}'s Lehigh team had a 2-9 record."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Check Lehigh season summaries or coaching bios listing year-by-year records."
    )

    leaf = evaluator.add_leaf(
        id="Lehigh_2024_Record_I2",
        desc="Posted a 9-4 record in second season (2024)",
        parent=lehigh_node,
        critical=True
    )
    claim = f"In the 2024 season, {name}'s Lehigh team had a 9-4 record."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Verify on official athletics or reputable sources."
    )

    leaf = evaluator.add_leaf(
        id="Lehigh_2025_Record_I2",
        desc="Posted a 12-1 record in third season (2025)",
        parent=lehigh_node,
        critical=True
    )
    claim = f"In the 2025 season, {name}'s Lehigh team had a 12-1 record."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Verify on official athletics or reputable sources."
    )

    # Eddie Robinson Award
    award_node = evaluator.add_parallel(
        id="Eddie_Robinson_Award_I2",
        desc="Won the Eddie Robinson Award as FCS Coach of the Year in 2025",
        parent=i2_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Award_Year_I2",
        desc="Received the award in 2025",
        parent=award_node,
        critical=True
    )
    claim = f"{name} won the Eddie Robinson Award as FCS Coach of the Year in 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Look for STATS Perform or other official FCS award announcements for 2025."
    )

    leaf = evaluator.add_leaf(
        id="Award_Title_I2",
        desc="Award recognizes top coach in FCS football",
        parent=award_node,
        critical=True
    )
    claim = "The Eddie Robinson Award recognizes the top coach in NCAA Division I FCS football."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Standard definition of the Eddie Robinson Award on reliable sources is acceptable."
    )

    # Yale head coaching (sequential)
    yale_head_node = evaluator.add_sequential(
        id="Yale_Head_Coaching_I2",
        desc="Hired as Yale head football coach in February 2026, succeeding Tony Reno",
        parent=i2_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Yale_Predecessor_Resignation_I2",
        desc="Previous Yale head coach Tony Reno resigned on February 17, 2026 due to health reasons",
        parent=yale_head_node,
        critical=True
    )
    claim = "Tony Reno resigned as Yale head football coach on February 17, 2026 due to health reasons."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Prefer Yale athletics or university announcements referencing Tony Reno’s resignation date and reason."
    )

    leaf = evaluator.add_leaf(
        id="Yale_HC_Hire_Date_I2",
        desc="Hired as Yale head coach on February 23, 2026",
        parent=yale_head_node,
        critical=True
    )
    claim = f"{name} was hired as Yale's head football coach on February 23, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Prefer an official Yale announcement or credible press coverage with the date Feb 23, 2026."
    )

    leaf = evaluator.add_leaf(
        id="Yale_Return_Status_I2",
        desc="Returned to Yale after previously serving as assistant coach there",
        parent=yale_head_node,
        critical=False
    )
    claim = f"{name} returned to Yale as head coach after previously serving as an assistant coach there."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Cross-reference prior Yale assistant tenure to establish that this was a return."
    )

    # Reference URLs presence checks (non-critical)
    ref_node = evaluator.add_parallel(
        id="Reference_URLs_I2",
        desc="Provide verifiable reference URLs for Individual 2's career information",
        parent=i2_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["wikipedia.org"])) >= 1,
        id="Wikipedia_Reference_I2",
        desc="Wikipedia page containing biographical and coaching career information (presence check)",
        parent=ref_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["yale.edu", "yalebulldogs.com"])) >= 1,
        id="Yale_Announcement_Reference_I2",
        desc="Official Yale University/athletics announcement of the head coaching appointment (presence check)",
        parent=ref_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["lehigh", "lehighsports.com"])) >= 1,
        id="Lehigh_Reference_I2",
        desc="Lehigh University athletics page or announcement (presence check)",
        parent=ref_node,
        critical=False
    )


async def verify_individual_3(evaluator: Evaluator, parent_node, i3: Individual3):
    i3_node = evaluator.add_parallel(
        id="Individual_3",
        desc="Identify the individual who became SMU's athletic director in March 2025 after serving as Maryland's athletic director",
        parent=parent_node,
        critical=False
    )

    name = i3.name or ""
    srcs = i3.sources if i3 and i3.sources else []

    # Gating nodes
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="i3_name_present",
        desc="Individual 3 name is provided",
        parent=i3_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_min_sources(srcs, 2),
        id="i3_min_two_sources",
        desc="Individual 3 has at least two reference URLs",
        parent=i3_node,
        critical=True
    )

    # Education and playing
    edu_node = evaluator.add_parallel(
        id="Education_Background_I3",
        desc="Verify undergraduate and graduate degrees from University of Georgia",
        parent=i3_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Georgia_Playing_Career_I3",
        desc="Played football at University of Georgia from 1988-1991",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} played football at the University of Georgia from 1988 to 1991."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="UGA bios or Wikipedia pages that state player years are acceptable."
    )

    leaf = evaluator.add_leaf(
        id="Undergraduate_Degree_I3",
        desc="Earned BBA in finance from Georgia's Terry College of Business in 1992",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} earned a BBA in finance from the University of Georgia's Terry College of Business in 1992."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Equivalent wording for the degree and school is acceptable."
    )

    leaf = evaluator.add_leaf(
        id="Graduate_Degree_I3",
        desc="Earned M.Ed. in sports management from Georgia in 1994",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} earned an M.Ed. in sports management from the University of Georgia in 1994."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Accept minor variations like 'Master of Education' and similar."
    )

    leaf = evaluator.add_leaf(
        id="Birth_Year_I3",
        desc="Individual was born in 1970",
        parent=edu_node,
        critical=True
    )
    claim = f"{name} was born in 1970."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow focus on the birth year being 1970."
    )

    # Georgia AD
    uga_ad_node = evaluator.add_parallel(
        id="Georgia_Athletic_Director_I3",
        desc="Served as University of Georgia athletic director from 2004-2010",
        parent=i3_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Georgia_AD_Start_Year_I3",
        desc="Became Georgia athletic director in 2004",
        parent=uga_ad_node,
        critical=True
    )
    claim = f"{name} became the University of Georgia athletic director in 2004."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=""
    )

    leaf = evaluator.add_leaf(
        id="Georgia_AD_Resignation_I3",
        desc="Resigned from Georgia AD position in 2010 (served 2004-2010)",
        parent=uga_ad_node,
        critical=True
    )
    claim = f"{name} resigned from the Georgia athletic director position in 2010."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=""
    )

    # Maryland AD
    umd_ad_node = evaluator.add_parallel(
        id="Maryland_Athletic_Director_I3",
        desc="Served as University of Maryland athletic director from 2018-2025",
        parent=i3_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Maryland_AD_Start_Year_I3",
        desc="Became permanent Maryland athletic director in 2018 (after interim role starting in 2017)",
        parent=umd_ad_node,
        critical=True
    )
    claim = f"{name} became the permanent athletic director at the University of Maryland in 2018, after serving in an interim role starting in 2017."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Accept wording that clearly indicates interim in 2017 and permanent in 2018."
    )

    leaf = evaluator.add_leaf(
        id="Maryland_AD_End_Year_I3",
        desc="Concluded Maryland athletic director role in 2025",
        parent=umd_ad_node,
        critical=True
    )
    claim = f"{name} concluded the Maryland athletic director role in 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs
    )

    leaf = evaluator.add_leaf(
        id="Maryland_AD_Duration_I3",
        desc="Served approximately 7 years as Maryland AD (2018-2025)",
        parent=umd_ad_node,
        critical=False
    )
    claim = f"{name} served approximately seven years as Maryland's AD from 2018 to 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs
    )

    # SMU AD (sequential)
    smu_node = evaluator.add_sequential(
        id="SMU_Athletic_Director_I3",
        desc="Appointed as SMU's athletic director in March 2025 with specific announcement and start dates",
        parent=i3_node,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="SMU_Announcement_Date_I3",
        desc="Appointment as SMU athletic director was announced on March 21, 2025",
        parent=smu_node,
        critical=True
    )
    claim = f"{name} was announced as SMU's director of athletics on March 21, 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Prefer SMU official press releases or athletics site."
    )

    leaf = evaluator.add_leaf(
        id="SMU_Start_Date_I3",
        desc="Began role as SMU athletic director on March 31, 2025",
        parent=smu_node,
        critical=True
    )
    claim = f"{name} began the role as SMU athletic director on March 31, 2025."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs
    )

    leaf = evaluator.add_leaf(
        id="SMU_Predecessor_I3",
        desc="Succeeded Rick Hart, who stepped down after 2024-25 academic year",
        parent=smu_node,
        critical=True
    )
    predecessor = (i3.smu_ad.predecessor_name if (i3 and i3.smu_ad) else None) or "Rick Hart"
    claim = f"{name} succeeded {predecessor}, who stepped down after the 2024–25 academic year."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Allow hyphen variations in '2024–25'; focus on predecessor name and step-down timing."
    )

    # Reference URLs presence checks (non-critical)
    ref_node = evaluator.add_parallel(
        id="Reference_URLs_I3",
        desc="Provide verifiable reference URLs for Individual 3's career information",
        parent=i3_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["wikipedia.org"])) >= 1,
        id="Wikipedia_Reference_I3",
        desc="Wikipedia page containing biographical and career information (presence check)",
        parent=ref_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["smu.edu", "smumustangs.com"])) >= 1,
        id="SMU_Announcement_Reference_I3",
        desc="Official SMU announcement of the athletic director appointment (presence check)",
        parent=ref_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=len(filter_urls(srcs, ["umd.edu", "umterps.com"])) >= 1,
        id="Maryland_Reference_I3",
        desc="University of Maryland athletics page or announcement (presence check)",
        parent=ref_node,
        critical=False
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
    Evaluate an answer for the 2025–2026 higher-ed/athletics career transitions task.
    """
    # Initialize evaluator (root: parallel aggregation to allow independent branches)
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

    # Optional top-level node reflecting the task completion rubric
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Correctly identify all three individuals who made significant career transitions in higher education during 2025-2026 based on the provided constraints",
        parent=root,
        critical=False  # Set non-critical to allow mixed criticality of children (per framework constraints)
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_individuals(),
        template_class=IndividualsExtraction,
        extraction_name="individuals_extraction"
    )

    # Add a small info block about counts of sources provided
    eval_info = {
        "i1_sources_count": len(extraction.individual_1.sources) if extraction and extraction.individual_1 and extraction.individual_1.sources else 0,
        "i2_sources_count": len(extraction.individual_2.sources) if extraction and extraction.individual_2 and extraction.individual_2.sources else 0,
        "i3_sources_count": len(extraction.individual_3.sources) if extraction and extraction.individual_3 and extraction.individual_3.sources else 0,
    }
    evaluator.add_custom_info(eval_info, info_type="source_counts", info_name="per_individual_source_counts")

    # Build verification subtrees
    await verify_individual_1(evaluator, task_node, extraction.individual_1 or Individual1())
    await verify_individual_2(evaluator, task_node, extraction.individual_2 or Individual2())
    await verify_individual_3(evaluator, task_node, extraction.individual_3 or Individual3())

    # Return summary
    return evaluator.get_summary()