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
TASK_ID = "astro_person_identification"
TASK_DESCRIPTION = """
Identify the full name of the astrophysicist who meets all of the following criteria:

Educational Background:
- Earned a Bachelor of Science degree in Astronomy and Physics from Yale University in 1994
- Earned a Ph.D. in Astronomy from Harvard University in 2000, with doctoral advisors Giovanni Fazio and Lee Hartmann

Career Positions:
- Held a Miller Research Fellowship at the University of California, Berkeley from 2000-2002
- Was appointed Canada Research Chair in Observational Astrophysics at the University of Toronto in 2008
- Served as Dean of the Faculty of Science at York University from 2014 to 2018
- Became the Harold Tanner Dean of the College of Arts and Sciences at Cornell University in 2018
- Was named Hans Bethe Professor at Cornell University in 2022
- Was appointed Provost of Johns Hopkins University, effective October 15, 2023
- Was appointed as the 10th President of the California Institute of Technology (Caltech), with the appointment effective July 1, 2026

Publications:
- Authored a book titled "Strange New Worlds" published in 2011, which was a finalist for the Lane Anderson Award
- Authored a book titled "Neutrino Hunters" that won the Canadian Science Writers Association Book Award in 2014
- Authored a children's picture book titled "Child of the Universe" published on March 17, 2020

Awards and Honors:
- Received the E.W.R. Steacie Memorial Fellowship in 2009
- Received the Carl Sagan Medal for Excellence in Public Communication in Planetary Science in 2020

Provide the person's full name (first and last name).
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class EducationYale(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    majors: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EducationHarvard(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    year: Optional[str] = None
    advisors: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class CareerMiller(BaseModel):
    fellowship_name: Optional[str] = None
    institution: Optional[str] = None
    period: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerCRC(BaseModel):
    chair_title: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerYork(BaseModel):
    position: Optional[str] = None
    institution: Optional[str] = None
    period: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerCornell(BaseModel):
    position: Optional[str] = None
    institution: Optional[str] = None
    start_year: Optional[str] = None
    bethe_professor_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerJHU(BaseModel):
    position: Optional[str] = None
    institution: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerCaltech(BaseModel):
    position: Optional[str] = None
    institution: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookSNW(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    recognition: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookNH(BaseModel):
    title: Optional[str] = None
    award: Optional[str] = None
    award_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookCOTU(BaseModel):
    title: Optional[str] = None
    publication_date: Optional[str] = None
    book_type: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AwardSteacie(BaseModel):
    name: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AwardSagan(BaseModel):
    name: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PersonExtraction(BaseModel):
    full_name: Optional[str] = None

    education_yale: Optional[EducationYale] = None
    education_harvard: Optional[EducationHarvard] = None

    miller_fellowship: Optional[CareerMiller] = None
    crc_toronto: Optional[CareerCRC] = None
    york_deanship: Optional[CareerYork] = None
    cornell_deanship: Optional[CareerCornell] = None
    jhu_provost: Optional[CareerJHU] = None
    caltech_presidency: Optional[CareerCaltech] = None

    snw_book: Optional[BookSNW] = None
    nh_book: Optional[BookNH] = None
    cotu_book: Optional[BookCOTU] = None

    steacie: Optional[AwardSteacie] = None
    sagan: Optional[AwardSagan] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person() -> str:
    return """
    Extract the identified person's full name and the detailed facts asserted in the answer, organized into the following JSON schema. Very important: also extract the specific URLs (as sources) that the answer cites for each fact group. Extract only URLs that are explicitly present in the answer text.

    Required JSON fields:
    - full_name: The person's full name (first and last name; include middle names/initials if present in the answer).

    - education_yale: {
        institution: the undergraduate institution string (e.g., "Yale University"),
        degree: the degree name (e.g., "B.S.", "Bachelor of Science"),
        majors: the majors as written (e.g., "Astronomy and Physics"),
        year: the graduation year string (e.g., "1994"),
        sources: [URLs that support the Yale undergraduate details]
      }

    - education_harvard: {
        institution: (e.g., "Harvard University"),
        degree: (e.g., "Ph.D."),
        field: (e.g., "Astronomy"),
        year: (e.g., "2000"),
        advisors: list of doctoral advisors' names as written in the answer,
        sources: [URLs that support the Harvard Ph.D. details]
      }

    - miller_fellowship: {
        fellowship_name: (e.g., "Miller Research Fellowship"),
        institution: (e.g., "University of California, Berkeley" or "UC Berkeley"),
        period: the overall period string as written (e.g., "2000-2002"),
        start_year: if available,
        end_year: if available,
        sources: [URLs that support the fellowship]
      }

    - crc_toronto: {
        chair_title: (e.g., "Canada Research Chair in Observational Astrophysics"),
        institution: (e.g., "University of Toronto"),
        year: (e.g., "2008"),
        sources: [URLs that support this appointment]
      }

    - york_deanship: {
        position: (e.g., "Dean of the Faculty of Science"),
        institution: (e.g., "York University"),
        period: (e.g., "2014 to 2018"),
        start_year: if available (e.g., "2014"),
        end_year: if available (e.g., "2018"),
        sources: [URLs that support this deanship]
      }

    - cornell_deanship: {
        position: (e.g., "Harold Tanner Dean of the College of Arts and Sciences"),
        institution: (e.g., "Cornell University"),
        start_year: (e.g., "2018"),
        bethe_professor_year: (e.g., "2022"),
        sources: [URLs that support Cornell info]
      }

    - jhu_provost: {
        position: (e.g., "Provost"),
        institution: (e.g., "Johns Hopkins University"),
        effective_date: (e.g., "October 15, 2023"),
        sources: [URLs that support the JHU provost appointment]
      }

    - caltech_presidency: {
        position: (e.g., "10th President"),
        institution: (e.g., "California Institute of Technology" or "Caltech"),
        effective_date: (e.g., "July 1, 2026"),
        sources: [URLs that support the Caltech presidency]
      }

    - snw_book: {
        title: (e.g., "Strange New Worlds"),
        year: (e.g., "2011"),
        recognition: any mention like "finalist for the Lane Anderson Award",
        sources: [URLs that support this book and recognition]
      }

    - nh_book: {
        title: (e.g., "Neutrino Hunters"),
        award: (e.g., "Canadian Science Writers Association Book Award"),
        award_year: (e.g., "2014"),
        sources: [URLs that support the book and the award]
      }

    - cotu_book: {
        title: (e.g., "Child of the Universe"),
        publication_date: (e.g., "March 17, 2020"),
        book_type: (e.g., "children's picture book"),
        sources: [URLs that support the book details]
      }

    - steacie: {
        name: (e.g., "E.W.R. Steacie Memorial Fellowship"),
        year: (e.g., "2009"),
        sources: [URLs that support this award]
      }

    - sagan: {
        name: (e.g., "Carl Sagan Medal for Excellence in Public Communication in Planetary Science"),
        year: (e.g., "2020"),
        sources: [URLs that support this medal]
      }

    Rules:
    - Extract exactly what is written in the answer; do not invent any information.
    - If a requested subfield is not mentioned, set it to null; for arrays, use an empty array.
    - For URL fields, include only valid URLs that are explicitly present in the answer text. Do not fabricate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


GENERAL_VERIFY_INSTRUCTION = (
    "Focus on whether the webpage explicitly supports the stated fact for the named person. "
    "Allow minor paraphrasing, abbreviations, and reasonable variants in naming or formatting "
    "(e.g., 'UC Berkeley' vs 'University of California, Berkeley'; 'Ph.D.' vs 'PhD'; "
    "date formats like '2000–02' vs '2000-2002'; 'Oct 15, 2023' vs 'October 15, 2023'). "
    "Match should be robust to small stylistic differences but must reflect the same real-world fact."
)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_education_branch(evaluator: Evaluator, parent, data: PersonExtraction) -> None:
    full_name = data.full_name or "the person"
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="The person's educational background matches the specified undergraduate and doctoral credentials",
        parent=parent,
        critical=True,
    )

    # Undergraduate (Yale)
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Degree",
        desc="The person earned a B.S. in Astronomy and Physics from Yale University in 1994",
        parent=edu_node,
        critical=True,
    )
    yale = data.education_yale or EducationYale()
    yale_sources = ensure_list(yale.sources)

    # Gate: sources provided for Yale
    evaluator.add_custom_node(
        result=len(yale_sources) > 0,
        id="Yale_Sources_Provided",
        desc="At least one cited source is provided for Yale undergraduate details",
        parent=ug_node,
        critical=True,
    )

    # Institution
    node = evaluator.add_leaf(
        id="Yale_BS_Institution",
        desc="The undergraduate degree was from Yale University",
        parent=ug_node,
        critical=True,
    )
    claim = f"{full_name} earned an undergraduate bachelor's degree from Yale University."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=yale_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Year
    node = evaluator.add_leaf(
        id="Yale_BS_Year",
        desc="The undergraduate degree was awarded in 1994",
        parent=ug_node,
        critical=True,
    )
    year_text = yale.year or ""
    claim = f"{full_name}'s Yale undergraduate degree was awarded in {year_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=yale_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Major(s)
    node = evaluator.add_leaf(
        id="Yale_BS_Major",
        desc="The degree was a B.S. in Astronomy and Physics (double major)",
        parent=ug_node,
        critical=True,
    )
    majors_text = yale.majors or ""
    degree_text = yale.degree or "Bachelor of Science"
    claim = f"{full_name} earned a {degree_text} with majors in {majors_text} at Yale."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=yale_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Doctoral (Harvard)
    phd_node = evaluator.add_parallel(
        id="Doctoral_Degree",
        desc="The person earned a Ph.D. in Astronomy from Harvard University in 2000 with specific advisors",
        parent=edu_node,
        critical=True,
    )
    harvard = data.education_harvard or EducationHarvard()
    harvard_sources = ensure_list(harvard.sources)

    # Gate: sources provided for Harvard
    evaluator.add_custom_node(
        result=len(harvard_sources) > 0,
        id="Harvard_Sources_Provided",
        desc="At least one cited source is provided for Harvard Ph.D. details",
        parent=phd_node,
        critical=True,
    )

    # Institution
    node = evaluator.add_leaf(
        id="Harvard_PhD_Institution",
        desc="The doctoral degree was from Harvard University",
        parent=phd_node,
        critical=True,
    )
    claim = f"{full_name} earned a doctoral degree (PhD or equivalent) from Harvard University."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=harvard_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Year
    node = evaluator.add_leaf(
        id="Harvard_PhD_Year",
        desc="The Ph.D. was awarded in 2000",
        parent=phd_node,
        critical=True,
    )
    phd_year = harvard.year or ""
    claim = f"{full_name}'s Harvard PhD was awarded in {phd_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=harvard_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Field
    node = evaluator.add_leaf(
        id="Harvard_PhD_Field",
        desc="The Ph.D. was in Astronomy",
        parent=phd_node,
        critical=True,
    )
    phd_field = harvard.field or ""
    claim = f"{full_name}'s Harvard PhD field was {phd_field}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=harvard_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Advisors
    node = evaluator.add_leaf(
        id="Harvard_PhD_Advisors",
        desc="The doctoral advisors were Giovanni Fazio and Lee Hartmann",
        parent=phd_node,
        critical=True,
    )
    advisors_text = ", ".join(harvard.advisors) if harvard.advisors else ""
    claim = f"{full_name}'s Harvard PhD advisors were {advisors_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=harvard_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )


async def build_career_branch(evaluator: Evaluator, parent, data: PersonExtraction) -> None:
    full_name = data.full_name or "the person"
    career_node = evaluator.add_parallel(
        id="Career_Trajectory",
        desc="The person's career includes specific academic and administrative positions at designated institutions",
        parent=parent,
        critical=True,
    )

    # Miller Research Fellowship (UC Berkeley, 2000-2002)
    miller_node = evaluator.add_parallel(
        id="Early_Career_Fellowship",
        desc="The person held a Miller Research Fellowship at UC Berkeley from 2000-2002",
        parent=career_node,
        critical=True,
    )
    miller = data.miller_fellowship or CareerMiller()
    miller_sources = ensure_list(miller.sources)

    evaluator.add_custom_node(
        result=len(miller_sources) > 0,
        id="Miller_Sources_Provided",
        desc="At least one cited source is provided for the Miller Research Fellowship",
        parent=miller_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="Miller_Fellowship_Institution",
        desc="The fellowship was at UC Berkeley",
        parent=miller_node,
        critical=True,
    )
    inst_text = miller.institution or "UC Berkeley"
    fellowship_text = miller.fellowship_name or "Miller Research Fellowship"
    claim = f"{full_name} held the {fellowship_text} at {inst_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=miller_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Miller_Fellowship_Period",
        desc="The fellowship was from 2000-2002",
        parent=miller_node,
        critical=True,
    )
    period_text = miller.period or ""
    # Also consider start/end if present
    if miller.start_year and miller.end_year:
        period_text = period_text or f"{miller.start_year}-{miller.end_year}"
    claim = f"{full_name}'s {fellowship_text} at {inst_text} took place during {period_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=miller_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Canada Research Chair (U Toronto, 2008)
    crc_node = evaluator.add_parallel(
        id="Canada_Research_Chair",
        desc="The person was appointed Canada Research Chair in Observational Astrophysics at University of Toronto in 2008",
        parent=career_node,
        critical=True,
    )
    crc = data.crc_toronto or CareerCRC()
    crc_sources = ensure_list(crc.sources)

    evaluator.add_custom_node(
        result=len(crc_sources) > 0,
        id="CRC_Sources_Provided",
        desc="At least one cited source is provided for the Canada Research Chair appointment",
        parent=crc_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="CRC_Institution",
        desc="The Canada Research Chair was at University of Toronto",
        parent=crc_node,
        critical=True,
    )
    inst_text = crc.institution or ""
    claim = f"{full_name} held a Canada Research Chair at {inst_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=crc_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="CRC_Year",
        desc="The appointment was in 2008",
        parent=crc_node,
        critical=True,
    )
    crc_year = crc.year or ""
    claim = f"{full_name}'s Canada Research Chair appointment was in {crc_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=crc_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="CRC_Title",
        desc="The chair was in Observational Astrophysics",
        parent=crc_node,
        critical=True,
    )
    title_text = crc.chair_title or ""
    claim = f"{full_name} held the {title_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=crc_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # York University Deanship (2014-2018)
    york_node = evaluator.add_parallel(
        id="York_University_Deanship",
        desc="The person served as Dean of the Faculty of Science at York University from 2014 to 2018",
        parent=career_node,
        critical=True,
    )
    york = data.york_deanship or CareerYork()
    york_sources = ensure_list(york.sources)

    evaluator.add_custom_node(
        result=len(york_sources) > 0,
        id="York_Sources_Provided",
        desc="At least one cited source is provided for the York University deanship",
        parent=york_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="York_Position",
        desc="The position was Dean of the Faculty of Science",
        parent=york_node,
        critical=True,
    )
    pos_text = york.position or ""
    claim = f"{full_name} served as {pos_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=york_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="York_Institution",
        desc="The institution was York University",
        parent=york_node,
        critical=True,
    )
    inst_text = york.institution or ""
    claim = f"{full_name} served at {inst_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=york_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="York_Period",
        desc="The deanship was from 2014 to 2018",
        parent=york_node,
        critical=True,
    )
    period_text = york.period or ""
    if york.start_year and york.end_year:
        period_text = period_text or f"{york.start_year} to {york.end_year}"
    claim = f"{full_name}'s deanship period was {period_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=york_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Cornell Deanship (2018) + Hans Bethe Professor (2022)
    cornell_node = evaluator.add_parallel(
        id="Cornell_Deanship",
        desc="The person became the Harold Tanner Dean of the College of Arts and Sciences at Cornell University in 2018 and was named Hans Bethe Professor in 2022",
        parent=career_node,
        critical=True,
    )
    cornell = data.cornell_deanship or CareerCornell()
    cornell_sources = ensure_list(cornell.sources)

    evaluator.add_custom_node(
        result=len(cornell_sources) > 0,
        id="Cornell_Sources_Provided",
        desc="At least one cited source is provided for the Cornell deanship and professorship",
        parent=cornell_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="Cornell_Position",
        desc="The position was Harold Tanner Dean of the College of Arts and Sciences",
        parent=cornell_node,
        critical=True,
    )
    pos_text = cornell.position or ""
    claim = f"{full_name} served as {pos_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cornell_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Cornell_Institution",
        desc="The institution was Cornell University",
        parent=cornell_node,
        critical=True,
    )
    inst_text = cornell.institution or ""
    claim = f"{full_name} served at {inst_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cornell_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Cornell_Start_Year",
        desc="The deanship began in 2018",
        parent=cornell_node,
        critical=True,
    )
    start_year = cornell.start_year or ""
    claim = f"{full_name}'s Cornell deanship began in {start_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cornell_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Cornell_Bethe_Professor",
        desc="The person was named Hans Bethe Professor in 2022",
        parent=cornell_node,
        critical=True,
    )
    bethe_year = cornell.bethe_professor_year or ""
    claim = f"{full_name} was named Hans Bethe Professor in {bethe_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cornell_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Johns Hopkins Provost (effective Oct 15, 2023)
    jhu_node = evaluator.add_parallel(
        id="Johns_Hopkins_Provost",
        desc="The person was appointed Provost of Johns Hopkins University effective October 15, 2023",
        parent=career_node,
        critical=True,
    )
    jhu = data.jhu_provost or CareerJHU()
    jhu_sources = ensure_list(jhu.sources)

    evaluator.add_custom_node(
        result=len(jhu_sources) > 0,
        id="JHU_Sources_Provided",
        desc="At least one cited source is provided for the Johns Hopkins provost appointment",
        parent=jhu_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="JHU_Position",
        desc="The position was Provost",
        parent=jhu_node,
        critical=True,
    )
    pos_text = jhu.position or ""
    claim = f"{full_name} was appointed {pos_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=jhu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="JHU_Institution",
        desc="The institution was Johns Hopkins University",
        parent=jhu_node,
        critical=True,
    )
    inst_text = jhu.institution or ""
    claim = f"This appointment was at {inst_text} for {full_name}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=jhu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="JHU_Effective_Date",
        desc="The appointment was effective October 15, 2023",
        parent=jhu_node,
        critical=True,
    )
    eff_date = jhu.effective_date or ""
    claim = f"The effective date for {full_name}'s provost appointment was {eff_date}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=jhu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Caltech Presidency (effective July 1, 2026)
    caltech_node = evaluator.add_parallel(
        id="Caltech_Presidency",
        desc="The person was appointed as the 10th President of Caltech, effective July 1, 2026",
        parent=career_node,
        critical=True,
    )
    caltech = data.caltech_presidency or CareerCaltech()
    caltech_sources = ensure_list(caltech.sources)

    evaluator.add_custom_node(
        result=len(caltech_sources) > 0,
        id="Caltech_Sources_Provided",
        desc="At least one cited source is provided for the Caltech presidency appointment",
        parent=caltech_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="Caltech_Position",
        desc="The position is 10th President",
        parent=caltech_node,
        critical=True,
    )
    pos_text = caltech.position or ""
    claim = f"{full_name} was appointed {pos_text} of Caltech."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=caltech_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Caltech_Institution",
        desc="The institution is California Institute of Technology (Caltech)",
        parent=caltech_node,
        critical=True,
    )
    inst_text = caltech.institution or ""
    claim = f"The presidency appointment for {full_name} is at {inst_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=caltech_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Caltech_Effective_Date",
        desc="The appointment is effective July 1, 2026",
        parent=caltech_node,
        critical=True,
    )
    eff_date = caltech.effective_date or ""
    claim = f"The effective date for {full_name}'s Caltech presidency is {eff_date}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=caltech_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )


async def build_publications_branch(evaluator: Evaluator, parent, data: PersonExtraction) -> None:
    full_name = data.full_name or "the person"
    pubs_node = evaluator.add_parallel(
        id="Publications",
        desc="The person authored specific books with documented publication years and recognition",
        parent=parent,
        critical=True,
    )

    # Strange New Worlds
    snw_node = evaluator.add_parallel(
        id="Strange_New_Worlds",
        desc="The person wrote 'Strange New Worlds' published in 2011, which was a finalist for the Lane Anderson Award",
        parent=pubs_node,
        critical=True,
    )
    snw = data.snw_book or BookSNW()
    snw_sources = ensure_list(snw.sources)

    evaluator.add_custom_node(
        result=len(snw_sources) > 0,
        id="SNW_Sources_Provided",
        desc="At least one cited source is provided for the 'Strange New Worlds' details",
        parent=snw_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="SNW_Title",
        desc="The book title is 'Strange New Worlds'",
        parent=snw_node,
        critical=True,
    )
    title_text = snw.title or ""
    claim = f"{full_name} authored a book titled '{title_text}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=snw_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="SNW_Year",
        desc="The book was published in 2011",
        parent=snw_node,
        critical=True,
    )
    year_text = snw.year or ""
    claim = f"'{snw.title or 'the book'}' by {full_name} was published in {year_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=snw_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="SNW_Award",
        desc="The book was a finalist for the Lane Anderson Award",
        parent=snw_node,
        critical=True,
    )
    recog_text = snw.recognition or ""
    claim = f"'{snw.title or 'the book'}' by {full_name} received recognition as {recog_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=snw_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Neutrino Hunters
    nh_node = evaluator.add_parallel(
        id="Neutrino_Hunters",
        desc="The person wrote 'Neutrino Hunters' which won the Canadian Science Writers Association Book Award in 2014",
        parent=pubs_node,
        critical=True,
    )
    nh = data.nh_book or BookNH()
    nh_sources = ensure_list(nh.sources)

    evaluator.add_custom_node(
        result=len(nh_sources) > 0,
        id="NH_Sources_Provided",
        desc="At least one cited source is provided for the 'Neutrino Hunters' details",
        parent=nh_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="NH_Title",
        desc="The book title is 'Neutrino Hunters'",
        parent=nh_node,
        critical=True,
    )
    title_text = nh.title or ""
    claim = f"{full_name} authored a book titled '{title_text}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=nh_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="NH_Award",
        desc="The book won the Canadian Science Writers Association Book Award in 2014",
        parent=nh_node,
        critical=True,
    )
    award_text = nh.award or ""
    award_year_text = nh.award_year or ""
    claim = f"'{nh.title or 'the book'}' by {full_name} won the {award_text} in {award_year_text}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=nh_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Child of the Universe
    cotu_node = evaluator.add_parallel(
        id="Child_of_the_Universe",
        desc="The person wrote the children's book 'Child of the Universe' published on March 17, 2020",
        parent=pubs_node,
        critical=True,
    )
    cotu = data.cotu_book or BookCOTU()
    cotu_sources = ensure_list(cotu.sources)

    evaluator.add_custom_node(
        result=len(cotu_sources) > 0,
        id="COTU_Sources_Provided",
        desc="At least one cited source is provided for 'Child of the Universe' details",
        parent=cotu_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="COTU_Title",
        desc="The book title is 'Child of the Universe'",
        parent=cotu_node,
        critical=True,
    )
    title_text = cotu.title or ""
    claim = f"{full_name} authored a book titled '{title_text}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cotu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="COTU_Publication_Date",
        desc="The book was published on March 17, 2020",
        parent=cotu_node,
        critical=True,
    )
    pub_date = cotu.publication_date or ""
    claim = f"'{cotu.title or 'the book'}' by {full_name} was published on {pub_date}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cotu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="COTU_Type",
        desc="The book is a children's picture book",
        parent=cotu_node,
        critical=True,
    )
    book_type = cotu.book_type or ""
    claim = f"'{cotu.title or 'the book'}' by {full_name} is a {book_type}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=cotu_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )


async def build_awards_branch(evaluator: Evaluator, parent, data: PersonExtraction) -> None:
    full_name = data.full_name or "the person"
    awards_node = evaluator.add_parallel(
        id="Major_Awards_and_Honors",
        desc="The person received specific prestigious awards and fellowships in documented years",
        parent=parent,
        critical=True,
    )

    # Steacie Fellowship (2009)
    steacie_node = evaluator.add_parallel(
        id="Steacie_Fellowship",
        desc="The person received the E.W.R. Steacie Memorial Fellowship in 2009",
        parent=awards_node,
        critical=True,
    )
    steacie = data.steacie or AwardSteacie()
    steacie_sources = ensure_list(steacie.sources)

    evaluator.add_custom_node(
        result=len(steacie_sources) > 0,
        id="Steacie_Sources_Provided",
        desc="At least one cited source is provided for the Steacie Fellowship",
        parent=steacie_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="Steacie_Award_Name",
        desc="The award is the E.W.R. Steacie Memorial Fellowship",
        parent=steacie_node,
        critical=True,
    )
    award_name = steacie.name or ""
    claim = f"{full_name} received the {award_name}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=steacie_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Steacie_Year",
        desc="The fellowship was awarded in 2009",
        parent=steacie_node,
        critical=True,
    )
    award_year = steacie.year or ""
    claim = f"{full_name} received this award in {award_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=steacie_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    # Carl Sagan Medal (2020)
    sagan_node = evaluator.add_parallel(
        id="Carl_Sagan_Medal",
        desc="The person received the Carl Sagan Medal for Excellence in Public Communication in Planetary Science in 2020",
        parent=awards_node,
        critical=True,
    )
    sagan = data.sagan or AwardSagan()
    sagan_sources = ensure_list(sagan.sources)

    evaluator.add_custom_node(
        result=len(sagan_sources) > 0,
        id="Sagan_Sources_Provided",
        desc="At least one cited source is provided for the Carl Sagan Medal",
        parent=sagan_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id="Sagan_Award_Name",
        desc="The award is the Carl Sagan Medal for Excellence in Public Communication in Planetary Science",
        parent=sagan_node,
        critical=True,
    )
    award_name = sagan.name or ""
    claim = f"{full_name} received the {award_name}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sagan_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )

    node = evaluator.add_leaf(
        id="Sagan_Year",
        desc="The medal was awarded in 2020",
        parent=sagan_node,
        critical=True,
    )
    sagan_year = sagan.year or ""
    claim = f"{full_name} received this medal in {sagan_year}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sagan_sources,
        additional_instruction=GENERAL_VERIFY_INSTRUCTION,
    )


# --------------------------------------------------------------------------- #
# Main verification tree builder                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, data: PersonExtraction) -> None:
    # Root level: wrap everything under a critical "Person_Identification" node
    person_node = evaluator.add_parallel(
        id="Person_Identification",
        desc="The correct person is identified who matches all specified educational, career, publication, and award criteria",
        parent=evaluator.root,
        critical=True,
    )

    # Name presence (critical)
    evaluator.add_custom_node(
        result=bool(data.full_name and data.full_name.strip()),
        id="Name_Present",
        desc="The answer provides a non-empty full name",
        parent=person_node,
        critical=True,
    )

    # Build four critical pillars
    await build_education_branch(evaluator, person_node, data)
    await build_career_branch(evaluator, person_node, data)
    await build_publications_branch(evaluator, person_node, data)
    await build_awards_branch(evaluator, person_node, data)


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
    Evaluate an answer for the astrophysicist identification task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_person(),
        template_class=PersonExtraction,
        extraction_name="person_extraction",
    )

    # Optional info recording
    evaluator.add_custom_info(
        info={"extracted_full_name": extracted.full_name or None},
        info_type="extraction_summary",
        info_name="extracted_overview",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()