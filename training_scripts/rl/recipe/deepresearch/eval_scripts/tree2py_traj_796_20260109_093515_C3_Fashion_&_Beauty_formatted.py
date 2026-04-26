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
TASK_ID = "designer_chloe_2023"
TASK_DESCRIPTION = """
In October 2023, a designer was appointed as creative director of a French luxury fashion house for their third tenure at the brand. This designer previously served as design director for women's ready-to-wear at Saint Laurent beginning in 2016, and holds a Master of Arts in Fashion from Central Saint Martins (2007, graduated with distinction, studied under Professor Louise Wilson). Identify this designer, then verify that the fashion house they joined was founded in 1952 by Gaby Aghion. Confirm that their debut runway collection was presented in February 2024 for the Fall 2024 season. Finally, identify the specific song by Kate Bush that was featured as the soundtrack for this debut runway show.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IdentitySection(BaseModel):
    designer_name: Optional[str] = None
    fashion_house: Optional[str] = None
    identity_sources: List[str] = Field(default_factory=list)


class AppointmentSection(BaseModel):
    appointment_date: Optional[str] = None  # e.g., "October 2023"
    third_tenure: Optional[bool] = None     # true if explicitly stated (e.g., "third tenure", "third time")
    appointment_sources: List[str] = Field(default_factory=list)


class SaintLaurentSection(BaseModel):
    role_title: Optional[str] = None  # e.g., "design director for women's ready-to-wear"
    start_year: Optional[str] = None  # e.g., "2016"
    sources: List[str] = Field(default_factory=list)


class EducationSection(BaseModel):
    degree: Optional[str] = None      # e.g., "Master of Arts in Fashion"
    institution: Optional[str] = None # e.g., "Central Saint Martins"
    graduation_year: Optional[str] = None  # e.g., "2007"
    distinction: Optional[bool] = None     # true if "with distinction" is stated
    professor: Optional[str] = None   # e.g., "Louise Wilson"
    sources: List[str] = Field(default_factory=list)


class HouseSection(BaseModel):
    house_name: Optional[str] = None
    founding_year: Optional[str] = None  # e.g., "1952"
    founder: Optional[str] = None        # e.g., "Gaby Aghion"
    sources: List[str] = Field(default_factory=list)


class DebutSection(BaseModel):
    month_year: Optional[str] = None     # e.g., "February 2024"
    season: Optional[str] = None         # e.g., "Fall 2024"
    sources: List[str] = Field(default_factory=list)


class SoundtrackSection(BaseModel):
    artist: Optional[str] = None         # ideally "Kate Bush"
    song_title: Optional[str] = None     # specific song name
    sources: List[str] = Field(default_factory=list)


class DesignerAppointmentExtraction(BaseModel):
    identity: Optional[IdentitySection] = None
    appointment: Optional[AppointmentSection] = None
    saint_laurent: Optional[SaintLaurentSection] = None
    education: Optional[EducationSection] = None
    house: Optional[HouseSection] = None
    debut_show: Optional[DebutSection] = None
    soundtrack: Optional[SoundtrackSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extraction() -> str:
    return """
    Extract structured data from the answer about the designer and related constraints. You must extract ONLY what is explicitly stated in the answer text and the URLs it cites.

    Required sections and fields:

    identity:
      - designer_name: The full name of the designer identified in the answer.
      - fashion_house: The French luxury fashion house they were appointed to.
      - identity_sources: All URLs cited that identify or confirm the designer and the fashion house.

    appointment:
      - appointment_date: The month and year of the appointment (e.g., "October 2023").
      - third_tenure: Return true if the answer explicitly states this was their "third tenure" (or equivalent phrasing like "third time") at the brand; otherwise false or null.
      - appointment_sources: All URLs cited that directly support the appointment details (date and third tenure).

    saint_laurent:
      - role_title: The role at Saint Laurent (e.g., "design director for women's ready-to-wear").
      - start_year: The start year mentioned (e.g., "2016").
      - sources: All URLs cited that support the Saint Laurent role and start year.

    education:
      - degree: The exact degree (e.g., "Master of Arts in Fashion").
      - institution: The institution (e.g., "Central Saint Martins").
      - graduation_year: The year (e.g., "2007").
      - distinction: Return true if "with distinction" is explicitly stated; otherwise false or null.
      - professor: The professor mentioned (e.g., "Louise Wilson").
      - sources: All URLs cited that support the education details.

    house:
      - house_name: The fashion house name (should match identity.fashion_house if present).
      - founding_year: The founding year claimed (e.g., "1952").
      - founder: The founder's name (e.g., "Gaby Aghion").
      - sources: All URLs cited that support founding info.

    debut_show:
      - month_year: The month and year of the debut runway collection (e.g., "February 2024").
      - season: The season (e.g., "Fall 2024").
      - sources: All URLs cited that support the debut show timing.

    soundtrack:
      - artist: The artist name mentioned (should be "Kate Bush" if stated).
      - song_title: The specific song title used as the soundtrack for the debut runway show.
      - sources: All URLs cited that support the soundtrack song information.

    RULES:
    - Extract only URLs explicitly present in the answer text.
    - If a required detail is missing, return null for that field.
    - If a sources list for a section has no URLs in the answer, return an empty list for that section's sources.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def list_or_empty(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: DesignerAppointmentExtraction) -> None:
    """
    Build the verification tree based on the rubric, adding leaf nodes for each atomic check,
    and verify claims against cited sources.
    """

    # Create a critical root child to reflect rubric root criticality
    task_root = evaluator.add_sequential(
        id="task_root",
        desc="Identify the designer and verify all stated constraints about their appointment, background, the fashion house, the debut collection timing, and the Kate Bush soundtrack song.",
        parent=evaluator.root,
        critical=True
    )

    # -------------------- Identify designer and house -------------------- #
    identify_node = evaluator.add_parallel(
        id="identify_designer_and_house",
        desc="Provide the designer's identity and clearly specify which French luxury fashion house they were appointed to as creative director.",
        parent=task_root,
        critical=True
    )

    designer_name = extraction.identity.designer_name if extraction.identity else None
    house_name = extraction.identity.fashion_house if extraction.identity else None

    # Critical existence check to gate subsequent verifications
    evaluator.add_custom_node(
        result=bool(designer_name and designer_name.strip()) and bool(house_name and house_name.strip()),
        id="designer_and_house_provided",
        desc="Designer name and fashion house are both provided in the answer.",
        parent=identify_node,
        critical=True
    )

    # -------------------- Verify constraints (parallel) ------------------ #
    constraints_node = evaluator.add_parallel(
        id="verify_constraints",
        desc="Verify each constraint stated in the question about the designer, the fashion house, and the debut runway show.",
        parent=task_root,
        critical=True
    )

    # 1) Appointment in Oct 2023 and third tenure
    appointment_node = evaluator.add_parallel(
        id="appointment_oct_2023_third_tenure",
        desc="Verify the designer was appointed creative director in October 2023 and that this marked their third tenure at the fashion house.",
        parent=constraints_node,
        critical=True
    )

    appointment_sources = list_or_empty(extraction.appointment.appointment_sources if extraction.appointment else [])
    # Existence gate for sources
    evaluator.add_custom_node(
        result=len(appointment_sources) > 0,
        id="appointment_sources_present",
        desc="Appointment sources are provided.",
        parent=appointment_node,
        critical=True
    )

    # 1.a Appointment month/year
    appt_date_leaf = evaluator.add_leaf(
        id="appointment_in_oct_2023",
        desc="The designer was appointed creative director in October 2023.",
        parent=appointment_node,
        critical=True
    )
    claim_appt_date = f"{designer_name} was appointed creative director of {house_name} in October 2023."
    await evaluator.verify(
        claim=claim_appt_date,
        node=appt_date_leaf,
        sources=appointment_sources,
        additional_instruction="Confirm the appointment month/year is October 2023. Allow minor phrasing variation but month/year must match."
    )

    # 1.b Third tenure
    third_tenure_leaf = evaluator.add_leaf(
        id="appointment_marked_third_tenure",
        desc="This appointment marked the designer's third tenure at the fashion house.",
        parent=appointment_node,
        critical=True
    )
    claim_third_tenure = f"This appointment marked {designer_name}'s third tenure at {house_name}."
    await evaluator.verify(
        claim=claim_third_tenure,
        node=third_tenure_leaf,
        sources=appointment_sources,
        additional_instruction="Look for wording such as 'third time', 'third tenure', or equivalent phrasing clearly indicating three distinct tenures."
    )

    # 2) Saint Laurent role beginning 2016
    sl_node = evaluator.add_parallel(
        id="saint_laurent_role_2016",
        desc="Verify the designer previously served as design director for women's ready-to-wear at Saint Laurent beginning in 2016.",
        parent=constraints_node,
        critical=True
    )

    sl_sources = list_or_empty(extraction.saint_laurent.sources if extraction.saint_laurent else [])
    evaluator.add_custom_node(
        result=len(sl_sources) > 0,
        id="saint_laurent_sources_present",
        desc="Saint Laurent role sources are provided.",
        parent=sl_node,
        critical=True
    )

    # 2.a Role title
    sl_role_leaf = evaluator.add_leaf(
        id="saint_laurent_role_title",
        desc="Designer served as design director for women's ready-to-wear at Saint Laurent.",
        parent=sl_node,
        critical=True
    )
    claim_sl_role = f"{designer_name} previously served as design director for women's ready-to-wear at Saint Laurent."
    await evaluator.verify(
        claim=claim_sl_role,
        node=sl_role_leaf,
        sources=sl_sources,
        additional_instruction="Allow synonyms like 'womenswear RTW'. Ensure the role is specifically design director for women's ready-to-wear."
    )

    # 2.b Start year 2016
    sl_year_leaf = evaluator.add_leaf(
        id="saint_laurent_start_year_2016",
        desc="Designer began the Saint Laurent role in 2016.",
        parent=sl_node,
        critical=True
    )
    claim_sl_year = f"{designer_name} began this role at Saint Laurent in 2016."
    await evaluator.verify(
        claim=claim_sl_year,
        node=sl_year_leaf,
        sources=sl_sources,
        additional_instruction="Confirm that the cited material explicitly indicates a start in 2016."
    )

    # 3) Education: CSM MA, 2007, distinction, Louise Wilson
    csm_node = evaluator.add_parallel(
        id="csm_ma_2007_distinction_louise_wilson",
        desc="Verify CSM MA Fashion (2007), graduated with distinction, studied under Professor Louise Wilson.",
        parent=constraints_node,
        critical=True
    )

    csm_sources = list_or_empty(extraction.education.sources if extraction.education else [])
    evaluator.add_custom_node(
        result=len(csm_sources) > 0,
        id="csm_sources_present",
        desc="Education sources are provided.",
        parent=csm_node,
        critical=True
    )

    # 3.a Degree and institution
    csm_degree_leaf = evaluator.add_leaf(
        id="csm_degree_ma_fashion",
        desc="Designer completed a Master of Arts in Fashion at Central Saint Martins.",
        parent=csm_node,
        critical=True
    )
    claim_csm_degree = f"{designer_name} completed a Master of Arts in Fashion at Central Saint Martins."
    await evaluator.verify(
        claim=claim_csm_degree,
        node=csm_degree_leaf,
        sources=csm_sources,
        additional_instruction="Confirm both the degree (MA Fashion) and the institution (Central Saint Martins)."
    )

    # 3.b Graduation year 2007
    csm_year_leaf = evaluator.add_leaf(
        id="csm_graduation_year_2007",
        desc="Designer graduated in 2007.",
        parent=csm_node,
        critical=True
    )
    claim_csm_year = f"{designer_name} graduated in 2007."
    await evaluator.verify(
        claim=claim_csm_year,
        node=csm_year_leaf,
        sources=csm_sources,
        additional_instruction="Confirm the graduation year as 2007."
    )

    # 3.c Distinction
    csm_dist_leaf = evaluator.add_leaf(
        id="csm_graduated_with_distinction",
        desc="Designer graduated with distinction.",
        parent=csm_node,
        critical=True
    )
    claim_csm_dist = f"{designer_name} graduated with distinction."
    await evaluator.verify(
        claim=claim_csm_dist,
        node=csm_dist_leaf,
        sources=csm_sources,
        additional_instruction="Look for 'with distinction' or an equivalent formal honor stated."
    )

    # 3.d Studied under Louise Wilson
    csm_prof_leaf = evaluator.add_leaf(
        id="csm_professor_louise_wilson",
        desc="Designer studied under Professor Louise Wilson.",
        parent=csm_node,
        critical=True
    )
    claim_csm_prof = f"{designer_name} studied under Professor Louise Wilson."
    await evaluator.verify(
        claim=claim_csm_prof,
        node=csm_prof_leaf,
        sources=csm_sources,
        additional_instruction="Confirm explicit mentorship or tutelage under Louise Wilson."
    )

    # 4) House founded in 1952 by Gaby Aghion
    house_node = evaluator.add_parallel(
        id="house_founded_1952_gaby_aghion",
        desc="Verify the identified fashion house was founded in 1952 by Gaby Aghion.",
        parent=constraints_node,
        critical=True
    )

    house_sources = list_or_empty(extraction.house.sources if extraction.house else [])
    evaluator.add_custom_node(
        result=len(house_sources) > 0,
        id="house_sources_present",
        desc="Fashion house founding sources are provided.",
        parent=house_node,
        critical=True
    )

    # 4.a Founding year 1952
    house_year_leaf = evaluator.add_leaf(
        id="house_founding_year_1952",
        desc="The fashion house was founded in 1952.",
        parent=house_node,
        critical=True
    )
    claim_house_year = f"{house_name} was founded in 1952."
    await evaluator.verify(
        claim=claim_house_year,
        node=house_year_leaf,
        sources=house_sources,
        additional_instruction="Confirm that the founding year is 1952."
    )

    # 4.b Founder Gaby Aghion
    house_founder_leaf = evaluator.add_leaf(
        id="house_founder_gaby_aghion",
        desc="The fashion house was founded by Gaby Aghion.",
        parent=house_node,
        critical=True
    )
    claim_house_founder = f"{house_name} was founded by Gaby Aghion."
    await evaluator.verify(
        claim=claim_house_founder,
        node=house_founder_leaf,
        sources=house_sources,
        additional_instruction="Confirm that the founder is Gaby Aghion."
    )

    # 5) Debut runway in Feb 2024 for Fall 2024
    debut_node = evaluator.add_parallel(
        id="debut_runway_feb_2024_fall_2024",
        desc="Verify the designer's debut runway collection was presented in February 2024 for the Fall 2024 season.",
        parent=constraints_node,
        critical=True
    )

    debut_sources = list_or_empty(extraction.debut_show.sources if extraction.debut_show else [])
    evaluator.add_custom_node(
        result=len(debut_sources) > 0,
        id="debut_sources_present",
        desc="Debut runway show sources are provided.",
        parent=debut_node,
        critical=True
    )

    # 5.a Presented in February 2024
    debut_date_leaf = evaluator.add_leaf(
        id="debut_presented_feb_2024",
        desc="Debut runway collection was presented in February 2024.",
        parent=debut_node,
        critical=True
    )
    claim_debut_date = f"{designer_name}'s debut runway collection for {house_name} was presented in February 2024."
    await evaluator.verify(
        claim=claim_debut_date,
        node=debut_date_leaf,
        sources=debut_sources,
        additional_instruction="Confirm the show took place in February 2024 (Paris Fashion Week timing is acceptable evidence)."
    )

    # 5.b For Fall 2024 season
    debut_season_leaf = evaluator.add_leaf(
        id="debut_for_fall_2024_season",
        desc="Debut runway collection was for the Fall 2024 season.",
        parent=debut_node,
        critical=True
    )
    claim_debut_season = f"The debut runway collection was for the Fall 2024 season."
    await evaluator.verify(
        claim=claim_debut_season,
        node=debut_season_leaf,
        sources=debut_sources,
        additional_instruction="Confirm the collection is labeled as Fall 2024 (FW24/Autumn-Winter 2024 should be considered equivalent)."
    )

    # 6) Kate Bush soundtrack song
    kb_node = evaluator.add_parallel(
        id="kate_bush_song_used",
        desc="Identify the specific song by Kate Bush used as the soundtrack for the debut runway show.",
        parent=constraints_node,
        critical=True
    )

    soundtrack_sources = list_or_empty(extraction.soundtrack.sources if extraction.soundtrack else [])
    song_title = extraction.soundtrack.song_title if extraction.soundtrack else None
    artist_name = extraction.soundtrack.artist if extraction.soundtrack else None

    # Existence gate for song title and sources
    evaluator.add_custom_node(
        result=bool(song_title and song_title.strip()) and len(soundtrack_sources) > 0,
        id="soundtrack_song_and_sources_present",
        desc="Kate Bush song title is provided and soundtrack sources are present.",
        parent=kb_node,
        critical=True
    )

    kb_leaf = evaluator.add_leaf(
        id="kate_bush_specific_song_verified",
        desc="The specific Kate Bush song used for the debut runway soundtrack is correctly identified.",
        parent=kb_node,
        critical=True
    )
    # Build the claim; enforce Kate Bush artist if stated
    if artist_name and artist_name.strip():
        claim_kb = f"The debut runway show soundtrack featured '{song_title}' by {artist_name}."
    else:
        claim_kb = f"The debut runway show soundtrack featured '{song_title}' by Kate Bush."
    await evaluator.verify(
        claim=claim_kb,
        node=kb_leaf,
        sources=soundtrack_sources,
        additional_instruction="Confirm that the specified Kate Bush track title was used for the debut runway show soundtrack."
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
) -> Dict[str, Any]:
    """
    Evaluate the provided answer against the rubric using the Mind2Web2 framework.
    """
    # Initialize evaluator with sequential root (per rubric)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extraction(),
        template_class=DesignerAppointmentExtraction,
        extraction_name="designer_appointment_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()