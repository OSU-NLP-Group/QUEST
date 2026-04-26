import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ryan_coogler_2025_film"
TASK_DESCRIPTION = (
    "In 2025, director Ryan Coogler released a film that went on to receive a record-breaking 16 Academy Award "
    "nominations at the 98th Academy Awards (nominations announced January 22, 2026). For this film, provide the "
    "following information: 1. The name of the lead actor who starred in dual roles, 2. The year in which the film is set, "
    "3. The specific city and state in Mississippi where the film is set, 4. The name of the cinematographer, "
    "5. The film format and camera systems used (including film gauge and specific camera types), 6. The name of the "
    "composer who created the film's score, 7. The studio that distributed the film, 8. The name of the production company, "
    "9. The production budget range (in millions of dollars), 10. The total worldwide box office gross (in millions of dollars), "
    "11. The film's running time in minutes, 12. The US theatrical release date, 13. The venue where the film's premiere took place, "
    "14. The date of the film's premiere, 15. The time period during which principal photography took place (month and year range)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FilmInfoExtraction(BaseModel):
    film_title: Optional[FieldValue] = None

    lead_actor_dual_roles: Optional[FieldValue] = None
    setting_year: Optional[FieldValue] = None
    setting_location_city_state: Optional[FieldValue] = None
    cinematographer: Optional[FieldValue] = None
    film_format_and_cameras: Optional[FieldValue] = None
    composer: Optional[FieldValue] = None
    distributor: Optional[FieldValue] = None
    production_company: Optional[FieldValue] = None
    budget_range_millions: Optional[FieldValue] = None
    box_office_total_millions: Optional[FieldValue] = None
    running_time_minutes: Optional[FieldValue] = None
    us_release_date: Optional[FieldValue] = None
    premiere_venue: Optional[FieldValue] = None
    premiere_date: Optional[FieldValue] = None
    principal_photography_period: Optional[FieldValue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return """
    Extract from the answer the details for the 2025 Ryan Coogler film that received a record 16 Academy Award nominations at the 98th Academy Awards.
    For each requested attribute, return an object with:
    - value: the exact text as stated in the answer for that attribute
    - sources: a list of URLs explicitly cited in the answer that support that attribute
      If the answer provides a general 'Sources' section, associate any URLs that plausibly support the attribute.
      Do not invent or infer URLs beyond those mentioned in the answer.

    Return a single JSON object with the following fields (each is an object { "value": string|null, "sources": string[] }):
    1. film_title
    2. lead_actor_dual_roles                  (name of the lead actor who starred in dual roles)
    3. setting_year                           (year in which the film is set)
    4. setting_location_city_state            (specific city and state in Mississippi where the film is set; or 'Mississippi Delta' if that is how the answer phrases it)
    5. cinematographer                        (name)
    6. film_format_and_cameras                (film gauge and specific camera systems used, e.g., '65mm; IMAX 15-perf and Ultra Panavision 70')
    7. composer                               (name of the composer who created the score)
    8. distributor                            (studio that distributed the film)
    9. production_company                     (name)
    10. budget_range_millions                 (range in millions of dollars as written, e.g., '$90–100 million')
    11. box_office_total_millions             (total worldwide gross in millions, e.g., '$369 million')
    12. running_time_minutes                  (e.g., '138 minutes' or '138 min')
    13. us_release_date                       (US theatrical release date)
    14. premiere_venue                        (venue where the premiere took place)
    15. premiere_date                         (date of the premiere)
    16. principal_photography_period          (month–year range, e.g., 'April–July 2024')

    If any value is missing from the answer, set value to null and sources to an empty array for that field.
    Strictly follow this structure.
    """


# --------------------------------------------------------------------------- #
# Helper functions for building claims and nodes                              #
# --------------------------------------------------------------------------- #
def _has_value_and_sources(f: Optional[FieldValue]) -> bool:
    return bool(f and f.value and str(f.value).strip() and f.sources and len(f.sources) > 0)


def _title_context(data: FilmInfoExtraction) -> str:
    # Use film title if provided; otherwise provide a general identifier
    if data.film_title and data.film_title.value and data.film_title.value.strip():
        return data.film_title.value.strip()
    return "the 2025 Ryan Coogler film"


def _common_additional_instruction(title_text: str) -> str:
    return (
        f"Verify this information specifically for the 2025 film directed by Ryan Coogler titled '{title_text}'. "
        "Allow minor variations in formatting (e.g., hyphens vs. en dashes, accents, or abbreviations). "
        "Rely on explicit statements or clear evidence on the provided webpage(s)."
    )


# --------------------------------------------------------------------------- #
# Build verification tree and schedule verifications                          #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, root_node, data: FilmInfoExtraction) -> None:
    # Root container for film information (make it non-critical to allow partial credit)
    film_root = evaluator.add_parallel(
        id="Film_Information_Verification",
        desc="Verify detailed information about the 2025 film directed by Ryan Coogler that received a record 16 Academy Award nominations",
        parent=root_node,
        critical=False
    )

    title_text = _title_context(data)
    common_ins = _common_additional_instruction(title_text)

    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = []

    # Helper to add a field verification block (existence + source-supported check)
    def add_field_block(
        node_id: str,
        node_desc: str,
        field: Optional[FieldValue],
        claim_text: str,
        add_ins: Optional[str] = None
    ):
        container = evaluator.add_parallel(
            id=node_id,
            desc=node_desc,
            parent=film_root,
            critical=False
        )

        exists = _has_value_and_sources(field)
        evaluator.add_custom_node(
            result=exists,
            id=f"{node_id}_exists",
            desc=f"Value and at least one source are provided for: {node_desc}",
            parent=container,
            critical=True
        )

        leaf = evaluator.add_leaf(
            id=f"{node_id}_supported",
            desc=node_desc,
            parent=container,
            critical=True
        )

        srcs: List[str] = field.sources if (field and field.sources) else []
        claims_and_sources.append((
            claim_text,
            srcs,
            leaf,
            add_ins or common_ins
        ))

    # 1) Lead Actor (dual roles)
    lead = data.lead_actor_dual_roles or FieldValue()
    add_field_block(
        node_id="Lead_Actor",
        node_desc="Identify that Michael B. Jordan stars in the film in dual roles",
        field=lead,
        claim_text=f"In '{title_text}', the lead actor who starred in dual roles is {lead.value}.",
        add_ins=(common_ins + " Confirm that the cited source(s) indicate that this actor is the lead and plays dual/two roles.")
    )

    # 2) Setting Year
    setting_year = data.setting_year or FieldValue()
    add_field_block(
        node_id="Setting_Year",
        node_desc="Specify that the film is set in 1932",
        field=setting_year,
        claim_text=f"The film '{title_text}' is set in {setting_year.value}.",
        add_ins=(common_ins + " Focus on the story's setting year (diegetic time), not production or release year.")
    )

    # 3) Setting Location (city/state in Mississippi)
    setting_loc = data.setting_location_city_state or FieldValue()
    add_field_block(
        node_id="Setting_Location",
        node_desc="Identify that the film is set in Clarksdale, Mississippi (or Mississippi Delta)",
        field=setting_loc,
        claim_text=f"The film '{title_text}' is set in {setting_loc.value}.",
        add_ins=(common_ins + " Accept either a specific city/state like 'Clarksdale, Mississippi' or a clear statement that the setting is the Mississippi Delta.")
    )

    # 4) Cinematographer
    cine = data.cinematographer or FieldValue()
    add_field_block(
        node_id="Cinematographer",
        node_desc="Identify Autumn Durald Arkapaw as the cinematographer",
        field=cine,
        claim_text=f"The cinematographer for '{title_text}' is {cine.value}.",
        add_ins=(common_ins + " Look for credits such as 'cinematography by' or 'director of photography'.")
    )

    # 5) Film Format and Camera Systems
    fmt = data.film_format_and_cameras or FieldValue()
    add_field_block(
        node_id="Film_Format",
        node_desc="Specify that the film was shot on 65mm film using IMAX 15-perf and Ultra Panavision 70 cameras",
        field=fmt,
        claim_text=f"The film '{title_text}' was shot using the following format and camera systems: {fmt.value}.",
        add_ins=(common_ins + " Look for mentions of 65mm (or 70mm presentation), IMAX 15‑perf/15/70, and Ultra Panavision 70 (or equivalent wording).")
    )

    # 6) Composer
    comp = data.composer or FieldValue()
    add_field_block(
        node_id="Composer",
        node_desc="Identify Ludwig Göransson as the composer",
        field=comp,
        claim_text=f"The composer who created the score for '{title_text}' is {comp.value}.",
        add_ins=(common_ins + " Look for 'music by' or 'score by' language.")
    )

    # 7) Distributor
    dist = data.distributor or FieldValue()
    add_field_block(
        node_id="Distributor",
        node_desc="Identify Warner Bros. Pictures as the distributor",
        field=dist,
        claim_text=f"The distributor of '{title_text}' is {dist.value}.",
        add_ins=(common_ins + " Confirm the studio that handled the film's theatrical distribution (especially in the U.S.).")
    )

    # 8) Production Company
    prodco = data.production_company or FieldValue()
    add_field_block(
        node_id="Production_Company",
        node_desc="Identify Proximity Media as the production company",
        field=prodco,
        claim_text=f"The production company for '{title_text}' is {prodco.value}.",
        add_ins=(common_ins + " Verify the credited production company/companies.")
    )

    # 9) Budget Range (in millions)
    budget = data.budget_range_millions or FieldValue()
    add_field_block(
        node_id="Budget_Range",
        node_desc="Specify the production budget was in the range of $90-100 million",
        field=budget,
        claim_text=f"The production budget for '{title_text}' was {budget.value}.",
        add_ins=(common_ins + " Allow ranges or approximations and minor formatting differences; focus on the stated range in millions of USD.")
    )

    # 10) Worldwide Box Office Total (in millions)
    gross = data.box_office_total_millions or FieldValue()
    add_field_block(
        node_id="Box_Office_Total",
        node_desc="Specify the film grossed $369 million worldwide",
        field=gross,
        claim_text=f"The total worldwide box office gross for '{title_text}' was {gross.value}.",
        add_ins=(common_ins + " Accept approximate phrasing (e.g., 'about' or rounding). Confirm it is the worldwide total.")
    )

    # 11) Running Time (minutes)
    runtime = data.running_time_minutes or FieldValue()
    add_field_block(
        node_id="Running_Time",
        node_desc="Specify the film's running time is 138 minutes",
        field=runtime,
        claim_text=f"The running time of '{title_text}' is {runtime.value}.",
        add_ins=(common_ins + " Accept 'minutes' or 'min' as equivalent.")
    )

    # 12) US Theatrical Release Date
    us_release = data.us_release_date or FieldValue()
    add_field_block(
        node_id="US_Release_Date",
        node_desc="Specify the US theatrical release date was April 18, 2025",
        field=us_release,
        claim_text=f"The US theatrical release date of '{title_text}' was {us_release.value}.",
        add_ins=(common_ins + " Confirm the U.S. theatrical release date (not international or streaming).")
    )

    # 13) Premiere Venue
    prem_venue = data.premiere_venue or FieldValue()
    add_field_block(
        node_id="Premiere_Venue",
        node_desc="Identify that the premiere took place at AMC Lincoln Square in New York City",
        field=prem_venue,
        claim_text=f"The premiere of '{title_text}' took place at {prem_venue.value}.",
        add_ins=(common_ins + " Accept variations like 'AMC Lincoln Square 13'. Confirm it is in New York City.")
    )

    # 14) Premiere Date
    prem_date = data.premiere_date or FieldValue()
    add_field_block(
        node_id="Premiere_Date",
        node_desc="Specify the premiere date was April 3, 2025",
        field=prem_date,
        claim_text=f"The premiere date of '{title_text}' was {prem_date.value}.",
        add_ins=(common_ins + " Confirm the stated premiere date (month day, year).")
    )

    # 15) Principal Photography Period
    principal = data.principal_photography_period or FieldValue()
    add_field_block(
        node_id="Principal_Photography_Period",
        node_desc="Specify that principal photography took place from April to July 2024",
        field=principal,
        claim_text=f"Principal photography for '{title_text}' took place during {principal.value}.",
        add_ins=(common_ins + " Confirm the month–year range for principal photography; allow minor formatting differences like 'April–July 2024'.")
    )

    # Run all verifications (those with missing preconditions will be skipped automatically)
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


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
    # Initialize evaluator with a parallel root (allows partial credit across attributes)
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

    # Extract structured film info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmInfoExtraction,
        extraction_name="film_info_extraction"
    )

    # Build verification tree according to the rubric and verify
    await _build_and_verify(evaluator, root, extracted)

    # Return structured summary including the verification tree and scores
    return evaluator.get_summary()