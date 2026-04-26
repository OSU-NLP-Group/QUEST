import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "major_2025_dog_shows_winners"
TASK_DESCRIPTION = (
    "For the three major AKC-sanctioned all-breed dog shows held in the United States in 2025 — the National Dog Show "
    "(held on Thanksgiving Day), the Westminster Kennel Club Dog Show (149th annual), and the AKC National "
    "Championship (held in December in Orlando) — identify the Best in Show winner for each show. For each winner, "
    "provide: (1) the dog's name and breed, (2) the handler's full name and location (city and state), and (3) the "
    "specific dog food brand and formula fed to the dog."
)


# -----------------------------------------------------------------------------
# Ground truth expectations encoded from rubric (used for "stated-as-expected")
# -----------------------------------------------------------------------------
NDS_EXPECT = {
    "annual_edition": "24th",
    "held_on_thanksgiving": True,
    "dog_name": "Soleil",
    "breed": "Belgian Sheepdog",
    "age": "5½ years old",
    "sex": "female",
    "handler_full_name": "Daniel Martin",
    "handler_location": "Princeton, North Carolina",
    "food_brand": "Purina Pro Plan",
    "food_formula": "Sport Performance 30/20 Chicken & Rice Formula",
}

WKC_EXPECT = {
    "annual_edition": "149th",
    "held_in_feb_2025": True,
    "dog_name": "Monty",
    "breed": "Giant Schnauzer",
    "age": "5 years old",
    "sex": "male",
    "owner_handled": True,
    "handler_full_name": "Katie Bernardin",
    "handler_location": "Chaplin, Connecticut",
    "historic_first": "first Giant Schnauzer to win Westminster Best in Show",
    "food_brand": "Purina Pro Plan",
    "food_formula": "All-Life Stages SPORT Performance 30/20 Salmon & Rice Formula",
}

AKCNC_EXPECT = {
    "held_in_dec_2025": True,
    "held_in_orlando": True,
    "dog_name": "JJ",
    "breed": "Lhasa Apso",
    "age": "5 years old",
    "handler_full_name_allowed": ["Susie Giles", "Susan Giles"],  # Accept either
    "group_before_bis": "Non-Sporting Group",
    "food_brand": "Purina Pro Plan",
    # Formula must be specific (any valid named formula, not just the brand)
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class ShowDetails(BaseModel):
    # General show details
    annual_edition: Optional[str] = None          # e.g., "24th" or "149th"
    event_timing: Optional[str] = None            # e.g., "Thanksgiving Day", "February 2025", "December 2025"
    location_city_state: Optional[str] = None     # e.g., "Princeton, North Carolina", "Chaplin, Connecticut", etc.

    # Winner info
    winner_dog_name: Optional[str] = None
    winner_breed: Optional[str] = None
    winner_age: Optional[str] = None
    winner_sex: Optional[str] = None

    # Handler info
    handler_full_name: Optional[str] = None
    handler_location_city_state: Optional[str] = None

    # Special statements (per-show)
    owner_handled_status: Optional[str] = None      # e.g., "owner-handled" (WKC only)
    historic_first_statement: Optional[str] = None  # e.g., "first Giant Schnauzer ..." (WKC only)
    group_before_bis: Optional[str] = None          # e.g., "Non-Sporting Group" (AKCNC only)

    # Food info
    food_brand: Optional[str] = None
    food_formula: Optional[str] = None

    # All URLs that the answer attributed for this show (any or all related claims)
    sources: List[str] = Field(default_factory=list)


class AllShowsExtraction(BaseModel):
    national_dog_show_2025: Optional[ShowDetails] = None
    westminster_2025: Optional[ShowDetails] = None
    akc_national_championship_2025: Optional[ShowDetails] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_shows() -> str:
    return """
    Extract structured details for each of the three 2025 US all-breed dog shows as presented in the answer text.
    Only extract information explicitly present in the answer. Do not invent anything.

    For each show, extract the following fields (use null if not present in the answer):
    - annual_edition: e.g., "24th", "149th" (string as written)
    - event_timing: e.g., "Thanksgiving Day", "February 2025", "December 2025"
    - location_city_state: a "City, State" string for the event location if stated at the show-level (optional)
    - winner_dog_name
    - winner_breed
    - winner_age (as written in the answer, e.g., "5½ years old", "5.5 years", "five and a half years")
    - winner_sex (e.g., "male", "female")
    - handler_full_name (full handler name as written)
    - handler_location_city_state (a "City, State" string as written)
    - owner_handled_status (e.g., "owner-handled") — primarily relevant for Westminster; null otherwise
    - historic_first_statement (e.g., "first Giant Schnauzer ...") — primarily relevant for Westminster; null otherwise
    - group_before_bis (e.g., "Non-Sporting Group") — primarily relevant for AKC National Championship; null otherwise
    - food_brand (e.g., "Purina Pro Plan")
    - food_formula (e.g., "Sport Performance 30/20 Chicken & Rice Formula"; a specific named formula)
    - sources: an array of all distinct URLs (plain or in markdown link form) that the answer cites for THIS show specifically

    Organize the output under these exact top-level keys:
    - national_dog_show_2025
    - westminster_2025
    - akc_national_championship_2025

    For each show, if a field is not stated in the answer, set it to null (or empty list for sources).
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
INS_STATED = (
    "Judge based only on the provided answer text. Accept case-insensitive or minor formatting/spelling variations. "
    "Do not use external knowledge."
)

INS_SUPPORTED_GENERIC = (
    "Judge based only on the content of the provided webpages/screenshots. Accept minor name/title variations. "
    "If the provided URLs are missing or irrelevant, this claim should not be considered supported."
)

INS_SUPPORTED_AGE = (
    INS_SUPPORTED_GENERIC
    + " For age, accept reasonable equivalents such as '5½ years', '5.5 years', or 'five and a half years'."
)

INS_SUPPORTED_HANDLER_NAME_VARIANT = (
    INS_SUPPORTED_GENERIC
    + " Accept 'Susie Giles' and 'Susan Giles' as the same person (nickname vs formal)."
)

def _safe_sources(s: Optional[List[str]]) -> List[str]:
    return [u for u in (s or []) if isinstance(u, str) and u.strip()]


def add_dual_check(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    base_desc: str,
    stated_claim: str,
    supported_claim: str,
    sources: List[str],
    *,
    sources_gate_node=None,
    critical: bool = True,
    addl_ins_stated: Optional[str] = None,
    addl_ins_supported: Optional[str] = None,
):
    """
    For a single rubric item, add a parallel aggregator with two leaf checks:
    - stated: whether the answer explicitly states the expected info
    - supported: whether the info is supported by the cited sources
    """
    agg = evaluator.add_parallel(
        id=base_id,
        desc=base_desc,
        parent=parent_node,
        critical=critical,
    )

    # stated-in-answer leaf
    stated_node = evaluator.add_leaf(
        id=f"{base_id}_stated",
        desc=f"{base_desc} — stated in the answer",
        parent=agg,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim=stated_claim,
        node=stated_node,
        additional_instruction=addl_ins_stated or INS_STATED,
    ))

    # supported-by-sources leaf
    supported_node = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=f"{base_desc} — supported by cited sources",
        parent=agg,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim=supported_claim,
        node=supported_node,
        sources=sources,
        additional_instruction=addl_ins_supported or INS_SUPPORTED_GENERIC,
        extra_prerequisites=[sources_gate_node] if sources_gate_node else None,
    ))
    return agg


# -----------------------------------------------------------------------------
# Show-specific verifications
# -----------------------------------------------------------------------------
async def verify_nds(evaluator: Evaluator, parent_node, nds: Optional[ShowDetails]) -> None:
    show_node = evaluator.add_parallel(
        id="National_Dog_Show_2025",
        desc="National Dog Show (2025) requirements and constraints",
        parent=parent_node,
        critical=False,
    )

    sources = _safe_sources(nds.sources if nds else [])
    sources_gate = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="NDS_Sources_Provided",
        desc="At least one source URL is provided for National Dog Show 2025 details",
        parent=show_node,
        critical=False,
    )

    # 1) Annual edition (24th)
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Annual_Edition",
        "States the National Dog Show is the 24th annual (as constrained).",
        stated_claim="The answer states that the 2025 National Dog Show is the 24th annual edition.",
        supported_claim="The 2025 National Dog Show was the 24th annual National Dog Show.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 2) Held on Thanksgiving Day
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Held_On_Thanksgiving",
        "States the National Dog Show is held on Thanksgiving Day (as constrained).",
        stated_claim="The answer states that the National Dog Show is held on Thanksgiving Day.",
        supported_claim="The National Dog Show is held on Thanksgiving Day.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 3) Winner dog's name: Soleil
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Winner_Dog_Name",
        "Identifies the National Dog Show 2025 Best in Show winner dog's name as Soleil (as constrained).",
        stated_claim="The answer identifies the 2025 National Dog Show Best in Show winner as 'Soleil'.",
        supported_claim="The 2025 National Dog Show Best in Show winner was 'Soleil'.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 4) Breed: Belgian Sheepdog
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Winner_Dog_Breed",
        "Identifies Soleil's breed as Belgian Sheepdog (as constrained).",
        stated_claim="The answer states that Soleil's breed is Belgian Sheepdog.",
        supported_claim="Soleil, the 2025 National Dog Show Best in Show winner, is a Belgian Sheepdog.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 5) Age: 5½ years old
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Winner_Age",
        "States Soleil's age as 5½ years old (as constrained).",
        stated_claim=(
            "The answer states that Soleil's age is about five and a half years old "
            "(e.g., '5½ years old', '5.5 years old', or equivalent)."
        ),
        supported_claim="Soleil's age at the time of the 2025 National Dog Show was approximately 5½ years.",
        sources=sources,
        sources_gate_node=sources_gate,
        addl_ins_supported=INS_SUPPORTED_AGE,
    )

    # 6) Sex: female
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Winner_Sex",
        "States Soleil's sex as female (as constrained).",
        stated_claim="The answer states that Soleil is female.",
        supported_claim="Soleil is a female.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 7) Handler full name: Daniel Martin
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Handler_Full_Name",
        "States Soleil's handler full name as Daniel Martin (as constrained).",
        stated_claim="The answer states that Soleil was handled by Daniel Martin.",
        supported_claim="Soleil was handled by Daniel Martin.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 8) Handler location: Princeton, North Carolina
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Handler_Location",
        "States Soleil's handler location as Princeton, North Carolina (city and state) (as constrained).",
        stated_claim="The answer states that Daniel Martin is from Princeton, North Carolina.",
        supported_claim="Daniel Martin is from Princeton, North Carolina.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 9) Food brand: Purina Pro Plan
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Food_Brand",
        "States Soleil's dog food brand as Purina Pro Plan (as constrained).",
        stated_claim="The answer states that Soleil eats Purina Pro Plan.",
        supported_claim="Soleil's dog food brand is Purina Pro Plan.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 10) Food formula: Sport Performance 30/20 Chicken & Rice Formula
    add_dual_check(
        evaluator,
        show_node,
        "NDS_Food_Formula",
        "States Soleil's specific dog food formula as Sport Performance 30/20 Chicken & Rice Formula (as constrained).",
        stated_claim="The answer specifies that Soleil's food formula is 'Sport Performance 30/20 Chicken & Rice Formula'.",
        supported_claim="Soleil's specific food formula is Purina Pro Plan Sport Performance 30/20 Chicken & Rice Formula.",
        sources=sources,
        sources_gate_node=sources_gate,
    )


async def verify_wkc(evaluator: Evaluator, parent_node, wkc: Optional[ShowDetails]) -> None:
    show_node = evaluator.add_parallel(
        id="Westminster_2025",
        desc="Westminster Kennel Club Dog Show (2025) requirements and constraints",
        parent=parent_node,
        critical=False,
    )

    sources = _safe_sources(wkc.sources if wkc else [])
    sources_gate = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="WKC_Sources_Provided",
        desc="At least one source URL is provided for Westminster 2025 details",
        parent=show_node,
        critical=False,
    )

    # 1) Annual edition: 149th
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Annual_Edition",
        "States the Westminster show is the 149th annual (as constrained).",
        stated_claim="The answer states that the 2025 Westminster Kennel Club Dog Show is the 149th annual.",
        supported_claim="The 2025 Westminster Kennel Club Dog Show was the 149th annual event.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 2) Held in February 2025
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Held_In_February",
        "States the Westminster show is held in February 2025 (as constrained).",
        stated_claim="The answer states that the 2025 Westminster Kennel Club Dog Show was held in February 2025.",
        supported_claim="The 2025 Westminster Kennel Club Dog Show took place in February 2025.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 3) Winner dog's name: Monty
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Winner_Dog_Name",
        "Identifies the Westminster 2025 Best in Show winner dog's name as Monty (as constrained).",
        stated_claim="The answer identifies the 2025 Westminster Best in Show winner as 'Monty'.",
        supported_claim="The 2025 Westminster Kennel Club Dog Show Best in Show winner was 'Monty'.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 4) Breed: Giant Schnauzer
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Winner_Dog_Breed",
        "Identifies Monty's breed as Giant Schnauzer (as constrained).",
        stated_claim="The answer states that Monty's breed is Giant Schnauzer.",
        supported_claim="Monty, the 2025 Westminster BIS winner, is a Giant Schnauzer.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 5) Age: 5 years old
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Winner_Age",
        "States Monty's age as 5 years old (as constrained).",
        stated_claim="The answer states that Monty's age is 5 years old.",
        supported_claim="Monty's age at the time of the 2025 Westminster show was 5 years old (approximately).",
        sources=sources,
        sources_gate_node=sources_gate,
        addl_ins_supported=INS_SUPPORTED_AGE,
    )

    # 6) Sex: male
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Winner_Sex",
        "States Monty's sex as male (as constrained).",
        stated_claim="The answer states that Monty is male.",
        supported_claim="Monty is a male.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 7) Owner-handled
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Owner_Handled_Status",
        "States Monty is owner-handled (as constrained).",
        stated_claim="The answer states that Monty is owner-handled.",
        supported_claim="Monty was owner-handled at Westminster 2025.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 8) Handler full name: Katie Bernardin
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Handler_Full_Name",
        "States Monty's handler full name as Katie Bernardin (as constrained).",
        stated_claim="The answer states that Monty was handled by Katie Bernardin.",
        supported_claim="Monty, the 2025 Westminster BIS winner, was handled by Katie Bernardin.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 9) Handler location: Chaplin, Connecticut
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Handler_Location",
        "States Monty's handler location as Chaplin, Connecticut (city and state) (as constrained).",
        stated_claim="The answer states that Katie Bernardin is from Chaplin, Connecticut.",
        supported_claim="Katie Bernardin is from Chaplin, Connecticut.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 10) Historic first: first Giant Schnauzer to win Westminster BIS
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Historic_First",
        "States Monty is the first Giant Schnauzer in history to win Westminster Best in Show (as constrained).",
        stated_claim="The answer states that Monty is the first Giant Schnauzer to win Westminster Best in Show.",
        supported_claim="Monty is the first Giant Schnauzer ever to win Westminster Best in Show.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 11) Food brand: Purina Pro Plan
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Food_Brand",
        "States Monty's dog food brand as Purina Pro Plan (as constrained).",
        stated_claim="The answer states that Monty eats Purina Pro Plan.",
        supported_claim="Monty's dog food brand is Purina Pro Plan.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 12) Food formula: All-Life Stages SPORT Performance 30/20 Salmon & Rice Formula
    add_dual_check(
        evaluator,
        show_node,
        "WKC_Food_Formula",
        "States Monty's specific dog food formula as All-Life Stages SPORT Performance 30/20 Salmon & Rice Formula (as constrained).",
        stated_claim=(
            "The answer specifies that Monty's food formula is 'All-Life Stages SPORT Performance 30/20 Salmon & Rice Formula'."
        ),
        supported_claim=(
            "Monty's specific food formula is Purina Pro Plan All-Life Stages SPORT Performance 30/20 Salmon & Rice Formula."
        ),
        sources=sources,
        sources_gate_node=sources_gate,
    )


async def verify_akc_nc(evaluator: Evaluator, parent_node, akc: Optional[ShowDetails]) -> None:
    show_node = evaluator.add_parallel(
        id="AKC_National_Championship_2025",
        desc="AKC National Championship (2025) requirements and constraints",
        parent=parent_node,
        critical=False,
    )

    sources = _safe_sources(akc.sources if akc else [])
    sources_gate = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="AKCNC_Sources_Provided",
        desc="At least one source URL is provided for AKC National Championship 2025 details",
        parent=show_node,
        critical=False,
    )

    # 1) Held in December 2025
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Held_In_December",
        "States the AKC National Championship is held in December 2025 (as constrained).",
        stated_claim="The answer states that the 2025 AKC National Championship is held in December 2025.",
        supported_claim="The 2025 AKC National Championship took place in December 2025.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 2) Held in Orlando
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Held_In_Orlando",
        "States the AKC National Championship is held in Orlando (as constrained).",
        stated_claim="The answer states that the AKC National Championship is held in Orlando.",
        supported_claim="The AKC National Championship is held in Orlando.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 3) Winner dog's name: JJ
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Winner_Dog_Name",
        "Identifies the AKC National Championship 2025 Best in Show winner dog's name as JJ (as constrained).",
        stated_claim="The answer identifies the 2025 AKC National Championship Best in Show winner as 'JJ'.",
        supported_claim="The 2025 AKC National Championship Best in Show winner was 'JJ'.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 4) Breed: Lhasa Apso
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Winner_Dog_Breed",
        "Identifies JJ's breed as Lhasa Apso (as constrained).",
        stated_claim="The answer states that JJ's breed is Lhasa Apso.",
        supported_claim="JJ, the 2025 AKC National Championship BIS winner, is a Lhasa Apso.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 5) Age: 5 years old
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Winner_Age",
        "States JJ's age as 5 years old (as constrained).",
        stated_claim="The answer states that JJ's age is 5 years old.",
        supported_claim="JJ's age at the time of the 2025 AKC National Championship was 5 years old (approximately).",
        sources=sources,
        sources_gate_node=sources_gate,
        addl_ins_supported=INS_SUPPORTED_AGE,
    )

    # 6) Handler full name: Susie/Susan Giles (accept either)
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Handler_Full_Name",
        "States JJ was handled by Susie Giles; accept Susan Giles as an equivalent reference (as constrained).",
        stated_claim="The answer states that JJ was handled by Susie (or Susan) Giles.",
        supported_claim="JJ, the 2025 AKC National Championship BIS winner, was handled by Susie Giles (also known as Susan Giles).",
        sources=sources,
        sources_gate_node=sources_gate,
        addl_ins_supported=INS_SUPPORTED_HANDLER_NAME_VARIANT,
    )

    # 7) Handler location city+state must be provided; also verify with sources
    #    Use extracted value if available for the supported-by-sources claim.
    extracted_loc = (akc.handler_location_city_state or "") if akc else ""
    loc_has_city_and_state_node = evaluator.add_parallel(
        id="AKCNC_Handler_Location_City_State",
        desc="Provides JJ handler's location including both city and state (required by the proposed question).",
        parent=show_node,
        critical=True,
    )
    # 7a stated
    stated_loc_node = evaluator.add_leaf(
        id="AKCNC_Handler_Location_City_State_stated",
        desc="Handler location includes both city and state — stated in the answer",
        parent=loc_has_city_and_state_node,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim=(
            "In the answer, for the 2025 AKC National Championship, the handler's location includes both a city and a state, "
            "formatted like 'City, State'."
        ),
        node=stated_loc_node,
        additional_instruction=INS_STATED,
    ))
    # 7b supported
    supported_loc_node = evaluator.add_leaf(
        id="AKCNC_Handler_Location_City_State_supported",
        desc="Handler location includes both city and state — supported by cited sources",
        parent=loc_has_city_and_state_node,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim=(
            f"JJ's handler (Susie/Susan Giles) is based in {extracted_loc}."
            if extracted_loc else
            "JJ's handler (Susie/Susan Giles) has a stated location that includes both a city and a state."
        ),
        node=supported_loc_node,
        sources=sources,
        additional_instruction=INS_SUPPORTED_HANDLER_NAME_VARIANT,
        extra_prerequisites=[sources_gate] if sources_gate else None,
    ))

    # 8) Group before BIS: Non-Sporting Group
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Group_Before_BIS",
        "States JJ won the Non-Sporting Group before winning Best in Show (as constrained).",
        stated_claim="The answer states that JJ won the Non-Sporting Group before Best in Show.",
        supported_claim="JJ won the Non-Sporting Group before winning Best in Show at the 2025 AKC National Championship.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 9) Food brand: Purina Pro Plan
    add_dual_check(
        evaluator,
        show_node,
        "AKCNC_Food_Brand",
        "States JJ's dog food brand as Purina Pro Plan (as constrained).",
        stated_claim="The answer states that JJ eats Purina Pro Plan.",
        supported_claim="JJ's dog food brand is Purina Pro Plan.",
        sources=sources,
        sources_gate_node=sources_gate,
    )

    # 10) Food formula: must be a specific formula (not just brand)
    extracted_formula = (akc.food_formula or "").strip() if akc else ""
    # 10a: stated specificity (answer-level)
    formula_node = evaluator.add_parallel(
        id="AKCNC_Food_Formula",
        desc="Provides a specific dog food formula name for JJ (not just the brand) (required by the proposed question).",
        parent=show_node,
        critical=True,
    )
    stated_formula_specific_node = evaluator.add_leaf(
        id="AKCNC_Food_Formula_stated",
        desc="Specific food formula is provided — stated in the answer",
        parent=formula_node,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim="The answer specifies a particular Purina Pro Plan formula for JJ (not just the brand).",
        node=stated_formula_specific_node,
        additional_instruction=INS_STATED,
    ))
    # 10b: supported by cited sources (use whatever formula string was extracted)
    supported_formula_node = evaluator.add_leaf(
        id="AKCNC_Food_Formula_supported",
        desc="Specific food formula is provided — supported by cited sources",
        parent=formula_node,
        critical=True,
    )
    asyncio.create_task(evaluator.verify(
        claim=(
            f"JJ's specific dog food formula is Purina Pro Plan {extracted_formula}."
            if extracted_formula else
            "JJ has a specific named Purina Pro Plan formula (beyond the brand) used as his food."
        ),
        node=supported_formula_node,
        sources=sources,
        additional_instruction=INS_SUPPORTED_GENERIC,
        extra_prerequisites=[sources_gate] if sources_gate else None,
    ))


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Build and execute the evaluation tree for the 2025 major US all-breed dog shows.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel aggregation for top-level combination
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

    # Extract structured info from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_shows(),
        template_class=AllShowsExtraction,
        extraction_name="shows_extraction",
    )

    # Add ground-truth info snapshot for transparency (not used for scoring directly)
    evaluator.add_ground_truth(
        {
            "National_Dog_Show_2025": NDS_EXPECT,
            "Westminster_2025": WKC_EXPECT,
            "AKC_National_Championship_2025": {
                **AKCNC_EXPECT,
                "food_formula": "Must be a specific named Purina Pro Plan formula (not just the brand).",
            },
        },
        gt_type="expected_constraints",
    )

    # Major parent node (reflecting rubric root)
    major_node = evaluator.add_parallel(
        id="Major_2025_Dog_Shows_Winners",
        desc=(
            "Verify Best in Show winners and required details for the three named 2025 US all-breed shows "
            "(National Dog Show, Westminster Kennel Club Dog Show, AKC National Championship), including all stated constraints."
        ),
        parent=root,
        critical=False,  # Keep non-critical to allow partial credit across shows
    )

    # Spawn verifications for each show
    await asyncio.gather(
        verify_nds(evaluator, major_node, extracted.national_dog_show_2025),
        verify_wkc(evaluator, major_node, extracted.westminster_2025),
        verify_akc_nc(evaluator, major_node, extracted.akc_national_championship_2025),
    )

    # Return structured summary
    return evaluator.get_summary()