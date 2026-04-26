import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "manalapan_camden_research"
TASK_DESCRIPTION = (
    "A billionaire investor made two significant real estate purchases in Manalapan, Florida between 2022 and 2024. "
    "The first purchase, completed in 2022, was a residential estate spanning 16 acres for $173 million, which set a Florida residential price record at that time. "
    "The second purchase, completed in August 2024, was a hotel resort property with 309 rooms for $277.4 million.\n\n"
    "Separately, identify the architectural firm that designed a LEED Platinum certified corporate headquarters building for a water utility company located in Camden, New Jersey.\n\n"
    "Provide the following information:\n\n"
    "1. The name of the hotel resort purchased in August 2024\n"
    "2. The address or property name of the 16-acre residential estate purchased in 2022\n"
    "3. The name of the billionaire who made both purchases\n"
    "4. The name of the architectural firm that designed the water company headquarters\n"
    "5. The name of the water utility company\n"
    "6. The year the headquarters building was completed and opened\n"
    "7. The square footage of the headquarters building"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FloridaHotelInfo(BaseModel):
    """Information about the 2024 hotel resort purchase."""
    name: Optional[str] = None
    location: Optional[str] = None
    purchase_date: Optional[str] = None   # e.g., "August 2024"
    room_count: Optional[str] = None      # keep as string to allow formats like "309", "309 rooms", "309 keys"
    price: Optional[str] = None           # e.g., "$277.4 million"
    buyer_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FloridaEstateInfo(BaseModel):
    """Information about the 2022 16-acre estate purchase."""
    name_or_address: Optional[str] = None
    location: Optional[str] = None
    acreage: Optional[str] = None         # e.g., "16 acres"
    purchase_year: Optional[str] = None   # e.g., "2022"
    price: Optional[str] = None           # e.g., "$173 million"
    buyer_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArchitectureHQInfo(BaseModel):
    """Information about the Camden, NJ water utility HQ and designer."""
    firm_name: Optional[str] = None
    water_company_name: Optional[str] = None
    location: Optional[str] = None        # e.g., "Camden, New Jersey"
    completion_year: Optional[str] = None
    square_footage: Optional[str] = None
    leed_certification_level: Optional[str] = None  # e.g., "LEED Platinum"
    sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    """Top-level extraction aggregating all required fields and sources."""
    hotel: Optional[FloridaHotelInfo] = None
    estate: Optional[FloridaEstateInfo] = None
    billionaire_name: Optional[str] = None
    hq: Optional[ArchitectureHQInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract the requested details from the answer. Return a JSON object with the following structure and fields.
    For each field, extract exactly what is stated in the answer text. If not specified, return null; for URL lists, return an empty list if none are provided.

    {
      "hotel": {
        "name": string | null,                   // Name of the hotel resort purchased in August 2024
        "location": string | null,               // Location as stated (e.g., "Manalapan, Florida")
        "purchase_date": string | null,          // Date/period as stated (e.g., "August 2024")
        "room_count": string | null,             // As stated (e.g., "309", "309 rooms", "309 keys")
        "price": string | null,                  // As stated (e.g., "$277.4 million")
        "buyer_name": string | null,             // Buyer name as stated for this transaction
        "sources": string[]                      // All URLs in the answer that specifically support the hotel transaction
      },
      "estate": {
        "name_or_address": string | null,        // The property name or address of the 16-acre estate
        "location": string | null,               // Location as stated (e.g., "Manalapan, Florida")
        "acreage": string | null,                // As stated (e.g., "16 acres")
        "purchase_year": string | null,          // As stated (e.g., "2022")
        "price": string | null,                  // As stated (e.g., "$173 million")
        "buyer_name": string | null,             // Buyer name as stated for this transaction
        "sources": string[]                      // All URLs in the answer that specifically support the estate transaction
      },
      "billionaire_name": string | null,         // The name of the billionaire who made both purchases
      "hq": {
        "firm_name": string | null,              // The architectural firm that designed the HQ
        "water_company_name": string | null,     // The water utility company
        "location": string | null,               // As stated (e.g., "Camden, New Jersey")
        "completion_year": string | null,        // Year completed/opened
        "square_footage": string | null,         // As stated
        "leed_certification_level": string | null, // As stated (e.g., "LEED Platinum")
        "sources": string[]                      // All URLs in the answer that support the HQ claims
      }
    }

    Special guidance:
    - Only extract information explicitly present in the answer.
    - For URLs: extract actual URLs (including those in markdown links). If no URL is present, return an empty list.
    - Do not infer or create data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_florida_properties_section(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    """Build and verify the Florida properties research section."""
    section_node = evaluator.add_parallel(
        id="Florida_Properties_Research",
        desc="Identify and verify information about the billionaire's two property purchases in Manalapan, Florida",
        parent=parent_node,
        critical=True
    )

    # ----- Hotel Resort Information -----
    hotel_info = data.hotel or FloridaHotelInfo()
    hotel_node = evaluator.add_parallel(
        id="Hotel_Resort_Information",
        desc="Identify the hotel resort purchased in August 2024 (309 rooms, $277.4M) in Manalapan, Florida",
        parent=section_node,
        critical=True
    )

    # Existence: Hotel name provided
    evaluator.add_custom_node(
        result=bool(hotel_info.name and hotel_info.name.strip()),
        id="Hotel_Name_Provided",
        desc="The answer provides the name of the hotel resort purchased in August 2024",
        parent=hotel_node,
        critical=True
    )

    # Create verification leaves
    hotel_loc_leaf = evaluator.add_leaf(
        id="Hotel_Location",
        desc="The identified hotel resort is located in Manalapan, Florida",
        parent=hotel_node,
        critical=True
    )
    hotel_date_leaf = evaluator.add_leaf(
        id="Hotel_Purchase_Date",
        desc="The identified hotel resort purchase occurred in August 2024",
        parent=hotel_node,
        critical=True
    )
    hotel_rooms_leaf = evaluator.add_leaf(
        id="Hotel_Room_Count",
        desc="The identified hotel resort has 309 rooms",
        parent=hotel_node,
        critical=True
    )
    hotel_price_leaf = evaluator.add_leaf(
        id="Hotel_Purchase_Price",
        desc="The identified hotel resort purchase price was $277.4 million",
        parent=hotel_node,
        critical=True
    )

    hotel_claims = [
        (
            f"The hotel resort named '{_safe(hotel_info.name)}' is located in Manalapan, Florida.",
            hotel_info.sources,
            hotel_loc_leaf,
            "Confirm the property is in Manalapan, Florida (Palm Beach County). Minor phrasing differences like 'Manalapan, FL' are acceptable."
        ),
        (
            f"The purchase of '{_safe(hotel_info.name)}' occurred in August 2024.",
            hotel_info.sources,
            hotel_date_leaf,
            "Confirm the transaction closed/occurred in August 2024. Accept variants like 'Aug. 2024'."
        ),
        (
            f"The hotel '{_safe(hotel_info.name)}' has 309 rooms.",
            hotel_info.sources,
            hotel_rooms_leaf,
            "Confirm the room count is 309. Accept synonyms like '309 keys' or '309 guestrooms'."
        ),
        (
            f"The purchase price for '{_safe(hotel_info.name)}' was $277.4 million.",
            hotel_info.sources,
            hotel_price_leaf,
            "Confirm the price as $277.4 million. Accept numeric equivalents like $277,400,000 or phrasing such as 'about $277.4 million'."
        ),
    ]
    await evaluator.batch_verify(hotel_claims)

    # ----- Estate Information -----
    estate_info = data.estate or FloridaEstateInfo()
    estate_node = evaluator.add_parallel(
        id="Estate_Information",
        desc="Identify the 16-acre residential estate purchased in 2022 for $173M in Manalapan, Florida, and that it set a Florida residential price record at the time",
        parent=section_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(estate_info.name_or_address and estate_info.name_or_address.strip()),
        id="Estate_Address_Or_Name_Provided",
        desc="The answer provides the address or property name of the 16-acre residential estate purchased in 2022",
        parent=estate_node,
        critical=True
    )

    estate_loc_leaf = evaluator.add_leaf(
        id="Estate_Location",
        desc="The identified estate is located in Manalapan, Florida",
        parent=estate_node,
        critical=True
    )
    estate_acre_leaf = evaluator.add_leaf(
        id="Estate_Acreage",
        desc="The identified estate spans 16 acres",
        parent=estate_node,
        critical=True
    )
    estate_year_leaf = evaluator.add_leaf(
        id="Estate_Purchase_Year",
        desc="The identified estate was purchased in 2022",
        parent=estate_node,
        critical=True
    )
    estate_price_leaf = evaluator.add_leaf(
        id="Estate_Purchase_Price",
        desc="The identified estate purchase price was $173 million",
        parent=estate_node,
        critical=True
    )
    estate_record_leaf = evaluator.add_leaf(
        id="Estate_Record_Claim",
        desc="The identified estate purchase set a Florida residential price record at the time",
        parent=estate_node,
        critical=True
    )

    estate_claims = [
        (
            f"The estate named/addressed '{_safe(estate_info.name_or_address)}' is located in Manalapan, Florida.",
            estate_info.sources,
            estate_loc_leaf,
            "Confirm the estate is in Manalapan, Florida. Accept minor phrasing variations."
        ),
        (
            "The estate spans 16 acres.",
            estate_info.sources,
            estate_acre_leaf,
            "Confirm the acreage is 16 acres. Accept small formatting variations."
        ),
        (
            "The estate purchase occurred in 2022.",
            estate_info.sources,
            estate_year_leaf,
            "Confirm the transaction date/year is 2022. Accept 'purchased in 2022' or equivalent phrasing."
        ),
        (
            "The estate purchase price was $173 million.",
            estate_info.sources,
            estate_price_leaf,
            "Confirm the price as $173 million. Accept numeric equivalent like $173,000,000."
        ),
        (
            "This estate purchase set a Florida residential price record at the time.",
            estate_info.sources,
            estate_record_leaf,
            "Confirm the sale was the highest/record residential price in Florida at that time. Accept equivalent phrasing indicating a state residential sale record."
        ),
    ]
    await evaluator.batch_verify(estate_claims)

    # ----- Billionaire Identification -----
    buyer_node = evaluator.add_parallel(
        id="Billionaire_Identification",
        desc="Identify the billionaire who made both Manalapan purchases",
        parent=section_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.billionaire_name and data.billionaire_name.strip()),
        id="Billionaire_Name_Provided",
        desc="The answer provides the name of the billionaire investor who made the purchases",
        parent=buyer_node,
        critical=True
    )

    same_buyer_leaf = evaluator.add_leaf(
        id="Same_Buyer_Verification",
        desc="The answer indicates/establishes that the same billionaire made both the 2022 estate purchase and the August 2024 hotel purchase",
        parent=buyer_node,
        critical=True
    )

    same_buyer_claim = (
        f"The same billionaire '{_safe(data.billionaire_name)}' made both purchases described: "
        f"the 2022 16-acre Manalapan estate for $173 million and the August 2024 309-room Manalapan hotel for $277.4 million."
    )
    await evaluator.verify(
        claim=same_buyer_claim,
        node=same_buyer_leaf,
        sources=None,
        additional_instruction=(
            "Verify this based on the provided answer text. Determine whether the answer itself clearly states or establishes "
            "that the same person made both purchases. The name may be stated once; focus on internal consistency."
        ),
    )


async def build_architecture_section(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    """Build and verify the Architecture research section."""
    section_node = evaluator.add_parallel(
        id="Architecture_Research",
        desc="Identify the architectural firm and provide details about the LEED Platinum water utility headquarters in Camden, New Jersey",
        parent=parent_node,
        critical=True
    )

    hq_info = data.hq or ArchitectureHQInfo()

    # ----- Architectural Firm Information -----
    firm_node = evaluator.add_parallel(
        id="Architectural_Firm_Information",
        desc="Identify the firm that designed the LEED Platinum corporate headquarters for a water utility company in Camden, NJ",
        parent=section_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hq_info.firm_name and hq_info.firm_name.strip()),
        id="Firm_Name_Provided",
        desc="The answer provides the name of the architectural firm",
        parent=firm_node,
        critical=True
    )

    leed_leaf = evaluator.add_leaf(
        id="LEED_Platinum_Certification",
        desc="The headquarters building is LEED Platinum certified",
        parent=firm_node,
        critical=True
    )
    water_hq_leaf = evaluator.add_leaf(
        id="Water_Utility_HQ",
        desc="The building is a corporate headquarters for a water utility company",
        parent=firm_node,
        critical=True
    )
    camden_leaf = evaluator.add_leaf(
        id="Camden_Location",
        desc="The headquarters is located in Camden, New Jersey",
        parent=firm_node,
        critical=True
    )

    firm_claims = [
        (
            "The corporate headquarters building is LEED Platinum certified.",
            hq_info.sources,
            leed_leaf,
            "Confirm the building achieved LEED Platinum certification. Accept explicit statements like 'LEED Platinum'."
        ),
        (
            "The building serves as the corporate headquarters for a water utility company.",
            hq_info.sources,
            water_hq_leaf,
            "Confirm the building is the HQ of a water utility company (not just an office or plant)."
        ),
        (
            "The headquarters building is located in Camden, New Jersey.",
            hq_info.sources,
            camden_leaf,
            "Confirm the location as Camden, New Jersey."
        ),
    ]
    await evaluator.batch_verify(firm_claims)

    # ----- Water Company Name -----
    company_node = evaluator.add_parallel(
        id="Water_Company_Name",
        desc="Provide the name of the water utility company",
        parent=section_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hq_info.water_company_name and hq_info.water_company_name.strip()),
        id="Company_Name_Provided",
        desc="The answer provides the name of the water utility company",
        parent=company_node,
        critical=True
    )

    # ----- Building Specifications -----
    specs_node = evaluator.add_parallel(
        id="Building_Specifications",
        desc="Provide the completion/opening year and the square footage of the headquarters building",
        parent=section_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hq_info.completion_year and hq_info.completion_year.strip()),
        id="Completion_Year_Provided",
        desc="The answer provides a specific year the headquarters building was completed and opened",
        parent=specs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hq_info.square_footage and hq_info.square_footage.strip()),
        id="Square_Footage_Provided",
        desc="The answer provides a specific square footage value for the headquarters building",
        parent=specs_node,
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
    Evaluate an answer for the Manalapan property purchases and Camden HQ architecture task.
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

    # Add a top-level critical node matching the rubric's root
    complete_node = evaluator.add_parallel(
        id="Complete_Research_Task",
        desc="Complete research identifying Florida property acquisitions and New Jersey architectural project details",
        parent=root,
        critical=True
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=ResearchExtraction,
        extraction_name="extracted_research"
    )

    # Build and verify sections
    await asyncio.gather(
        build_florida_properties_section(evaluator, complete_node, extracted),
        build_architecture_section(evaluator, complete_node, extracted)
    )

    # Return evaluation summary
    return evaluator.get_summary()