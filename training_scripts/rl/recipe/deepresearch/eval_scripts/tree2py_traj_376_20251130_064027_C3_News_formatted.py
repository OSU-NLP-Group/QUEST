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
TASK_ID = "steve_witkoff_career"
TASK_DESCRIPTION = """
Research and document the professional career development of Steve Witkoff, the real estate developer who was appointed as U.S. Special Envoy to the Middle East in 2024. Your research should trace his path from legal education through founding his real estate development company and completing a major Manhattan building acquisition. Specifically, provide the following information with supporting reference URLs: (1) The name of the law school from which Steve Witkoff earned his Juris Doctor (JD) degree and the year he graduated, (2) The name of the New York City law firm where he worked as his first legal employment after graduating from law school, (3) The name of the real estate company he co-founded in 1985, the name of his co-founding partner (who was also an attorney from the same law firm), and explain the origin of the company name, (4) The name of the real estate development company he founded in 1997 after leaving the company mentioned in item 3, (5) Identify one major historic Manhattan building that Steve Witkoff's company (founded in 1997) acquired in 1998, and provide: the building's name, the year of acquisition, the purchase price, the name of his business partner in this acquisition, the Manhattan neighborhood where the building is located, and the name of the building's original architect. All information must be supported by reference URLs from publicly available and verifiable sources.
"""

# Ground truth expectations used to validate constraints
EXPECTED_FACTS = {
    "jd_school": "Hofstra University School of Law",  # Also known as "Maurice A. Deane School of Law at Hofstra University"
    "jd_year": "1983",
    "first_firm": "Dreyer & Traub",
    "company_1985": "Stellar Management",
    "cofounder_1985": "Larry Gluck",
    "cofounder_1985_firm": "Dreyer & Traub",
    "stellar_name_origin": "Steve + Larry",
    "company_1997": "The Witkoff Group",
    "acquisition_building_1998": "Woolworth Building",
    "acquisition_year_1998": "1998",
    "purchase_price_1998": "$155 million",
    "business_partner_1998": "Rubin Schron",
    "neighborhood_1998": "Tribeca",
    "architect_woolworth": "Cass Gilbert",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LawSchoolInfo(BaseModel):
    jd_school_name: Optional[str] = None
    jd_graduation_year: Optional[str] = None
    sources_school: List[str] = Field(default_factory=list)
    sources_year: List[str] = Field(default_factory=list)

class FirstLawFirmInfo(BaseModel):
    first_law_firm_name: Optional[str] = None
    city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class Company1985Info(BaseModel):
    company_name: Optional[str] = None
    cofounder_name: Optional[str] = None
    cofounder_law_firm: Optional[str] = None
    name_origin_explanation: Optional[str] = None
    sources_company: List[str] = Field(default_factory=list)
    sources_cofounder: List[str] = Field(default_factory=list)
    sources_origin: List[str] = Field(default_factory=list)

class Company1997Info(BaseModel):
    company_name: Optional[str] = None
    year_founded: Optional[str] = None
    left_previous_company_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class Building1998Info(BaseModel):
    building_name: Optional[str] = None
    acquisition_year: Optional[str] = None
    purchase_price: Optional[str] = None
    business_partner: Optional[str] = None
    neighborhood: Optional[str] = None
    original_architect: Optional[str] = None
    sources_building: List[str] = Field(default_factory=list)
    sources_year: List[str] = Field(default_factory=list)
    sources_price: List[str] = Field(default_factory=list)
    sources_partner: List[str] = Field(default_factory=list)
    sources_neighborhood: List[str] = Field(default_factory=list)
    sources_architect: List[str] = Field(default_factory=list)

class CareerExtraction(BaseModel):
    item1: Optional[LawSchoolInfo] = None
    item2: Optional[FirstLawFirmInfo] = None
    item3: Optional[Company1985Info] = None
    item4: Optional[Company1997Info] = None
    item5: Optional[Building1998Info] = None

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career() -> str:
    return """
    Extract the career-development facts about Steve Witkoff exactly as presented in the answer, and collect the supporting reference URLs for each fact. Return a single JSON object with the following nested structure. If any field is missing from the answer, return null for that field; if a sources list is missing, return an empty array for that sources list.

    Required JSON schema:
    {
      "item1": {
        "jd_school_name": string | null,
        "jd_graduation_year": string | null,
        "sources_school": string[]  // URLs supporting the JD school
        "sources_year": string[]    // URLs supporting the graduation year
      },
      "item2": {
        "first_law_firm_name": string | null,
        "city": string | null,  // e.g., "New York City" if stated
        "sources": string[]     // URLs supporting the first law firm employment after law school
      },
      "item3": {
        "company_name": string | null, // e.g., "Stellar Management"
        "cofounder_name": string | null, // e.g., "Larry Gluck" (or "Laurence Gluck")
        "cofounder_law_firm": string | null, // e.g., "Dreyer & Traub"
        "name_origin_explanation": string | null, // e.g., "Stellar is from Steve + Larry"
        "sources_company": string[],   // URLs supporting co-founding in 1985 and company name
        "sources_cofounder": string[], // URLs supporting cofounder identity and law firm
        "sources_origin": string[]     // URLs supporting name origin explanation
      },
      "item4": {
        "company_name": string | null, // e.g., "The Witkoff Group"
        "year_founded": string | null, // e.g., "1997"
        "left_previous_company_name": string | null, // e.g., "Stellar Management", if stated
        "sources": string[] // URLs supporting leaving Stellar and founding The Witkoff Group in 1997
      },
      "item5": {
        "building_name": string | null,          // e.g., "Woolworth Building"
        "acquisition_year": string | null,       // e.g., "1998"
        "purchase_price": string | null,         // e.g., "$155 million"
        "business_partner": string | null,       // e.g., "Rubin Schron"
        "neighborhood": string | null,           // e.g., "Tribeca"
        "original_architect": string | null,     // e.g., "Cass Gilbert"
        "sources_building": string[],            // URLs supporting the building identity in Witkoff acquisition context
        "sources_year": string[],                // URLs supporting the acquisition year
        "sources_price": string[],               // URLs supporting the purchase price
        "sources_partner": string[],             // URLs supporting the business partner identity
        "sources_neighborhood": string[],        // URLs supporting the building neighborhood/location
        "sources_architect": string[]            // URLs supporting the original architect
      }
    }

    Special rules for sources extraction:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - Accept plain URLs or markdown links [text](url). Always extract the actual URL string.
    - If a URL is missing a protocol (http/https), prepend http://
    - If the answer references a source without a URL (e.g., "according to Wikipedia"), return an empty array for that sources list.
    - Do not deduplicate; include all URLs mentioned, even if repeated.

    Notes:
    - Preserve names as stated in the answer (e.g., "Maurice A. Deane School of Law at Hofstra University" is equivalent to Hofstra Law School).
    - For amounts (e.g., "$155 million"), keep the string as-is (including symbols or words like "about", "approximately", etc.).
    """

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]

# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_item_1(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Item_1_Law_School_Education",
        desc="Verify JD law school and graduation year match constraints, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    item1 = ex.item1 or LawSchoolInfo()

    # Source presence checks (critical single-step checks)
    evaluator.add_custom_node(
        result=len(non_empty_urls(item1.sources_school)) > 0,
        id="Item_1_school_urls_present",
        desc="At least one URL is provided supporting the JD school claim",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item1.sources_year)) > 0,
        id="Item_1_year_urls_present",
        desc="At least one URL is provided supporting the JD graduation year claim",
        parent=item_node,
        critical=True
    )

    # JD school verification
    jd_school_leaf = evaluator.add_leaf(
        id="JD_From_Hofstra_Law_School_With_URL",
        desc="States that Witkoff earned his JD from Hofstra Law School (or Hofstra University School of Law) AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    jd_school_claim = (
        "Steve Witkoff earned his Juris Doctor (JD) from Hofstra University School of Law "
        "(also known as the Maurice A. Deane School of Law at Hofstra University)."
    )
    await evaluator.verify(
        claim=jd_school_claim,
        node=jd_school_leaf,
        sources=non_empty_urls(item1.sources_school),
        additional_instruction="Accept synonyms like 'Hofstra Law School' or 'Maurice A. Deane School of Law at Hofstra University'. The source must explicitly tie Steve Witkoff to earning a JD from this law school."
    )

    # JD graduation year verification
    jd_year_leaf = evaluator.add_leaf(
        id="JD_Graduation_Year_1983_With_URL",
        desc="States that Witkoff graduated (JD) in 1983 AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    jd_year_claim = "Steve Witkoff received his JD in 1983."
    await evaluator.verify(
        claim=jd_year_claim,
        node=jd_year_leaf,
        sources=non_empty_urls(item1.sources_year),
        additional_instruction="The page must clearly indicate the year 1983 for Witkoff’s JD graduation. Accept phrases like 'Class of 1983' or 'earned his JD in 1983'."
    )

async def verify_item_2(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Item_2_First_Law_Firm",
        desc="Verify first legal employment after law school matches constraints, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    item2 = ex.item2 or FirstLawFirmInfo()

    evaluator.add_custom_node(
        result=len(non_empty_urls(item2.sources)) > 0,
        id="Item_2_first_firm_urls_present",
        desc="At least one URL is provided supporting the first legal employment firm",
        parent=item_node,
        critical=True
    )

    first_firm_leaf = evaluator.add_leaf(
        id="First_Legal_Job_Dreyer_And_Traub_With_URL",
        desc="States that Witkoff's first legal employment after law school was at Dreyer & Traub in New York City AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    first_firm_claim = "After graduating from law school, Steve Witkoff’s first legal employment was at Dreyer & Traub in New York City."
    await evaluator.verify(
        claim=first_firm_claim,
        node=first_firm_leaf,
        sources=non_empty_urls(item2.sources),
        additional_instruction="Accept equivalent phrasing like 'began his legal career at Dreyer & Traub.' If the specific 'New York City' phrasing is not explicit, it is acceptable if Dreyer & Traub is clearly a NYC firm."
    )

async def verify_item_3(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Item_3_Company_1985",
        desc="Verify 1985 co-founding details match constraints (company, co-founder/shared law-firm background, and name origin), each with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    item3 = ex.item3 or Company1985Info()

    # Source presence checks per subfact
    evaluator.add_custom_node(
        result=len(non_empty_urls(item3.sources_company)) > 0,
        id="Item_3_company_urls_present",
        desc="At least one URL is provided supporting Stellar co-founding in 1985",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item3.sources_cofounder)) > 0,
        id="Item_3_cofounder_urls_present",
        desc="At least one URL is provided supporting cofounder identity and law firm",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item3.sources_origin)) > 0,
        id="Item_3_origin_urls_present",
        desc="At least one URL is provided supporting name origin explanation",
        parent=item_node,
        critical=True
    )

    # 3.1 Co-founded Stellar Management in 1985
    cofound_stellar_leaf = evaluator.add_leaf(
        id="CoFounded_Stellar_Management_In_1985_With_URL",
        desc="States that Witkoff co-founded Stellar Management in 1985 AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    cofound_stellar_claim = "In 1985, Steve Witkoff co-founded the real estate company Stellar Management."
    await evaluator.verify(
        claim=cofound_stellar_claim,
        node=cofound_stellar_leaf,
        sources=non_empty_urls(item3.sources_company),
        additional_instruction="The source must explicitly mention both 'Stellar Management' and the co-founding year 1985 in the context of Steve Witkoff."
    )

    # 3.2 Cofounder Larry Gluck from Dreyer & Traub
    cofounder_leaf = evaluator.add_leaf(
        id="CoFounder_Larry_Gluck_From_Dreyer_And_Traub_With_URL",
        desc="States that the co-founding partner was Larry (Laurence) Gluck and that he was also an attorney from Dreyer & Traub AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    cofounder_claim = "Witkoff’s cofounding partner at Stellar Management was Larry (Laurence) Gluck, who was also an attorney at Dreyer & Traub."
    await evaluator.verify(
        claim=cofounder_claim,
        node=cofounder_leaf,
        sources=non_empty_urls(item3.sources_cofounder),
        additional_instruction="The source should clearly associate Larry (or Laurence) Gluck as cofounder with Witkoff and indicate his affiliation with Dreyer & Traub."
    )

    # 3.3 Stellar name origin: Steve + Larry
    name_origin_leaf = evaluator.add_leaf(
        id="Stellar_Name_Origin_Steve_Plus_Larry_With_URL",
        desc="Explains that 'Stellar' was derived from combining 'Steve' and 'Larry' AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    name_origin_claim = "The name 'Stellar' originated by combining 'Steve' and 'Larry' (a portmanteau of their names)."
    await evaluator.verify(
        claim=name_origin_claim,
        node=name_origin_leaf,
        sources=non_empty_urls(item3.sources_origin),
        additional_instruction="Accept phrasings such as 'Stellar comes from Steve + Larry' or 'a portmanteau of Steve and Larry.'"
    )

async def verify_item_4(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Item_4_Company_1997",
        desc="Verify post-Stellar 1997 company founding matches constraints, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    item4 = ex.item4 or Company1997Info()

    evaluator.add_custom_node(
        result=len(non_empty_urls(item4.sources)) > 0,
        id="Item_4_company_1997_urls_present",
        desc="At least one URL is provided supporting leaving Stellar and founding The Witkoff Group in 1997",
        parent=item_node,
        critical=True
    )

    founded_leaf = evaluator.add_leaf(
        id="Founded_Witkoff_Group_In_1997_After_Leaving_Stellar_With_URL",
        desc="States that Witkoff left Stellar Management and founded the Witkoff Group in 1997 AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    founded_claim = "After leaving Stellar Management, Steve Witkoff founded The Witkoff Group in 1997."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=non_empty_urls(item4.sources),
        additional_instruction="The source should clearly indicate that Witkoff departed Stellar Management and founded The Witkoff Group in 1997."
    )

async def verify_item_5(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Item_5_Manhattan_Building_1998",
        desc="Verify the constrained 1998 acquisition and required attributes, each with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    item5 = ex.item5 or Building1998Info()

    # Source presence checks per subfact
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_building)) > 0,
        id="Item_5_building_urls_present",
        desc="At least one URL is provided supporting the acquired building identification",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_year)) > 0,
        id="Item_5_acq_year_urls_present",
        desc="At least one URL is provided supporting the acquisition year",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_price)) > 0,
        id="Item_5_price_urls_present",
        desc="At least one URL is provided supporting the purchase price",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_partner)) > 0,
        id="Item_5_partner_urls_present",
        desc="At least one URL is provided supporting the business partner identity",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_neighborhood)) > 0,
        id="Item_5_neighborhood_urls_present",
        desc="At least one URL is provided supporting the neighborhood/location",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(non_empty_urls(item5.sources_architect)) > 0,
        id="Item_5_architect_urls_present",
        desc="At least one URL is provided supporting the original architect",
        parent=item_node,
        critical=True
    )

    # 5.1 Building identity: Woolworth Building
    building_leaf = evaluator.add_leaf(
        id="Building_Woolworth_Building_With_URL",
        desc="Identifies the acquired building as the Woolworth Building AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    building_claim = "The major historic Manhattan building acquired by The Witkoff Group was the Woolworth Building."
    await evaluator.verify(
        claim=building_claim,
        node=building_leaf,
        sources=non_empty_urls(item5.sources_building),
        additional_instruction="The source must clearly associate The Witkoff Group or Steve Witkoff with acquiring the Woolworth Building."
    )

    # 5.2 Acquisition year: 1998
    year_leaf = evaluator.add_leaf(
        id="Acquisition_Year_1998_With_URL",
        desc="States that the Woolworth Building acquisition year was 1998 AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    year_claim = "The acquisition of the Woolworth Building occurred in 1998."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=non_empty_urls(item5.sources_year),
        additional_instruction="The source should explicitly state the acquisition year as 1998."
    )

    # 5.3 Purchase price: $155 million
    price_leaf = evaluator.add_leaf(
        id="Purchase_Price_155_Million_With_URL",
        desc="States that the purchase price was $155 million AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    price_claim = "The purchase price for the Woolworth Building acquisition was $155 million."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=non_empty_urls(item5.sources_price),
        additional_instruction="Allow reasonable wording variations such as 'about $155 million' or 'approximately $155 million'. The number should clearly be 155 million in USD."
    )

    # 5.4 Business partner: Rubin Schron
    partner_leaf = evaluator.add_leaf(
        id="Business_Partner_Rubin_Schron_With_URL",
        desc="States that the business partner in the acquisition was Rubin Schron AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    partner_claim = "Steve Witkoff’s business partner in the Woolworth Building acquisition was Rubin Schron."
    await evaluator.verify(
        claim=partner_claim,
        node=partner_leaf,
        sources=non_empty_urls(item5.sources_partner),
        additional_instruction="The source must clearly identify Rubin Schron as the acquisition partner with Witkoff."
    )

    # 5.5 Neighborhood: Tribeca
    neighborhood_leaf = evaluator.add_leaf(
        id="Neighborhood_Tribeca_With_URL",
        desc="States that the Woolworth Building is located in Tribeca, Manhattan AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    neighborhood_claim = "The Woolworth Building is located in the Tribeca neighborhood of Manhattan."
    await evaluator.verify(
        claim=neighborhood_claim,
        node=neighborhood_leaf,
        sources=non_empty_urls(item5.sources_neighborhood),
        additional_instruction="Pass only if the evidence explicitly states 'Tribeca' or 'TriBeCa'. Do not accept 'Financial District' or 'Civic Center' as equivalent."
    )

    # 5.6 Original architect: Cass Gilbert
    architect_leaf = evaluator.add_leaf(
        id="Original_Architect_Cass_Gilbert_With_URL",
        desc="States that the building's original architect was Cass Gilbert AND is supported by cited URLs.",
        parent=item_node,
        critical=True
    )
    architect_claim = "The Woolworth Building’s original architect was Cass Gilbert."
    await evaluator.verify(
        claim=architect_claim,
        node=architect_leaf,
        sources=non_empty_urls(item5.sources_architect),
        additional_instruction="The source should explicitly name Cass Gilbert as the building’s original architect."
    )

async def verify_global_citation(evaluator: Evaluator, parent_node, ex: CareerExtraction) -> None:
    item_node = evaluator.add_parallel(
        id="Global_Citation_Requirement",
        desc="All factual claims included in the response are supported by publicly available reference URLs.",
        parent=parent_node,
        critical=True
    )

    item1 = ex.item1 or LawSchoolInfo()
    item2 = ex.item2 or FirstLawFirmInfo()
    item3 = ex.item3 or Company1985Info()
    item4 = ex.item4 or Company1997Info()
    item5 = ex.item5 or Building1998Info()

    all_sources_lists = [
        non_empty_urls(item1.sources_school),
        non_empty_urls(item1.sources_year),
        non_empty_urls(item2.sources),
        non_empty_urls(item3.sources_company),
        non_empty_urls(item3.sources_cofounder),
        non_empty_urls(item3.sources_origin),
        non_empty_urls(item4.sources),
        non_empty_urls(item5.sources_building),
        non_empty_urls(item5.sources_year),
        non_empty_urls(item5.sources_price),
        non_empty_urls(item5.sources_partner),
        non_empty_urls(item5.sources_neighborhood),
        non_empty_urls(item5.sources_architect),
    ]
    urls_present_for_all = all(len(lst) > 0 for lst in all_sources_lists)

    evaluator.add_custom_node(
        result=urls_present_for_all,
        id="URLs_Present_For_All_Factual_Claims",
        desc="No required fact is presented without at least one publicly accessible supporting URL.",
        parent=item_node,
        critical=True
    )

    # Record helpful counts as custom info
    evaluator.add_custom_info(
        info={
            "total_required_facts": len(all_sources_lists),
            "facts_with_urls": sum(1 for lst in all_sources_lists if len(lst) > 0),
        },
        info_type="citation_stats",
        info_name="global_citation_counts"
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
    Evaluate an answer for the Steve Witkoff career development task.
    """
    # Initialize evaluator and root
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

    # Create the main critical node reflecting the rubric root
    main_node = evaluator.add_parallel(
        id="Professional_Career_Research",
        desc="Verify all required career-development facts about Steve Witkoff, each supported by publicly accessible reference URL(s), matching the provided constraints.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_career(),
        template_class=CareerExtraction,
        extraction_name="career_extraction"
    )

    # Add ground truth / expectations for transparency
    evaluator.add_ground_truth({
        "expected_facts": EXPECTED_FACTS
    }, gt_type="ground_truth")

    # Build and run verification subtrees
    await verify_item_1(evaluator, main_node, extraction)
    await verify_item_2(evaluator, main_node, extraction)
    await verify_item_3(evaluator, main_node, extraction)
    await verify_item_4(evaluator, main_node, extraction)
    await verify_item_5(evaluator, main_node, extraction)
    await verify_global_citation(evaluator, main_node, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()