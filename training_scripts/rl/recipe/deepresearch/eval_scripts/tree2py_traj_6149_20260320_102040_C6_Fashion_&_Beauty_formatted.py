import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sustainable_fashion_brand_identification"
TASK_DESCRIPTION = """
Identify a fashion brand that meets ALL of the following criteria:

1. The brand must be a Certified B Corporation
2. The brand must publish comprehensive annual sustainability reports or benefit corporation reports that detail environmental and social impact metrics
3. The brand must have won an award at the CNMI (Camera Nazionale della Moda Italiana) Sustainable Fashion Awards in 2022, 2023, or 2024
4. At least 50% of the brand's materials must be certified sustainable materials (such as GOTS certified organic cotton, Fair Trade certified materials, recycled materials, or other third-party certified sustainable materials)
5. The brand must have Fair Trade Certified factories OR publicly disclose detailed supplier lists with worker welfare information
6. The brand must be headquartered in or have significant operations in the United States
7. The brand must have been founded before the year 2000
8. The brand must publicly disclose its supply chain, including publishing factory or supplier lists on platforms like Open Supply Hub or its own website

What is the name of this fashion brand?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandVerificationExtraction(BaseModel):
    # Core brand identification
    brand_name: Optional[str] = None
    brand_website: Optional[str] = None

    # 1. B Corp certification
    bcorp_urls: List[str] = Field(default_factory=list)

    # 2. Sustainability / Benefit Corporation reporting
    sustainability_report_urls: List[str] = Field(default_factory=list)

    # 3. CNMI Sustainable Fashion Awards (2022, 2023, 2024)
    cnmi_award_year: Optional[str] = None
    cnmi_award_category: Optional[str] = None
    cnmi_award_urls: List[str] = Field(default_factory=list)

    # 4. Sustainable materials threshold
    materials_percentage_statement: Optional[str] = None  # e.g., "over 60%" or "majority of materials"
    materials_certifications: List[str] = Field(default_factory=list)  # e.g., ["GOTS", "recycled polyester"]
    materials_urls: List[str] = Field(default_factory=list)

    # 5. Fair labor OR supplier disclosure
    labor_practices_desc: Optional[str] = None  # e.g., "Fair Trade Certified factories" or "published supplier list"
    labor_practices_urls: List[str] = Field(default_factory=list)

    # 6. US presence
    us_presence_desc: Optional[str] = None  # e.g., "Headquartered in San Francisco, USA"
    us_presence_urls: List[str] = Field(default_factory=list)

    # 7. Founded before 2000
    founding_year: Optional[str] = None
    founding_year_urls: List[str] = Field(default_factory=list)

    # 8. Supply chain disclosure (supplier/factory lists)
    supply_chain_disclosure_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_info() -> str:
    return """
    Extract the fashion brand proposed in the answer and all URLs the answer uses as evidence for each of the eight required criteria below.
    Return strictly what is explicitly present in the answer text. Do not invent or infer unseen URLs.

    Required fields to extract:
    - brand_name: The name of the fashion brand identified as meeting the criteria.
    - brand_website: The brand's official website URL if provided.

    Criterion 1 (Certified B Corporation):
    - bcorp_urls: List all URLs provided that confirm B Corp certification (prefer bcorporation.net directory profile or brand's official B Corp announcement/info page).

    Criterion 2 (Sustainability/Benefit Corporation reports):
    - sustainability_report_urls: List all URLs pointing to annual sustainability/impact/benefit corporation reports (HTML or PDF). These should contain environmental/social metrics if present.

    Criterion 3 (CNMI Sustainable Fashion Awards 2022/2023/2024):
    - cnmi_award_year: The award year if stated (e.g., "2023").
    - cnmi_award_category: The award category name if stated (e.g., "Circular Economy Award").
    - cnmi_award_urls: List all URLs confirming the CNMI recognition (official CNMI site, reputable news, or brand page).

    Criterion 4 (≥50% certified sustainable materials):
    - materials_percentage_statement: The exact phrase/number describing the share (e.g., "over 50%", "majority", "60%+").
    - materials_certifications: List the named certifications or certified material types (e.g., "GOTS organic cotton", "recycled polyester", "Fair Trade cotton").
    - materials_urls: List all URLs evidencing the material share and certifications.

    Criterion 5 (Fair Trade factories OR supplier list with worker welfare info):
    - labor_practices_desc: The textual description (e.g., "Fair Trade Certified factory" or "supplier list with welfare info").
    - labor_practices_urls: List all URLs evidencing Fair Trade certification or detailed supplier disclosure (supplier/factory list pages, audit summaries, etc.).

    Criterion 6 (US headquarters or significant operations):
    - us_presence_desc: The textual statement related to US HQ/operations if provided.
    - us_presence_urls: List all URLs evidencing US HQ/operations (about/contact pages, press releases, facility pages).

    Criterion 7 (Founded before 2000):
    - founding_year: The founding year as explicitly stated in the answer (string).
    - founding_year_urls: List all URLs evidencing the founding year.

    Criterion 8 (Supply chain disclosure):
    - supply_chain_disclosure_urls: List all URLs that publish supplier/factory lists (on Open Supply Hub or the brand's own site or in reports).

    IMPORTANT URL RULES:
    - Only extract URLs explicitly present in the answer (plain or within markdown links). Do not paraphrase or infer.
    - If a required URL list is not provided in the answer, return an empty list [] for that field.
    - Preserve full URLs including http/https protocol.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def coalesce(value: Optional[str], fallback: str) -> str:
    return value if value and value.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, ext: BrandVerificationExtraction) -> None:
    brand = coalesce(ext.brand_name, "the brand identified in the answer")

    # Create top-level task node (critical gate for all criteria)
    task_node = evaluator.add_parallel(
        id="sustainable_fashion_brand_identification",
        desc="Identify a fashion brand that meets all specified sustainability criteria including B Corp certification, award recognition, supply chain transparency, and ethical production standards",
        parent=evaluator.root,
        critical=True,
    )

    # Brand existence check (critical)
    evaluator.add_custom_node(
        result=bool(ext.brand_name and ext.brand_name.strip()),
        id="brand_name_present",
        desc="A single fashion brand name is clearly provided in the answer",
        parent=task_node,
        critical=True,
    )

    # ----------------------------- Group 1 -------------------------------- #
    # B Corp certification + Sustainability Reporting
    bcorp_reporting = evaluator.add_parallel(
        id="b_corp_and_reporting",
        desc="The brand must be B Corp certified and publish comprehensive sustainability reports",
        parent=task_node,
        critical=True,
    )

    # 1A. B Corp certification (sequential)
    bcorp_seq = evaluator.add_sequential(
        id="b_corp_certification_status",
        desc="The brand holds active Certified B Corporation status",
        parent=bcorp_reporting,
        critical=True,
    )

    # Existence of B Corp URL(s)
    evaluator.add_custom_node(
        result=non_empty(ext.bcorp_urls),
        id="b_corp_url_provided",
        desc="At least one URL is provided that is intended to confirm B Corp certification",
        parent=bcorp_seq,
        critical=True,
    )

    # b_corp_verified
    leaf_bcorp_verified = evaluator.add_leaf(
        id="b_corp_verified",
        desc="Verification that the brand is a Certified B Corporation",
        parent=bcorp_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand} is a Certified B Corporation (B Corp).",
        node=leaf_bcorp_verified,
        sources=ext.bcorp_urls,
        additional_instruction="Prefer evidence from the official B Lab directory (bcorporation.net). Accept the brand's official page only if it explicitly states an active B Corp certification.",
    )

    # b_corp_verification_url
    leaf_bcorp_url = evaluator.add_leaf(
        id="b_corp_verification_url",
        desc="URL reference confirming B Corp certification from B Lab or official brand source",
        parent=bcorp_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided source(s) explicitly confirm that {brand} is B Corp certified (e.g., B Lab directory profile or the brand’s official certification announcement/info page).",
        node=leaf_bcorp_url,
        sources=ext.bcorp_urls,
        additional_instruction="If available, verify that at least one URL is the brand’s profile on bcorporation.net; otherwise, a brand page that clearly states Certified B Corporation status is acceptable.",
    )

    # 1B. Sustainability Reporting (sequential)
    reporting_seq = evaluator.add_sequential(
        id="sustainability_reporting",
        desc="The brand publishes annual sustainability or benefit corporation reports",
        parent=bcorp_reporting,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.sustainability_report_urls),
        id="sustainability_report_url_provided",
        desc="At least one URL to sustainability/impact/benefit corporation report(s) is provided",
        parent=reporting_seq,
        critical=True,
    )

    leaf_reports_published = evaluator.add_leaf(
        id="public_reports_published",
        desc="The brand makes detailed sustainability reports publicly accessible",
        parent=reporting_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand} publishes comprehensive annual sustainability or benefit corporation reports that include environmental and social impact metrics.",
        node=leaf_reports_published,
        sources=ext.sustainability_report_urls,
        additional_instruction="Confirm the URLs point to annual sustainability/impact/benefit reports (PDF or HTML) and that they contain concrete metrics, goals, or results (e.g., GHG emissions, materials share, worker data).",
    )

    leaf_report_url = evaluator.add_leaf(
        id="sustainability_report_url",
        desc="URL reference to the brand's sustainability or benefit corporation report",
        parent=reporting_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) point to {brand}'s sustainability/impact/benefit report pages or PDFs.",
        node=leaf_report_url,
        sources=ext.sustainability_report_urls,
        additional_instruction="The pages should be clearly labeled as sustainability/impact/benefit reports and be accessible.",
    )

    # ----------------------------- Group 2 -------------------------------- #
    # Award recognition at CNMI Sustainable Fashion Awards (2022-2024)
    award_grp = evaluator.add_parallel(
        id="award_recognition",
        desc="The brand received recognition at CNMI Sustainable Fashion Awards between 2022-2024",
        parent=task_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.cnmi_award_urls),
        id="cnmi_award_url_provided",
        desc="At least one URL is provided confirming CNMI Sustainable Fashion Award recognition",
        parent=award_grp,
        critical=True,
    )

    leaf_cnmi_won = evaluator.add_leaf(
        id="cnmi_award_won",
        desc="The brand won an award at CNMI Sustainable Fashion Awards in 2022, 2023, or 2024",
        parent=award_grp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand} won an award at the CNMI Sustainable Fashion Awards in 2022, 2023, or 2024.",
        node=leaf_cnmi_won,
        sources=ext.cnmi_award_urls,
        additional_instruction="Confirm that the brand is a WINNER (not just a nominee or finalist). The page should clearly indicate 'winner' or equivalent language for 2022/2023/2024.",
    )

    # Note: Marked critical to satisfy framework's constraint for critical parents
    leaf_award_cat = evaluator.add_leaf(
        id="award_category_identified",
        desc="The specific award category is identified (e.g., Pioneer Award, Circular Economy Award)",
        parent=award_grp,
        critical=True,
    )
    category_txt = coalesce(ext.cnmi_award_category, "a specific CNMI Sustainable Fashion Award category")
    await evaluator.verify(
        claim=f"The CNMI recognition for {brand} identifies the award category as {category_txt}.",
        node=leaf_award_cat,
        sources=ext.cnmi_award_urls,
        additional_instruction="Look for the explicit category name on the CNMI site, press releases, or credible media coverage.",
    )

    leaf_cnmi_url = evaluator.add_leaf(
        id="cnmi_award_url",
        desc="URL reference confirming the CNMI Sustainable Fashion Award recognition",
        parent=award_grp,
        critical=True,
    )
    year_txt = coalesce(ext.cnmi_award_year, "one of 2022, 2023, or 2024")
    await evaluator.verify(
        claim=f"The provided URL(s) confirm that {brand} won a CNMI Sustainable Fashion Award in {year_txt}.",
        node=leaf_cnmi_url,
        sources=ext.cnmi_award_urls,
        additional_instruction="Accept official CNMI pages or reputable media/brand pages clearly stating the award and year.",
    )

    # ----------------------------- Group 3 -------------------------------- #
    # Materials and ethical production
    materials_grp = evaluator.add_parallel(
        id="materials_and_production",
        desc="The brand meets requirements for sustainable materials and ethical production",
        parent=task_node,
        critical=True,
    )

    # 3A. Sustainable materials threshold (parallel)
    materials_threshold = evaluator.add_parallel(
        id="sustainable_materials_threshold",
        desc="At least 50% of materials are certified sustainable (GOTS, Fair Trade, recycled, or other third-party certified)",
        parent=materials_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.materials_urls),
        id="materials_cert_url_provided",
        desc="At least one URL is provided documenting sustainable material share/certifications",
        parent=materials_threshold,
        critical=True,
    )

    leaf_materials_threshold = evaluator.add_leaf(
        id="materials_meet_threshold",
        desc="The brand publicly states at least 50% of materials meet sustainability certification standards",
        parent=materials_threshold,
        critical=True,
    )
    percent_txt = coalesce(ext.materials_percentage_statement, "at least half (≥50%)")
    await evaluator.verify(
        claim=f"{brand} publicly states that {percent_txt} of its materials are certified sustainable (e.g., GOTS, Fair Trade, recycled, or similar third-party certified materials), meeting or exceeding 50%.",
        node=leaf_materials_threshold,
        sources=ext.materials_urls,
        additional_instruction="Accept phrasing indicating 'majority', 'over half', '>=50%', or explicit numeric statements ≥50%. The evidence must cite third-party certified materials.",
    )

    # Marked critical to satisfy immediate parent critical constraint
    leaf_material_types = evaluator.add_leaf(
        id="material_types_identified",
        desc="Specific material certifications are identified (e.g., GOTS organic cotton, recycled materials)",
        parent=materials_threshold,
        critical=True,
    )
    certs_txt = ", ".join(ext.materials_certifications) if ext.materials_certifications else "specific certified materials (e.g., GOTS organic cotton, recycled materials)"
    await evaluator.verify(
        claim=f"The sources identify specific certified sustainable material types used by {brand}, such as {certs_txt}.",
        node=leaf_material_types,
        sources=ext.materials_urls,
        additional_instruction="Look for explicit mentions like 'GOTS-certified organic cotton', 'recycled nylon/polyester', or Fair Trade-certified materials.",
    )

    leaf_materials_url = evaluator.add_leaf(
        id="materials_certification_url",
        desc="URL reference documenting the brand's sustainable material usage and certifications",
        parent=materials_threshold,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) document {brand}'s sustainable material usage and the relevant third-party certifications.",
        node=leaf_materials_url,
        sources=ext.materials_urls,
        additional_instruction="Reports, sustainability pages, or certification pages should explicitly discuss certified material shares/types.",
    )

    # 3B. Fair labor practices (sequential)
    fair_labor_seq = evaluator.add_sequential(
        id="fair_labor_practices",
        desc="The brand demonstrates Fair Trade certification or published supplier transparency",
        parent=materials_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.labor_practices_urls),
        id="labor_practices_url_provided",
        desc="At least one URL is provided evidencing Fair Trade certification or supplier disclosure",
        parent=fair_labor_seq,
        critical=True,
    )

    leaf_labor_or_disclosure = evaluator.add_leaf(
        id="fair_trade_or_supplier_disclosure",
        desc="The brand has Fair Trade Certified factories OR publicly discloses detailed supplier lists",
        parent=fair_labor_seq,
        critical=True,
    )
    labor_txt = coalesce(ext.labor_practices_desc, "Fair Trade Certified factories or publicly disclosed supplier lists with worker welfare information")
    await evaluator.verify(
        claim=f"{brand} has {labor_txt}.",
        node=leaf_labor_or_disclosure,
        sources=ext.labor_practices_urls,
        additional_instruction="Accept EITHER: (1) Fair Trade Certified factories/production OR (2) a detailed supplier/factory list that includes worker-related information (e.g., audit/welfare notes).",
    )

    leaf_labor_url = evaluator.add_leaf(
        id="labor_practices_url",
        desc="URL reference showing Fair Trade certification or supplier disclosure",
        parent=fair_labor_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) show Fair Trade certification for {brand} or a detailed supplier/factory list with worker welfare information.",
        node=leaf_labor_url,
        sources=ext.labor_practices_urls,
        additional_instruction="Look for authoritative certification pages or brand transparency pages/lists (including Open Supply Hub links) that include worker welfare details.",
    )

    # ----------------------------- Group 4 -------------------------------- #
    # Operational & transparency requirements
    ops_transparency_grp = evaluator.add_parallel(
        id="operational_and_transparency_requirements",
        desc="The brand meets geographic, temporal, and supply chain transparency criteria",
        parent=task_node,
        critical=True,
    )

    # 4A. US headquarters or operations (sequential)
    us_ops_seq = evaluator.add_sequential(
        id="us_headquarters_or_operations",
        desc="The brand is headquartered in or has significant operations in the United States",
        parent=ops_transparency_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.us_presence_urls),
        id="us_presence_url_provided",
        desc="At least one URL is provided evidencing US HQ/operations/facilities",
        parent=us_ops_seq,
        critical=True,
    )

    leaf_us_presence = evaluator.add_leaf(
        id="us_presence_confirmed",
        desc="Evidence of US headquarters, retail operations, or manufacturing facilities",
        parent=us_ops_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand} is headquartered in or has significant operations in the United States.",
        node=leaf_us_presence,
        sources=ext.us_presence_urls,
        additional_instruction="Accept About/Contact/Locations pages, press releases, or facility pages clearly indicating US HQ or major US operations.",
    )

    leaf_us_url = evaluator.add_leaf(
        id="us_presence_url",
        desc="URL reference showing the brand's US presence or headquarters location",
        parent=us_ops_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) clearly show {brand}'s US headquarters or significant operations.",
        node=leaf_us_url,
        sources=ext.us_presence_urls,
        additional_instruction="Evidence can include explicit HQ address in the US or multiple US facilities/operations.",
    )

    # 4B. Founded before 2000 (sequential)
    founded_seq = evaluator.add_sequential(
        id="established_before_2000",
        desc="The brand was founded before the year 2000",
        parent=ops_transparency_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.founding_year_urls),
        id="founding_year_url_provided",
        desc="At least one URL is provided evidencing the brand's founding year",
        parent=founded_seq,
        critical=True,
    )

    leaf_founding_verified = evaluator.add_leaf(
        id="founding_year_verified",
        desc="The brand's founding year is confirmed to be before 2000",
        parent=founded_seq,
        critical=True,
    )
    fy_text = coalesce(ext.founding_year, "a year before 2000")
    await evaluator.verify(
        claim=f"{brand} was founded before the year 2000 (founding year: {fy_text}).",
        node=leaf_founding_verified,
        sources=ext.founding_year_urls,
        additional_instruction="Confirm the founding year is strictly < 2000 (not equal to 2000). Prefer official brand pages or reputable sources.",
    )

    leaf_founding_url = evaluator.add_leaf(
        id="founding_year_url",
        desc="URL reference showing the brand's founding date",
        parent=founded_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) show {brand}'s founding year and that it is before 2000.",
        node=leaf_founding_url,
        sources=ext.founding_year_urls,
        additional_instruction="The page should clearly state the founding year/date.",
    )

    # 4C. Supply chain transparency (sequential)
    supply_seq = evaluator.add_sequential(
        id="supply_chain_transparency",
        desc="The brand publicly discloses its supply chain, including factory or supplier lists",
        parent=ops_transparency_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(ext.supply_chain_disclosure_urls),
        id="supply_chain_disclosure_url_provided",
        desc="At least one URL is provided to a supplier/factory list or supply chain disclosure",
        parent=supply_seq,
        critical=True,
    )

    leaf_supply_disclosed = evaluator.add_leaf(
        id="supply_chain_disclosed",
        desc="The brand publishes supplier lists on Open Supply Hub, its website, or in reports",
        parent=supply_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand} publicly discloses its supply chain by publishing factory or supplier lists (e.g., on Open Supply Hub or on its own website/reports).",
        node=leaf_supply_disclosed,
        sources=ext.supply_chain_disclosure_urls,
        additional_instruction="Look for an explicit supplier or factory list (or Open Supply Hub brand page) that reveals supplier/factory names/locations.",
    )

    leaf_supply_url = evaluator.add_leaf(
        id="supply_chain_disclosure_url",
        desc="URL reference to the brand's supply chain disclosure or published supplier list",
        parent=supply_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL(s) point to {brand}'s published supplier/factory list or supply chain disclosure.",
        node=leaf_supply_url,
        sources=ext.supply_chain_disclosure_urls,
        additional_instruction="Ensure the URLs actually show the list or clear disclosure content.",
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator with root node (non-critical by design)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brand_info(),
        template_class=BrandVerificationExtraction,
        extraction_name="brand_verification_extraction",
    )

    # Optionally record the criteria set for transparency
    evaluator.add_custom_info(
        {
            "criteria": [
                "Certified B Corporation",
                "Publishes annual sustainability/benefit reports with metrics",
                "CNMI Sustainable Fashion Award (2022/2023/2024) winner",
                "≥50% certified sustainable materials",
                "Fair Trade factories OR detailed supplier list with worker welfare info",
                "US HQ or significant operations",
                "Founded before 2000",
                "Public supply chain disclosure (supplier/factory list)",
            ]
        },
        info_type="criteria",
        info_name="evaluation_criteria",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return the final structured summary
    return evaluator.get_summary()