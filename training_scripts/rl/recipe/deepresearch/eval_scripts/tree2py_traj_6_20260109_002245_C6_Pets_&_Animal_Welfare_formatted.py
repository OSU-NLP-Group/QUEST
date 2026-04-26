import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chicago_foster_rescue_charitynavigator_100"
TASK_DESCRIPTION = (
    "Identify a 501(c)(3) nonprofit animal rescue organization that meets ALL of the following criteria: "
    "1. Located in Chicago, Illinois; "
    "2. Operates as a foster home-based rescue (animals are housed in volunteer foster homes rather than a central shelter facility); "
    "3. Entirely volunteer-run organization; "
    "4. Focuses on rescuing companion animals (dogs and cats); "
    "5. Has earned a Four-Star rating from Charity Navigator; "
    "6. Has achieved a perfect 100% overall score on Charity Navigator; "
    "7. Has achieved a perfect score of 100 in the Impact & Measurement beacon on Charity Navigator; "
    "8. Has achieved a perfect score of 100 in the Culture & Community beacon on Charity Navigator; "
    "9. Has achieved a perfect score of 100 in the Leadership & Adaptability beacon on Charity Navigator; "
    "10. Provides comprehensive services including foster care, spay/neuter programs, medical care, and adoption services. "
    "Provide the organization's name and the following supporting evidence: the organization's official website URL, the Charity Navigator profile URL showing the 100% score and Four-Star rating, "
    "evidence of its 501(c)(3) status and EIN, evidence of its foster-based volunteer-run operational model, and evidence of its Chicago, Illinois location."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OrgExtraction(BaseModel):
    # Core identification
    organization_name: Optional[str] = None
    official_website_url: Optional[str] = None
    charity_navigator_url: Optional[str] = None

    # Legal/EIN
    ein: Optional[str] = None
    nonprofit_status_evidence_urls: List[str] = Field(default_factory=list)
    ein_evidence_urls: List[str] = Field(default_factory=list)

    # Location
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_evidence_urls: List[str] = Field(default_factory=list)

    # Operational model and focus
    foster_based_evidence_urls: List[str] = Field(default_factory=list)
    volunteer_run_evidence_urls: List[str] = Field(default_factory=list)
    companion_animals_evidence_urls: List[str] = Field(default_factory=list)

    # Services
    foster_care_evidence_urls: List[str] = Field(default_factory=list)
    spay_neuter_evidence_urls: List[str] = Field(default_factory=list)
    medical_care_evidence_urls: List[str] = Field(default_factory=list)
    adoption_services_evidence_urls: List[str] = Field(default_factory=list)

    # CN additional evidence (if any)
    cn_additional_evidence_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_org_info() -> str:
    return """
Extract the organization and evidence details explicitly provided in the answer. Do not infer or fabricate any information.

Return the following fields:
- organization_name: The single nonprofit organization name proposed in the answer.
- official_website_url: The organization's official website URL.
- charity_navigator_url: The Charity Navigator profile URL for this organization.
- ein: The organization's EIN (Employer Identification Number) exactly as stated, if provided (allow formats like '12-3456789' or '123456789').
- nonprofit_status_evidence_urls: All URLs cited that specifically support its 501(c)(3) status (IRS, Charity Navigator, Guidestar, official site, etc.).
- ein_evidence_urls: All URLs cited that specifically display/confirm the EIN as belonging to the organization (Charity Navigator, IRS, official site, etc.).

- location_city: The city stated for the organization (e.g., 'Chicago').
- location_state: The state stated for the organization (e.g., 'Illinois' or 'IL').
- location_evidence_urls: All URLs cited that show its Chicago, Illinois location.

- foster_based_evidence_urls: URLs cited that explicitly show animals are housed in volunteer foster homes (not a central shelter facility).
- volunteer_run_evidence_urls: URLs cited that explicitly show the organization is entirely volunteer-run (no paid staff).
- companion_animals_evidence_urls: URLs cited that show the organization focuses on companion animals (dogs and cats).

- foster_care_evidence_urls: URLs cited that show foster care is provided until adoption.
- spay_neuter_evidence_urls: URLs cited that show a spay/neuter program is provided.
- medical_care_evidence_urls: URLs cited that show medical care including vaccinations, deworming, and blood testing.
- adoption_services_evidence_urls: URLs cited that show adoption services/program are provided.

- cn_additional_evidence_urls: Any additional URLs in the answer that relate to Charity Navigator ratings, beacons, or scoring details.

General rules:
1) Only extract what is explicitly present in the answer. If something is not present, use null for single-value fields or an empty array for URL lists.
2) Extract only valid URLs mentioned in the answer (plain or in markdown links). If a URL lacks protocol, prepend http://.
3) Do not include duplicate URLs; preserve unique URLs only.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_urls(*parts: Any) -> List[str]:
    urls: List[str] = []
    seen = set()
    for part in parts:
        if not part:
            continue
        if isinstance(part, str):
            candidates = [part]
        elif isinstance(part, list):
            candidates = part
        else:
            continue
        for u in candidates:
            if not u:
                continue
            s = u.strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                urls.append(s)
    return urls


def org_ref_name(extracted: OrgExtraction) -> str:
    return extracted.organization_name or "the organization"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_required_response_fields(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="Response includes the required named fields/URLs requested in the prompt",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.organization_name and extracted.organization_name.strip()),
        id="Organization_Name_Provided",
        desc="Provide the organization's name",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.official_website_url and extracted.official_website_url.strip()),
        id="Official_Website_URL_Provided",
        desc="Provide the organization's official website URL",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.charity_navigator_url and extracted.charity_navigator_url.strip()),
        id="Charity_Navigator_Profile_URL_Provided",
        desc="Provide the Charity Navigator profile URL for the organization",
        parent=node,
        critical=True
    )


async def build_legal_status(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Legal_Status",
        desc="Verify the organization is a 501(c)(3) and provides EIN with evidence",
        parent=parent_node,
        critical=True
    )

    # 501(c)(3) status
    leaf_501c3 = evaluator.add_leaf(
        id="501c3_Status_With_Evidence",
        desc="Evidence shows the organization is registered as a 501(c)(3) nonprofit with the IRS (citation/URL provided)",
        parent=node,
        critical=True
    )
    sources_501 = combine_urls(
        extracted.charity_navigator_url,
        extracted.official_website_url,
        extracted.nonprofit_status_evidence_urls,
        extracted.ein_evidence_urls
    )
    claim_501 = f"Evidence shows that {org_ref_name(extracted)} is a 501(c)(3) nonprofit organization (IRS tax-exempt status 501(c)(3) is explicitly indicated)."
    await evaluator.verify(
        claim=claim_501,
        node=leaf_501c3,
        sources=sources_501,
        additional_instruction="Accept explicit mentions like '501(c)(3)' or '501c3' or 'IRS-designated 501(c)(3)'. Pages such as the official site, Charity Navigator, IRS, or Guidestar are valid evidence."
    )

    # EIN with evidence
    leaf_ein = evaluator.add_leaf(
        id="EIN_With_Evidence",
        desc="The EIN is provided and evidence/URL supports that EIN belongs to the organization",
        parent=node,
        critical=True
    )
    ein_str = extracted.ein or "UNKNOWN"
    sources_ein = combine_urls(
        extracted.charity_navigator_url,
        extracted.ein_evidence_urls,
        extracted.official_website_url
    )
    claim_ein = f"The Employer Identification Number (EIN) for {org_ref_name(extracted)} is '{ein_str}', and the provided source shows that this EIN belongs to this organization."
    await evaluator.verify(
        claim=claim_ein,
        node=leaf_ein,
        sources=sources_ein,
        additional_instruction="The page should explicitly show the EIN. Charity Navigator profile often lists the EIN. Only pass if the shown EIN matches the stated EIN and is clearly associated with the organization."
    )


async def build_geographic_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Geographic_Requirement",
        desc="Verify the organization is in the required location with evidence",
        parent=parent_node,
        critical=True
    )

    leaf_loc = evaluator.add_leaf(
        id="Located_In_Chicago_Illinois_With_Evidence",
        desc="Evidence shows the organization is located in Chicago, Illinois (citation/URL provided)",
        parent=node,
        critical=True
    )
    sources_loc = combine_urls(
        extracted.official_website_url,
        extracted.charity_navigator_url,
        extracted.location_evidence_urls
    )
    claim_loc = f"Evidence shows that {org_ref_name(extracted)} is located in Chicago, Illinois (Chicago, IL)."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=sources_loc,
        additional_instruction="Look for a physical address or location statement explicitly stating 'Chicago, Illinois' or 'Chicago, IL'. If the location is a different city (e.g., Evanston, Palatine), the claim is not supported."
    )


async def build_operational_model_and_focus(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Operational_Model_and_Focus",
        desc="Verify foster-based, volunteer-run model and dogs/cats focus with evidence",
        parent=parent_node,
        critical=True
    )

    # Foster home-based, no central shelter
    leaf_foster = evaluator.add_leaf(
        id="Foster_Home_Based_No_Central_Shelter_With_Evidence",
        desc="Evidence shows animals are housed in volunteer foster homes rather than a central shelter facility (citation/URL provided)",
        parent=node,
        critical=True
    )
    sources_foster = combine_urls(
        extracted.official_website_url,
        extracted.foster_based_evidence_urls
    )
    claim_foster = f"Evidence shows that {org_ref_name(extracted)} operates as a foster home-based rescue where animals live in volunteer foster homes instead of a centralized shelter facility."
    await evaluator.verify(
        claim=claim_foster,
        node=leaf_foster,
        sources=sources_foster,
        additional_instruction="Accept explicit phrases like 'foster-based rescue', 'no central shelter', 'animals live in foster homes'. If the organization operates a permanent shelter facility as primary housing, do not support."
    )

    # Entirely volunteer-run
    leaf_volunteer = evaluator.add_leaf(
        id="Entirely_Volunteer_Run_With_Evidence",
        desc="Evidence shows the organization is entirely volunteer-run (citation/URL provided)",
        parent=node,
        critical=True
    )
    sources_volunteer = combine_urls(
        extracted.official_website_url,
        extracted.volunteer_run_evidence_urls,
        extracted.charity_navigator_url
    )
    claim_volunteer = f"Evidence shows that {org_ref_name(extracted)} is entirely volunteer-run (no paid staff)."
    await evaluator.verify(
        claim=claim_volunteer,
        node=leaf_volunteer,
        sources=sources_volunteer,
        additional_instruction="Look for explicit statements like 'entirely volunteer-run', '100% volunteer-run', or 'no paid staff'. If staff or employees are indicated, the claim is not supported."
    )

    # Companion animals focus (dogs and cats)
    leaf_companion = evaluator.add_leaf(
        id="Companion_Animals_Dogs_And_Cats_With_Evidence",
        desc="Evidence shows the organization rescues companion animals focused on dogs and cats (citation/URL provided)",
        parent=node,
        critical=True
    )
    sources_companion = combine_urls(
        extracted.official_website_url,
        extracted.companion_animals_evidence_urls,
        extracted.charity_navigator_url
    )
    claim_companion = f"Evidence shows that {org_ref_name(extracted)} focuses on rescuing companion animals, specifically dogs and cats."
    await evaluator.verify(
        claim=claim_companion,
        node=leaf_companion,
        sources=sources_companion,
        additional_instruction="Accept explicit mentions that the rescue focuses on dogs and cats (companion animals)."
    )


async def build_charity_navigator_ratings_and_scores(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Charity_Navigator_Ratings_And_Scores",
        desc="Verify Charity Navigator Four-Star rating, 100% overall score, and required beacon scores (using the provided Charity Navigator URL as evidence)",
        parent=parent_node,
        critical=True
    )

    cn_sources = combine_urls(extracted.charity_navigator_url, extracted.cn_additional_evidence_urls)

    # Four-Star Rating
    leaf_star = evaluator.add_leaf(
        id="Four_Star_Rating",
        desc="Charity Navigator shows the organization has a Four-Star rating",
        parent=node,
        critical=True
    )
    claim_star = "On the Charity Navigator profile, the organization has a Four-Star rating (4-star)."
    await evaluator.verify(
        claim=claim_star,
        node=leaf_star,
        sources=cn_sources,
        additional_instruction="Look for star icons or 'Four-Star' wording on the Charity Navigator profile."
    )

    # Overall 100% Score
    leaf_overall = evaluator.add_leaf(
        id="Overall_100_Percent_Score",
        desc="Charity Navigator shows a perfect 100% overall score",
        parent=node,
        critical=True
    )
    claim_overall = "On the Charity Navigator profile, the organization's overall score is 100 (i.e., 100 out of 100, 100%)."
    await evaluator.verify(
        claim=claim_overall,
        node=leaf_overall,
        sources=cn_sources,
        additional_instruction="Check the overall score on the page; accept '100', '100/100', or '100%'."
    )

    # Impact & Measurement 100
    leaf_impact = evaluator.add_leaf(
        id="Impact_And_Measurement_100",
        desc="Charity Navigator shows a perfect score of 100 in the Impact & Measurement beacon",
        parent=node,
        critical=True
    )
    claim_impact = "On the Charity Navigator profile, the Impact & Measurement beacon score is 100."
    await evaluator.verify(
        claim=claim_impact,
        node=leaf_impact,
        sources=cn_sources,
        additional_instruction="Locate the 'Impact & Measurement' beacon section and verify the score is exactly 100."
    )

    # Culture & Community 100
    leaf_culture = evaluator.add_leaf(
        id="Culture_And_Community_100",
        desc="Charity Navigator shows a perfect score of 100 in the Culture & Community beacon",
        parent=node,
        critical=True
    )
    claim_culture = "On the Charity Navigator profile, the Culture & Community beacon score is 100."
    await evaluator.verify(
        claim=claim_culture,
        node=leaf_culture,
        sources=cn_sources,
        additional_instruction="Locate the 'Culture & Community' beacon section and verify the score is exactly 100."
    )

    # Leadership & Adaptability 100
    leaf_lead = evaluator.add_leaf(
        id="Leadership_And_Adaptability_100",
        desc="Charity Navigator shows a perfect score of 100 in the Leadership & Adaptability beacon",
        parent=node,
        critical=True
    )
    claim_lead = "On the Charity Navigator profile, the Leadership & Adaptability beacon score is 100."
    await evaluator.verify(
        claim=claim_lead,
        node=leaf_lead,
        sources=cn_sources,
        additional_instruction="Locate the 'Leadership & Adaptability' beacon section and verify the score is exactly 100."
    )

    # Additional constraints under CN section (as specified by rubric)
    leaf_cost = evaluator.add_leaf(
        id="Cost_Effectiveness_Threshold",
        desc="Evidence shows cost-effective rescue operations as specified (cost per animal rescue < 75% of household pet-saving costs)",
        parent=node,
        critical=True
    )
    claim_cost = "Evidence shows the organization's cost per animal rescue is under 75% of standard household pet-saving costs."
    await evaluator.verify(
        claim=claim_cost,
        node=leaf_cost,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Only pass if a source explicitly provides a comparable cost metric demonstrating the threshold. If no such explicit cost comparison is found, mark as not supported."
    )

    leaf_feedback = evaluator.add_leaf(
        id="Constituent_Feedback_Collected_And_Used",
        desc="Evidence shows the organization collects and uses feedback from the people/animals it serves (as specified in constraints)",
        parent=node,
        critical=True
    )
    claim_feedback = "Evidence shows the organization collects feedback from constituents and uses that feedback to improve services."
    await evaluator.verify(
        claim=claim_feedback,
        node=leaf_feedback,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Look for a 'Constituent Feedback' section or similar descriptions on Charity Navigator or official site indicating feedback collection and use."
    )

    leaf_quality = evaluator.add_leaf(
        id="Quality_Feedback_Practices",
        desc="Evidence shows quality feedback practices (collecting feedback broadly, ensuring comfort, and acting on feedback) as specified in constraints",
        parent=node,
        critical=True
    )
    claim_quality = "Evidence shows quality feedback practices: collecting feedback broadly, ensuring comfort/safety when providing feedback, and acting on feedback."
    await evaluator.verify(
        claim=claim_quality,
        node=leaf_quality,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Look for explicit mention of broad collection, safe/comfortable feedback environment, and examples of acting on feedback."
    )

    leaf_strategy = evaluator.add_leaf(
        id="Strategic_Thinking_Mission_Vision_Goals",
        desc="Evidence shows strategic thinking via clear mission, vision, and strategic goals (as specified in constraints)",
        parent=node,
        critical=True
    )
    claim_strategy = "Evidence shows clear mission, vision, and strategic goals that demonstrate strategic thinking."
    await evaluator.verify(
        claim=claim_strategy,
        node=leaf_strategy,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Accept explicit mission/vision statements and strategic plan goals on official site or relevant Charity Navigator sections."
    )

    leaf_lead_dev = evaluator.add_leaf(
        id="Leadership_Development_External_Focus",
        desc="Evidence shows investment in leadership development and an external focus on mobilizing its mission (as specified in constraints)",
        parent=node,
        critical=True
    )
    claim_lead_dev = "Evidence shows that the organization invests in leadership development and maintains an external focus on mobilizing its mission."
    await evaluator.verify(
        claim=claim_lead_dev,
        node=leaf_lead_dev,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Look for explicit descriptions of leadership development efforts and external mission mobilization."
    )

    leaf_adapt = evaluator.add_leaf(
        id="Adaptability_Practices",
        desc="Evidence shows adaptability via technology integration, operational flexibility, or other innovative practices (as specified in constraints)",
        parent=node,
        critical=True
    )
    claim_adapt = "Evidence shows adaptability through technology integration, operational flexibility, or other innovative practices."
    await evaluator.verify(
        claim=claim_adapt,
        node=leaf_adapt,
        sources=combine_urls(cn_sources, extracted.official_website_url),
        additional_instruction="Look for explicit mentions of adapting processes, technology adoption, or innovations that improved operations/services."
    )


async def build_services_provided(
    evaluator: Evaluator,
    parent_node,
    extracted: OrgExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Services_Provided",
        desc="Verify the required animal rescue services are provided with evidence",
        parent=parent_node,
        critical=True
    )

    # Foster homes until adoption
    leaf_safe_foster = evaluator.add_leaf(
        id="Safe_Caring_Foster_Homes_Until_Adoption",
        desc="Evidence shows the organization provides safe, caring foster homes for rescued animals until adoption",
        parent=node,
        critical=True
    )
    sources_foster = combine_urls(extracted.official_website_url, extracted.foster_care_evidence_urls)
    claim_safe_foster = f"Evidence shows {org_ref_name(extracted)} provides safe, caring foster homes for rescued animals until they are adopted."
    await evaluator.verify(
        claim=claim_safe_foster,
        node=leaf_safe_foster,
        sources=sources_foster,
        additional_instruction="Accept explicit statements about providing foster homes until adoption."
    )

    # Spay/Neuter program
    leaf_spay = evaluator.add_leaf(
        id="Spay_Neuter_Program",
        desc="Evidence shows the organization operates a spay/neuter program for adopted animals",
        parent=node,
        critical=True
    )
    sources_spay = combine_urls(extracted.official_website_url, extracted.spay_neuter_evidence_urls)
    claim_spay = f"Evidence shows {org_ref_name(extracted)} operates a spay/neuter program (e.g., adopted animals are spayed/neutered)."
    await evaluator.verify(
        claim=claim_spay,
        node=leaf_spay,
        sources=sources_spay,
        additional_instruction="Look for explicit mention of spay/neuter services or requirements for adopted animals."
    )

    # Medical care including vaccinations, deworming, blood testing
    leaf_med = evaluator.add_leaf(
        id="Medical_Care_Includes_Listed_Care",
        desc="Evidence shows the organization provides medical care including vaccinations, deworming, and blood testing",
        parent=node,
        critical=True
    )
    sources_med = combine_urls(extracted.official_website_url, extracted.medical_care_evidence_urls)
    claim_med = f"Evidence shows {org_ref_name(extracted)} provides medical care including vaccinations, deworming, and blood testing."
    await evaluator.verify(
        claim=claim_med,
        node=leaf_med,
        sources=sources_med,
        additional_instruction="All three types of care should be supported: vaccinations, deworming, and blood testing."
    )

    # Adoption services
    leaf_adopt = evaluator.add_leaf(
        id="Adoption_Services",
        desc="Evidence shows the organization facilitates adoptions (e.g., an adoption program/process is described) consistent with the constraint intent",
        parent=node,
        critical=True
    )
    sources_adopt = combine_urls(extracted.official_website_url, extracted.adoption_services_evidence_urls)
    claim_adopt = f"Evidence shows {org_ref_name(extracted)} facilitates adoptions, with an adoption program or process described."
    await evaluator.verify(
        claim=claim_adopt,
        node=leaf_adopt,
        sources=sources_adopt,
        additional_instruction="Look for an explicit adoption process/program page or description."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for the Chicago foster-based rescue Charity Navigator 100% task.
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
        default_model=model
    )

    # Extraction
    extracted: OrgExtraction = await evaluator.extract(
        prompt=prompt_extract_org_info(),
        template_class=OrgExtraction,
        extraction_name="organization_extraction"
    )

    # Build main critical node that mirrors rubric top-level
    org_node = evaluator.add_parallel(
        id="Organization_Identification",
        desc="Response identifies exactly one nonprofit animal rescue organization that satisfies all stated criteria and provides the required supporting evidence/URLs",
        parent=root,
        critical=True
    )

    # Subtrees in the order of natural gating
    await build_required_response_fields(evaluator, org_node, extracted)
    await build_legal_status(evaluator, org_node, extracted)
    await build_geographic_requirement(evaluator, org_node, extracted)
    await build_operational_model_and_focus(evaluator, org_node, extracted)
    await build_charity_navigator_ratings_and_scores(evaluator, org_node, extracted)
    await build_services_provided(evaluator, org_node, extracted)

    # Add optional context info
    evaluator.add_custom_info(
        info={
            "note": "All verifications rely primarily on the official website and Charity Navigator profile URL(s) provided in the answer. "
                    "If the answer omits required URLs or evidence, related nodes will likely fail or be skipped due to critical gating."
        },
        info_type="eval_context"
    )

    return evaluator.get_summary()