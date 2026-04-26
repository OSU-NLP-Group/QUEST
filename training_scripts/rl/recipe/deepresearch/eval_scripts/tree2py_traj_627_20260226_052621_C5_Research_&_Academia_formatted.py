import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "neuralink_prime_sites_us_2026"
TASK_DESCRIPTION = (
    "As of February 2026, Neuralink has selected exactly two facilities in the United States to conduct its PRIME Study "
    "(Precise Robotically IMplanted Brain-Computer InterfacE) clinical trial. Identify both US-based PRIME Study sites "
    "by providing the following information for each: For the site that was announced first (in 2024): the name of the "
    "neurological facility, the complete street address (including street number, street name, city, state, and ZIP code), "
    "the month and year when it was announced as a PRIME Study site, and a reference URL confirming this site's "
    "participation in the PRIME Study. For the site that was announced second (in 2025): the name of the specific research "
    "facility or program, the name of the medical school it is affiliated with, the city where it is located, the month "
    "and year when it was announced as a PRIME Study site, and a reference URL confirming this site's participation in the PRIME Study."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FirstSiteInfo(BaseModel):
    # The first US site (announced in 2024; Phoenix, AZ neurological facility)
    name: Optional[str] = None
    address_full: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    announcement_month: Optional[str] = None  # e.g., "April"
    announcement_year: Optional[str] = None   # e.g., "2024"
    url: Optional[str] = None                 # Reference URL confirming participation


class SecondSiteInfo(BaseModel):
    # The second US site (announced in 2025; University of Miami-affiliated paralysis research facility)
    facility_name: Optional[str] = None                 # e.g., "The Miami Project to Cure Paralysis"
    host_institution: Optional[str] = None              # e.g., "Miller School of Medicine"
    city: Optional[str] = None                          # e.g., "Miami"
    announcement_month: Optional[str] = None            # e.g., "January"
    announcement_year: Optional[str] = None             # e.g., "2025"
    url: Optional[str] = None                           # Reference URL confirming participation


class PrimeSitesExtraction(BaseModel):
    first_site: Optional[FirstSiteInfo] = None
    second_site: Optional[SecondSiteInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_prime_sites() -> str:
    return """
    Extract the two US-based Neuralink PRIME Study clinical trial sites described in the answer, filling the fields
    for the first site (announced in 2024) and the second site (announced in 2025) as follows.

    IMPORTANT MAPPING RULES:
    - "first_site" must be the US site announced in 2024 (Phoenix, AZ neurological facility).
    - "second_site" must be the US site announced in 2025 (University of Miami–affiliated paralysis research facility/program).

    For first_site (announced in 2024), extract:
    - name: The neurological facility's name (e.g., the institute/center name).
    - address_full: A single-line complete street address string including street number, street name, city, state, and ZIP code.
    - street: The street address line (e.g., "350 W Thomas Rd").
    - city: The city (expected to be "Phoenix").
    - state: The state (expected to be "AZ" or "Arizona").
    - zip: The ZIP code (5 digits).
    - announcement_month: The month when it was announced as a PRIME Study site (should be "April").
    - announcement_year: The year when it was announced as a PRIME Study site (should be "2024").
    - url: A single reference URL (from the facility's official website or a credible news/press release) that explicitly confirms this site's participation in Neuralink's PRIME Study.

    For second_site (announced in 2025), extract:
    - facility_name: The specific research facility or program name (e.g., "The Miami Project to Cure Paralysis").
    - host_institution: The medical school it is affiliated with (expected to be "Miller School of Medicine"; variations like "University of Miami Miller School of Medicine" are acceptable).
    - city: The city where the facility is located (expected to be "Miami").
    - announcement_month: The month when it was announced as a PRIME Study site (should be "January").
    - announcement_year: The year when it was announced as a PRIME Study site (should be "2025").
    - url: A single reference URL (from the university's official website or a credible news/press source) that explicitly confirms this site's participation in Neuralink's PRIME Study.

    GENERAL RULES:
    - Extract ONLY what is explicitly present in the answer text. Do not invent or infer missing information.
    - Normalize months to full English names where possible (e.g., "Apr" -> "April", "Jan" -> "January").
    - For URLs: return one single, valid URL string for each site; if multiple are listed, choose the most authoritative/official one; if none present, set to null.
    - If a field is not in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def month_year_str(month: Optional[str], year: Optional[str]) -> str:
    m = (month or "").strip()
    y = (year or "").strip()
    if m and y:
        return f"{m} {y}"
    return (m or y or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_first_site(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    info: Optional[FirstSiteInfo],
) -> None:
    # Parent node for the first site (parallel)
    first_node = evaluator.add_parallel(
        id="First_US_Site",
        desc="Information about the first US site announced for the PRIME Study (announced in 2024)",
        parent=parent_node,
        critical=False,
    )

    # Convenience values
    name = (info.name if info else "") or ""
    url = (info.url if info else "") or ""
    addr_full = (info.address_full if info else "") or ""
    city = (info.city if info else "") or ""
    state = (info.state if info else "") or ""
    ann_m = (info.announcement_month if info else "") or ""
    ann_y = (info.announcement_year if info else "") or ""
    ann_my = month_year_str(ann_m, ann_y)

    # Critical existence of URL reference (used as precondition for URL-grounded checks)
    url_exists = evaluator.add_custom_node(
        result=bool(url.strip()),
        id="first_url_provided",
        desc="First site: URL reference is provided",
        parent=first_node,
        critical=True,
    )

    # Facility Name verification (critical)
    name_leaf = evaluator.add_leaf(
        id="first_facility_name",
        desc="Correct identification of the neurological facility in Phoenix, Arizona that was announced as the first US PRIME Study site",
        parent=first_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The facility named '{name}' located in Phoenix, Arizona is a US PRIME Study clinical trial site for Neuralink."
        ),
        node=name_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Verify only using the provided page. It must explicitly support that the named facility participates "
            "as a Neuralink PRIME Study site and that it is located in Phoenix, Arizona. Allow minor naming variants "
            "(e.g., inclusion/exclusion of the parent hospital name), but the identity must be clear."
        ),
    )

    # Complete Address verification (critical)
    addr_leaf = evaluator.add_leaf(
        id="first_complete_address",
        desc="Provision of the complete street address including street number, street name, city, state, and ZIP code",
        parent=first_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The complete street address of '{name}' is '{addr_full}'."
        ),
        node=addr_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Confirm that the page supports the exact or obviously equivalent address for the facility. "
            "The provided address string must include: street number, street name, city, state, and ZIP code. "
            "Accept minor punctuation/capitalization variants. If the page shows these components separately, "
            "that is acceptable as long as they unambiguously form the same full address."
        ),
    )

    # Announcement Date verification (critical)
    ann_leaf = evaluator.add_leaf(
        id="first_announcement_date",
        desc="Correct identification of the announcement date in April 2024",
        parent=first_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The PRIME Study site announcement for '{name}' occurred in {ann_my}."
        ),
        node=ann_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Confirm that the page indicates the site's PRIME Study participation announcement took place in April 2024. "
            "Use the press release/article date or explicit announcement wording. If the page indicates a month/year "
            "different from the claim, mark as not supported."
        ),
    )

    # URL Reference confirmation (critical)
    url_leaf = evaluator.add_leaf(
        id="first_url_reference_confirms",
        desc="Provision of a valid URL from the facility's official website or credible news source confirming the PRIME Study site announcement",
        parent=first_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This page confirms that '{name}' participates as a site in Neuralink's PRIME Study."
        ),
        node=url_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Decide strictly based on the provided page. It must explicitly confirm this facility's participation "
            "as a Neuralink PRIME Study site (PRIME = Precise Robotically IMplanted Brain-Computer InterfacE)."
        ),
    )


async def verify_second_site(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    info: Optional[SecondSiteInfo],
) -> None:
    # Parent node for the second site (parallel)
    second_node = evaluator.add_parallel(
        id="Second_US_Site",
        desc="Information about the second US site announced for the PRIME Study (announced in 2025)",
        parent=parent_node,
        critical=False,
    )

    # Convenience values
    facility = (info.facility_name if info else "") or ""
    host = (info.host_institution if info else "") or ""
    city = (info.city if info else "") or ""
    url = (info.url if info else "") or ""
    ann_m = (info.announcement_month if info else "") or ""
    ann_y = (info.announcement_year if info else "") or ""
    ann_my = month_year_str(ann_m, ann_y)

    # Critical existence of URL reference (used as precondition)
    url_exists = evaluator.add_custom_node(
        result=bool(url.strip()),
        id="second_url_provided",
        desc="Second site: URL reference is provided",
        parent=second_node,
        critical=True,
    )

    # Facility Name verification (critical)
    facility_leaf = evaluator.add_leaf(
        id="second_facility_name",
        desc="Correct identification of the University of Miami-affiliated paralysis research facility announced as the second US PRIME Study site",
        parent=second_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The facility/program named '{facility}' is a University of Miami–affiliated paralysis research facility that participates as a site in Neuralink's PRIME Study."
        ),
        node=facility_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Verify that the page explicitly supports that the named facility/program is affiliated with the University of Miami "
            "and is a Neuralink PRIME Study site. Accept minor naming variants (e.g., with/without 'The')."
        ),
    )

    # Host Institution verification (critical)
    host_leaf = evaluator.add_leaf(
        id="second_host_institution",
        desc="Identification of the Miller School of Medicine as the host medical school",
        parent=second_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The affiliated medical school for this PRIME Study site is '{host}'."
        ),
        node=host_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Confirm that the page shows affiliation with the (University of Miami) Miller School of Medicine. "
            "The provided value should correspond to 'Miller School of Medicine' (allowing reasonable variants such as "
            "'University of Miami Miller School of Medicine'). If the page indicates a different school, mark as not supported."
        ),
    )

    # City Location verification (critical)
    city_leaf = evaluator.add_leaf(
        id="second_city_location",
        desc="Identification of Miami as the city location",
        parent=second_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The facility is located in the city of {city}, Florida."
        ),
        node=city_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Verify that the page indicates the facility is in Miami, Florida. If the claimed city differs from what the page shows, mark as not supported."
        ),
    )

    # Announcement Date verification (critical)
    ann_leaf = evaluator.add_leaf(
        id="second_announcement_date",
        desc="Correct identification of the announcement date in January 2025",
        parent=second_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The PRIME Study site announcement for '{facility}' occurred in {ann_my}."
        ),
        node=ann_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Confirm that the page indicates the site's PRIME Study participation announcement took place in January 2025. "
            "Use the press release/article date or explicit announcement wording. If the page indicates a different month/year "
            "than the claim, mark as not supported."
        ),
    )

    # URL Reference confirmation (critical)
    url_leaf = evaluator.add_leaf(
        id="second_url_reference_confirms",
        desc="Provision of a valid URL from the university's official website or credible news source confirming the PRIME Study site announcement",
        parent=second_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This page confirms that '{facility}' participates as a site in Neuralink's PRIME Study."
        ),
        node=url_leaf,
        sources=url,
        extra_prerequisites=[url_exists],
        additional_instruction=(
            "Decide strictly based on the provided page. It must explicitly confirm this facility's participation "
            "as a Neuralink PRIME Study site (PRIME = Precise Robotically IMplanted Brain-Computer InterfacE)."
        ),
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
    # Initialize evaluator (root is non-critical parallel to avoid child criticality constraints)
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

    # Add a rubric root grouping node (non-critical, parallel)
    rubric_root = evaluator.add_parallel(
        id="US_PRIME_Study_Sites",
        desc="Identification and detailed information about the two US-based Neuralink PRIME Study clinical trial sites",
        parent=root,
        critical=False,
    )

    # Extract the structured sites info from the answer
    sites_info = await evaluator.extract(
        prompt=prompt_extract_prime_sites(),
        template_class=PrimeSitesExtraction,
        extraction_name="prime_sites_extraction",
    )

    # Optionally record constraints as "ground truth info" context for transparency
    evaluator.add_ground_truth({
        "constraints": {
            "first_site_expected": {
                "announcement_month": "April",
                "announcement_year": "2024",
                "city": "Phoenix",
                "state": "Arizona"
            },
            "second_site_expected": {
                "announcement_month": "January",
                "announcement_year": "2025",
                "city": "Miami",
                "host_institution_includes": "Miller School of Medicine"
            }
        }
    })

    # Verify first and second sites according to rubric
    await verify_first_site(evaluator, rubric_root, sites_info.first_site)
    await verify_second_site(evaluator, rubric_root, sites_info.second_site)

    # Return the evaluation summary with verification tree
    return evaluator.get_summary()