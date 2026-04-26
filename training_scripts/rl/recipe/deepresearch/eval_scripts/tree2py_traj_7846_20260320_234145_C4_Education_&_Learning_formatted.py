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
TASK_ID = "tx_three_universities_transfer_housing_accreditation"
TASK_DESCRIPTION = """I am a community college student in Texas planning to transfer to a four-year university after completing my associate degree. I prefer to live off-campus rather than in university housing, and I want to ensure that most of my community college credits will transfer.

Please identify three public universities in Texas that meet all of the following requirements:

1. The university must be located in Texas.
2. The university must be a public institution (not a private university).
3. The university must be accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).
4. The university must NOT require first-year undergraduate students to live on campus. (Some universities mandate on-campus housing for freshmen, but I need institutions where this is optional or not required.)
5. The university must accept at least 60 semester credit hours as transfer credits from accredited community colleges toward a bachelor's degree.

For each university, please provide:
- The university's full name
- A link to the official university webpage confirming their housing policy (specifically stating whether on-campus housing is required for first-year students)
- A link to the official university webpage or transfer guide confirming their transfer credit policy (specifically stating the minimum number of transfer credits accepted)
- A link to verify the university's SACSCOC accreditation status
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    housing_policy_url: Optional[str] = None
    transfer_policy_url: Optional[str] = None
    accreditation_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities listed in the answer that are intended to meet the user's constraints.
    For each university, extract the following fields:

    - name: The university's full name as stated in the answer.
    - housing_policy_url: A URL to an official university page that explicitly addresses first-year/freshman on-campus housing requirements (or the lack thereof). This should be on the university's official website (typically a .edu domain or an official subdomain).
    - transfer_policy_url: A URL to an official university page (e.g., transfer admissions, catalog, transfer guide) that explicitly states how many transfer credits are accepted. Prefer pages that specify maxima like "up to 60/64/66 hours" or "at least 60 credits".
    - accreditation_url: A URL that verifies SACSCOC accreditation. Prefer the SACSCOC institution directory/listing (sacscoc.org) or the university's official accreditation statement page that explicitly names SACSCOC.
    - additional_urls: Any other URLs provided in the answer that are directly relevant to this specific university (e.g., official "About", admissions, or transfer equivalency pages). Include only valid URLs not already used above.

    Rules:
    - Only include URLs explicitly present in the answer (plain links or markdown links). Extract the actual URL strings.
    - If the answer lists more than three universities, include ONLY the first three that appear.
    - If any field is missing for a university, set it to null (for URLs) or leave the list empty (for additional_urls).
    - Ensure URLs are complete (prepend http:// if protocol missing).

    Return a JSON object with a single field:
    {
      "universities": [
        {
          "name": ...,
          "housing_policy_url": ...,
          "transfer_policy_url": ...,
          "accreditation_url": ...,
          "additional_urls": [...]
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_sources_for_uni(u: UniversityItem) -> List[str]:
    """Collect all available URLs for a university item, removing duplicates."""
    urls: List[str] = []
    if u.housing_policy_url:
        urls.append(u.housing_policy_url)
    if u.transfer_policy_url:
        urls.append(u.transfer_policy_url)
    if u.accreditation_url:
        urls.append(u.accreditation_url)
    urls.extend(u.additional_urls or [])
    return _dedup_preserve_order(urls)


def uni_display_name(u: UniversityItem, index: int) -> str:
    return u.name if (u and u.name) else f"University #{index + 1}"


def ordinal_name(i: int) -> str:
    return ["First", "Second", "Third"][i] if 0 <= i < 3 else f"University_{i+1}"


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_official_documentation_checks(
    evaluator: Evaluator,
    parent_node,
    universities: List[UniversityItem],
) -> None:
    """
    Build the 'Official_Documentation' critical node:
      - Ensure each required URL exists
      - Verify that housing/transfer pages are official university pages
      - Verify accreditation URL is authoritative (SACSCOC or official university accreditation page)
    """
    node = evaluator.add_parallel(
        id="Official_Documentation",
        desc="All housing policies and transfer credit policies are verified from official university websites or official accreditation databases",
        parent=parent_node,
        critical=True,
    )

    for i, uni in enumerate(universities[:3]):
        disp = uni_display_name(uni, i)

        # Existence checks (critical under Official_Documentation)
        evaluator.add_custom_node(
            result=bool(uni.housing_policy_url and uni.housing_policy_url.strip()),
            id=f"uni_{i}_housing_url_provided",
            desc=f"{disp}: Housing policy URL is provided",
            parent=node,
            critical=True
        )
        evaluator.add_custom_node(
            result=bool(uni.transfer_policy_url and uni.transfer_policy_url.strip()),
            id=f"uni_{i}_transfer_url_provided",
            desc=f"{disp}: Transfer policy URL is provided",
            parent=node,
            critical=True
        )
        evaluator.add_custom_node(
            result=bool(uni.accreditation_url and uni.accreditation_url.strip()),
            id=f"uni_{i}_accreditation_url_provided",
            desc=f"{disp}: Accreditation verification URL is provided",
            parent=node,
            critical=True
        )

        # Housing URL is official university page
        if uni.housing_policy_url:
            leaf = evaluator.add_leaf(
                id=f"uni_{i}_housing_url_official",
                desc=f"{disp}: Housing policy page is an official university page",
                parent=node,
                critical=True,
            )
            claim = f"The URL is an official page belonging to {disp}'s website (e.g., a .edu domain or clearly branded as the university) and discusses housing policy."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=uni.housing_policy_url,
                additional_instruction="Confirm the page branding and domain indicate an official university page. A .edu domain or a clearly branded official subdomain is strong evidence."
            )

        # Transfer URL is official university page
        if uni.transfer_policy_url:
            leaf = evaluator.add_leaf(
                id=f"uni_{i}_transfer_url_official",
                desc=f"{disp}: Transfer policy page is an official university page",
                parent=node,
                critical=True,
            )
            claim = f"The URL is an official page belonging to {disp}'s website (e.g., a .edu domain or clearly branded as the university) and discusses transfer credit policies."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=uni.transfer_policy_url,
                additional_instruction="Confirm the page branding and domain indicate an official university page, and that the content clearly pertains to transfer credit policy."
            )

        # Accreditation URL is authoritative (SACSCOC or official statement)
        if uni.accreditation_url:
            leaf = evaluator.add_leaf(
                id=f"uni_{i}_accreditation_url_authoritative",
                desc=f"{disp}: Accreditation verification page is authoritative (SACSCOC or official university accreditation statement)",
                parent=node,
                critical=True,
            )
            claim = f"The URL is either on sacscoc.org showing {disp}'s accreditation listing, or an official {disp} page that clearly states SACSCOC accreditation."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=uni.accreditation_url,
                additional_instruction="Accept sacscoc.org institution listings, or an official university accreditation page explicitly mentioning SACSCOC. If the page is unrelated or unauthoritative, mark as incorrect."
            )


async def build_common_property_checks(
    evaluator: Evaluator,
    parent_node,
    universities: List[UniversityItem],
) -> None:
    """
    Build critical checks that apply to all three universities:
      - Located in Texas
      - Public institution
      - Accredited by SACSCOC
    """
    # Located in Texas
    loc_node = evaluator.add_parallel(
        id="All_Universities_Located_in_Texas",
        desc="Verify that all three provided universities are located in the state of Texas",
        parent=parent_node,
        critical=True,
    )
    for i, uni in enumerate(universities[:3]):
        disp = uni_display_name(uni, i)
        leaf = evaluator.add_leaf(
            id=f"uni_{i}_located_in_texas",
            desc=f"{disp} is located in Texas",
            parent=loc_node,
            critical=True,
        )
        claim = f"The institution named '{disp}' is located in the U.S. state of Texas (TX)."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=collect_sources_for_uni(uni),
            additional_instruction="Accept evidence such as address or state field indicating Texas (TX), including on SACSCOC listing or official university pages."
        )

    # Public institution
    public_node = evaluator.add_parallel(
        id="All_Universities_Are_Public",
        desc="Verify that all three provided universities are public institutions (not private)",
        parent=parent_node,
        critical=True,
    )
    for i, uni in enumerate(universities[:3]):
        disp = uni_display_name(uni, i)
        leaf = evaluator.add_leaf(
            id=f"uni_{i}_is_public",
            desc=f"{disp} is a public institution (not private)",
            parent=public_node,
            critical=True,
        )
        claim = f"The institution named '{disp}' is a public university (not private)."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=collect_sources_for_uni(uni),
            additional_instruction="SACSCOC directory often lists 'Control: Public/Private'—that is sufficient. Also accept explicit 'public university' statements on official pages."
        )

    # SACSCOC accreditation
    accred_node = evaluator.add_parallel(
        id="SACSCOC_Accreditation",
        desc="Verify that all three universities are accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)",
        parent=parent_node,
        critical=True,
    )
    for i, uni in enumerate(universities[:3]):
        disp = uni_display_name(uni, i)
        leaf = evaluator.add_leaf(
            id=f"uni_{i}_sacscoc_accredited",
            desc=f"{disp} is accredited by SACSCOC",
            parent=accred_node,
            critical=True,
        )
        claim = f"The institution named '{disp}' is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)."
        sources = [uni.accreditation_url] if uni.accreditation_url else collect_sources_for_uni(uni)
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction="Prefer sacscoc.org listings. Alternatively, accept official university accreditation pages explicitly naming SACSCOC."
        )


async def build_policy_checks_per_university(
    evaluator: Evaluator,
    parent_node,
    universities: List[UniversityItem],
) -> None:
    """
    Build non-critical policy checks for each university:
      - Freshman housing NOT required
      - Accepts at least 60 semester credit hours in transfer
    Each university block uses a sequential node with a critical existence gate.
    """
    for i, uni in enumerate(universities[:3]):
        title = f"{ordinal_name(i)}_University_Policies"
        disp = uni_display_name(uni, i)

        uni_seq = evaluator.add_sequential(
            id=title,
            desc=f"The {ordinal_name(i).lower()} university does not require first-year students to live on campus AND accepts at least 60 transfer credits from community colleges",
            parent=parent_node,
            critical=False,  # Non-critical overall; allows partial credit if some universities pass and others don't
        )

        # Gate: require both housing and transfer URLs to proceed
        gate_ok = bool(uni.housing_policy_url and uni.transfer_policy_url)
        evaluator.add_custom_node(
            result=gate_ok,
            id=f"uni_{i}_policies_gate_urls_present",
            desc=f"{disp}: Both housing and transfer policy URLs are provided to evaluate policies",
            parent=uni_seq,
            critical=True
        )

        # Housing: NOT required for first-year undergraduates
        housing_leaf = evaluator.add_leaf(
            id=f"uni_{i}_housing_no_requirement",
            desc=f"{disp}: First-year undergrads are NOT required to live on campus",
            parent=uni_seq,
            critical=True
        )
        housing_claim = (
            "This page shows that first-year (freshman) undergraduate students are not required to live on campus. "
            "On-campus housing is optional rather than mandatory."
        )
        await evaluator.verify(
            claim=housing_claim,
            node=housing_leaf,
            sources=uni.housing_policy_url if uni.housing_policy_url else None,
            additional_instruction=(
                "Treat policies that say 'required' or 'mandatory' (even with standard exemptions like living with parents, age, distance, etc.) as a requirement (therefore this claim would be false). "
                "Accept as true only if the page clearly indicates there is no general freshman live-on requirement (e.g., 'optional', 'not required', 'no policy mandating on-campus living')."
            )
        )

        # Transfer: accepts at least 60 semester credit hours
        transfer_leaf = evaluator.add_leaf(
            id=f"uni_{i}_accepts_60plus_transfer",
            desc=f"{disp}: Accepts at least 60 semester credit hours in transfer",
            parent=uni_seq,
            critical=True
        )
        transfer_claim = (
            "This page indicates the university accepts at least 60 semester credit hours in transfer toward a bachelor's degree, "
            "including credits from accredited community colleges."
        )
        await evaluator.verify(
            claim=transfer_claim,
            node=transfer_leaf,
            sources=uni.transfer_policy_url if uni.transfer_policy_url else None,
            additional_instruction=(
                "It is sufficient if the policy states a maximum transfer limit of 60 or more hours (e.g., 60, 64, 66). "
                "Phrases like 'up to 66 lower-division hours' or 'up to 60 credits' satisfy the criterion. "
                "The page should clearly pertain to undergraduate transfer policy."
            )
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Texas transfer/housing/accreditation universities task.
    """
    # Initialize evaluator with a non-critical root (to allow mixed critical/non-critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three public universities in Texas that meet all specified criteria regarding accreditation, housing policies, and transfer credit acceptance",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract up to three universities and their URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Prepare exactly three slots
    universities: List[UniversityItem] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Record a compact view of what was extracted
    evaluator.add_custom_info(
        info={
            "universities": [
                {
                    "name": u.name,
                    "housing_policy_url": u.housing_policy_url,
                    "transfer_policy_url": u.transfer_policy_url,
                    "accreditation_url": u.accreditation_url,
                    "additional_urls_count": len(u.additional_urls or []),
                }
                for u in universities
            ]
        },
        info_type="extraction_summary",
        info_name="extracted_universities_overview"
    )

    # Build critical documentation checks
    await build_official_documentation_checks(evaluator, root, universities)

    # Build critical common property checks (TX location, public, SACSCOC)
    await build_common_property_checks(evaluator, root, universities)

    # Build non-critical policy checks per university (housing not required + >=60 transfer credits)
    await build_policy_checks_per_university(evaluator, root, universities)

    # Return structured evaluation summary
    return evaluator.get_summary()