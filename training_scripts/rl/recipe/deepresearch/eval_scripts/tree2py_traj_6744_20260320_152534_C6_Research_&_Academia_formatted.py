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
TASK_ID = "aastcs11_eclipse_visibility_2026"
TASK_DESCRIPTION = """
Identify four astronomers from the confirmed plenary speaker list for the AASTCS 11: Exoplanet Atmospheres 2026 conference (scheduled for March 16-20, 2026 in Denver, Colorado) whose primary institutional affiliations are located in geographic regions where the total lunar eclipse on March 3, 2026 will have its totality phase visible. For each astronomer, provide: (1) Their full name, (2) Confirmation that they are a plenary speaker at AASTCS 11 (with supporting URL), (3) Their primary institutional affiliation name, (4) The institution's location (city and state/country), (5) Verification that the totality phase of the March 3, 2026 eclipse is visible from this location (with supporting URL), and (6) A description of their research focus and how it relates to exoplanet atmospheres (with supporting URL). Note: According to NASA visibility information, the totality phase of the March 3, 2026 lunar eclipse is visible from North America (early morning), eastern Asia (evening), Australia (evening), and throughout the Pacific region. The eclipse is NOT visible from Africa or Europe during totality.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AstronomerItem(BaseModel):
    name: Optional[str] = None
    speaker_urls: List[str] = Field(default_factory=list, description="URLs confirming plenary speaker status for AASTCS 11: Exoplanet Atmospheres 2026")
    institution_name: Optional[str] = None
    institution_urls: List[str] = Field(default_factory=list, description="URLs verifying affiliation with the stated institution")
    location_city: Optional[str] = None
    location_region: Optional[str] = None  # state/province for US, or country otherwise
    visibility_urls: List[str] = Field(default_factory=list, description="URLs (e.g., NASA) supporting totality visibility for the location on 3 Mar 2026")
    research_description: Optional[str] = None
    research_urls: List[str] = Field(default_factory=list, description="URLs supporting research focus relevant to exoplanet atmospheres")


class AstronomersExtraction(BaseModel):
    astronomers: List[AstronomerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_astronomers() -> str:
    return """
    Extract up to four astronomers from the answer who are stated to be confirmed plenary speakers for
    “AASTCS 11: Exoplanet Atmospheres 2026”. For each astronomer, extract:

    - name: Full name as written in the answer.
    - speaker_urls: All URLs that the answer cites to confirm they are a plenary speaker at AASTCS 11: Exoplanet Atmospheres 2026.
      These should be direct pages (e.g., official AAS/AASTCS site, conference program page, or comparable official confirmation).
    - institution_name: The primary institutional affiliation name (e.g., University/Institute/Observatory).
    - institution_urls: URLs that verify the astronomer’s affiliation with that institution (e.g., official profile page, lab page).
    - location_city: The city of that institution (as stated in the answer).
    - location_region: The state/province (for US/Canada/Australia) or country (for non-US) as stated in the answer.
    - visibility_urls: URLs that support that the totality phase of the 3 March 2026 lunar eclipse is visible from the stated location
      (e.g., NASA eclipse map, reputable astronomy orgs). Use URLs explicitly present in the answer.
    - research_description: A brief description (1–2 sentences) of the astronomer’s research focus as stated in the answer.
    - research_urls: URLs cited in the answer that support the research focus description (e.g., official profile, publications, group pages).

    Rules:
    - Extract ONLY what is explicitly present in the answer. Do not infer or invent details.
    - For URLs: return exactly the URLs present in the answer (including those inside markdown links). Do not fabricate URLs.
    - If any field is missing in the answer for an astronomer, set it to null (for strings) or [] (for URL lists).
    - Return a JSON object with a single field: "astronomers": an array of up to 4 such objects in the same order as presented in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
NASA_REGION_NOTE = (
    "According to widely-cited NASA visibility information, totality for the 3 March 2026 lunar eclipse is visible from "
    "North America (early morning), eastern Asia (evening), Australia (evening), and across the Pacific. It is not visible "
    "from Africa or Europe during totality. Use the provided visibility URL(s) to confirm that the stated city/region falls "
    "within a totality-visible region, as shown or stated on the page(s). Prefer explicit maps or region lists."
)


async def verify_astronomer(
    evaluator: Evaluator,
    parent_node,
    item: AstronomerItem,
    idx: int,
) -> None:
    """
    Build and run the verification subtree for a single astronomer.
    The top-level astronomer node is sequential: if identity fails, remaining checks are skipped.
    """
    # Top-level astronomer sequential node
    astro_node = evaluator.add_sequential(
        id=f"Astronomer_{idx+1}",
        desc=f"Astronomer #{idx+1} meeting all criteria",
        parent=parent_node,
        critical=False,  # allow partial credit across four astronomers
    )

    # --------------------------- Identity (Critical) --------------------------- #
    identity_parent = evaluator.add_parallel(
        id=f"A{idx+1}_Identity",
        desc="The astronomer is explicitly named as a confirmed plenary speaker for AASTCS 11: Exoplanet Atmospheres 2026",
        parent=astro_node,
        critical=True
    )

    # Existence of at least one confirmation URL
    evaluator.add_custom_node(
        result=bool(item.speaker_urls),
        id=f"A{idx+1}_Identity_URL",
        desc="A URL is provided that confirms this person is a plenary speaker at AASTCS 11",
        parent=identity_parent,
        critical=True
    )

    # Content verification: page confirms plenary speaker status for AASTCS 11
    identity_check = evaluator.add_leaf(
        id=f"A{idx+1}_Identity_Verified",
        desc="Confirmation page(s) explicitly list this person as a plenary speaker for AASTCS 11: Exoplanet Atmospheres 2026",
        parent=identity_parent,
        critical=True
    )
    name_for_claim = item.name or "the person"
    await evaluator.verify(
        claim=f"The provided page(s) confirm that {name_for_claim} is a confirmed plenary speaker for 'AASTCS 11: Exoplanet Atmospheres 2026'.",
        node=identity_check,
        sources=item.speaker_urls,
        additional_instruction="Focus on whether the page explicitly indicates 'plenary' for AASTCS 11: Exoplanet Atmospheres 2026. "
                               "Accept minor naming or formatting variations. If the page is for a different year or only lists "
                               "an invited/contributed talk (not plenary), treat as unsupported."
    )

    # ------------------------- Affiliation (Critical) ------------------------- #
    affiliation_parent = evaluator.add_parallel(
        id=f"A{idx+1}_Affiliation",
        desc="Verification of the astronomer's current institutional affiliation",
        parent=astro_node,
        critical=True
    )

    # Institution name provided
    evaluator.add_custom_node(
        result=bool(item.institution_name and item.institution_name.strip()),
        id=f"A{idx+1}_Institution_Name",
        desc="The specific name of the astronomer's primary affiliated institution is provided",
        parent=affiliation_parent,
        critical=True
    )

    # Affiliation URL provided (existence)
    evaluator.add_custom_node(
        result=bool(item.institution_urls),
        id=f"A{idx+1}_Affiliation_URL",
        desc="A URL is provided that verifies the astronomer's affiliation with the stated institution",
        parent=affiliation_parent,
        critical=True
    )

    # Content verification: page supports the affiliation claim
    affiliation_verified = evaluator.add_leaf(
        id=f"A{idx+1}_Affiliation_Verified",
        desc="The URL(s) confirm the astronomer's affiliation with the stated institution",
        parent=affiliation_parent,
        critical=True
    )
    inst_name = item.institution_name or "the stated institution"
    await evaluator.verify(
        claim=f"The provided page(s) confirm that {name_for_claim} is affiliated with {inst_name} (current primary affiliation).",
        node=affiliation_verified,
        sources=item.institution_urls,
        additional_instruction="Look for clear signals such as an official profile page, group page, department listing, or "
                               "institutional news acknowledging the person's current affiliation. Minor title/role differences "
                               "are acceptable as long as the affiliation is clear."
    )

    # --------------------- Eclipse Visibility (Critical) ---------------------- #
    visibility_parent = evaluator.add_parallel(
        id=f"A{idx+1}_Eclipse_Visibility",
        desc="The astronomer's institution is in a region where the March 3, 2026 total lunar eclipse totality is visible",
        parent=astro_node,
        critical=True
    )

    # Location provided (city and state/country)
    has_location = bool(item.location_city and item.location_city.strip()) and bool(item.location_region and item.location_region.strip())
    evaluator.add_custom_node(
        result=has_location,
        id=f"A{idx+1}_Institution_Location",
        desc="The geographic location (city and state/country) of the institution is provided",
        parent=visibility_parent,
        critical=True
    )

    # Visibility URL provided (existence)
    evaluator.add_custom_node(
        result=bool(item.visibility_urls),
        id=f"A{idx+1}_Visibility_URL",
        desc="A URL is provided that confirms eclipse visibility for this geographic region",
        parent=visibility_parent,
        critical=True
    )

    # Content verification: visibility confirmed by the provided source
    location_str = ""
    if item.location_city or item.location_region:
        parts = [p for p in [item.location_city, item.location_region] if p]
        location_str = ", ".join(parts)
    else:
        location_str = "the stated location"

    visibility_verified = evaluator.add_leaf(
        id=f"A{idx+1}_Visibility_Verification",
        desc="Verification that totality is visible from this location, consistent with NASA visibility information",
        parent=visibility_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The totality phase of the March 3, 2026 lunar eclipse is visible from {location_str}.",
        node=visibility_verified,
        sources=item.visibility_urls,
        additional_instruction=NASA_REGION_NOTE + " Prefer explicit confirmation for the specific city/region. "
                                                   "If the page only shows partial visibility or explicitly excludes the region, mark as unsupported."
    )

    # ------------------- Research Focus (Non-critical) ----------------------- #
    research_parent = evaluator.add_parallel(
        id=f"A{idx+1}_Research_Focus",
        desc="The astronomer's research focus is relevant to exoplanet atmospheres",
        parent=astro_node,
        critical=False
    )

    # Research description provided
    evaluator.add_custom_node(
        result=bool(item.research_description and item.research_description.strip()),
        id=f"A{idx+1}_Research_Description",
        desc="A description of the astronomer's research area is provided",
        parent=research_parent,
        critical=False
    )

    # Research URL(s) support that focus
    research_supported = evaluator.add_leaf(
        id=f"A{idx+1}_Research_URL",
        desc="A URL is provided supporting the astronomer's research focus",
        parent=research_parent,
        critical=False
    )
    research_desc_for_claim = item.research_description or "a focus related to exoplanet atmospheres"
    await evaluator.verify(
        claim=f"The provided page(s) indicate that {name_for_claim}'s research is relevant to exoplanet atmospheres (e.g., atmospheric characterization, spectra, retrievals, chemistry, climate, clouds/aerosols). Stated focus: {research_desc_for_claim}.",
        node=research_supported,
        sources=item.research_urls,
        additional_instruction="Look for explicit mentions of exoplanet atmospheres or clearly related topics (atmospheric retrievals, spectroscopy of exoplanets, cloud/aerosol modeling, atmospheric chemistry, climate of exoplanets, JWST atmospheric spectra analysis). "
                               "If only general astrophysics with no atmospheric relevance is shown, mark as unsupported."
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
    """
    Evaluate an answer for the AASTCS 11 eclipse-visibility task.
    """
    # Initialize evaluator (root is non-critical by design in framework)
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

    # Extract astronomers list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_astronomers(),
        template_class=AstronomersExtraction,
        extraction_name="astronomers_extraction"
    )

    # Ground truth policy note (for context in output)
    evaluator.add_ground_truth({
        "eclipse_totality_visibility_regions": "North America (early morning), eastern Asia (evening), Australia (evening), and throughout the Pacific; not Africa/Europe during totality.",
        "eclipse_date": "2026-03-03",
        "conference": "AASTCS 11: Exoplanet Atmospheres 2026"
    })

    # Prepare up to 4 astronomers (pad with empty if fewer)
    items: List[AstronomerItem] = list(extracted.astronomers[:4])
    while len(items) < 4:
        items.append(AstronomerItem())

    # Build and verify each astronomer subtree
    for i in range(4):
        await verify_astronomer(evaluator, root, items[i], i)

    # Return evaluation summary
    return evaluator.get_summary()