import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "two_publications_oct_nov_2025_mars_and_creta_sharks"
TASK_DESCRIPTION = """
Identify two research publications from October–November 2025 that meet the following criteria:

Publication 1: A study reporting the first in situ detection of electrical activity in the Martian atmosphere, using acoustic data from the SuperCam microphone instrument aboard NASA's Perseverance rover. The lead author must be affiliated with the Institut de Recherche en Astrophysique et Planétologie, Université de Toulouse, France. The publication must appear in Nature journal.

Publication 2: A study on Cretaceous-period shark fossils discovered in the Darwin Formation of Northern Territory, Australia. The fossils must be from the upper Aptian geological stage (approximately 115 million years ago) and belong to the family Cardabiodontidae (lamniform sharks). The lead author must be affiliated with the Department of Earth and Planetary Sciences at Stanford University. The publication must appear in Communications Biology journal.

For each publication, provide:
1. The full name of the lead author
2. The complete institutional affiliation (including department/institute, university, and country)
3. The exact publication title
4. The journal name and publication date
5. A valid URL reference to the publication
6. For Publication 1: The number of electrical discharge events detected and the atmospheric phenomena they were associated with
7. For Publication 2: The geological formation name, the specific territory/region in Australia, the geological time period (stage and approximate age), the shark family classification, estimated body length range, the number of vertebral centra analyzed, and the museum housing the specimens
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Publication1Info(BaseModel):
    lead_author_name: Optional[str] = None
    affiliation: Optional[str] = None  # the full affiliation string as provided in the answer
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None  # keep as free-form string from the answer
    url: Optional[str] = None
    discharge_event_count: Optional[str] = None  # e.g., "2", "two", "two events"
    associated_phenomena: Optional[str] = None  # e.g., "dust devils", "convective vortices"


class Publication2Info(BaseModel):
    lead_author_name: Optional[str] = None
    affiliation: Optional[str] = None  # full affiliation string
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None
    formation: Optional[str] = None  # e.g., "Darwin Formation"
    region: Optional[str] = None  # e.g., "Northern Territory, Australia"
    stage: Optional[str] = None  # e.g., "upper Aptian"
    age: Optional[str] = None  # e.g., "~115 Ma", "around 115 million years"
    classification_order: Optional[str] = None  # e.g., "Lamniformes" or "lamniform sharks"
    classification_family: Optional[str] = None  # e.g., "Cardabiodontidae"
    body_length_range: Optional[str] = None  # e.g., "6–8 m", "6.3–7.6 m"
    vertebral_centra_count: Optional[str] = None  # e.g., "6"
    museum: Optional[str] = None  # e.g., "Northern Territory Museum", "MAGNT"


class PublicationsExtraction(BaseModel):
    pub1: Optional[Publication1Info] = None
    pub2: Optional[Publication2Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_publications() -> str:
    return """
    Extract structured information for two publications (Publication 1 and Publication 2) exactly as stated in the answer text. If multiple candidate publications are mentioned, pick the ones the answer claims meet the constraints; do not invent or infer.

    For Publication 1 (Mars electrical activity, SuperCam microphone, Perseverance, Nature, Oct–Nov 2025):
    - lead_author_name: Full name of the lead/first author as given
    - affiliation: The full affiliation string for the lead author as given (do not split)
    - title: Exact publication title text as presented
    - journal: Journal name as presented (e.g., "Nature")
    - publication_date: The publication date string as presented in the answer (e.g., "20 November 2025")
    - url: A URL to the publication page provided in the answer (if any)
    - discharge_event_count: The number of electrical discharge events detected as stated in the answer
    - associated_phenomena: The atmospheric phenomena they were associated with as stated (e.g., "dust devils", "convective vortices")

    For Publication 2 (Cretaceous sharks, Darwin Formation, Northern Territory, Australia; upper Aptian ~115 Ma; Cardabiodontidae; Communications Biology, Oct–Nov 2025):
    - lead_author_name: Full name of the lead/first author as given
    - affiliation: Full affiliation string for the lead author as given
    - title: Exact publication title text as presented
    - journal: Journal name as presented (e.g., "Communications Biology")
    - publication_date: The publication date string as presented in the answer
    - url: A URL to the publication page provided in the answer (if any)
    - formation: Geological formation name as stated (e.g., "Darwin Formation")
    - region: Specific region/territory in Australia (e.g., "Northern Territory, Australia")
    - stage: Geological stage (e.g., "upper Aptian")
    - age: Approximate age phrasing as stated (e.g., "~115 Ma", "approximately 115 million years ago")
    - classification_order: The order/group as stated (e.g., "Lamniformes", "lamniform sharks")
    - classification_family: The family (e.g., "Cardabiodontidae")
    - body_length_range: Estimated body length range as stated (e.g., "6–8 m", "6.3–7.6 m")
    - vertebral_centra_count: Number of vertebral centra analyzed as stated (extract the number in the answer)
    - museum: Museum housing the specimens as stated (e.g., "Northern Territory Museum", "MAGNT")

    Rules:
    - Return null for any field missing from the answer.
    - Do not add or infer any content not explicitly present in the answer.
    - For URLs, extract the actual URL string that is present; if missing protocol, keep as-is.
    - Keep all fields as strings (do not convert numbers to numeric types).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


def extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d+", text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_pub1_tree_and_verify(evaluator: Evaluator, root_node, pub1: Publication1Info) -> None:
    # Publication 1 parent node (parallel, non-critical at this level)
    pub1_node = evaluator.add_parallel(
        id="publication_1",
        desc="Publication 1 (Mars atmospheric electrical activity using SuperCam microphone acoustic data on Perseverance; Nature; Oct–Nov 2025; lead author affiliation IRAP/Université de Toulouse/France) and all required reported fields.",
        parent=root_node,
        critical=False,
    )

    # ---------------- Bibliographic info (critical) ----------------
    biblio_node = evaluator.add_parallel(
        id="pub1_bibliographic_info",
        desc="Provide required bibliographic fields for Publication 1.",
        parent=pub1_node,
        critical=True,
    )

    # URL validity as critical custom node (acts as a gate for other URL-based checks)
    evaluator.add_custom_node(
        result=is_valid_url(pub1.url),
        id="pub1_url",
        desc="Provide a valid URL reference to Publication 1.",
        parent=biblio_node,
        critical=True,
    )

    # Title leaf
    title_leaf = evaluator.add_leaf(
        id="pub1_title",
        desc="Provide the exact publication title for Publication 1.",
        parent=biblio_node,
        critical=True,
    )

    # Journal leaf
    journal_leaf = evaluator.add_leaf(
        id="pub1_journal",
        desc="Journal name for Publication 1 is Nature.",
        parent=biblio_node,
        critical=True,
    )

    # Publication date leaf
    pub_date_leaf = evaluator.add_leaf(
        id="pub1_pub_date",
        desc="Publication date for Publication 1 is in October–November 2025.",
        parent=biblio_node,
        critical=True,
    )

    # Prepare claims for batch verify (title, journal, date)
    title_claim = f'The publication title is "{pub1.title}".'
    journal_claim = "This publication appears in the journal Nature."
    pub_date_claim = "The publication date is in October or November 2025 (between 2025-10-01 and 2025-11-30)."

    await evaluator.batch_verify([
        (
            title_claim,
            pub1.url,
            title_leaf,
            "Match the article's main title. Allow minor punctuation, casing, hyphen, and subtitle variations. Do not accept a different paper."
        ),
        (
            journal_claim,
            pub1.url,
            journal_leaf,
            "Verify the journal/brand on the publication page (e.g., Nature). Do not confuse with Nature Communications or other Nature Portfolio journals."
        ),
        (
            pub_date_claim,
            pub1.url,
            pub_date_leaf,
            "Use the official 'Published' date or 'Published online' date shown. Accept any date in Oct or Nov 2025."
        )
    ])

    # ---------------- Lead author & affiliation (critical) ----------------
    auth_aff_node = evaluator.add_parallel(
        id="pub1_lead_author_and_affiliation",
        desc="Provide lead author full name and complete institutional affiliation for Publication 1, meeting the stated affiliation constraint.",
        parent=pub1_node,
        critical=True,
    )

    lead_author_leaf = evaluator.add_leaf(
        id="pub1_lead_author_name",
        desc="Provide the full name of Publication 1 lead author.",
        parent=auth_aff_node,
        critical=True,
    )
    aff_inst_leaf = evaluator.add_leaf(
        id="pub1_affiliation_institute",
        desc="Lead author affiliation includes Institut de Recherche en Astrophysique et Planétologie (IRAP).",
        parent=auth_aff_node,
        critical=True,
    )
    aff_univ_leaf = evaluator.add_leaf(
        id="pub1_affiliation_university",
        desc="Lead author affiliation includes Université de Toulouse.",
        parent=auth_aff_node,
        critical=True,
    )
    aff_country_leaf = evaluator.add_leaf(
        id="pub1_affiliation_country",
        desc="Lead author affiliation country is France.",
        parent=auth_aff_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            f"The lead (first) author of the paper is {pub1.lead_author_name}.",
            pub1.url,
            lead_author_leaf,
            "Check the author list and identify the first/lead author name. Allow minor variations (middle initials, accents)."
        ),
        (
            "The lead author's affiliation includes Institut de Recherche en Astrophysique et Planétologie (IRAP).",
            pub1.url,
            aff_inst_leaf,
            "Verify that the affiliation of the first/lead author explicitly mentions IRAP. Accept 'IRAP' acronym. Ensure it is for the lead author."
        ),
        (
            "The lead author's affiliation includes Université de Toulouse.",
            pub1.url,
            aff_univ_leaf,
            "Verify the lead author's affiliation includes Université de Toulouse (accept variants like Université Toulouse III - Paul Sabatier / UPS)."
        ),
        (
            "The lead author's affiliation country is France.",
            pub1.url,
            aff_country_leaf,
            "Confirm the affiliation indicates France (country). It may appear in the address line."
        )
    ])

    # ---------------- Topic & method constraints (critical) ----------------
    topic_node = evaluator.add_parallel(
        id="pub1_topic_and_method_constraints",
        desc="Verify Publication 1 satisfies the Mars topic/method constraints.",
        parent=pub1_node,
        critical=True,
    )

    first_in_situ_leaf = evaluator.add_leaf(
        id="pub1_first_in_situ_detection",
        desc="Publication 1 reports the first in situ detection of electrical activity in the Martian atmosphere.",
        parent=topic_node,
        critical=True,
    )
    supercam_leaf = evaluator.add_leaf(
        id="pub1_uses_supercam_microphone_acoustic_data",
        desc="Publication 1 uses acoustic data from the SuperCam microphone instrument.",
        parent=topic_node,
        critical=True,
    )
    perseverance_leaf = evaluator.add_leaf(
        id="pub1_uses_perseverance_data",
        desc="Publication 1 uses data from NASA’s Perseverance rover.",
        parent=topic_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            "The study reports the first in situ detection of electrical activity in the Martian atmosphere.",
            pub1.url,
            first_in_situ_leaf,
            "Look for explicit statements like 'first in situ detection' in abstract or main text."
        ),
        (
            "The study uses acoustic data collected by the SuperCam microphone instrument.",
            pub1.url,
            supercam_leaf,
            "Confirm SuperCam microphone acoustic data are used to detect electrical activity."
        ),
        (
            "The study uses data from NASA's Perseverance rover.",
            pub1.url,
            perseverance_leaf,
            "Confirm Perseverance rover is the mission/instrument platform providing the data."
        )
    ])

    # ---------------- Pub1-specific requested result fields (critical) ----------------
    results_node = evaluator.add_parallel(
        id="pub1_requested_results_fields",
        desc="Provide the additional Pub1-specific requested result fields.",
        parent=pub1_node,
        critical=True,
    )

    discharge_count_leaf = evaluator.add_leaf(
        id="pub1_discharge_event_count",
        desc="Provide the number of electrical discharge events detected (as stated by the publication).",
        parent=results_node,
        critical=True,
    )
    associated_phenomena_leaf = evaluator.add_leaf(
        id="pub1_associated_phenomena",
        desc="State what atmospheric phenomena the detected events were associated with (as stated by the publication).",
        parent=results_node,
        critical=True,
    )

    dis_count_text = pub1.discharge_event_count if pub1.discharge_event_count else ""
    phenomena_text = pub1.associated_phenomena if pub1.associated_phenomena else ""

    await evaluator.batch_verify([
        (
            f"The study reports {dis_count_text} electrical discharge event(s).",
            pub1.url,
            discharge_count_leaf,
            "Find the reported number of detected electrical discharge events. Accept numerals or words (e.g., 'two')."
        ),
        (
            f"The detected electrical events were associated with {phenomena_text}.",
            pub1.url,
            associated_phenomena_leaf,
            "Confirm the atmospheric phenomena (e.g., dust devils/convective vortices) explicitly associated with these events."
        )
    ])


async def build_pub2_tree_and_verify(evaluator: Evaluator, root_node, pub2: Publication2Info) -> None:
    # Publication 2 parent node (parallel, non-critical at this level)
    pub2_node = evaluator.add_parallel(
        id="publication_2",
        desc="Publication 2 (Cretaceous shark fossils; Darwin Formation, Northern Territory, Australia; upper Aptian ~115 Ma; Cardabiodontidae lamniforms; Communications Biology; Oct–Nov 2025; lead author affiliation Stanford EPS/USA) and all required reported fields.",
        parent=root_node,
        critical=False,
    )

    # ---------------- Bibliographic info (critical) ----------------
    biblio_node = evaluator.add_parallel(
        id="pub2_bibliographic_info",
        desc="Provide required bibliographic fields for Publication 2.",
        parent=pub2_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_valid_url(pub2.url),
        id="pub2_url",
        desc="Provide a valid URL reference to Publication 2.",
        parent=biblio_node,
        critical=True,
    )

    title_leaf = evaluator.add_leaf(
        id="pub2_title",
        desc="Provide the exact publication title for Publication 2.",
        parent=biblio_node,
        critical=True,
    )
    journal_leaf = evaluator.add_leaf(
        id="pub2_journal",
        desc="Journal name for Publication 2 is Communications Biology.",
        parent=biblio_node,
        critical=True,
    )
    pub_date_leaf = evaluator.add_leaf(
        id="pub2_pub_date",
        desc="Publication date for Publication 2 is in October–November 2025.",
        parent=biblio_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            f'The publication title is "{pub2.title}".',
            pub2.url,
            title_leaf,
            "Match the article's main title. Allow small punctuation/casing differences. Do not accept a different paper."
        ),
        (
            "This publication appears in the journal Communications Biology.",
            pub2.url,
            journal_leaf,
            "Verify the journal shown is Communications Biology (Nature Portfolio)."
        ),
        (
            "The publication date is in October or November 2025 (between 2025-10-01 and 2025-11-30).",
            pub2.url,
            pub_date_leaf,
            "Use the official 'Published' or 'Published online' date shown. Accept any date in Oct or Nov 2025."
        )
    ])

    # ---------------- Lead author & affiliation (critical) ----------------
    auth_aff_node = evaluator.add_parallel(
        id="pub2_lead_author_and_affiliation",
        desc="Provide lead author full name and complete institutional affiliation for Publication 2, meeting the stated affiliation constraint.",
        parent=pub2_node,
        critical=True,
    )

    lead_author_leaf = evaluator.add_leaf(
        id="pub2_lead_author_name",
        desc="Provide the full name of Publication 2 lead author.",
        parent=auth_aff_node,
        critical=True,
    )
    dept_leaf = evaluator.add_leaf(
        id="pub2_affiliation_department",
        desc="Lead author affiliation includes Department of Earth and Planetary Sciences.",
        parent=auth_aff_node,
        critical=True,
    )
    univ_leaf = evaluator.add_leaf(
        id="pub2_affiliation_university",
        desc="Lead author affiliation includes Stanford University.",
        parent=auth_aff_node,
        critical=True,
    )
    country_leaf = evaluator.add_leaf(
        id="pub2_affiliation_country",
        desc="Lead author affiliation country is USA.",
        parent=auth_aff_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            f"The lead (first) author of the paper is {pub2.lead_author_name}.",
            pub2.url,
            lead_author_leaf,
            "Check the author list and identify the first/lead author name. Allow minor variations (initials, accents)."
        ),
        (
            "The lead author's affiliation includes the Department of Earth and Planetary Sciences.",
            pub2.url,
            dept_leaf,
            "Verify the affiliation text for the lead author includes 'Department of Earth and Planetary Sciences' (EPS)."
        ),
        (
            "The lead author's affiliation includes Stanford University.",
            pub2.url,
            univ_leaf,
            "Verify the lead author's affiliation includes Stanford University."
        ),
        (
            "The lead author's affiliation country is USA.",
            pub2.url,
            country_leaf,
            "Confirm the affiliation indicates United States (USA). It may appear in address lines."
        )
    ])

    # ---------------- Geology & geography constraints (critical) ----------------
    geo_node = evaluator.add_parallel(
        id="pub2_geology_geography_constraints",
        desc="Verify Publication 2 satisfies the stated geographic and geologic constraints.",
        parent=pub2_node,
        critical=True,
    )

    formation_leaf = evaluator.add_leaf(
        id="pub2_formation",
        desc="Fossils are from the Darwin Formation.",
        parent=geo_node,
        critical=True,
    )
    region_leaf = evaluator.add_leaf(
        id="pub2_region",
        desc="Discovery location is Northern Territory, Australia.",
        parent=geo_node,
        critical=True,
    )
    stage_leaf = evaluator.add_leaf(
        id="pub2_stage",
        desc="Geological stage is upper Aptian.",
        parent=geo_node,
        critical=True,
    )
    age_leaf = evaluator.add_leaf(
        id="pub2_age",
        desc="Approximate age is about 115 million years ago (Cretaceous).",
        parent=geo_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            "The fossils are from the Darwin Formation.",
            pub2.url,
            formation_leaf,
            "Find 'Darwin Formation' as the formation source of the fossils."
        ),
        (
            "The discovery location/region is in the Northern Territory, Australia.",
            pub2.url,
            region_leaf,
            "Confirm the geographic region is Northern Territory in Australia."
        ),
        (
            "The geological stage is upper Aptian.",
            pub2.url,
            stage_leaf,
            "Look for 'upper Aptian' (Early Cretaceous)."
        ),
        (
            "The age is approximately 115 million years ago (about 115 Ma).",
            pub2.url,
            age_leaf,
            "Accept approximate ranges that center around ~115 Ma (e.g., 113–116 Ma)."
        )
    ])

    # ---------------- Taxonomy constraints (critical) ----------------
    tax_node = evaluator.add_parallel(
        id="pub2_taxonomy_constraints",
        desc="Verify Publication 2 satisfies the stated taxonomic constraints.",
        parent=pub2_node,
        critical=True,
    )

    lamniform_leaf = evaluator.add_leaf(
        id="pub2_lamniform",
        desc="Specimens are lamniform sharks.",
        parent=tax_node,
        critical=True,
    )
    cardabio_leaf = evaluator.add_leaf(
        id="pub2_cardabiodontidae",
        desc="Specimens belong to family Cardabiodontidae.",
        parent=tax_node,
        critical=True,
    )

    await evaluator.batch_verify([
        (
            "The specimens are lamniform sharks (order Lamniformes).",
            pub2.url,
            lamniform_leaf,
            "Confirm the order/group is lamniform (Lamniformes)."
        ),
        (
            "The specimens belong to the family Cardabiodontidae.",
            pub2.url,
            cardabio_leaf,
            "Look for Cardabiodontidae family assignment."
        )
    ])

    # ---------------- Specimen/result specific fields (critical) ----------------
    spec_node = evaluator.add_parallel(
        id="pub2_requested_specimen_fields",
        desc="Provide the additional Pub2-specific requested specimen/result fields (including stated quantitative constraints).",
        parent=pub2_node,
        critical=True,
    )

    body_len_leaf = evaluator.add_leaf(
        id="pub2_body_length_range",
        desc="Provide estimated body length range; must be in the 6–8 meter range per constraints.",
        parent=spec_node,
        critical=True,
    )
    centra_leaf = evaluator.add_leaf(
        id="pub2_vertebral_centra_count",
        desc="Provide the number of vertebral centra analyzed; must be at least five per constraints.",
        parent=spec_node,
        critical=True,
    )
    museum_leaf = evaluator.add_leaf(
        id="pub2_museum_housing_specimens",
        desc="Identify the museum housing the specimens; must be Northern Territory Museum collections per constraints.",
        parent=spec_node,
        critical=True,
    )

    centra_text = pub2.vertebral_centra_count if pub2.vertebral_centra_count else ""
    body_text = pub2.body_length_range if pub2.body_length_range else ""
    museum_text = pub2.museum if pub2.museum else ""

    await evaluator.batch_verify([
        (
            "The estimated body length is in the range of approximately 6 to 8 meters.",
            pub2.url,
            body_len_leaf,
            f"Verify the publication's stated body length range. Accept values that fall within about 6–8 m (e.g., {body_text} such as 6.3–7.6 m)."
        ),
        (
            f"The study analyzed {centra_text} vertebral centra, which is at least five.",
            pub2.url,
            centra_leaf,
            "Find the number of vertebral centra studied and confirm it is ≥ 5. Accept numerals or words."
        ),
        (
            "The specimens are housed in the Northern Territory Museum collections.",
            pub2.url,
            museum_leaf,
            "Accept 'Museum and Art Gallery of the Northern Territory (MAGNT)' as equivalent to Northern Territory Museum collections."
        )
    ])


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
    Evaluate an answer for the two-publication task.
    """
    # Initialize evaluator (root is non-critical to avoid critical-children restriction)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root node aggregation per rubric
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify and report two research publications (Publication 1 and Publication 2) that meet all stated date/journal/topic/affiliation constraints and provide all requested fields.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record constraint info as ground truth/context
    evaluator.add_ground_truth({
        "pub1_constraints": {
            "journal": "Nature",
            "date_window": "2025-10-01 to 2025-11-30",
            "topic": "First in situ detection of electrical activity in Martian atmosphere using SuperCam microphone on Perseverance",
            "lead_affiliation_required": ["IRAP", "Université de Toulouse", "France"]
        },
        "pub2_constraints": {
            "journal": "Communications Biology",
            "date_window": "2025-10-01 to 2025-11-30",
            "geology": ["Darwin Formation", "Northern Territory, Australia", "upper Aptian (~115 Ma)"],
            "taxonomy": ["Lamniformes (lamniform sharks)", "Cardabiodontidae"],
            "specimens": {
                "body_length_range": "within ~6–8 m",
                "vertebral_centra_min": 5,
                "museum": "Northern Territory Museum collections (MAGNT acceptable)"
            }
        }
    })

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_publications(),
        template_class=PublicationsExtraction,
        extraction_name="publications_extraction",
    )

    # Build trees and verify both publications (create nodes even if some fields are missing)
    await build_pub1_tree_and_verify(evaluator, root, extracted.pub1 or Publication1Info())
    await build_pub2_tree_and_verify(evaluator, root, extracted.pub2 or Publication2Info())

    # Return structured summary
    return evaluator.get_summary()