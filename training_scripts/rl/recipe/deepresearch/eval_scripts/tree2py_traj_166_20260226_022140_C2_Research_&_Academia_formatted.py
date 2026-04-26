import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "discoveries_2025_comet_shark"
TASK_DESCRIPTION = """
In 2025, two significant scientific discoveries made headlines in the fields of astronomy and paleontology. The first was the detection of interstellar comet 3I/ATLAS, discovered in July 2025. The second was the discovery of giant shark fossils near Darwin, Australia, representing an ancient lamniform shark from the age of dinosaurs.

For your research task:

1. Identify the first academic paper published about the discovery of interstellar comet 3I/ATLAS. Provide:
   - The name of the journal in which it was published
   - The exact publication date
   - The institutional affiliation of the lead author

2. Identify the academic study about the giant shark fossil discovered near Darwin, Australia (the 115-million-year-old lamniform shark). Provide:
   - The name of the journal in which it was published
   - The lead institution responsible for the study
   - The age of the fossil (in millions of years)

Provide reference URLs for all information.
"""

# Ground-truth expectations (used for clarity in breakdown; verification uses URLs)
GT_COMET = {
    "journal": "Monthly Notices of the Royal Astronomical Society Letters (MNRAS Letters)",
    "publication_date": "July 18, 2025",
    "lead_author_affiliation": "Eureka Scientific Incorporated",
}
GT_SHARK = {
    "journal": "Communications Biology",
    "lead_institution": "Swedish Museum of Natural History",
    "fossil_age_mya": "115"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CometPaperInfo(BaseModel):
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SharkStudyInfo(BaseModel):
    journal_name: Optional[str] = None
    lead_institution: Optional[str] = None
    fossil_age_mya: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    comet_3i_atlas_paper: Optional[CometPaperInfo] = None
    darwin_shark_study: Optional[SharkStudyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
    Extract publication details for two parts from the answer text. Return a single JSON object with two top-level objects: 'comet_3i_atlas_paper' and 'darwin_shark_study'.

    1) comet_3i_atlas_paper:
       - journal_name: the journal name of the academic paper about the discovery of interstellar comet 3I/ATLAS (e.g., "MNRAS Letters" / "Monthly Notices of the Royal Astronomical Society: Letters")
       - publication_date: the publication date as written in the answer (any clear human-readable form is fine, e.g., "July 18, 2025" or "18 July 2025")
       - lead_author_affiliation: the institutional affiliation for the lead author, as stated in the answer (e.g., "Eureka Scientific Inc." / "Eureka Scientific Incorporated")
       - sources: array of all URLs explicitly provided in the answer that directly support this comet paper (journal pages, paper pages, DOI pages, or strong authoritative references)

    2) darwin_shark_study:
       - journal_name: the journal name where the giant shark fossil study near Darwin was published (e.g., "Communications Biology")
       - lead_institution: the lead institution for the study (e.g., "Swedish Museum of Natural History"; also accept native-language forms if appearing in the answer)
       - fossil_age_mya: the age (in millions of years) for the fossil as explicitly stated in the answer (e.g., "115", "~115", "approximately 115 million years", or "115 Ma")
       - sources: array of all URLs explicitly provided in the answer that directly support this shark study (journal pages, paper pages, DOI pages, or strong authoritative references)

    STRICT RULES:
    - Only extract values explicitly present in the answer; do not infer.
    - For any missing field, return null (for strings) or [] (for arrays).
    - For sources, include only actual URLs present in the answer (any format is acceptable, including markdown links). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_comet_section(
    evaluator: Evaluator,
    parent_node,
    comet: Optional[CometPaperInfo],
) -> None:
    """
    Build and verify the comet (3I/ATLAS) subtree.
    Parent node is critical; all children must be critical to satisfy the rubric.
    """
    # Existence / sources gate (critical)
    sources_exist = bool(comet and comet.sources and len(comet.sources) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id="comet_sources_provided",
        desc="3I/ATLAS paper: at least one supporting reference URL is provided in the answer",
        parent=parent_node,
        critical=True,
    )

    # Journal verification leaf
    journal_node = evaluator.add_leaf(
        id="paper_journal_and_reference",
        desc="The first academic paper on 3I/ATLAS was published in Monthly Notices of the Royal Astronomical Society Letters (MNRAS Letters), with a valid reference URL provided",
        parent=parent_node,
        critical=True,
    )
    journal_claim = (
        "The academic paper about the discovery of interstellar comet 3I/ATLAS was "
        "published in Monthly Notices of the Royal Astronomical Society Letters "
        "(also known as MNRAS Letters)."
    )
    await evaluator.verify(
        claim=journal_claim,
        node=journal_node,
        sources=(comet.sources if comet else []),
        additional_instruction=(
            "Verify that at least one of the cited URLs is an official paper/journal page or authoritative record "
            "showing that the publication venue is Monthly Notices of the Royal Astronomical Society Letters "
            "(MNRAS Letters). Accept reasonable naming variants like 'MNRAS Letters', "
            "'Monthly Notices of the Royal Astronomical Society: Letters', 'MNRAS (Letters)'. "
            "Ensure the page is about the paper reporting the discovery of interstellar comet 3I/ATLAS."
        ),
    )

    # Publication date verification leaf
    date_node = evaluator.add_leaf(
        id="paper_publication_date",
        desc="The first academic paper was published on July 18, 2025",
        parent=parent_node,
        critical=True,
    )
    date_claim = (
        "The publication date of the academic paper about the discovery of interstellar comet 3I/ATLAS is July 18, 2025."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=(comet.sources if comet else []),
        additional_instruction=(
            "Check the publication date on the cited paper/journal/DOI page. "
            "Accept common date formatting variants such as '18 July 2025'. "
            "The date must correspond to the publication on the journal/platform, not a preprint news article date."
        ),
    )

    # Lead author affiliation verification leaf
    affiliation_node = evaluator.add_leaf(
        id="lead_author_affiliation",
        desc="The lead author of the first paper is affiliated with Eureka Scientific Incorporated",
        parent=parent_node,
        critical=True,
    )
    affiliation_claim = (
        "The lead author of the academic paper about the discovery of interstellar comet 3I/ATLAS is affiliated with "
        "Eureka Scientific Incorporated (also acceptable as 'Eureka Scientific Inc.')."
    )
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_node,
        sources=(comet.sources if comet else []),
        additional_instruction=(
            "Look for author affiliation on the paper/journal page. "
            "Treat 'Eureka Scientific Inc.' and 'Eureka Scientific Incorporated' as equivalent. "
            "If multiple affiliations are listed, confirm that Eureka Scientific appears for the lead/first author."
        ),
    )


async def verify_shark_section(
    evaluator: Evaluator,
    parent_node,
    shark: Optional[SharkStudyInfo],
) -> None:
    """
    Build and verify the Darwin shark study subtree.
    Parent node is critical; all children must be critical to satisfy the rubric.
    """
    # Existence / sources gate (critical)
    sources_exist = bool(shark and shark.sources and len(shark.sources) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id="darwin_sources_provided",
        desc="Darwin shark study: at least one supporting reference URL is provided in the answer",
        parent=parent_node,
        critical=True,
    )

    # Journal verification leaf
    journal_node = evaluator.add_leaf(
        id="publication_journal_and_reference",
        desc="The Darwin shark fossil study was published in Communications Biology, with a valid reference URL provided",
        parent=parent_node,
        critical=True,
    )
    journal_claim = (
        "The academic study on the giant lamniform shark fossils near Darwin, Australia was published in Communications Biology."
    )
    await evaluator.verify(
        claim=journal_claim,
        node=journal_node,
        sources=(shark.sources if shark else []),
        additional_instruction=(
            "Verify that at least one cited URL clearly shows the journal as Communications Biology (Nature Portfolio). "
            "Ensure the page pertains to the Darwin-area giant shark fossil study."
        ),
    )

    # Lead institution verification leaf
    lead_inst_node = evaluator.add_leaf(
        id="lead_institution",
        desc="The lead institution for the study is the Swedish Museum of Natural History",
        parent=parent_node,
        critical=True,
    )
    lead_inst_claim = (
        "The lead institution for the study is the Swedish Museum of Natural History."
    )
    await evaluator.verify(
        claim=lead_inst_claim,
        node=lead_inst_node,
        sources=(shark.sources if shark else []),
        additional_instruction=(
            "Check the article, journal page, or institutional announcement for institutional leadership or "
            "corresponding lead. Accept the native-language name 'Naturhistoriska riksmuseet' as equivalent."
        ),
    )

    # Fossil age verification leaf
    fossil_age_node = evaluator.add_leaf(
        id="fossil_age",
        desc="The fossil is dated to 115 million years ago",
        parent=parent_node,
        critical=True,
    )
    fossil_age_claim = (
        "The fossil is approximately 115 million years old (around 115 Ma)."
    )
    await evaluator.verify(
        claim=fossil_age_claim,
        node=fossil_age_node,
        sources=(shark.sources if shark else []),
        additional_instruction=(
            "Confirm that the study or authoritative reference explicitly states an age of ~115 million years "
            "(accept '115 Ma', 'approximately 115 million years', or near-equivalents)."
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
    """
    Evaluate an answer for the 2025 comet + Darwin shark research task.
    """
    # Initialize evaluator with a CRITICAL root (as per rubric) and parallel aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete research task on two recent scientific discoveries and provide academic publication details with reference URLs",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )
    # Set root to critical manually since initialize defaults to non-critical
    root.critical = True

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="research_extraction",
    )

    # Add ground truth info for transparency (not used directly for scoring)
    evaluator.add_ground_truth(
        {
            "comet_expected": GT_COMET,
            "shark_expected": GT_SHARK,
            "notes": "Verification relies on cited URLs; acceptable synonym forms allowed per instructions.",
        },
        gt_type="expected_values",
    )

    # Build comet subtree (critical, parallel)
    comet_node = evaluator.add_parallel(
        id="comet_3i_atlas_research",
        desc="Research interstellar comet 3I/ATLAS and identify the first academic paper published about its discovery, with supporting reference URL",
        parent=root,
        critical=True,
    )
    await verify_comet_section(evaluator, comet_node, extraction.comet_3i_atlas_paper)

    # Build shark subtree (critical, parallel)
    shark_node = evaluator.add_parallel(
        id="darwin_shark_research",
        desc="Research the giant shark fossil discovered near Darwin, Australia and identify its publication details, with supporting reference URL",
        parent=root,
        critical=True,
    )
    await verify_shark_section(evaluator, shark_node, extraction.darwin_shark_study)

    # Return the evaluation summary
    return evaluator.get_summary()