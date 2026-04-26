import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "quantum_breakthrough_2025"
TASK_DESCRIPTION = "In 2025, multiple quantum computing breakthroughs were announced by various research organizations. Identify the specific quantum computing breakthrough that meets ALL of the following criteria: (1) The announcement was made in 2025 and published in the peer-reviewed scientific journal Nature, (2) The system demonstrated a computational speedup exceeding 10,000 times compared to classical supercomputers, (3) The breakthrough involved running a verifiable algorithm on quantum hardware. For the breakthrough you identify, provide the following verified information: the exact announcement date, the specific speedup factor claimed, the name of the algorithm or technology, the number of qubits used in the demonstration, the name of the quantum chip or hardware platform, the organization(s) responsible for the breakthrough, a reference URL from an official source, and a brief description of the practical application or experiment performed. Your answer must be fully supported by verifiable sources and include reference URLs."


class BreakthroughExtraction(BaseModel):
    announcement_date: Optional[str] = None
    speedup_factor: Optional[str] = None
    algorithm_name: Optional[str] = None
    qubit_count: Optional[str] = None
    quantum_chip_name: Optional[str] = None
    organizations: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)

    journal_name: Optional[str] = None
    application_description: Optional[str] = None
    comparison_baseline: Optional[str] = None


def prompt_extract_breakthrough() -> str:
    return (
        "Your task is to extract a single quantum computing breakthrough from the answer that satisfies ALL of the "
        "following criteria: (1) the announcement was made in 2025, (2) it was published in the peer‑reviewed "
        "scientific journal Nature (the flagship journal), (3) it demonstrated a computational speedup exceeding "
        "10,000× compared to classical supercomputers, and (4) the breakthrough involved running a verifiable "
        "algorithm on quantum hardware.\n"
        "Extract the following fields exactly as stated in the answer:\n"
        "1. announcement_date: The exact announcement date string as provided (e.g., 'January 15, 2025').\n"
        "2. speedup_factor: The specific speedup factor claim (e.g., '11,000×', 'over 10,000 times').\n"
        "3. algorithm_name: The name of the algorithm or technology used in the demonstration.\n"
        "4. qubit_count: The number of qubits used (string form; keep formatting as in the answer).\n"
        "5. quantum_chip_name: The name/code of the quantum chip or hardware platform.\n"
        "6. organizations: A list of organization(s) responsible (each as a separate string).\n"
        "7. reference_urls: A list of URLs that the answer cites as sources. Include official sources if present, "
        "   such as Nature journal pages (nature.com) and official organization press releases/webpages.\n"
        "8. journal_name: The journal name (should be 'Nature' if claimed).\n"
        "9. application_description: A short description of the practical application or experiment performed.\n"
        "10. comparison_baseline: The performance comparison baseline (e.g., 'classical supercomputers').\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly present in the answer. Do not invent.\n"
        "- If any field is missing in the answer, set it to null (or empty list for list fields).\n"
        "- For URLs, include full URLs; accept markdown links by extracting the actual URL.\n"
        "- If the answer mentions multiple breakthroughs, select the one that most clearly meets ALL criteria above "
        "  and extract its details."
    )


def _has_nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip() != "" for u in urls)


def _joined_orgs(orgs: List[str]) -> str:
    return ", ".join([o.strip() for o in orgs if _has_nonempty_str(o)]) if orgs else ""


async def verify_breakthrough(evaluator: Evaluator, root_node, info: BreakthroughExtraction) -> None:
    urls = info.reference_urls or []

    main_critical = evaluator.add_parallel(
        id="qci_main",
        desc="Correctly identify and provide verified information about the 2025 Nature‑published quantum computing breakthrough with >10,000× speedup and a verifiable algorithm on hardware",
        parent=root_node,
        critical=True,
    )

    announcement_node = evaluator.add_parallel(
        id="announcement_main",
        desc="Announcement date verification (must be in 2025)",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.announcement_date) and _has_valid_urls(urls),
        id="announcement_date_exists",
        desc="Announcement date is provided and sources are present",
        parent=announcement_node,
        critical=True,
    )

    ann_date_leaf = evaluator.add_leaf(
        id="announcement_date_correct",
        desc="Provide the correct announcement date of the quantum computing breakthrough",
        parent=announcement_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official announcement of the breakthrough was made on {info.announcement_date}.",
        node=ann_date_leaf,
        sources=urls,
        additional_instruction="Confirm the stated announcement date using the provided official sources. Minor timezone differences are acceptable; the core calendar date must match the sources.",
    )

    ann_year_leaf = evaluator.add_leaf(
        id="announcement_year_2025",
        desc="Announcement was made in the year 2025",
        parent=announcement_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The announcement was made in 2025.",
        node=ann_year_leaf,
        sources=urls,
        additional_instruction="Use the official sources to verify the announcement/publication timing is within calendar year 2025.",
    )

    performance_node = evaluator.add_parallel(
        id="performance_main",
        desc="Performance metric verification (>10,000× speedup vs classical supercomputers)",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.speedup_factor) and _has_valid_urls(urls),
        id="speedup_factor_exists",
        desc="Speedup factor is provided and sources are present",
        parent=performance_node,
        critical=True,
    )

    speedup_threshold_leaf = evaluator.add_leaf(
        id="speedup_exceeds_10000",
        desc="Confirm the speedup factor exceeds 10,000× compared to classical supercomputers",
        parent=performance_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The demonstrated computational speedup exceeds 10,000 times compared to classical supercomputers.",
        node=speedup_threshold_leaf,
        sources=urls,
        additional_instruction="Look for phrasing like '10,000×', 'over ten thousand times', 'more than 10,000×' in the sources, and ensure it is relative to classical supercomputers.",
    )

    speedup_specific_leaf = evaluator.add_leaf(
        id="speedup_specific_factor",
        desc="Verify the specific speedup factor claimed",
        parent=performance_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The specific speedup factor claimed was {info.speedup_factor}.",
        node=speedup_specific_leaf,
        sources=urls,
        additional_instruction="Verify the exact numeric claim or phrasing of the speedup factor. Minor rounding is acceptable.",
    )

    publication_node = evaluator.add_parallel(
        id="publication_main",
        desc="Publication and reference URL verification",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_valid_urls(urls),
        id="reference_url_exists",
        desc="Provide valid reference URL(s) from official sources",
        parent=publication_node,
        critical=True,
    )

    journal_leaf = evaluator.add_leaf(
        id="journal_is_nature",
        desc="Confirm publication in Nature journal",
        parent=publication_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This breakthrough was published in the journal Nature.",
        node=journal_leaf,
        sources=urls,
        additional_instruction="Confirm that the article is in the flagship 'Nature' journal at nature.com (not just a Nature-branded family journal unless the page explicitly shows 'Nature').",
    )

    official_source_leaf = evaluator.add_leaf(
        id="reference_url_official_source",
        desc="At least one reference URL is from an official source (Nature journal page or official organization site)",
        parent=publication_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided reference URLs is an official source: a Nature journal page or an official announcement page from the responsible organization(s).",
        node=official_source_leaf,
        sources=urls,
        additional_instruction="Check the URL domains (nature.com for Nature; official organization domains for press releases) and the page content to confirm official provenance.",
    )

    technical_node = evaluator.add_parallel(
        id="technical_specs_main",
        desc="Technical specifications verification",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.qubit_count) and _has_valid_urls(urls),
        id="qubit_count_exists",
        desc="Qubit count is provided and sources are present",
        parent=technical_node,
        critical=True,
    )
    qubits_leaf = evaluator.add_leaf(
        id="qubit_count_correct",
        desc="Specify the number of qubits used in the demonstration",
        parent=technical_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The demonstration used {info.qubit_count} qubits.",
        node=qubits_leaf,
        sources=urls,
        additional_instruction="Confirm the stated number of qubits in the sources. Accept phrases like 'about N qubits' if consistent with the claimed value.",
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.algorithm_name) and _has_valid_urls(urls),
        id="algorithm_name_exists",
        desc="Algorithm or technology name is provided and sources are present",
        parent=technical_node,
        critical=True,
    )
    algorithm_leaf = evaluator.add_leaf(
        id="algorithm_verifiable_on_hw",
        desc="Identify the algorithm/technology and confirm it was a verifiable algorithm run on quantum hardware",
        parent=technical_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The demonstration ran a verifiable algorithm named '{info.algorithm_name}' on quantum hardware.",
        node=algorithm_leaf,
        sources=urls,
        additional_instruction="Confirm that the named algorithm was actually executed on quantum hardware and involved verifiability/verification steps as stated in the sources.",
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.quantum_chip_name) and _has_valid_urls(urls),
        id="quantum_chip_name_exists",
        desc="Quantum chip/hardware platform name is provided and sources are present",
        parent=technical_node,
        critical=True,
    )
    chip_leaf = evaluator.add_leaf(
        id="quantum_chip_name_correct",
        desc="Identify the quantum chip or hardware platform used",
        parent=technical_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The quantum chip or hardware platform used was '{info.quantum_chip_name}'.",
        node=chip_leaf,
        sources=urls,
        additional_instruction="Verify the chip/platform name from the sources; code names or model numbers are acceptable if they match the sources.",
    )

    org_node = evaluator.add_parallel(
        id="organization_main",
        desc="Organization(s) responsible for the breakthrough",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(info.organizations) > 0) and _has_valid_urls(urls),
        id="organizations_exist",
        desc="Organization(s) are identified and sources are present",
        parent=org_node,
        critical=True,
    )
    orgs_leaf = evaluator.add_leaf(
        id="organizations_correct",
        desc="Identify the organization(s) responsible for the breakthrough",
        parent=org_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The organizations responsible for the breakthrough were { _joined_orgs(info.organizations) }.",
        node=orgs_leaf,
        sources=urls,
        additional_instruction="Confirm the credited organizations from the Nature article and/or official announcements.",
    )

    app_node = evaluator.add_parallel(
        id="application_main",
        desc="Practical application or experiment performed",
        parent=main_critical,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.application_description) and _has_valid_urls(urls),
        id="application_desc_exists",
        desc="Application/experiment description is provided and sources are present",
        parent=app_node,
        critical=True,
    )
    app_leaf = evaluator.add_leaf(
        id="application_desc_correct",
        desc="Describe the practical application or experiment performed",
        parent=app_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The practical application or experiment performed was: {info.application_description}.",
        node=app_leaf,
        sources=urls,
        additional_instruction="Confirm the described application/experiment details from the sources; concise summary is acceptable if it matches the sources.",
    )

    soft_node = evaluator.add_parallel(
        id="additional_soft_details",
        desc="Non‑critical additional verification: comparison baseline",
        parent=root_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(info.comparison_baseline) and _has_valid_urls(urls),
        id="comparison_baseline_exists",
        desc="Comparison baseline is provided and sources are present",
        parent=soft_node,
        critical=False,
    )
    baseline_leaf = evaluator.add_leaf(
        id="comparison_baseline_classical",
        desc="Identify what the quantum system was compared against (classical supercomputers)",
        parent=soft_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The performance comparison baseline was classical supercomputers.",
        node=baseline_leaf,
        sources=urls,
        additional_instruction="Look for mentions of comparisons against classical computing or supercomputers in the sources.",
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_breakthrough(),
        template_class=BreakthroughExtraction,
        extraction_name="breakthrough_extraction",
    )

    evaluator.add_ground_truth(
        {
            "required_criteria": [
                "Announcement in 2025",
                "Published in Nature (flagship journal)",
                "Speedup > 10,000× compared to classical supercomputers",
                "Verifiable algorithm executed on quantum hardware",
            ],
            "requested_fields": [
                "announcement_date",
                "speedup_factor",
                "algorithm_name",
                "qubit_count",
                "quantum_chip_name",
                "organizations",
                "reference_urls",
                "journal_name",
                "application_description",
                "comparison_baseline",
            ],
        },
        gt_type="task_requirements",
    )

    await verify_breakthrough(evaluator, root, extracted)

    return evaluator.get_summary()