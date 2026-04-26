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
TASK_ID = "journalist_identification"
TASK_DESCRIPTION = (
    "Identify the full name of the broadcast journalist who meets all of the following criteria: "
    "was born in 1977 and graduated from a high school in Richmond, California in 1995; "
    "graduated from American University School of Communication in 1999; began journalism career that same year working "
    "as a print reporter for the San Francisco Chronicle, where they covered Mayor Gavin Newsom's administration; "
    "transitioned to television news in 2007 at KGO-TV in San Francisco; won a Northern California Emmy Award in 2010 "
    "for Best Daytime Newscast in a Large Market; joined ABC News in 2011 as a Los Angeles-based correspondent; "
    "became anchor for the Saturday edition of World News Tonight on March 2, 2015; served as the lead correspondent "
    "for Hillary Clinton's 2016 presidential campaign, logging more than 239,000 miles in the air and spending more "
    "than 500 days on the campaign trail; was named Chief White House Correspondent for ABC News in January 2021; "
    "and joined CBS's 60 Minutes program on January 19, 2023, becoming the first Latina correspondent for that program, "
    "with their first story airing on May 14, 2023."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JournalistExtraction(BaseModel):
    full_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journalist_info() -> str:
    return (
        "Extract the identified journalist's full name and any URLs explicitly cited in the answer.\n"
        "Return JSON with fields:\n"
        "- full_name: The complete name of the journalist identified in the answer (include first and last name; include middle names/initials if present).\n"
        "- source_urls: A list of all URLs mentioned in the answer text. Include only valid URLs. If none are provided, return an empty list.\n"
        "If the answer contains multiple names, choose the one the answer asserts as the solution to the task.\n"
        "Do not invent any URLs. Only extract those explicitly present in the answer (plain URLs or markdown links)."
    )


# --------------------------------------------------------------------------- #
# Helper: Build additional instruction                                        #
# --------------------------------------------------------------------------- #
def build_additional_instruction(base: str, sources_present: bool) -> str:
    suffix = (
        "Use the cited URLs to verify this claim. If no sources are provided in the answer, you may use general world "
        "knowledge to judge the claim."
    )
    if sources_present:
        return base + " Use the cited URLs to verify this claim."
    else:
        return base + " If no sources are provided, you may use general world knowledge to judge the claim."


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    full_name: str,
    sources: Optional[List[str]],
) -> None:
    """
    Build the 'Meets_All_Criteria' parallel node and verify each leaf criterion.
    """
    sources_present = bool(sources) and len(sources) > 0

    meets_node = evaluator.add_parallel(
        id="Meets_All_Criteria",
        desc="The identified journalist satisfies all required biographical/career criteria.",
        parent=parent_node,
        critical=True
    )

    # Create leaf nodes
    leaf_nodes_and_payloads = []

    # 1. Born in 1977
    born_1977 = evaluator.add_leaf(
        id="Born_1977",
        desc="The journalist was born in 1977.",
        parent=meets_node,
        critical=True
    )
    claim_born = f"{full_name} was born in 1977."
    add_ins_born = build_additional_instruction(
        "Match if the person's birth year is 1977. Minor formatting variations of birthdate are acceptable.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_born, sources, born_1977, add_ins_born))

    # 2. High School in Richmond, CA in 1995
    hs_richmond = evaluator.add_leaf(
        id="High_School_Richmond_CA_1995",
        desc="The journalist graduated from a high school in Richmond, California in 1995.",
        parent=meets_node,
        critical=True
    )
    claim_hs = f"{full_name} graduated from a high school in Richmond, California in 1995."
    add_ins_hs = build_additional_instruction(
        "Accept De Anza High School (Richmond, CA) or any Richmond-based high school with graduation year 1995.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_hs, sources, hs_richmond, add_ins_hs))

    # 3. Graduated American University SOC in 1999
    au_soc = evaluator.add_leaf(
        id="Graduated_American_University_SOC_1999",
        desc="The journalist graduated from American University School of Communication in 1999.",
        parent=meets_node,
        critical=True
    )
    claim_au = f"{full_name} graduated from American University's School of Communication in 1999."
    add_ins_au = build_additional_instruction(
        "Allow minor naming variants such as 'American University School of Communication' vs 'AU SOC'.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_au, sources, au_soc, add_ins_au))

    # 4. Began career in 1999 as print reporter at SF Chronicle
    sf_start = evaluator.add_leaf(
        id="Began_Career_1999_As_Print_Reporter_SF_Chronicle",
        desc="The journalist began their journalism career in 1999 as a print reporter for the San Francisco Chronicle.",
        parent=meets_node,
        critical=True
    )
    claim_sf_start = f"{full_name} began a journalism career in 1999 as a print reporter at the San Francisco Chronicle."
    add_ins_sf_start = build_additional_instruction(
        "Accept phrasing such as 'started at the San Francisco Chronicle as a print reporter in 1999'.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_sf_start, sources, sf_start, add_ins_sf_start))

    # 5. Covered Gavin Newsom's administration at SF Chronicle
    newsom_cov = evaluator.add_leaf(
        id="Covered_Gavin_Newsom_Administration_At_SF_Chronicle",
        desc="While at the San Francisco Chronicle, the journalist covered Mayor Gavin Newsom's administration.",
        parent=meets_node,
        critical=True
    )
    claim_newsom = f"While at the San Francisco Chronicle, {full_name} covered Mayor Gavin Newsom's administration."
    add_ins_newsom = build_additional_instruction(
        "Coverage phrasing like 'covered City Hall/the Newsom administration' counts as a match.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_newsom, sources, newsom_cov, add_ins_newsom))

    # 6. Transitioned to TV in 2007 at KGO-TV (San Francisco)
    kgo_2007 = evaluator.add_leaf(
        id="Transitioned_To_TV_2007_At_KGO_TV_SF",
        desc="The journalist transitioned to television news in 2007 at KGO-TV in San Francisco.",
        parent=meets_node,
        critical=True
    )
    claim_kgo = f"{full_name} transitioned to television news in 2007 at KGO-TV in San Francisco."
    add_ins_kgo = build_additional_instruction(
        "Accept 'joined KGO-TV in 2007' or equivalent wording indicating the TV transition.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_kgo, sources, kgo_2007, add_ins_kgo))

    # 7. Won Northern California Emmy in 2010 for Best Daytime Newscast (Large Market)
    emmy_2010 = evaluator.add_leaf(
        id="Won_NorCal_Emmy_2010_For_Best_Daytime_Newscast_Large_Market",
        desc="The journalist won a Northern California Emmy Award in 2010 specifically for Best Daytime Newscast in a Large Market.",
        parent=meets_node,
        critical=True
    )
    claim_emmy = f"In 2010, {full_name} won a Northern California Emmy Award for Best Daytime Newscast in a Large Market."
    add_ins_emmy = build_additional_instruction(
        "Exact category wording may appear as 'Daytime Newscast (Large Market)'. Minor wording variants are acceptable.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_emmy, sources, emmy_2010, add_ins_emmy))

    # 8. Joined ABC News in 2011 as LA-based correspondent
    abc_2011 = evaluator.add_leaf(
        id="Joined_ABC_News_2011_As_LA_Based_Correspondent",
        desc="The journalist joined ABC News in 2011 as a Los Angeles-based correspondent.",
        parent=meets_node,
        critical=True
    )
    claim_abc = f"{full_name} joined ABC News in 2011 as a Los Angeles-based correspondent."
    add_ins_abc = build_additional_instruction(
        "Confirm ABC News entry year 2011 and role as LA-based correspondent.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_abc, sources, abc_2011, add_ins_abc))

    # 9. Became anchor for World News Tonight Saturday on March 2, 2015
    wnt_anchor = evaluator.add_leaf(
        id="Anchor_WNT_Saturday_March_2_2015",
        desc="The journalist became anchor for the Saturday edition of World News Tonight on March 2, 2015.",
        parent=meets_node,
        critical=True
    )
    claim_wnt = f"{full_name} became anchor for the Saturday edition of ABC World News Tonight on March 2, 2015."
    add_ins_wnt = build_additional_instruction(
        "Match if appointed/started as Saturday anchor on March 2, 2015. Allow minor phrasing variants.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_wnt, sources, wnt_anchor, add_ins_wnt))

    # 10. Lead correspondent for Clinton 2016; >239,000 miles and >500 days
    clinton_2016 = evaluator.add_leaf(
        id="Lead_Correspondent_Clinton_2016_With_Miles_And_Days",
        desc="The journalist served as the lead correspondent for Hillary Clinton's 2016 presidential campaign and, in that role, logged more than 239,000 miles and spent more than 500 days on the campaign trail.",
        parent=meets_node,
        critical=True
    )
    claim_clinton = (
        f"In 2016, {full_name} served as the lead correspondent for Hillary Clinton's presidential campaign, "
        "logging more than 239,000 miles in the air and spending more than 500 days on the campaign trail."
    )
    add_ins_clinton = build_additional_instruction(
        "Numbers may appear as 'more than 239,000 miles' and 'more than 500 days'; accept equivalent phrasing conveying those thresholds.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_clinton, sources, clinton_2016, add_ins_clinton))

    # 11. Named Chief White House Correspondent in January 2021
    whc_2021 = evaluator.add_leaf(
        id="Named_Chief_White_House_Correspondent_Jan_2021",
        desc="The journalist was named Chief White House Correspondent for ABC News in January 2021.",
        parent=meets_node,
        critical=True
    )
    claim_whc = f"In January 2021, {full_name} was named Chief White House Correspondent for ABC News."
    add_ins_whc = build_additional_instruction(
        "Confirm title change occurred in January 2021. Accept minor phrasing variants.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_whc, sources, whc_2021, add_ins_whc))

    # 12. Joined CBS 60 Minutes on Jan 19, 2023; first Latina correspondent
    sixty_join = evaluator.add_leaf(
        id="Joined_60_Minutes_Jan_19_2023_First_Latina_Correspondent",
        desc="The journalist joined CBS's 60 Minutes on January 19, 2023, becoming the first Latina correspondent for that program.",
        parent=meets_node,
        critical=True
    )
    claim_sixty = (
        f"On January 19, 2023, {full_name} joined CBS's 60 Minutes and became the program's first Latina correspondent."
    )
    add_ins_sixty = build_additional_instruction(
        "Confirm both the date (Jan 19, 2023) and the 'first Latina correspondent' distinction.",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_sixty, sources, sixty_join, add_ins_sixty))

    # 13. First 60 Minutes story aired May 14, 2023 about sperm whales
    sixty_story = evaluator.add_leaf(
        id="First_60_Minutes_Story_May_14_2023_About_Sperm_Whales",
        desc="The journalist's first 60 Minutes story aired on May 14, 2023, and that first story was about sperm whales.",
        parent=meets_node,
        critical=True
    )
    claim_sixty_story = (
        f"{full_name}'s first 60 Minutes story aired on May 14, 2023 and was about sperm whales."
    )
    add_ins_sixty_story = build_additional_instruction(
        "Require both the air date (May 14, 2023) and the topic (sperm whales).",
        sources_present
    )
    leaf_nodes_and_payloads.append((claim_sixty_story, sources, sixty_story, add_ins_sixty_story))

    # Execute batch verification in parallel for efficiency
    await evaluator.batch_verify(leaf_nodes_and_payloads)


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
) -> Dict[str, Any]:
    """
    Evaluate the provided answer for the journalist identification task.

    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container
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

    # Add the main sequential critical node for the task
    main_seq = evaluator.add_sequential(
        id="Journalist_Identification",
        desc="Identify the full name of the broadcast journalist who satisfies all criteria stated in the proposed question and constraints.",
        parent=root,
        critical=True
    )

    # Extract name and sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_journalist_info(),
        template_class=JournalistExtraction,
        extraction_name="journalist_info"
    )

    # Provide_Full_Name check (critical)
    full_name_val = extraction.full_name.strip() if extraction.full_name else ""
    is_full = bool(full_name_val) and len([t for t in full_name_val.split() if t.strip()]) >= 2

    evaluator.add_custom_node(
        result=is_full,
        id="Provide_Full_Name",
        desc="Answer provides the journalist's full name (not only a first name, last name, or role).",
        parent=main_seq,
        critical=True
    )

    # Build and verify all criteria (parallel, critical), conditioned by sequential gating
    await build_and_verify_criteria(
        evaluator=evaluator,
        parent_node=main_seq,
        full_name=full_name_val if full_name_val else "the identified journalist",
        sources=extraction.source_urls if extraction.source_urls else None,
    )

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_full_name": full_name_val or None,
            "num_sources": len(extraction.source_urls),
            "sources": extraction.source_urls,
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    return evaluator.get_summary()