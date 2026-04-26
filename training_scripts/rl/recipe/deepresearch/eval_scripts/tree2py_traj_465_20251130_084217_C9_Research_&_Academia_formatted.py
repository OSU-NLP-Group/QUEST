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
TASK_ID = "paleo_shark_oct2025"
TASK_DESCRIPTION = """
A recent paleontological research paper published in October 2025 reports the discovery of giant lamniform shark fossils from the Darwin Formation in Northern Territory, Australia, dated to approximately 115 million years ago (upper Aptian period). The research represents an international collaboration involving authors from multiple institutions across different countries.

Your task is to identify this research paper and provide comprehensive details including:

1. Paper Information: Complete title, journal name (must be Communications Biology), publication date, DOI, and confirm open access status
2. Lead Author: Name and affiliation with Stanford University's Department of Earth and Planetary Sciences
3. Co-author from Georgia, USA: Name, affiliation with Columbus State University in Columbus, Georgia, department (must be Biology), and expertise in shark vertebrae research
4. Student/Alumni Co-author (if applicable): Name and status related to Columbus State University
5. Co-author from Western Australia: Name, affiliation with Western Australian Museum, and confirm the museum's Collections and Research Centre location in Welshpool
6. Co-author from Sweden: Name, affiliation with Swedish Museum of Natural History's Department of Palaeobiology in Stockholm
7. International Collaboration: Confirm authors represent at least 4 different countries
8. Specimen Details: Repository institution (Museum and Art Gallery of the Northern Territory), specimen type (vertebrae), number of specimens (5), and at least two specific specimen catalog numbers with "NTM P" prefix
9. Taxonomic Classification: Shark order (Lamniformes) and family (Cardabiodontidae)
10. Size Estimates: Estimated body length range (6-8 meters) and mass (over 3 tons)
11. Geological Age: Confirmation of upper Aptian period, approximately 115 million years ago
12. Evolutionary Significance: Explanation of how this discovery pushes back the timeline of lamniform shark gigantism by approximately 15 million years

Provide all information with supporting URL references for verification.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None  # e.g., "October 2025" or "2025-10-12"
    doi: Optional[str] = None
    open_access: Optional[str] = None  # e.g., "Open Access" or "yes"
    paper_urls: List[str] = Field(default_factory=list)  # nature.com page, etc.


class ScienceClaims(BaseModel):
    locality: Optional[str] = None  # e.g., "Darwin Formation, Northern Territory, Australia"
    age: Optional[str] = None  # e.g., "upper Aptian, ~115 Ma"
    order: Optional[str] = None  # e.g., "Lamniformes"
    family: Optional[str] = None  # e.g., "Cardabiodontidae"
    length_range: Optional[str] = None  # e.g., "6–8 m"
    mass: Optional[str] = None  # e.g., "over 3 tons"
    significance: Optional[str] = None  # e.g., "~15 Myr pushback of lamniform gigantism"
    urls: List[str] = Field(default_factory=list)


class LeadAuthor(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None  # Expect "Stanford University, Department of Earth and Planetary Sciences"
    urls: List[str] = Field(default_factory=list)  # paper page or Stanford profile


class CSUAuthor(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None  # "Columbus State University, Columbus, Georgia, USA"
    department: Optional[str] = None  # "Biology"
    expertise: Optional[str] = None  # "shark vertebrae research"
    urls: List[str] = Field(default_factory=list)


class StudentAlumni(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None  # "student", "alumni", "graduate student", etc.
    urls: List[str] = Field(default_factory=list)


class WAMAuthor(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None  # "Western Australian Museum"
    crc_location: Optional[str] = None  # "Welshpool"
    urls: List[str] = Field(default_factory=list)


class SwedishAuthor(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None  # "Swedish Museum of Natural History"
    department: Optional[str] = None  # "Department of Palaeobiology"
    location: Optional[str] = None  # "Stockholm"
    urls: List[str] = Field(default_factory=list)


class Collaboration(BaseModel):
    countries: List[str] = Field(default_factory=list)  # e.g., ["Australia","USA","Sweden","UK"]


class Specimens(BaseModel):
    repository: Optional[str] = None  # "Museum and Art Gallery of the Northern Territory"
    specimen_type: Optional[str] = None  # "vertebrae"
    number_of_specimens: Optional[str] = None  # "5" (keep as string)
    catalog_numbers: List[str] = Field(default_factory=list)  # e.g., ["NTM P.12345","NTM P.67890"]
    urls: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    paper: PaperInfo = PaperInfo()
    science: ScienceClaims = ScienceClaims()
    authors_lead: LeadAuthor = LeadAuthor()
    authors_csu: CSUAuthor = CSUAuthor()
    authors_student: StudentAlumni = StudentAlumni()
    authors_wam: WAMAuthor = WAMAuthor()
    authors_swedish: SwedishAuthor = SwedishAuthor()
    collaboration: Collaboration = Collaboration()
    specimens: Specimens = Specimens()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
    Extract structured information exactly as presented in the answer. Do not invent anything. If a field is missing, return null for that field or an empty list for arrays.

    Required JSON structure:
    {
      "paper": {
        "title": string or null,
        "journal": string or null,
        "publication_date": string or null,
        "doi": string or null,
        "open_access": string or null,
        "paper_urls": [array of URLs explicitly shown in the answer]
      },
      "science": {
        "locality": string or null,
        "age": string or null,
        "order": string or null,
        "family": string or null,
        "length_range": string or null,
        "mass": string or null,
        "significance": string or null,
        "urls": [array of URLs explicitly shown in the answer and relevant to these scientific claims]
      },
      "authors_lead": {
        "name": string or null,
        "affiliation": string or null,
        "urls": [array of URLs explicitly shown in the answer for the lead author's affiliation or paper author list]
      },
      "authors_csu": {
        "name": string or null,
        "affiliation": string or null,
        "department": string or null,
        "expertise": string or null,
        "urls": [array of URLs explicitly shown in the answer relevant to CSU author details]
      },
      "authors_student": {
        "name": string or null,
        "status": string or null,
        "urls": [array of URLs explicitly shown in the answer for student/alumni evidence]
      },
      "authors_wam": {
        "name": string or null,
        "affiliation": string or null,
        "crc_location": string or null,
        "urls": [array of URLs explicitly shown in the answer relevant to WAM author and CRC location]
      },
      "authors_swedish": {
        "name": string or null,
        "affiliation": string or null,
        "department": string or null,
        "location": string or null,
        "urls": [array of URLs explicitly shown in the answer relevant to Swedish author and department/location]
      },
      "collaboration": {
        "countries": [array of country names explicitly mentioned or implied via affiliations in the answer]
      },
      "specimens": {
        "repository": string or null,
        "specimen_type": string or null,
        "number_of_specimens": string or null,
        "catalog_numbers": [array of catalog numbers explicitly shown in the answer, prefer those beginning with "NTM P", include at least two if present],
        "urls": [array of URLs explicitly shown in the answer relevant to repository/specimen evidence]
      }
    }

    URL extraction rules:
    - Extract only URLs explicitly present in the answer text (plain URLs or URLs inside markdown links).
    - Do not infer or create URLs.
    - If a URL is missing the protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Return primary if non-empty, else fallback."""
    if primary and len(primary) > 0:
        return primary
    return fallback or []


def combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists into a unique list, preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def first_two_ntm_p(catalog_numbers: List[str]) -> List[str]:
    """Pick the first two catalog numbers starting with 'NTM P' (case-insensitive, allow punctuation)."""
    if not catalog_numbers:
        return []
    picked: List[str] = []
    for num in catalog_numbers:
        if not num:
            continue
        normalized = num.strip().upper().replace("NTM P.", "NTM P").replace("NTM  P", "NTM P")
        if normalized.startswith("NTM P"):
            picked.append(num)
        if len(picked) >= 2:
            break
    return picked


# --------------------------------------------------------------------------- #
# Verification phases                                                         #
# --------------------------------------------------------------------------- #
async def build_phase1_paper(evaluator: Evaluator, parent_node, ex: FullExtraction) -> None:
    phase1 = evaluator.add_parallel(
        id="Phase1_Paper_Identification_and_Access",
        desc="Identify the exact paper and verify required bibliographic/access constraints.",
        parent=parent_node,
        critical=True
    )

    paper = ex.paper
    paper_sources = paper.paper_urls

    # Paper Title
    node_title = evaluator.add_leaf(
        id="Paper_Title",
        desc="Provide the complete title of the research paper.",
        parent=phase1,
        critical=True
    )
    claim_title = f"The paper has the title '{paper.title}'."
    await evaluator.verify(
        claim=claim_title,
        node=node_title,
        sources=paper_sources,
        additional_instruction="Verify that the article page shows exactly this title or a clearly equivalent title."
    )

    # Journal Is Communications Biology
    node_journal = evaluator.add_leaf(
        id="Journal_Is_Communications_Biology",
        desc="Confirm the journal is Communications Biology.",
        parent=phase1,
        critical=True
    )
    await evaluator.verify(
        claim="The journal of this article is Communications Biology.",
        node=node_journal,
        sources=paper_sources,
        additional_instruction="Check the article page's journal name; it should say Communications Biology (a Nature Research journal)."
    )

    # Publication Date In October 2025
    node_pubdate = evaluator.add_leaf(
        id="Publication_Date_In_October_2025",
        desc="Confirm the publication date is in October 2025 (and therefore within Q4 2025).",
        parent=phase1,
        critical=True
    )
    await evaluator.verify(
        claim="This article was published in October 2025.",
        node=node_pubdate,
        sources=paper_sources,
        additional_instruction="Check the publication date on the article page. Accept any day in October 2025."
    )

    # DOI Nature Domain
    node_doi = evaluator.add_leaf(
        id="DOI_Nature_Domain",
        desc="Provide the paper DOI and confirm it is a valid nature.com DOI (e.g., doi.org resolving to a nature.com article).",
        parent=phase1,
        critical=True
    )
    claim_doi = f"The DOI of the article is '{paper.doi}', and the Nature/Communications Biology page lists this DOI."
    await evaluator.verify(
        claim=claim_doi,
        node=node_doi,
        sources=paper_sources,
        additional_instruction="Confirm the DOI string appears on the Nature (nature.com) article page and corresponds to Communications Biology."
    )

    # Open Access
    node_oa = evaluator.add_leaf(
        id="Open_Access",
        desc="Confirm the paper is open access.",
        parent=phase1,
        critical=True
    )
    await evaluator.verify(
        claim="This Communications Biology article is Open Access.",
        node=node_oa,
        sources=paper_sources,
        additional_instruction="Check the article page for Open Access labeling; accept clear indicators like 'Open Access'."
    )

    # Supporting URLs For Paper (existence check)
    evaluator.add_custom_node(
        result=bool(paper_sources),
        id="Supporting_URLs_For_Paper",
        desc="Provide at least one supporting URL (e.g., the nature.com article page) for bibliographic/access verification.",
        parent=phase1,
        critical=True
    )


async def build_phase2_science(evaluator: Evaluator, parent_node, ex: FullExtraction) -> None:
    phase2 = evaluator.add_parallel(
        id="Phase2_Site_Age_Taxonomy_Size_Significance",
        desc="Verify the required scientific constraints about locality, age, taxonomy, size estimates, and significance.",
        parent=parent_node,
        critical=True
    )

    paper_sources = ex.paper.paper_urls
    sci_sources = choose_sources(ex.science.urls, paper_sources)

    # Locality: Darwin Formation, NT, Australia
    node_locality = evaluator.add_leaf(
        id="Locality_Darwin_Formation_NT_Australia",
        desc="Confirm fossils are from the Darwin Formation in Northern Territory, Australia.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The fossils described are from the Darwin Formation in the Northern Territory, Australia.",
        node=node_locality,
        sources=sci_sources,
        additional_instruction="Verify locality wording on the article page or supplemental materials; accept close variants that clearly indicate Darwin Formation in NT, Australia."
    )

    # Age: Upper Aptian ~115 Ma
    node_age = evaluator.add_leaf(
        id="Age_Upper_Aptian_Approx_115_Ma",
        desc="Confirm the fossils are dated to the upper Aptian period, approximately 115 million years ago.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The fossils are dated to the upper Aptian, approximately 115 million years ago.",
        node=node_age,
        sources=sci_sources,
        additional_instruction="Confirm both the period (upper Aptian) and the approximate age (~115 Ma) from the article."
    )

    # Taxonomy: Order Lamniformes
    node_order = evaluator.add_leaf(
        id="Taxonomy_Order_Lamniformes",
        desc="Confirm the shark order is Lamniformes.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The sharks described belong to the order Lamniformes.",
        node=node_order,
        sources=sci_sources,
        additional_instruction="Verify the taxonomic order on the article page."
    )

    # Taxonomy: Family Cardabiodontidae
    node_family = evaluator.add_leaf(
        id="Taxonomy_Family_Cardabiodontidae",
        desc="Confirm the shark family is Cardabiodontidae.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The sharks are assigned to the family Cardabiodontidae.",
        node=node_family,
        sources=sci_sources,
        additional_instruction="Verify the family assignment on the article page."
    )

    # Size: Length 6–8 m
    node_length = evaluator.add_leaf(
        id="Size_Estimate_Length_6_to_8_m",
        desc="Report/confirm the estimated body length range is 6–8 meters.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The estimated body length range is 6–8 meters.",
        node=node_length,
        sources=sci_sources,
        additional_instruction="Accept variants like '6 to 8 m' or '6–8 m'."
    )

    # Size: Mass > 3 tons
    node_mass = evaluator.add_leaf(
        id="Size_Estimate_Mass_Over_3_Tons",
        desc="Report/confirm the estimated mass is over 3 tons.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="The estimated body mass is over 3 tons.",
        node=node_mass,
        sources=sci_sources,
        additional_instruction="Accept equivalents like '> 3 t' or '> 3000 kg'."
    )

    # Significance: Gigantism pushback ~15 Myr
    node_signif = evaluator.add_leaf(
        id="Gigantism_Timeline_Pushback_Approx_15_Myr",
        desc="Explain/confirm the discovery pushes back lamniform gigantism by approximately 15 million years.",
        parent=phase2,
        critical=True
    )
    await evaluator.verify(
        claim="This discovery pushes back the timeline of lamniform shark gigantism by approximately 15 million years.",
        node=node_signif,
        sources=sci_sources,
        additional_instruction="Verify language in the paper (main text or discussion) stating the ~15 Myr pushback."
    )

    # Supporting URLs for Science Claims (existence check)
    evaluator.add_custom_node(
        result=bool(sci_sources),
        id="Supporting_URLs_For_Science_Claims",
        desc="Provide supporting URL reference(s) for locality/age/taxonomy/size/significance claims (paper URL acceptable if it contains these details).",
        parent=phase2,
        critical=True
    )


async def build_phase3_authors(evaluator: Evaluator, parent_node, ex: FullExtraction) -> None:
    phase3 = evaluator.add_parallel(
        id="Phase3_Authors_and_International_Collaboration",
        desc="Provide required author identities/affiliations and confirm international collaboration threshold.",
        parent=parent_node,
        critical=True
    )

    paper_sources = ex.paper.paper_urls

    # Lead author group (critical)
    lead_group = evaluator.add_parallel(
        id="Lead_Author_Stanford_Earth_and_Planetary_Sciences",
        desc="Identify the lead author and verify Stanford University Department of Earth and Planetary Sciences affiliation.",
        parent=phase3,
        critical=True
    )
    lead = ex.authors_lead
    lead_sources = combine_sources(lead.urls, paper_sources)

    node_lead_name = evaluator.add_leaf(
        id="Lead_Author_Name",
        desc="Provide the lead author name.",
        parent=lead_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The lead author (first-listed author) is '{lead.name}'.",
        node=node_lead_name,
        sources=paper_sources,
        additional_instruction="Interpret 'lead author' as the first-listed author on the article page."
    )

    node_lead_affil = evaluator.add_leaf(
        id="Lead_Author_Affiliation_Stanford_EPS",
        desc="Confirm the lead author is affiliated with Stanford University, Department of Earth and Planetary Sciences.",
        parent=lead_group,
        critical=True
    )
    await evaluator.verify(
        claim="The lead author is affiliated with Stanford University, Department of Earth and Planetary Sciences.",
        node=node_lead_affil,
        sources=lead_sources,
        additional_instruction="Accept evidence from the article affiliations or a Stanford departmental profile clearly stating Earth & Planetary Sciences."
    )

    evaluator.add_custom_node(
        result=bool(lead_sources),
        id="URL_Lead_Author_Affiliation",
        desc="Provide a supporting URL for the lead author affiliation (paper page or institutional profile).",
        parent=lead_group,
        critical=True
    )

    # CSU co-author group (critical)
    csu_group = evaluator.add_parallel(
        id="Coauthor_Columbus_State_Biology_Shark_Vertebrae",
        desc="Identify the Columbus State University co-author and verify required department and expertise constraints.",
        parent=phase3,
        critical=True
    )
    csu = ex.authors_csu
    csu_sources = combine_sources(csu.urls, paper_sources)

    node_csu_name = evaluator.add_leaf(
        id="CSU_Coauthor_Name",
        desc="Provide the CSU co-author name.",
        parent=csu_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The co-author from Columbus State University is '{csu.name}'.",
        node=node_csu_name,
        sources=csu_sources,
        additional_instruction="Verify that the named person is an author and associated with CSU."
    )

    node_csu_affil = evaluator.add_leaf(
        id="CSU_Affiliation_Columbus_Georgia_USA",
        desc="Confirm affiliation with Columbus State University in Columbus, Georgia, USA.",
        parent=csu_group,
        critical=True
    )
    await evaluator.verify(
        claim="The co-author is affiliated with Columbus State University in Columbus, Georgia, USA.",
        node=node_csu_affil,
        sources=csu_sources,
        additional_instruction="Accept article affiliation lines or CSU profile pages explicitly indicating Columbus, Georgia, USA."
    )

    node_csu_dept = evaluator.add_leaf(
        id="CSU_Department_Biology",
        desc="Confirm the CSU co-author is in the Biology department.",
        parent=csu_group,
        critical=True
    )
    await evaluator.verify(
        claim="The CSU co-author is in the Biology department.",
        node=node_csu_dept,
        sources=csu_sources,
        additional_instruction="Look for departmental listing indicating 'Biology'."
    )

    node_csu_expertise = evaluator.add_leaf(
        id="CSU_Expertise_Shark_Vertebrae",
        desc="Confirm the CSU co-author has expertise in shark vertebrae research.",
        parent=csu_group,
        critical=True
    )
    await evaluator.verify(
        claim="The CSU co-author has expertise in shark vertebrae research.",
        node=node_csu_expertise,
        sources=csu_sources,
        additional_instruction="Accept profile pages, CVs, articles, or departmental news stating expertise related to shark vertebrae."
    )

    evaluator.add_custom_node(
        result=bool(csu_sources),
        id="URL_CSU_Author_Details",
        desc="Provide supporting URL(s) for CSU affiliation/department/expertise.",
        parent=csu_group,
        critical=True
    )

    # WAM co-author group (critical)
    wam_group = evaluator.add_parallel(
        id="Coauthor_Western_Australian_Museum_Welshpool",
        desc="Identify the Western Australian Museum co-author and confirm Welshpool CRC location constraint.",
        parent=phase3,
        critical=True
    )
    wam = ex.authors_wam
    wam_sources = combine_sources(wam.urls, paper_sources)

    node_wam_name = evaluator.add_leaf(
        id="WAM_Coauthor_Name",
        desc="Provide the Western Australian Museum co-author name.",
        parent=wam_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Western Australian Museum co-author is '{wam.name}'.",
        node=node_wam_name,
        sources=wam_sources,
        additional_instruction="Confirm the named person is an author and affiliated with WAM."
    )

    node_wam_affil = evaluator.add_leaf(
        id="WAM_Affiliation",
        desc="Confirm affiliation with the Western Australian Museum.",
        parent=wam_group,
        critical=True
    )
    await evaluator.verify(
        claim="This co-author is affiliated with the Western Australian Museum.",
        node=node_wam_affil,
        sources=wam_sources,
        additional_instruction="Accept article affiliations or WAM institutional pages."
    )

    node_wam_crc = evaluator.add_leaf(
        id="WAM_CRC_Location_Welshpool",
        desc="Confirm the Western Australian Museum Collections and Research Centre is located in Welshpool.",
        parent=wam_group,
        critical=True
    )
    await evaluator.verify(
        claim="The Western Australian Museum Collections and Research Centre is located in Welshpool.",
        node=node_wam_crc,
        sources=wam_sources,
        additional_instruction="Verify WAM site or official pages indicating Welshpool as CRC location."
    )

    evaluator.add_custom_node(
        result=bool(wam_sources),
        id="URL_WAM_Author_and_CRC",
        desc="Provide supporting URL(s) for WAM affiliation and Welshpool location.",
        parent=wam_group,
        critical=True
    )

    # Swedish co-author group (critical)
    swe_group = evaluator.add_parallel(
        id="Coauthor_Swedish_Museum_Palaeobiology_Stockholm",
        desc="Identify the Swedish Museum of Natural History co-author and confirm department/location constraint.",
        parent=phase3,
        critical=True
    )
    swe = ex.authors_swedish
    swe_sources = combine_sources(swe.urls, paper_sources)

    node_swe_name = evaluator.add_leaf(
        id="Swedish_Coauthor_Name",
        desc="Provide the Swedish Museum of Natural History co-author name.",
        parent=swe_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Swedish Museum of Natural History co-author is '{swe.name}'.",
        node=node_swe_name,
        sources=swe_sources,
        additional_instruction="Confirm the named person is an author and affiliated with the Swedish Museum of Natural History."
    )

    node_swe_affil = evaluator.add_leaf(
        id="Swedish_Museum_Affiliation",
        desc="Confirm affiliation with the Swedish Museum of Natural History.",
        parent=swe_group,
        critical=True
    )
    await evaluator.verify(
        claim="This co-author is affiliated with the Swedish Museum of Natural History.",
        node=node_swe_affil,
        sources=swe_sources,
        additional_instruction="Accept article affiliations or Swedish Museum official pages."
    )

    node_swe_dept = evaluator.add_leaf(
        id="Swedish_Department_Palaeobiology_Stockholm",
        desc="Confirm affiliation with the Department of Palaeobiology in Stockholm.",
        parent=swe_group,
        critical=True
    )
    await evaluator.verify(
        claim="The affiliation includes the Department of Palaeobiology in Stockholm.",
        node=node_swe_dept,
        sources=swe_sources,
        additional_instruction="Look for department name 'Palaeobiology' and location 'Stockholm' on affiliation lines or institutional pages."
    )

    evaluator.add_custom_node(
        result=bool(swe_sources),
        id="URL_Swedish_Author_and_Department",
        desc="Provide supporting URL(s) for Swedish affiliation/department/location.",
        parent=swe_group,
        critical=True
    )

    # International collaboration (critical)
    node_intl = evaluator.add_leaf(
        id="International_Collaboration_At_Least_4_Countries",
        desc="Confirm the author affiliations represent at least 4 different countries.",
        parent=phase3,
        critical=True
    )
    await evaluator.verify(
        claim="The author affiliations represent at least four different countries.",
        node=node_intl,
        sources=paper_sources,
        additional_instruction="Use the affiliations on the article page; count distinct countries (e.g., Australia, USA, Sweden, etc.). Accept synonyms like 'USA' / 'United States'."
    )


async def build_optional_student_group(evaluator: Evaluator, root_node, ex: FullExtraction) -> None:
    """
    Optional student/alumni group placed outside the critical chain to allow partial credit without failing.
    This avoids violating the framework rule that a critical parent cannot have non-critical children.
    """
    optional_node = evaluator.add_parallel(
        id="Optional_Student_or_Alumni_Coauthor",
        desc="If applicable: provide the student/alumni co-author name, CSU status, and supporting URL.",
        parent=root_node,
        critical=False
    )

    stu = ex.authors_student
    stu_sources = stu.urls

    node_stu_name = evaluator.add_leaf(
        id="Student_Alumni_Name",
        desc="Provide the student/alumni co-author name (if applicable).",
        parent=optional_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The student/alumni co-author is '{stu.name}'.",
        node=node_stu_name,
        sources=stu_sources,
        additional_instruction="Only mark as supported if the provided URL or article page indicates this person is a student or alumni co-author."
    )

    node_stu_status = evaluator.add_leaf(
        id="Student_Alumni_Status_Related_To_CSU",
        desc="State the person’s student or alumni status related to Columbus State University (if applicable).",
        parent=optional_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"This person has CSU status indicated as '{stu.status}'.",
        node=node_stu_status,
        sources=stu_sources,
        additional_instruction="Verify the CSU student or alumni status from the provided URL."
    )

    evaluator.add_custom_node(
        result=bool(stu_sources),
        id="URL_Student_Alumni_Evidence",
        desc="Provide a supporting URL for the student/alumni status (if applicable).",
        parent=optional_node,
        critical=False
    )


async def build_phase4_specimens(evaluator: Evaluator, parent_node, ex: FullExtraction) -> None:
    phase4 = evaluator.add_parallel(
        id="Phase4_Specimen_Repository_and_Catalog",
        desc="Verify repository and specimen constraints including type, count, and catalog prefix requirements.",
        parent=parent_node,
        critical=True
    )

    paper_sources = ex.paper.paper_urls
    spec = ex.specimens
    spec_sources = choose_sources(spec.urls, paper_sources)

    # Repository: Museum and Art Gallery of the Northern Territory
    node_repo = evaluator.add_leaf(
        id="Repository_Museum_and_Art_Gallery_NT",
        desc="Confirm the specimens are housed at the Museum and Art Gallery of the Northern Territory.",
        parent=phase4,
        critical=True
    )
    await evaluator.verify(
        claim="The specimens are housed at the Museum and Art Gallery of the Northern Territory (MAGNT).",
        node=node_repo,
        sources=spec_sources,
        additional_instruction="Confirm repository listing on the article page or MAGNT page."
    )

    # Specimen Type: Vertebrae
    node_type = evaluator.add_leaf(
        id="Specimen_Type_Vertebrae",
        desc="Confirm specimen type is vertebrae (plural).",
        parent=phase4,
        critical=True
    )
    await evaluator.verify(
        claim="The specimen type is vertebrae.",
        node=node_type,
        sources=spec_sources,
        additional_instruction="Verify explicit mention of 'vertebrae' in the article/specimen description."
    )

    # Number of Specimens: 5
    node_count = evaluator.add_leaf(
        id="Number_of_Specimens_5",
        desc="Confirm the number of specimens is 5.",
        parent=phase4,
        critical=True
    )
    await evaluator.verify(
        claim="The number of specimens described is five (5).",
        node=node_count,
        sources=spec_sources,
        additional_instruction="Confirm count of specimens from the article/specimen list."
    )

    # At least two catalog numbers with "NTM P" prefix
    node_catalogs = evaluator.add_leaf(
        id="At_Least_Two_Catalog_Numbers_NTM_P",
        desc='Provide at least two specimen catalog numbers with the prefix "NTM P".',
        parent=phase4,
        critical=True
    )
    ntm_two = first_two_ntm_p(spec.catalog_numbers)
    if len(ntm_two) >= 2:
        claim_catalogs = f"The specimen catalog numbers include '{ntm_two[0]}' and '{ntm_two[1]}', both with the 'NTM P' prefix."
    else:
        # If insufficient numbers provided, still form a claim that will fail during verification
        claim_catalogs = f"The specimen catalog numbers include {spec.catalog_numbers}, with at least two having the 'NTM P' prefix."
    await evaluator.verify(
        claim=claim_catalogs,
        node=node_catalogs,
        sources=spec_sources,
        additional_instruction="Verify the exact catalog numbers and the required 'NTM P' prefix on the paper page or repository listings. Allow minor punctuation variants like 'NTM P.' vs 'NTM P'."
    )

    evaluator.add_custom_node(
        result=bool(spec_sources),
        id="URL_Specimen_Evidence",
        desc="Provide supporting URL reference(s) for repository and specimen details (paper URL acceptable if it contains these details).",
        parent=phase4,
        critical=True
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
    Evaluate an answer for the October 2025 Communications Biology paleontological shark paper.
    Builds a verification tree and returns a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as PARALLEL holder (non-critical by framework default)
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

    # Extract comprehensive information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="full_extraction"
    )

    # Record helpful custom info for transparency
    evaluator.add_custom_info(
        info={
            "paper_urls": extracted.paper.paper_urls,
            "science_urls": extracted.science.urls,
            "lead_author_urls": extracted.authors_lead.urls,
            "csu_author_urls": extracted.authors_csu.urls,
            "wam_urls": extracted.authors_wam.urls,
            "swedish_urls": extracted.authors_swedish.urls,
            "specimen_urls": extracted.specimens.urls,
            "catalog_numbers": extracted.specimens.catalog_numbers,
            "collaboration_countries": extracted.collaboration.countries
        },
        info_type="extraction_metadata",
        info_name="extraction_metadata"
    )

    # Create main critical sequential root node for the task (to enforce phase ordering)
    main_root = evaluator.add_sequential(
        id="Root_Paleontological_Research_Paper",
        desc="Identify the specified October 2025 Communications Biology paper about giant lamniform shark vertebrae from the Darwin Formation and report all required constrained details with supporting URLs.",
        parent=root,
        critical=True
    )

    # Build phases under the main critical root
    await build_phase1_paper(evaluator, main_root, extracted)
    await build_phase2_science(evaluator, main_root, extracted)
    await build_phase3_authors(evaluator, main_root, extracted)
    await build_phase4_specimens(evaluator, main_root, extracted)

    # Build optional student/alumni group outside the critical chain (non-critical)
    await build_optional_student_group(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()