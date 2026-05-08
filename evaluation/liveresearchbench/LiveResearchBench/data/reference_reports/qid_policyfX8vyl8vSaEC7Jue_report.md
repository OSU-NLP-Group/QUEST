# U.S. ransomware payment disclosure and reporting: a comprehensive map, comparison, and playbook (financial services, healthcare, and critical infrastructure)

## Executive summary

- No single U.S. law universally compels private companies to publicly disclose ransomware payments. However, multiple frameworks create overlapping disclosure/reporting duties when ransomware incidents are material to investors, involve regulated data, occur in regulated sectors, or implicate sanctions/AML obligations. Key regimes include SEC issuer disclosure rules (Form 8-K Item 1.05; Reg S-K Item 106) [1][2][3][5][6][7], Treasury/OFAC sanctions guidance and licensing (updated ransomware advisory; sanctions designations; compliance expectations) [21][22][23][24], FinCEN BSA/SAR reporting guidance and typologies (2020 and 2021 ransomware advisories; BSA e-filing; SAR timing) [25][26][27][28][29], sectoral incident-reporting regimes (CIRCIA—proposed 72-hour incident and 24-hour ransom payment; TSA security directives; FERC/NERC CIP; NRC notifications) [30][31][32][33][34][35][36][37][38][39][40][41][42][45][46][47], healthcare HIPAA/HITECH breach notification (breach presumption for ransomware; 60-day notices; HHS portal) [49][50][51][52][53][54][55][56][57][58][59], financial sector (NYDFS 23 NYCRR 500, including 24-hour extortion payment notice and 30-day follow-up; federal banking regulators’ 36-hour rule; FFIEC guidance) [60][61][65][66][67][68][69].

- Public and investor communications should be aligned with SEC rules and staff expectations: Item 1.05 applies only to material incidents; use Item 8.01 for voluntary/immaterial updates; disclose material nature/scope/timing/impacts without technical details that would impede response; consider Attorney General (AG) delay only via the DOJ process; and manage Reg FD in parallel with law enforcement engagement [1][2][3][4][5][6].

- Cross-industry playbook priorities: rapid materiality governance; synchronized multi-regulator notifications (SEC, CISA/TSA/FERC/NRC, HHS/OCR, NYDFS, federal bank regulators) on different clocks; sanctions/AML checks and escalation before any payment (OFAC license where required; SAR filing where appropriate); careful handling of ransom payment details (mandatory for NYDFS timing/justification; generally not required in SEC or HIPAA notices); and controls to avoid misleading statements or omissions (lessons from SEC/OCR/DFS/Reg SCI enforcement) [3][21][26][60][65][67][8][9][10][11][12][13][14][15][16][20][22][23][24][49][62][63][64][75].

## SEC cybersecurity disclosure (issuers and FPIs)

Who is covered
- Domestic SEC registrants (including SRCs) via Exchange Act Form 8-K Item 1.05 and annual Regulation S-K Item 106; FPIs via Form 6-K/20-F Item 16K analogs [1][2][3].

Triggers and definitions
- Item 1.05: file within four business days after determining a “cybersecurity incident” is material (determination must be made without unreasonable delay). “Cybersecurity incident” means an unauthorized occurrence, or series of related unauthorized occurrences, on or conducted through a registrant’s information systems that jeopardizes confidentiality, integrity, or availability of information systems or information [1][2].
- Payment of a ransom does not eliminate disclosure obligations; ransom size alone is not determinative of materiality; related incidents can be aggregated for materiality [3].
- FPIs furnish material incident information on Form 6-K when publicized/required in home jurisdiction, on an exchange, or to security holders (no four-business-day trigger under U.S. rules) [2].

Timing and delays
- File a Form 8-K within four business days after the materiality determination; AG delay is available only if the U.S. Attorney General determines in writing that disclosure poses a substantial risk to national security or public safety (initial up to 30 days; extension possible); the deadline shifts to four business days after the delay expires [3][4].
- Staff guidance: use Item 8.01 for voluntary/immaterial incident updates; if later determined material, file Item 1.05 within four business days and amend as impact becomes known [5][6].

Where/how to file
- EDGAR (Form 8-K Item 1.05; 8-K/A for updates; Inline XBRL tagging phased in per the rule) [1][2].

Required content
- Briefly describe the material aspects of the incident’s nature, scope, and timing, and the incident’s material impact or reasonably likely material impact (including on financial condition and results); do not include specific technical details that would impede response/remediation [1][2].
- Ransom payment amount is not required by the SEC rule; do not assume disclosing or withholding payment details alters materiality [1][3].

Safe harbors and penalties
- Failure to timely file Item 1.05 does not affect Form S-3 eligibility; there is no new safe harbor from antifraud liability for Item 1.05 disclosures; traditional materiality and antifraud standards apply [1].
- Enforcement focus includes misleading statements/omissions and disclosure controls; see Blackbaud (2023), Pearson (2021), First American (2021), SolarWinds litigation developments (2024), ICE/NYSE (Reg SCI notice failures), R.R. Donnelley, and ICBC FS (books/records) [8][9][10][11][12][13][14][15][16][75].

Illustrative issuer filings (handling of ransom details)
- VF Corporation, UnitedHealth/Change Healthcare, and others disclosed material incidents and impacts; their SEC filings did not state ransom amounts, focusing on nature and impacts; Caesars (pre-Item 1.05) referenced steps “to ensure the stolen data was deleted,” widely read as implying payment [17][18][19][20].

## Treasury/OFAC sanctions-compliance and licensing (ransom payments)

Who is covered
- U.S. persons (including U.S. companies and their foreign branches) and transactions with a U.S. nexus; non-U.S. persons risk secondary sanctions exposure depending on program [21][23].

Triggers and expectations
- Paying ransom to a sanctioned person, a comprehensively sanctioned jurisdiction, or blocked wallet is generally prohibited absent OFAC authorization; U.S. sanctions are strict liability (no intent required); OFAC may consider a robust risk-based sanctions compliance program, prompt self-disclosure, and cooperation as mitigating in enforcement [21][23].
- OFAC has taken actions against facilitators of ransomware payments (e.g., SUEX and related virtual currency exchange designations) [22].

Licensing and where/how to file
- OFAC may issue specific licenses authorizing transactions otherwise prohibited; apply through OFAC’s online licensing portal; licensing decisions are discretionary and fact-specific [24][21].

Enforcement/disclosure implications
- OFAC’s updated ransomware advisory encourages prompt reporting to law enforcement (FBI, CISA) and to OFAC in the event of a potential sanctions nexus; while not a public disclosure regime, sanctions exposure, licensing status, and law enforcement engagement are materiality considerations for SEC filings and investor communications [21][6][1][2].

## FinCEN SAR filing and ransomware-related reporting

Who is covered
- Financial institutions subject to the BSA (e.g., banks, broker-dealers, MSBs, certain virtual asset service providers) [28][27].

Triggers and timing
- Financial institutions file SARs when transactions aggregate to thresholds and involve suspected criminal activity, including ransomware-related payments; for banks, SARs are generally due within 30 calendar days of initial detection of facts (extendable to 60 days if no suspect identified) [28].
- FinCEN’s 2020 and 2021 ransomware advisories instruct institutions and others facilitating payments to file SARs on ransomware-related activity and to include ransomware-specific key terms and data (e.g., wallet addresses, transaction IDs, malware/ransomware family, IOCs) [25][26][29].

Where/how to file and content
- File via FinCEN’s BSA E-Filing System; include “Cyber Event” and other relevant SAR fields; include advisory reference numbers and narrative details per the advisories [27][26][25].

Enforcement/trend analyses
- FinCEN publishes trend analyses on ransomware in BSA data (e.g., 2021 analysis) to inform typologies and detection; these analyses guide institutions’ monitoring and reporting practices [29].

## Federal cyber incident reporting regimes (ransom payment reporting and sector directives)

CIRCIA (proposed; not yet effective)
- Covered entities: “Covered entities” in critical infrastructure sectors (as defined by rule) would report covered cyber incidents and ransom payments to CISA; coverage criteria include size-based and sector/function-based tests (entity-level) [30][77].
- Triggers and timing: 72-hour report after the entity reasonably believes a covered cyber incident occurred; 24-hour ransom payment report after payment; supplemental reports for significant updates; option to submit a joint report when both occur [30].
- Where/how to file: CISA web-based CIRCIA incident reporting form (with protected channels); until final rule, CISA encourages voluntary incident reporting via its reporting portal [30][34].
- Protections and enforcement: statutory FOIA exemption, privilege protections, and restrictions on governmental use of reports; enforcement via information requests, subpoena, and DOJ civil action for noncompliance (6 U.S.C. 681d–681e) [32][33][30].
- Status: NPRM published April 4, 2024; DHS/CISA currently target May 2026 for the final rule per the Spring 2025 Unified Agenda (timelines subject to change); reporting is not yet mandatory under CIRCIA until the final rule is effective [30][31].

TSA Security Directives (pipelines, rail/transit, aviation)
- Covered entities and requirements: TSA emergency Security Directives and program amendments require specified owners/operators (pipelines, rail/transit, certain aviation entities) to implement cybersecurity measures and report cyber incidents to CISA; pipeline SDs updated/ratified in 2023; rail directives ratified in 2025; aviation added performance-based cyber measures in 2023 [35][36][37][38].
- Timing: reporting of significant/reportable cybersecurity incidents to CISA as soon as practicable, typically no later than 24 hours after identification; see surface cyber proposed rule to codify 24-hour reporting for certain surface modes [36][37][39].
- Penalties: TSA civil penalties for violations of directives/requirements under 49 CFR 1503.401 [40].

FERC/NERC (electric sector)
- Covered entities: NERC-registered owners, operators, and users of the Bulk Electric System; must comply with CIP standards approved by FERC, including CIP-008-6 (Incident Reporting and Response Planning) [41][42].
- Triggers and timing: reportable cyber security incidents reported to E-ISAC and DHS/CISA; initial notification for reportable incidents within 1 hour of determination, with follow-up content within seven days if not initially available; attempted compromises reported by end of next calendar day per entity criteria [42][41][43].
- Penalties: violations of Reliability Standards are subject to penalties up to $1 million per day per violation under Federal Power Act §215, enforced through NERC/Regional Entities with FERC oversight [44].

NRC (nuclear sector)
- Covered entities: NRC power reactor licensees must maintain a cybersecurity program per 10 CFR 73.54 and report cyber security events per 10 CFR 73.77, with additional immediate notification requirements under 10 CFR 50.72 for specified plant events [46][45][47].
- Triggers and timing: telephonic notification to NRC Headquarters Operations Center via ENS within 1 hour for events that adversely impact safety/security/EP functions; 4-hour and 8-hour notifications for lesser categories; written reports within 60 days in specified cases [45].
- Relevance to ransomware: ransomware that adversely impacts (or could have impacted) safety/security/EP functions or compromises support systems with such effect triggers 73.77 reporting [45][46][47].

CISA guidance (voluntary; not obligations)
- CISA’s Shields Up and StopRansomware hubs provide mitigation guidance and encourage reporting to CISA (a single federal report is designed to notify partner agencies), including a dedicated Report Ransomware page and incident reporting portal [76][48][34].

## Healthcare sector: HIPAA/HITECH breach notification and ransomware

Who is covered
- HIPAA covered entities (health plans, most providers conducting standard transactions, clearinghouses) and their business associates [51][53].

Triggers and definitions
- “Breach” of unsecured PHI triggers notification unless the entity documents a low probability of compromise via the required four-factor risk assessment; ransomware incidents that encrypt ePHI are generally presumed breaches (data was “acquired”) absent a documented low probability of compromise [50][49].
- “Unsecured PHI” excludes PHI rendered unusable/unreadable/indecipherable by HHS-specified encryption or destruction (safe harbor) [56].

Timing and recipients
- Individuals: without unreasonable delay, no later than 60 calendar days after discovery; media notice if >500 residents of a state/jurisdiction; HHS Secretary: for 500+, contemporaneous within 60 days; for fewer than 500, by 60 days after year-end (via HHS portal) [51][52][57][58].
- Business associates: notify covered entity without unreasonable delay, no later than 60 days after discovery, and provide information for individual notices [53].
- Law enforcement delay is available upon written request (or up to 30 days on oral request pending written) [54].

Where/how to file and required content
- HHS breach portal (separate forms for ≥500 and <500) and public breach postings; required content for individual notices includes what happened (including breach and discovery dates), types of PHI, steps individuals should take, mitigation/remediation steps, and contacts (plain language) [57][58][51].
- HIPAA does not require disclosing whether a ransom was paid in breach notices; content focuses on the incident, PHI involved, and mitigation steps [51].

Enforcement and guidance
- OCR ransomware guidance emphasizes law enforcement coordination (FBI/USSS) and the presumption of breach in ransomware unless low-probability analysis supports otherwise [49].
- Enforcement themes include risk analysis, access controls, audit log review, incident response, and timeliness/adequacy of notifications; ransomware-related settlements include Doctors’ Management Services (BA), and other large breach settlements (e.g., OSU-CHS, Excellus, Anthem, Premera) and 2024–2025 cases (e.g., Guam Memorial Hospital Authority, Syracuse ASC, Comprehensive Neurology, BST & Co.) [49][20][21][22][24][25][26][34][35][36][55][56][57][58].

## Financial services sector: NYDFS, federal banking regulators, and FFIEC

NYDFS Cybersecurity Regulation (23 NYCRR Part 500)
- Who is covered: “Covered entities” regulated by DFS (e.g., NY-chartered banks, licensed lenders, insurers) [60][61].
- Incident notice: 72-hour notice to DFS of a cybersecurity event that is either required to be reported to any government body/agency or that has a reasonable likelihood of materially harming any material part of the normal operation(s) [60].
- Ransom/extortion payment notice: within 24 hours of making an extortion payment, notify DFS; within 30 days, provide a written description of the reasons payment was necessary, alternatives considered, and diligence performed to ensure compliance with applicable laws (e.g., OFAC) [60].
- Where/how: DFS cyber incident reporting portal (DFS website) [61].
- Enforcement: DFS has brought numerous consent orders for cybersecurity/control failings and notification issues (e.g., First American Title 2021; EyeMed 2023; OneMain 2023) [62][63][64].

Federal banking regulators’ Computer-Security Incident Notification Rule (effective May 1, 2022)
- Who is covered: Banking organizations supervised by the OCC, FDIC, and Federal Reserve; bank service providers [65][66].
- Triggers and timing:
  - Banking organizations must notify their primary federal regulator as soon as possible and no later than 36 hours after determining that a “notification incident” has occurred (a computer-security incident resulting in material disruption/degradation of certain operations/services) [65].
  - Bank service providers must notify affected banking organization customers as soon as possible after determining the service provider has experienced a computer-security incident that has materially disrupted or is reasonably likely to materially disrupt covered services for 4+ hours [65].
- Where/how: regulator-specific channels per each agency’s instructions (see OCC news release and rule materials) [66].

FFIEC guidance
- Extortion/ransomware joint statement (2020): outlines supervisory expectations for risk management and response to cyber extortion, including ransomware; emphasizes prompt notifications to law enforcement and relevant regulators, incident response playbooks, and resilience [67].
- Authentication and access management (2021): sets expectations for layered security programs to mitigate risks including ransomware [68].
- Interagency incident response guidance (2005): response programs for unauthorized access and customer notice under GLBA; still instructive for notification content/process even as sector-specific rules (e.g., 36-hour) now apply [69].

## State-level laws on ransomware payment and reporting (public vs. private sector)

- Several states restrict public entities’ ability to pay ransom and/or require rapid reporting of cyber incidents. For example:
  - North Carolina law prohibits state agencies and local governments from paying a ransom and requires reporting ransomware incidents to the State CIO; it also restricts communication with the threat actor except as directed by law enforcement [70].
  - Florida law requires state agencies and local governments to report cybersecurity incidents promptly to the state’s Cybersecurity Operations Center and prohibits state agencies from paying or otherwise complying with a ransom demand [71].
- For private-sector ransomware payment, there is generally no state-level prohibition; instead, sectoral rules (e.g., NYDFS Part 500) and general data breach laws apply in parallel.

## Case studies: what went wrong and disclosure implications

SEC issuer/enforcement examples
- Blackbaud (2023): Misleading statements minimized the scope of stolen data in a ransomware incident; inadequate disclosure controls prevented critical information (e.g., bank account and SSNs) from reaching management in time for accurate reporting; $3 million penalty and cease-and-desist; lesson: avoid minimizing impacts and ensure robust disclosure controls channel ransomware facts to decision-makers [8][9].
- Pearson (2021): Described a known data breach as hypothetical and downplayed the stolen data; $1 million penalty; lesson: don’t describe realized incidents as hypothetical risk [10].
- First American (2021): Disclosure controls failures led to public statements before senior management had the full context of a significant vulnerability; $487,616 penalty; lesson: governance and escalation before public statements [11].
- SolarWinds litigation (2024): Court dismissed most SEC claims, limiting the use of “internal accounting controls” theories to police general cybersecurity; surviving claims focused on alleged misstatements in a “Security Statement”; lesson: antifraud theories will hinge on specific public statements; obligations under Item 1.05/Item 106 remain in place [12][13].
- ICE/NYSE (Reg SCI) (2024): $10 million for failing to timely inform the SEC of a cyber intrusion at SCI entities; lesson: market infrastructures have separate rapid notice duties beyond issuer rules [14].
- R.R. Donnelley (2024) and ICBC Financial Services (2024): Controls and books/records failures tied to cyber/ransomware disruptions; lessons: continuity and recordkeeping requirements can be violated by ransomware if not properly mitigated [15][16].

Issuer filings and staff comments (practical signaling)
- Staff has pushed issuers to reserve Item 1.05 for material incidents and to use Item 8.01 for immaterial/early disclosures; staff comment letters and director statements underscore this discipline [5][72][73].
- Example: UnitedHealth/Change Healthcare used Item 1.05 with multiple amendments focused on impacts and remediation rather than ransom details; Progress Software disclosed SEC inquiry closure regarding MOVEit without enforcement [18][74].

Healthcare OCR settlements
- Doctors’ Management Services (BA; ransomware), OSU-CHS, Excellus, Anthem, Premera, and 2024–2025 publicized OCR actions (e.g., Guam Memorial Hospital Authority; Syracuse ASC; Comprehensive Neurology; BST & Co.) highlight recurring failures in risk analysis, access controls, incident response, and timeliness/adequacy of notices; penalties and corrective action plans followed [49][20][21][22][24][25][26][34][35][36][55][56][57][58].

NYDFS enforcement
- First American Title (2021), EyeMed (2023), OneMain (2023): DFS found deficiencies in cybersecurity programs, access controls, vendor oversight, and notification practices; consent orders imposed penalties, remediation, and attestations; with the 2023 amendments, DFS can now also scrutinize 24-hour ransom payment notifications and 30-day justifications [62][63][64][60].

## Cross-industry lessons and actionable playbook

Materiality and disclosure governance
- Establish a standing materiality committee (Legal, CISO, IR, Finance, Risk) with triggers to convene upon any ransomware/extortion event and processes to document “without unreasonable delay” materiality analyses using quantitative and qualitative factors (e.g., operational disruption, reputational harm, regulatory/litigation exposure) consistent with SEC standards [1][2][7].
- Pre-clear law enforcement engagement and AG delay option: coordinate early with FBI/DOJ/CISA; consider whether the incident qualifies for AG delay; follow DOJ’s AG delay process; do not assume that paying ransom or a law enforcement request itself changes materiality or tolls the 8-K deadline (only the AG’s written determination does) [3][4][6].

Sequencing, timing, and parallel filings
- Map regulatory clocks in an “hour-by-hour” runbook:
  - SEC: 4 business days after materiality determination (issuers) [2].
  - CIRCIA (proposed): 72 hours for covered incidents; 24 hours for ransom payment [30].
  - TSA: 24 hours to CISA for reportable incidents (pipelines/rail; aviation program directives) [36][37][38][39].
  - NERC: within 1 hour of determination for reportable incidents; next-day for attempted compromises; seven-day follow-up details [42].
  - NRC: 1-hour/4-hour/8-hour ENS notifications; 60-day written follow-up for specified events [45].
  - HIPAA: 60 days to individuals/HHS (≥500 contemporaneous), media if >500 residents; BA to CE within 60 days; law enforcement delay available [51][52][53][54][57][58].
  - NYDFS: 72 hours for reportable events; 24-hour ransom payment notice; 30-day justification [60].
  - Federal banking regulators: 36 hours to primary regulator for “notification incidents”; service providers notify banks ASAP [65].
- Build a centralized “single source of truth” narrative synchronized across all filings and regulators to avoid inconsistencies (e.g., SEC Item 1.05 vs. HIPAA notices vs. DFS and bank regulator notices) [1][51][60][65].

Handling ransom payment details
- NYDFS requires notification within 24 hours of any extortion payment and a 30-day written description of why payment was necessary, alternatives considered, and sanctions diligence; prepare a pre-approved template and an internal decision memo documenting alternatives and sanctions checks (OFAC screening/licensing analysis) [60][21][24].
- SEC Item 1.05 does not require disclosing ransom payments or amounts; focus on material nature/scope/timing and impacts; avoid unnecessary technical detail that could impede response [1][2][3].
- HIPAA breach notices do not require stating whether a ransom was paid; focus on required content, mitigation, and protective steps for individuals [51].

Sanctions and AML integration
- Before any payment: conduct OFAC risk assessment, screen indicators (wallets, intermediaries), consider whether a specific license is required, and evaluate legal risks; document decisions with counsel; consider contacting OFAC and law enforcement (mitigating factor) [21][23][24].
- If financial institutions or VASPs are involved: ensure BSA/SAR obligations (30-day deadline for banks; include ransomware typologies and indicators per FinCEN guidance); coordinate with counterparties to gather transaction details for SAR narrative (e.g., wallet addresses, variant, IOCs) [28][26][27][25][29].

Communications discipline
- Use Item 8.01 for early/immaterial issuer updates; avoid implying materiality by using Item 1.05 prematurely; if later determined material, file Item 1.05 within four business days and amend as impacts are known [5][6][72][73].
- Avoid speculative or misleading statements (e.g., describing realized incidents as “hypothetical” risks) and ensure disclosure controls surface accurate scope (exfiltrated data types, operational impacts) to senior management; lessons from Blackbaud, Pearson, First American, SolarWinds [8][9][10][11][12][13][15].
- Manage Reg FD and selective disclosure risk: sharing additional details privately with customers, vendors, and law enforcement is permissible; apply Reg FD as usual for communications to covered market participants/shareholders (public dissemination or appropriate safeguards) [6].

Operational resilience and control baselines
- Adopt sector benchmarks (e.g., FFIEC layered controls; HIPAA Security Rule and HHS 405(d) practices; NERC/NRC/TSA sector directives) and test response-to-recovery for ransomware (including vaulting, segmentation, and OT/IT interplay) to mitigate both operational harm and disclosure failures [67][68][69][49][46][42][36][37][38][39].
- For healthcare, reinforce HIPAA Security Rule risk analysis, access controls, audit logs, and incident response to avoid OCR penalties and to support timely breach notifications [49][51][55].

Training and rehearsals
- Run disclosure-focused tabletop exercises including Legal/IR/Compliance/CISO and external counsel, covering: materiality committee decisioning, AG delay process, OFAC licensing decision tree, NYDFS 24-hour ransom notice, bank 36-hour notices, HIPAA 60-day notices, and sectoral reports (TSA/NERC/NRC) [4][60][65][51][42][45][36].

## Quick-reference mapping (who is covered; triggers; timing; where; content; penalties)

- SEC issuers (public companies)
  - Trigger: material cybersecurity incident; 4 business days after materiality determination; Item 1.05 on EDGAR; describe nature/scope/timing and impacts; AG delay requires DOJ written determination; antifraud liability applies; S-3 eligibility not affected by late 1.05 [1][2][3][4].

- Treasury/OFAC (all U.S. persons; ransom facilitators)
  - Trigger: sanctions nexus (blocked person/jurisdiction); licensing may be required; apply via OFAC portal; VSD/cooperation/compliance program are mitigating; designations (e.g., SUEX) underscore risks [21][24][23][22].

- FinCEN (BSA entities)
  - Trigger: suspicious activity including ransomware payments; SAR due generally within 30 days (banks); file via BSA E-Filing; include ransomware indicators and typologies per advisories [28][27][26][25][29].

- CIRCIA (critical infrastructure—proposed)
  - Trigger: covered cyber incident (72 hours after reasonably believing) and any ransom payment (24 hours after payment); report via CISA portal; statutory protections and enforcement via subpoena/DOJ; final rule targeted May 2026 (subject to change) [30][31][32][33][34].

- TSA (pipelines/rail/aviation)
  - Trigger: significant/reportable incidents; report to CISA within about 24 hours; implement performance-based cybersecurity; penalties under 49 CFR 1503.401 [35][36][37][38][39][40].

- FERC/NERC (electric)
  - Trigger: reportable cyber security incidents; initial within 1 hour of determination to E-ISAC and DHS/CISA; penalties up to $1M/day/violation [42][41][44].

- NRC (nuclear)
  - Trigger: cyber security events impacting (or potentially impacting) safety/security/EP; 1/4/8-hour ENS calls; 60-day written follow-ups; robust cyber program under 73.54 [45][46][47].

- HIPAA/HITECH (healthcare)
  - Trigger: breach of unsecured PHI (presumed for ransomware absent low-probability-of-compromise); 60-day notices to individuals (and HHS ≥500 and media >500 residents); HHS reporting portal; law enforcement delay available; OCR enforcement frequent [51][52][57][54][49].

- NYDFS Part 500 (financial services)
  - Trigger: 72-hour event notice; 24-hour ransom payment notice; 30-day written justification; DFS portal; consent orders and penalties for deficiencies [60][61][62][63][64].

- Federal banking regulators (all banking organizations and service providers)
  - Trigger: “notification incident” (material disruption) → 36-hour regulator notice; service providers ASAP to banks; regulator-specific channels [65][66].

- State public sector bans/reporting
  - Examples: NC ban on ransom payments by public entities; FL bans ransom payments by state agencies and imposes reporting; private sector typically governed by sectoral or general breach laws rather than payment bans [70][71].

## Sector-specific takeaways

Financial services
- Expect layered and fast reporting: DFS 72-hour and 24-hour payment notice/30-day justification; federal 36-hour rule; potential SARs; and SEC 8-K materiality determinations for issuers [60][65][26][2].
- Align extortion decisions with OFAC/FinCEN: complete sanctions checks and SAR planning before contemplating payment; prepare DFS 30-day justification memo contemporaneously [21][26][60].

Healthcare
- Treat ransomware as a presumed breach; conduct and document low-probability analysis if asserting no breach; meet HIPAA 60-day clocks and HHS portal filing; consider state breach laws and media notices; coordinate with OCR and law enforcement [49][50][51][52][57][54].
- Structure forensics under counsel, but be prepared to provide compliance-related records to OCR; repeated OCR enforcement underscores timeliness and security program expectations [55][49][20][21][22][24][25][26][34][35][36].

Critical infrastructure
- Map TSA/NERC/NRC reporting clocks; prepare runbooks to file within 1 hour (NERC) or 1–24 hours (NRC/TSA) as applicable; monitor CIRCIA’s finalization and harmonization options; use CISA reporting portals [42][45][36][39][30][34].

## Appendix: public/investor messaging guardrails

- Don’t conflate incident discovery with materiality; only use Item 1.05 when materiality is determined “without unreasonable delay”; otherwise, use Item 8.01 [2][5][6].
- Avoid minimizing or hypothetical framing when facts are known (Pearson); ensure disclosure controls surface accurate exfiltration/impact info (Blackbaud; First American); avoid overbroad security marketing statements (SolarWinds) [10][8][9][11][12][13].
- Law enforcement coordination and confidentiality: continue engaging FBI/DOJ/CISA; Item 1.05 does not preclude private sharing; manage Reg FD [6].
- Ransom details: disclose payment only where required (e.g., DFS 24-hour/30-day); be cautious in public statements to avoid inviting further extortion or compromising investigations; coordinate with OFAC/FinCEN counsel [60][21][26].

### Sources
[1] SEC Final Rule: Cybersecurity Risk Management, Strategy, Governance, and Incident Disclosure (Release No. 33-11216): https://www.sec.gov/files/rules/final/2023/33-11216.pdf  
[2] SEC Small Entity Compliance Guide (Cybersecurity): https://www.sec.gov/corpfin/secg-cybersecurity  
[3] SEC Exchange Act Form 8-K C&DIs (Section 104B – Item 1.05): https://www.sec.gov/rules-regulations/staff-guidance/compliance-disclosure-interpretations/exchange-act-form-8-k  
[4] DOJ Attorney General Delay Determination Process: https://www.justice.gov/media/1328226/dl?inline  
[5] SEC Corp Fin Director Statement (May 21, 2024): https://www.sec.gov/newsroom/speeches-statements/gerding-cybersecurity-incidents-05212024  
[6] SEC Corp Fin Director Statement (June 20, 2024): https://www.sec.gov/newsroom/whats-new/gerding-cybersecurity-incidents-06202024  
[7] 2018 Commission Statement and Guidance on Public Company Cybersecurity Disclosures: https://www.sec.gov/rules-regulations/2018/02/commission-statement-guidance-public-company-cybersecurity-disclosures  
[8] SEC Press Release 2023-48 (Blackbaud): https://www.sec.gov/newsroom/press-releases/2023-48  
[9] SEC Blackbaud complaint/order: https://www.sec.gov/litigation/complaints/2023/comp-pr2023-48.pdf  
[10] SEC Press Release 2021-154 (Pearson plc): https://www.sec.gov/newsroom/press-releases/2021-154  
[11] SEC Press Release 2021-102 (First American Financial): https://www.sec.gov/newsroom/press-releases/2021-102  
[12] SEC Press Release 2023-227 (SolarWinds complaint): https://www.sec.gov/newsroom/press-releases/2023-227  
[13] Reuters: Judge dismisses most of SEC lawsuit against SolarWinds (July 18, 2024): https://www.reuters.com/legal/us-judge-dismisses-most-sec-lawsuit-against-solarwinds-concerning-cyberattack-2024-07-18/  
[14] SEC Press Release 2024-63 (ICE/NYSE Reg SCI): https://www.sec.gov/newsroom/press-releases/2024-63  
[15] SEC Press Release 2024-75 (R.R. Donnelley): https://www.sec.gov/newsroom/press-releases/2024-75  
[16] SEC Administrative Proceeding (ICBC Financial Services): https://www.sec.gov/enforcement-litigation/administrative-proceedings/34-101794-s  
[17] VF Corporation Form 8-K (Dec. 18, 2023): https://www.sec.gov/Archives/edgar/data/103379/000095012323011228/d659095d8k.htm  
[18] UnitedHealth Group Form 8-K/A (Apr. 22, 2024): https://www.sec.gov/Archives/edgar/data/731766/000073176624000150/unh-20240221.htm  
[19] Caesars Entertainment Form 8-K (Sept. 14, 2023): https://www.sec.gov/Archives/edgar/data/1590895/000119312523235015/d537840d8k.htm  
[20] Globe Life Inc. Form 8-K (Item 8.01) (June 14, 2024): https://www.sec.gov/Archives/edgar/data/320335/000032033524000029/gl-20240614.htm  
[21] OFAC Updated Advisory: Potential Sanctions Risks for Facilitating Ransomware Payments (Sept. 21, 2021): https://home.treasury.gov/system/files/126/ofac_ransomware_advisory.pdf  
[22] OFAC Press Release: Sanctions on SUEX for facilitating ransomware (Sept. 21, 2021): https://home.treasury.gov/news/press-releases/jy0364  
[23] OFAC Framework for OFAC Compliance Commitments (May 2, 2019): https://home.treasury.gov/system/files/126/framework_ofac_cc.pdf  
[24] OFAC Online Licensing Application: https://ofac.treasury.gov/online-application  
[25] FinCEN Advisory on Ransomware (Oct. 1, 2020): https://www.fincen.gov/sites/default/files/advisory/2020-10-01/Advisory%20Ransomware%20FINAL%20508.pdf  
[26] FinCEN Updated Advisory on Ransomware (Nov. 8, 2021): https://www.fincen.gov/sites/default/files/advisory/2021-11-08/FinCEN%20Ransomware%20Advisory_FINAL_508.pdf  
[27] FinCEN BSA E-Filing System: https://bsaefiling.fincen.treas.gov  
[28] 31 CFR 1020.320 (SAR by banks): https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1020/section-1020.320  
[29] FinCEN Financial Trend Analysis: Ransomware (Oct. 2021): https://www.fincen.gov/sites/default/files/2021-10/Financial%20Trend%20Analysis_Ransomware%20508%20FINAL.pdf  
[30] Federal Register: CIRCIA Reporting Requirements (NPRM) (Apr. 4, 2024): https://www.federalregister.gov/documents/2024/04/04/2024-06526/cyber-incident-reporting-for-critical-infrastructure-act-circia-reporting-requirements  
[31] Reginfo Unified Agenda (RIN 1670-AA04; Spring 2025): https://www.reginfo.gov/public/do/eAgendaViewRule?RIN=1670-AA04&pubId=202504  
[32] 6 U.S.C. § 681d (Enforcement): https://www.law.cornell.edu/uscode/text/6/681d  
[33] 6 U.S.C. § 681e (Protections): https://www.law.cornell.edu/uscode/text/6/681e  
[34] CISA: Reporting a Cyber Incident: https://www.cisa.gov/reporting-cyber-incident  
[35] Federal Register: Ratification of Pipeline Security Directives (Apr. 19, 2024): https://www.govinfo.gov/content/pkg/FR-2024-04-19/html/2024-08393.htm  
[36] Federal Register: Ratification of Rail/Transit Security Directives (Jan. 21, 2025): https://www.federalregister.gov/documents/2025/01/21/2025-01422/ratification-of-security-directives  
[37] TSA Press Release (Rail) (Oct. 23, 2023): https://www.tsa.gov/news/press/releases/2023/10/23/tsa-renews-cybersecurity-requirements-passenger-and-freight-railroad  
[38] TSA Press Release (Aviation) (Mar. 7, 2023): https://www.tsa.gov/news/press/releases/2023/03/07/tsa-issues-new-cybersecurity-requirements-airport-and-aircraft  
[39] Federal Register: Enhancing Surface Cyber Risk Management (NPRM) (Nov. 7, 2024): https://www.govinfo.gov/content/pkg/FR-2024-11-07/html/2024-24704.htm  
[40] 49 CFR 1503.401 (Civil penalties): https://www.law.cornell.edu/cfr/text/49/1503.401  
[41] FERC News Release (Order No. 848): https://www.ferc.gov/news-events/news/ferc-requires-expanded-cyber-security-incident-reporting  
[42] Federal Register: FERC Approval of CIP-008-6 (June 26, 2019): https://www.govinfo.gov/content/pkg/FR-2019-06-26/html/2019-13587.htm  
[43] NERC E-ISAC About: https://www.nerc.com/pa/CI/ESISAC/Pages/default.aspx  
[44] FERC Enforcement and Reliability overview: https://www.ferc.gov/enforcement-reliability  
[45] 10 CFR 73.77 (Cyber security event notifications): https://www.law.cornell.edu/cfr/text/10/73.77  
[46] 10 CFR 73.54 (Cyber security program requirements): https://www.law.cornell.edu/cfr/text/10/73.54  
[47] 10 CFR 50.72 (Immediate notification requirements): https://www.law.cornell.edu/cfr/text/10/50.72  
[48] CISA: Report Ransomware: https://www.cisa.gov/stopransomware/report-ransomware  
[49] HHS OCR Fact Sheet: Ransomware and HIPAA: https://www.hhs.gov/hipaa/for-professionals/security/guidance/cybersecurity/ransomware-fact-sheet/index.html  
[50] 45 CFR 164.402 (Breach; risk assessment): https://www.law.cornell.edu/cfr/text/45/164.402  
[51] 45 CFR 164.404 (Notice to individuals; content/method): https://www.law.cornell.edu/cfr/text/45/164.404  
[52] 45 CFR 164.408 (Notice to HHS): https://www.law.cornell.edu/cfr/text/45/164.408  
[53] 45 CFR 164.410 (BA to CE notice): https://www.law.cornell.edu/cfr/text/45/164.410  
[54] 45 CFR 164.412 (Law-enforcement delay): https://www.law.cornell.edu/cfr/text/45/164.412  
[55] 45 CFR 164.414 (Burden of proof; documentation): https://www.law.cornell.edu/cfr/text/45/164.414  
[56] HHS Guidance: Encryption and Destruction (safe harbor): https://www.hhs.gov/hipaa/for-professionals/breach-notification/guidance/index.html  
[57] HHS Breach Reporting (Overview/Instructions): https://www.hhs.gov/hipaa/for-professionals/breach-notification/breach-reporting/index.html  
[58] HHS OCR Breach Report Submission Portal: https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf  
[59] HHS “Change Healthcare” Cybersecurity Incident FAQs: https://www.hhs.gov/hipaa/for-professionals/special-topics/change-healthcare-cybersecurity-incident-frequently-asked-questions/index.html  
[60] 23 NYCRR Part 500 (NYDFS Cybersecurity Regulation) (including §500.17): https://www.dfs.ny.gov/industry_guidance/cybersecurity/23nycrr500  
[61] NYDFS Cybersecurity Regulation portal: https://www.dfs.ny.gov/industry_guidance/cybersecurity  
[62] NYDFS Consent Order: First American Title (Oct. 21, 2021): https://www.dfs.ny.gov/system/files/documents/2021/10/ea20211021_first_american_title_insurance_company.pdf  
[63] NYDFS Press Release: EyeMed (Jan. 17, 2023): https://www.dfs.ny.gov/reports_and_publications/press_releases/pr20230117  
[64] NYDFS Consent Order: OneMain Financial (Dec. 15, 2023): https://www.dfs.ny.gov/system/files/documents/2023/12/ea20231215_onemain_financial_group_llc.pdf  
[65] Federal Register: Computer-Security Incident Notification (Final Rule) (Nov. 23, 2021): https://www.federalregister.gov/documents/2021/11/23/2021-25501/computer-security-incident-notification-requirements-for-banking-organizations-and-their  
[66] OCC News Release 2021-133: https://www.occ.treas.gov/news-issuances/news-releases/2021/nr-occ-2021-133.html  
[67] FFIEC Joint Statement: Cyber Attacks Involving Extortion (Nov. 3, 2020): https://www.ffiec.gov/press/PDF/FFIEC%20Joint%20Statement%20-%20Cyber%20Attacks%20Involving%20Extortion.pdf  
[68] FFIEC Authentication and Access Management (Nov. 2, 2021): https://www.ffiec.gov/press/pr110221.htm  
[69] FFIEC IT Handbook: Interagency Guidance on Response Programs (2005): https://ithandbook.ffiec.gov/media/274841/2005-incident-response-guidance.pdf  
[70] N.C. Gen. Stat. § 143B-1379 (State Information Technology Services; ransomware prohibition): https://www.ncleg.gov/EnactedLegislation/Statutes/PDF/BySection/Chapter_143B/GS_143B-1379.pdf  
[71] Fla. Stat. § 282.3186 (Cybersecurity incident notification; ransom ban for state agencies): https://www.flsenate.gov/Laws/Statutes/2024/282.3186  
[72] SEC Staff Comment Correspondence: Microsoft (2024): https://www.sec.gov/Archives/edgar/data/789019/000119312524172450/filename1.htm  
[73] SEC Staff Comment Correspondence: SouthState (2024): https://www.sec.gov/Archives/edgar/data/764038/000095010324007629/filename1.htm  
[74] Progress Software: SEC Investigation Conclusion (Aug. 7, 2024): https://investors.progress.com/news-releases/news-release-details/progress-announces-conclusion-sec-investigation-moveit  
[75] SEC FY23 Enforcement Results (Press Release 2023-234): https://www.sec.gov/newsroom/press-releases/2023-234  
[76] CISA Shields Up: https://www.cisa.gov/shields-up  
[77] CISA CIRCIA FAQs: https://www.cisa.gov/topics/cyber-threats-and-advisories/information-sharing/circia/faqs  
[78] NERC Enforcement Actions 2016 table (examples including CIP-008): https://www.nerc.com/pa/comp/CE/Pages/Actions_2016/2016_TABLE.htm