# Is AI Transforming Teaching and Assessment (2022–present)? Evidence on Adoption, Outcomes, Equity, and Policy

## Executive summary

- Adoption at scale: General-purpose AI assistants are now mainstream among learners and educators (e.g., ChatGPT reported 400M weekly active users in Feb 2025; Google’s Gemini app reported 450M monthly active users in Q2 2025), with major education deployments such as California State University’s systemwide ChatGPT Edu rollout (~460k students, 63k staff) and ubiquitous course-level use of GitHub Copilot via GitHub Education programs. Usage surveys show rapid growth among K–12 teachers, teens, and higher-education students. These trends are persistent, not transient spikes. [62][63][64][61][65][66][57][58][59][60]  
- Learning outcomes: High-quality causal evidence since 2022 shows small-to-moderate improvements when AI is implemented well and used sufficiently. Examples include 0.12–0.22 SD gains in elementary math with structured practice (Khan Academy RCT), a 4 percentage-point increase in mastery from an AI “copilot” assisting human tutors (larger gains for students supported by lower-rated tutors), moderate gains for practical skills in medical/nursing meta-analyses, and reduced teacher planning time (~31%) without lowering judged lesson quality in a school-randomized trial. Effects vary by context and design; some studies report neutral knowledge outcomes or autonomy risks for lower-achieving students without careful scaffolding. [1][2][3][15][16][7][5][6]  
- Equity: AI can improve instruction where baseline teaching quality is weakest (human–AI hybrid tutoring), but adoption and benefits are uneven across schools and students. Nationally representative surveys indicate guidance and use lag in higher-poverty schools; detectors for “AI-written text” produce high false positives for non-native writers; and early institutional studies report overall writing gains post-LLM with mixed distributional effects (e.g., SES differences). Equity depends on resourcing, usage design, and integrity policy choices. [3][27][29][28]  
- Institutional adaptation: Systems moved from initial bans to integration with guardrails. Ministries and quality bodies (US ED, UK DfE/Ofqual, OECD, UNESCO, TEQSA in Australia) issued guidance; universities and school systems revised academic integrity policies, syllabi language, and assessment design (e.g., “secure” in-person exams vs “open-AI” assessments requiring acknowledgement). Many institutions caution against relying on AI detectors as sole evidence. Assessment is shifting toward authenticity, process evidence, and oral defenses. [30][31][34][35][55][56][38][39][41][50][51][52][53][54]  

Overall conclusion: Since 2022, AI has begun to fundamentally change parts of teaching and assessment—in planning, tutoring support, research/writing workflows, and assessment design—supported by credible, verifiable evidence. The most consistent, policy-relevant gains appear when AI augments (not replaces) human instruction and when institutions redesign assessments and integrity practices accordingly. Significant evidence gaps remain for large, multi-site trials of full LLM tutors across K–12 core subjects and for precise cost-effectiveness estimates across diverse contexts. [1][2][3][7][15][16][38][39]

## 1) Popular tools and student workflows (2022–present)

### 1.1 Most-used tools by category and scale signals

- General-purpose LLM chatbots and multimodal assistants  
  - OpenAI ChatGPT: ~400M weekly active users (Feb 2025), with system-wide EDU deployments such as California State University’s ChatGPT Edu (≈460k students; 63k staff). [62][61]  
  - Google Gemini: Gemini app reported ~450M monthly active users (Q2 2025 earnings). [63][64]  
  - Microsoft Copilot and GitHub Copilot: GitHub Copilot adopted by 77,000+ organizations (Oct 2024); GitHub Education community includes 5M+ students and 200k educators, with Copilot Pro benefits for verified learners and teachers. [65][66]  

- Writing and paraphrasing  
  - Grammarly: “40M+ daily users” and >$700M revenue (2025), widely used by students and institutions. [67]  
  - QuillBot and similar tools see heavy student traffic (traffic proxies indicate strong student adoption, though active user figures vary by vendor). [67]

- AI search and citation assistants  
  - Perplexity: 780M queries in May 2025 (~26M/day); 15M MAU (early 2025). [68][69]  
  - Elicit, Scite, Semantic Scholar: Platform-reported users indicate growing researcher and student use (e.g., Elicit “5M+ researchers,” Scite “~1M users,” Semantic Scholar ~8M MAU). [70][71][72][73]

- Math/solver apps  
  - Photomath: 220M+ lifetime downloads by 2021; acquisition formalized under Google in 2024; multi-million monthly downloads continue. [75][74][76]  
  - Wolfram|Alpha: multi-million monthly visits; Symbolab claims “300M+ users worldwide.” [79][80]  
  - Chegg (bundle including Mathway): 2024 subscriber base ~6.6M (portfolio-level). [77][78]

- Adaptive/AI-driven learning platforms  
  - Khan Academy/Khanmigo: SY24–25 reports ~2.0M users of Khanmigo globally; ~770k U.S. students via district partnerships; Microsoft-funded “Khanmigo for Teachers” scaled nationally. [81][82]  
  - ALEKS (McGraw Hill): >7M users in FY2024; 64M+ lifetime learners. [83][84]  
  - Carnegie Learning MATHia, DreamBox: MATHia supports ~500k students annually; DreamBox reaches ~6M students in the U.S. within Discovery Education’s broader 45M-student footprint. [85][86]

Notes on measurement: Vendor-reported MAU/WAU include non-education use; education seat counts indicate potential reach rather than daily active use. Traffic/downloads are proxies and should be interpreted cautiously. [61][63][64][76]

### 1.2 Who is using AI (surveys, 2024–2025) and how

- K–12 teachers and students  
  - U.S. teachers: 60% report using AI tools, 32% weekly; regular users report ~5.9 hours/week saved (nationally representative Gallup/Walton FF, Mar–Apr 2025, n=2,232). [57]  
  - U.S. teens: About 1 in 4 used ChatGPT for schoolwork in 2024, double 2023; usage patterns vary by task and demographics. [58]  

- Higher education  
  - UK students: Use of generative AI “in assessments” jumped from 53% (2024) to 88% (2025); 92% report using some AI tool (HEPI/Kortext, n=1,041). [59]  
  - U.S. campuses: By April 2023, 67% of surveyed IT professionals reported using genAI at work while only 34% said their institution had a policy—highlighting demand outpacing governance. [60]  
  - Policy adoption: Institutional AI policies in U.S. higher ed rose markedly 2023→2025 (Tyton Partners). [89]

### 1.3 Measured workflow changes

- Teacher planning/productivity  
  - School-randomized trial (UK, Year 7–8 science): Access to ChatGPT with a usage guide cut weekly lesson-prep time by ~31% (56 vs 81.5 minutes) without observed degradation in sampled lesson quality; perceived workload burden also fell. [7]

- Human tutoring with AI “copilots”  
  - RCT: Tutors using an LLM-based copilot during live sessions achieved +4 percentage points in student topic mastery overall (p<0.01), with the largest gains (+9 pp) for students assigned to lower-rated tutors; tutors shifted toward better pedagogical moves. [2][3]

- Student research and writing  
  - Large institutional analysis post-LLM introduction found overall writing gains and narrowed gaps for certain student groups, though benefits were uneven by SES—underscoring the need for targeted support and transparent AI acknowledgement practices. [28]  
  - AI detectors are unreliable for high-stakes decisions and can yield high false positives for non-native writers, creating equity and due-process risks; many institutions advise against using detectors as sole evidence. [29][50][51][52][53][54][49]

- Programming and problem-solving  
  - Classroom deployments show LLM-generated hints can accelerate issue resolution and align with target knowledge components when well designed, but require guardrails to limit overreliance. [20]

## 2) Learning outcomes, cost-effectiveness, and equity

### 2.1 K–12 RCTs and systematic evidence (2022–2025)

- Mastery practice and coaching (Khan Academy): Two field experiments (teacher-level randomization; 10,979 students, Grades 3–8) found 0.12–0.22 SD gains in Grades 3–6 when students averaged ≥35 minutes/week; weaker effects in Grades 7–8 tied to lower usage. Dosage and implementation fidelity were critical. [1]  
- Human–AI hybrid tutoring (Tutor CoPilot): RCT showed +4 pp mastery overall, +9 pp for students with lower-rated tutors, and improved tutor pedagogy. Low per-tutor cost (~$20/year) implies strong cost-effectiveness where it upgrades lower-quality instruction. [2][3]  
- LLM as homework tutor (HS ESL): Class-level RCT replacing conventional homework with GPT‑4 sessions improved grammar learning and engagement, though replication is needed. [4]  
- AI-generated feedback (HS physics): Two RCTs over five weeks found strong heterogeneity: compulsory AI hints benefited lower achievers (d≈0.67) but reduced autonomy for higher achievers; on-demand hints benefited higher achievers (d≈0.38) but reduced autonomy for lower achievers—showing design choices have equity implications. [5]  
- AI explanations (Grade 6 math/physics): RCT found no clear learning advantage versus textbook explanations but improved positive emotions, situational interest, self-efficacy, and lowered cognitive load. [6]  
- Teacher workload RCT (UK): ~31% reduction in planning time with ChatGPT access and no observed decline in lesson quality—indicating productivity gains without short-term quality tradeoffs. [7]  

- Broader syntheses  
  - K–12 intelligent tutoring systems (ITS) review (mostly quasi-experimental) reports generally positive effects, smaller vs non-intelligent systems comparisons, and calls for larger, longer trials in diverse contexts. [25]  
  - Meta-analysis across “personal tutor” AI interventions reports average Hedges g≈0.30 with wide heterogeneity and stronger effects in short-duration interventions. [26]

- Benchmarks (human tutoring)  
  - Virtual 1:3 math tutoring RCT (Greenville County, SC; >2,000 middle-school students) produced +0.11 SD gains—useful as a contemporary benchmark for high-impact human tutoring against which AI and hybrid models can be compared. [23]

- Named adaptive platforms (post-2022 causal evidence status)  
  - Khanmigo: Large district RCT registered; results pending. [8]  
  - ASSISTments: Ongoing A/B experimentation with AI/crowd-sourced supports; at-scale LLM-specific RCTs still emerging. [9]  
  - MATHia: Large-scale product improvement (2022–24) indicated human-edited word problems improved outcomes (including for students with disabilities); generative AI rewrites had uneven effects—caution on content accuracy and accessibility. [10]  
  - ALEKS: Recent work focuses on placement and persistence rather than LLM tutoring per se; internal engine comparisons reported in EDM 2022. [12][11]

### 2.2 Higher education and adult learning

- Meta-analyses (RCTs) in medical and nursing education report moderate improvements for practical/skills outcomes (SMD ~0.6) and mixed or small knowledge gains; satisfaction and lowered cognitive load commonly observed. Effects strengthen in practice-oriented, longer interventions. [15][16]  
- Small RCTs suggest short-term performance boosts from ChatGPT-4 as a study aid without clear retention gains at one week—highlighting the need for longer follow-ups and transfer measures. [17]  
- Professional decision-making RCTs (physicians) show LLM assistance improves performance on vignette-based tasks by ~6–12 pp but sometimes increases time—relevant for advanced professional training where accuracy and rationale matter. [21]  
- Programming courses report faster issue resolution with well-designed AI feedback; course designs emphasize scaffolding to prevent overreliance—consistent with the K–12 physics heterogeneity around autonomy. [20]

### 2.3 Equity, heterogeneity, and integrity

- Adoption gaps: National surveys show AI use and guidance lag in higher-poverty schools, risking unequal benefits without deliberate supports and infrastructure. [27]  
- Heterogeneous effects: HS physics RCTs show benefits and harms vary by prior achievement and by compulsory/on-demand design; careful scaffolding and oversight are crucial to protect autonomy for lower-achieving learners. [5]  
- Writing outcomes and distribution: Large institutional analyses report overall writing improvements post-LLM introduction but uneven gains across SES; transparent AI acknowledgement and targeted support can mitigate disparities. [28]  
- Integrity and detectors: AI-text detectors can disproportionately flag non-native writing (high false positives), and many universities advise against using detector scores as sole evidence to avoid wrongful allegations. [29][50][51][52][53][54]

### 2.4 Cost-effectiveness and workload

- Human–AI hybrid tutoring (CoPilot) shows promising cost-effectiveness at ~$20 per tutor/year with measurable student mastery benefits, especially where it raises the floor of tutoring quality. [3][2]  
- Teacher productivity: A school-randomized RCT found material time savings at near-zero direct cost, implying favorable cost-per-minute saved. [7]  
- Fully automated LLM tutors: Robust, comparable cost-per-SD estimates are limited; many deployments are in pilot or preprint stages; large, peer-reviewed RCTs and cost studies are priorities. [8][9]

## 3) Institutional adaptation: policies, assessment, grading, and support

### 3.1 System-level guidance and sector shifts (global)

- International frameworks: UNESCO and OECD issued guidance urging responsible, age-appropriate, human-centered AI, assessment implications, and national guardrails. [56][55]  
- United States: US Department of Education released a comprehensive guidance report (May 2023) and a 2025 letter clarifying funding and priorities, emphasizing AI literacy, equity, and safeguards. [30][31]  
- UK schools and universities:  
  - Department for Education guidance addresses classroom use, age restrictions, and exam integrity; Ofqual clarified authenticity requirements, prohibited AI as sole marker, and records AI misuse in malpractice data. [34][35]  
  - Russell Group universities adopted sector-wide principles committing to AI literacy and ethical use; QAA advises assessment redesign rather than bans. [36][37]  
- Australia: National higher-education regulator TEQSA led a sector-wide “Assessment reform for the age of AI” promoting authentic/iterative/oral assessments and caution with detection. [38]  
- Exemplars of integrated policy: University of Sydney’s “two-lane” assessment model—Secure (AI-prohibited, in-person) vs Open (AI-permitted with acknowledgement)—with student-facing acknowledgement rules and integrity policy updates. [39][40]  
- New Zealand: Ministry guidance cautions schools against relying on AI detectors, requiring local policies to specify acceptable AI uses. [41]  
- Singapore, Korea, Hong Kong, Flanders: Ministries launched tools (e.g., SLS generative feedback), adaptive assessment pilots, AI digital textbooks roadmaps, and curricular guidance to integrate AI while safeguarding integrity and privacy. [42][43][44][45]

Trajectory: Many systems moved from early 2023 restrictions (e.g., NYC’s initial ban) toward integration with guardrails (NYC reversal and educator supports). [32][33]

### 3.2 Assessment design, academic integrity, and grading

- Assessment redesign priorities  
  - Authentic tasks tied to process evidence (“show your working”), in-class drafting, oral defenses, and “open-AI” assignments with acknowledgement, while retaining secure in-person exams for high stakes—codified in TEQSA guidance and the University of Sydney model; echoed in QAA advice. [38][39][37]  
  - Course-level policy language: Institutions distribute sample syllabus statements clarifying AI-permitted/prohibited uses and citation/acknowledgement rules (e.g., MIT TLL; Harvard CS50’s integrated assistant with boundaries). [47][48]

- Detection/proctoring practices  
  - Turnitin enabled AI writing detection in April 2023 and reports widespread scanning; however, universities increasingly caution that detector scores should not be used as sole evidence due to false positives/negatives and equity concerns. [49][87][50][51][52][53][54]

### 3.3 Faculty development and student support

- National and sector initiatives (e.g., U.S., U.K., Australia, Singapore, Hong Kong) emphasize AI literacy, educator training, and curricular integration; policy suites often include student-facing guidance on acceptable use and acknowledgement requirements. [30][34][38][42][44]  
- Surveys indicate institutional AI policies and student guidance coverage have expanded rapidly 2023→2025, though clarity gaps for learners remain. [89][46]

## 4) Synthesis: transformation versus transient trend

- Not a transient trend: Persistent, large-scale user bases (ChatGPT, Gemini), formal institutional deployments (CSU system), and sector surveys showing rapid mainstreaming (teachers, teens, university students) argue against a short-lived fad. [62][63][61][57][58][59]  
- Teaching and assessment practices are changing:  
  - Teaching: RCT-verified productivity gains (planning time), human–AI hybrid tutoring that elevates low-performing tutors, and course-level integrations (coding feedback, writing supports). [7][2][3][20][19]  
  - Assessment: Sector-wide shifts to authentic tasks, process evidence, and oral assessment; institutional policy models for “secure vs open-AI” assessments; caution on detectors. [38][39][37][50][51][52][53][54]  
  - Learning outcomes: Small-to-moderate gains where usage is sufficient and design is sound—particularly for practice-oriented skills—alongside neutral effects or autonomy risks when design is suboptimal. [1][15][16][5][6]

- Where AI’s comparative advantage is clearest today  
  - Augmenting human instruction (e.g., coaching the coach; scalable formative feedback) rather than replacing it. [2][3][20]  
  - Reducing educator workload for routine planning and resource drafting with evidence of time savings and no loss in quality in short-run trials. [7]

- Guardrails and equity-by-design are essential  
  - Adoption, accuracy, and integrity risks are not evenly distributed; design choices (compulsory vs on-demand feedback), access to devices/connectivity, and institutional resourcing determine who benefits. [5][27][29][41]

## 5) Uncertainties and evidence gaps

- LLM tutor RCTs at K–12 scale across core subjects with long-run outcomes, subgroup effects, and cost-effectiveness comparisons to high-impact human tutoring. [8]  
- Replications across subjects and regions; standardized measures of reasoning quality and retention beyond short-term quizzes. [15][16][17]  
- Robust, comparable cost-per-SD and implementation cost data for AI tools, including total cost of ownership for institutions. [3][7]  
- Valid, fair, and transparent approaches to academic integrity that do not rely on unreliable AI detectors, and clearer institutional processes for AI acknowledgement and process evidence. [49][50][51][52][53][54]

## 6) Practical recommendations

- For K–12 systems and districts  
  - Prioritize human–AI hybrid uses that raise instructional quality (e.g., tutor copilots, teacher planning aids), and invest in PD and AI literacy. [2][3][7][34]  
  - Build equity-by-design: ensure device/connectivity access, provide clear student guidance, and monitor subgroup impacts; avoid detector-driven discipline without corroborating process evidence. [27][41][50][51][52][53][54]

- For higher education  
  - Adopt “secure vs open-AI” assessment models with explicit acknowledgement requirements; redesign tasks to emphasize process and authentic application; supply sample syllabus language and student supports. [39][40][47][37]  
  - Evaluate learning outcomes beyond immediate performance (retention, transfer) and track heterogeneity across student groups. [17][28]

- For policymakers and funders  
  - Support large, multi-site RCTs and quasi-experiments of LLM tutoring and feedback systems; require equity and cost-effectiveness analyses and publish protocols. [8][25]  
  - Issue or update national guidance that balances innovation with integrity and transparency, and fund educator upskilling at scale. [30][34][38][56][55]

- For vendors and researchers  
  - Improve accuracy with retrieval and curriculum alignment, surface citations, and build features for acknowledgement and process logging; co-design with teachers and learners to protect autonomy and avoid overreliance. [10][20]

### Bottom line
AI is already reshaping teaching and assessment where it reduces friction (planning), augments instruction (human–AI tutoring), and prompts assessment redesign (secure vs open-AI). Verified outcome gains exist but are context-sensitive and design-dependent; equity risks are real without deliberate policy and support. The weight of credible, recent evidence and widespread policy/practice change indicates structural transformation is underway rather than a passing edtech trend. [1][2][3][7][15][16][38][39][62][63][61]

### Sources
[1] NBER Working Paper 32388 – Teaching Teachers to Use Computer-Assisted Learning (Khan Academy RCT): https://www.nber.org/papers/w32388  
[2] Tutor CoPilot RCT (arXiv): https://arxiv.org/abs/2410.03017  
[3] Tutor CoPilot – NSSA study page: https://nssa.stanford.edu/studies/tutor-copilot-human-ai-approach-scaling-real-time-expertise  
[4] GPT‑4 as Homework Tutor RCT (ar5iv): https://ar5iv.org/pdf/2409.15981  
[5] High school physics AI feedback RCTs (arXiv): https://arxiv.org/abs/2505.08672  
[6] AI-generated explanations RCT (Grade 6) (arXiv): https://arxiv.org/abs/2412.15747  
[7] EEF/NFER Teacher Choices Trial – ChatGPT for KS3 science planning: https://educationendowmentfoundation.org.uk/projects-and-evaluation/projects/choices-in-edtech-using-generative-ai-chatgpt-for-ks3-science-lesson-preparation-2024-teacher-choices-trial  
[8] AEA RCT Registry – Khanmigo: https://www.socialscienceregistry.org/trials/13519  
[9] ASSISTments – Student Supports (beta): https://www.assistments.org/individual-resource/student-supports-beta  
[10] THE Journal – IES/Carnegie Learning study on AI/human revisions to MATHia problems: https://thejournal.com/Articles/2024/02/26/IES-Carnegie-Learning-Study-Exploring-the-Use-of-AI-to-Help-Students-with-Reading-Disabilities.aspx  
[11] EDM 2022 – ALEKS assessment engine randomized comparison: https://educationaldatamining.org/edm2022/proceedings/2022.EDM-industry-track.109/index.html  
[12] Education Sciences (2025) – ALEKS placement fuzzy RDD study: https://www.mdpi.com/2227-7102/15/2/154  
[13] CENTURY Tech report – SATs outcomes (correlational): https://www.century.tech/news/century-tech-report-reveals-significant-positive-impact-of-ai-powered-learning-on-primary-sats-outcomes/  
[14] Squirrel AI press (post‑2022 evidence gap note): https://www.abc27.com/business/press-releases/cision/20240516CN15049/squirrel-ai-explores-potential-of-large-language-models-in-education-at-premier-ai-for-education-event-aied/  
[15] BMC Medical Education (2025) – Meta-analysis of RCTs (GAI in medical education): https://bmcmededuc.biomedcentral.com/articles/10.1186/s12909-025-07750-2  
[16] AIMS Press – Meta-analysis of RCTs in nursing education: https://www.aimspress.com/article/id/6840e997ba35de6e26810363  
[17] PubMed – Three-arm RCT with ChatGPT‑4 in medical education: https://pubmed.ncbi.nlm.nih.gov/40656250/  
[18] Emotionally enriched AI feedback RCT (arXiv): https://arxiv.org/abs/2410.15077  
[19] International Journal of Ed Tech in Higher Education – AI vs human writing feedback: https://educationaltechnologyjournal.springeropen.com/articles/10.1186/s41239-023-00425-2  
[20] ArXiv (Qi et al., 2024) – GPT‑4 feedback in programming exercises: https://arxiv.org/abs/2406.05603  
[21] medRxiv (2024) – LLM assistance RCTs for clinical decision-making: https://www.medrxiv.org/content/10.1101/2024.08.05.24311485v1.full  
[22] AEA RCT Registry – Generative AI for high-skilled work (developers): https://www.socialscienceregistry.org/trials/14530  
[23] PR Newswire – RCT finds significant gains from Littera’s virtual math tutoring: https://www.prnewswire.com/news-releases/randomized-controlled-trial-finds-statistically-significant-gains-from-litteras-virtual-math-tutoring-for-middle-school-students-302550830.html  
[24] IRAL (De Gruyter) – Meta-analysis: AI-assisted L2 learning (2025): https://www.degruyter.com/document/doi/10.1515/iral-2024-0213/html  
[25] PubMed – K–12 ITS systematic review (2025): https://pubmed.ncbi.nlm.nih.gov/40368938/  
[26] Open Praxis (2024) – Meta-analysis (AI in education; “personal tutors”): https://openpraxis.org/articles/10.55982/openpraxis.17.3.842  
[27] RAND (2025) – AI adoption in schools and equity implications: https://www.rand.org/pubs/research_reports/RRA134-25.html  
[28] arXiv (2024) – Large institutional study of writing outcomes post‑LLM: https://arxiv.org/abs/2410.22282  
[29] arXiv (2023) – AI-text detector bias against non-native English writing: https://ui.adsabs.harvard.edu/abs/2023arXiv230402819L/abstract  
[30] US Dept of Education (2023) – Artificial Intelligence and the Future of Teaching and Learning: https://digital.library.unt.edu/ark:/67531/metadc2114121/  
[31] US Dept of Education (2025) – AI guidance and proposed priority: https://www.ed.gov/about/news/press-release/us-department-of-education-issues-guidance-artificial-intelligence-use-schools-proposes-additional-supplemental-priority  
[32] Chalkbeat (2023) – NYC schools ban ChatGPT: https://www.chalkbeat.org/newyork/2023/1/3/23537987/nyc-schools-ban-chatgpt-writing-artificial-intelligence/  
[33] Chalkbeat (2023) – NYC reverses ChatGPT ban: https://www.chalkbeat.org/newyork/2023/5/18/23727942/chatgpt-nyc-schools-david-banks/  
[34] UK Department for Education – Generative AI in education: https://www.gov.uk/government/publications/generative-artificial-intelligence-in-education  
[35] Ofqual – Guide for schools and colleges 2025: https://www.gov.uk/government/publications/ofqual-guide-for-schools-and-colleges-2025/ofqual-guide-for-schools-and-colleges-2025  
[36] Russell Group – Principles on generative AI tools in education: https://www.russellgroup.ac.uk/policy/policy-briefings/principles-use-generative-ai-tools-education  
[37] QAA – Additional advice on generative AI tools: https://www.qaa.ac.uk/news-events/news/qaa-publishes-additional-advice-on-generative-artificial-intelligence-tools  
[38] TEQSA – Assessment reform for the age of AI: https://www.teqsa.gov.au/about-us/news-and-events/latest-news/assessment-reform-age-artificial-intelligence  
[39] University of Sydney – AI assessment policy (“two‑lane” model): https://www.sydney.edu.au/news-opinion/news/2024/11/27/university-of-sydney-ai-assessment-policy.html  
[40] University of Sydney – Responsible AI use (students): https://www.sydney.edu.au/students/responsible-ai-use.html  
[41] NZ Ministry of Education – Generative AI guidance for schools: https://www.education.govt.nz/school/digital-technology/generative-ai/  
[42] Singapore MOE – Speech (2024) on SLS genAI assistant and adaptive pilots: https://www.moe.gov.sg/news/speeches/20240528-speech-by-minister-chan-chun-sing-at-the-redesigning-pedagogy-international-conference-2024  
[43] Republic of Korea MOE – AI Digital Textbooks roadmap: https://english.moe.go.kr/boardCnts/viewRenewal.do?boardID=254&boardSeq=95291&lev=0&m=0202&opType=N&page=1  
[44] Hong Kong EDB – Using AI in Science Education: https://www.edb.gov.hk/en/curriculum-development/kla/science-edu/use-ai.html  
[45] Government of Flanders – Responsible AI in Flemish education (vision): https://www.vlaanderen.be/publicaties/responsible-ai-in-flemish-education-a-collaborative-process-from-development-to-use  
[46] Jisc – Student perceptions of AI 2025: https://www.jisc.ac.uk/reports/student-perceptions-of-ai-2025  
[47] MIT Teaching + Learning Lab – Generative AI and your course (sample syllabus statements): https://tll.mit.edu/teaching-resources/course-design/gen-ai-your-course/  
[48] Harvard CS50 – AI notes/policy (CS50.ai): https://cs50.harvard.edu/college/2025/spring/notes/ai/  
[49] Turnitin – Turns on AI writing detection (press): https://www.turnitin.com/press/turnitin-turns-on-ai-writing-detection-capabilities-for-educators-and-institutions  
[50] King’s College London – Macro-level AI guidance for education: https://www.kcl.ac.uk/about/strategy/learning-and-teaching/ai-guidance/macro-level  
[51] University of Greenwich – Opt-out of Turnitin’s AI detection: https://www.gre.ac.uk/articles/public-relations/our-decision-to-opt-out-of-using-turnitins-ai-detection-tool  
[52] University of Lincoln – Investigating academic integrity in Turnitin: https://digital.lincoln.ac.uk/resources/investigating-academic-integrity-in-turnitin/  
[53] University of Bristol – Other considerations when using AI in assessment: https://www.bristol.ac.uk/bilt/sharing-practice/guides/guidance-on-ai/using-ai-in-assessment/other-considerations-when-using-ai/  
[54] Bath Spa University – AI in teaching: https://www.bathspa.ac.uk/artificial-intelligence/ai-in-teaching/  
[55] OECD (2024) – Regulation and guidance on generative AI in education: https://www.oecd-ilibrary.org/en/education/regulation-and-guidance-on-generative-ai-in-education-2024_4cca1e36-en  
[56] UNESCO – Guidance on generative AI in education and research: https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research  
[57] Gallup – Teachers’ AI use and time saved: https://news.gallup.com/poll/691967/three-teachers-weekly-saving-six-weeks-year.aspx  
[58] Pew Research Center – Teens’ ChatGPT use for schoolwork (2024 vs 2023): https://www.pewresearch.org/short-reads/2025/01/15/about-a-quarter-of-us-teens-have-used-chatgpt-for-schoolwork-double-the-share-in-2023/  
[59] HEPI/Kortext (2025) – Explosive increase in student use of genAI tools: https://www.hepi.ac.uk/2025/02/26/hepi-kortext-ai-survey-shows-explosive-increase-in-the-use-of-generative-ai-tools-by-students/  
[60] EDUCAUSE QuickPoll (2023) – Adopting and adapting to genAI in higher ed tech: https://er.educause.edu/articles/2023/4/educause-quickpoll-results-adopting-and-adapting-to-generative-ai-in-higher-ed-tech  
[61] OpenAI – CSU system adopts ChatGPT Edu: https://openai.com/index/openai-and-the-csu-system/  
[62] TechCrunch – ChatGPT now serves 400M weekly users: https://techcrunch.com/2025/02/20/openai-now-serves-400-million-users-every-week/  
[63] Alphabet IR – Q2 2025 earnings call (Gemini app MAU): https://abc.xyz/2025-q2-earnings-call/  
[64] CNBC – Alphabet Q2 2025 coverage: https://www.cnbc.com/2025/07/23/alphabet-google-q2-earnings.html  
[65] GitHub Universe 2024 – Copilot adoption: https://github.com/newsroom/press-releases/github-universe-2024  
[66] GitHub Education – Program scale: https://github.com/edu  
[67] Reuters – Grammarly acquires Superhuman; 40M DAU and revenue: https://www.reuters.com/business/grammarly-acquires-email-startup-superhuman-ai-platform-push-2025-07-01/  
[68] TechCrunch – Perplexity 780M queries in May 2025: https://techcrunch.com/2025/06/05/perplexity-received-780-million-queries-last-month-ceo-says/  
[69] Financial Times – Perplexity MAU and growth context: https://www.ft.com/content/d4fb70f9-b971-433b-884c-2f01d1d08968  
[70] Elicit – Product and user base: https://elicit.com/  
[71] Scite – Pricing/users: https://scite.ai/pricing  
[72] Semantic Scholar – About (MAU): https://www.semanticscholar.org/about  
[73] AI2 Blog – Semantic Scholar leadership: https://blog.allenai.org/new-leadership-for-semantic-scholar-49d2d51bd761  
[74] AndroidHeadlines – Google formalizes Photomath acquisition in Play Store: https://www.androidheadlines.com/2024/03/google-makes-photomath-acquisition-official-play-store.html  
[75] TechCrunch (2021) – Photomath reaches 220M downloads: https://techcrunch.com/2021/02/18/math-learning-app-photomath-raises-23-million-as-it-reaches-220-million-downloads/  
[76] Crunchbase – Photomath tech profile (Apptopia downloads): https://www.crunchbase.com/organization/photomath-inc-2/technology  
[77] Chegg – FY2024 results (subscribers): https://www.chegg.com/about/newsroom/press-release/chegg-reports-2024-fourth-quarter-and-full-year-financial-results  
[78] Chegg IR – FY2024 results (alternate filing): https://investor.chegg.com/Press-Releases/press-release-details/2025/Chegg-Reports-2024-Fourth-Quarter-and-Full-Year-Financial-Results/default.aspx  
[79] Semrush – Wolfram|Alpha traffic overview: https://www.semrush.com/website/wolframalpha.com/overview/  
[80] Symbolab – About (users): https://www.symbolab.com/about  
[81] Khan Academy – Annual Report (Khanmigo metrics): https://annualreport.khanacademy.org/  
[82] CNBC – Microsoft funds free Khanmigo for US teachers: https://www.cnbc.com/2024/05/21/microsoft-khan-academy-launch-free-ai-assistant-for-all-us-teachers.html  
[83] GlobeNewswire – McGraw Hill FY2024 results (ALEKS users): https://www.globenewswire.com/news-release/2024/05/30/2890787/0/en/McGraw-Hill-Reports-Fourth-Quarter-and-Full-Year-Fiscal-2024-Financial-Results.html  
[84] SEC – McGraw Hill filing (ALEKS lifetime learners): https://www.sec.gov/Archives/edgar/data/1951070/000162828025035928/mcgrawhillinc-424b4.htm  
[85] IES – MATHia platform profile: https://ies.ed.gov/use-work/awards/mathia-digital-learning-platform-supporting-core-and-supplemental-instruction-middle-and-high-school  
[86] PR Newswire – Discovery Education completes acquisition of DreamBox Learning: https://www.prnewswire.com/news-releases/clearlake-capital-backed-discovery-education-completes-acquisition-of-dreambox-learning-301954475.html  
[87] Turnitin Blog – One year of AI writing detection (>200M papers; flagged shares): https://www.turnitin.com/blog/turnitin-celebrates-one-year-for-ai-writing-detection  
[88] TIME – AFT national AI training hub for educators: https://time.com/7301335/ai-education-microsoft-openai-anthropic/  
[89] Tyton Partners – Racing Forward (AI proficiency vs practice; institutional policy adoption): https://tytonpartners.com/racing-forward-bridging-the-gap-between-generative-ai-proficiency-and-educational-practice/