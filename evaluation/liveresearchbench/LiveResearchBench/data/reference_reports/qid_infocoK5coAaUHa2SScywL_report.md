# Client-Facing AI Portfolio/Planning Copilots Launched by U.S. and Canadian Banks and Registered Broker-Dealers (Jan 1, 2023–Sep 16, 2025)

## Executive Summary
- Confirmed U.S. GA launches: 3
  - Public (broker-dealer): Alpha (May 2023)
  - Stash (broker-dealer/RIA): Money Coach AI (May 2025)
  - Moomoo (broker-dealer): Moomoo AI (U.S. product page live; GA in 2025; exact U.S. GA month not disclosed)
- Tentative/partial: 2
  - Robinhood (broker-dealer): Cortex Digests GA in UK; U.S. rollout in 2025 appears phased and not fully GA as of Sep 16, 2025
  - eToro USA (broker-dealer): Tori rolling out to “select users in select regions” (U.S. GA unclear)
- Canada: No confirmed GA client-facing generative-AI portfolio/planning copilots by Canadian banks or registered broker-dealers within the window. Multiple institutions announced internal/advisor-only AI tools or non-copilot features.

Key takeaways:
- Activity is led by U.S. fintech broker-dealers targeting self-directed retail investors with in-app chat assistants and AI-generated insights.
- Disclosures about core stack (hosting, RAG, guardrails) are sparse; only Public explicitly cites GPT-4.
- Safety/compliance positioning centers on “not investment advice,” with limited public detail on model risk governance. A notable exception: Moomoo discloses on‑prem prompt handling and PII filtering.
- Early KPIs indicate strong engagement (Public’s “90% of MAUs used Alpha”; Stash’s conversion-on-action metric).

---

## Inclusion Criteria and Scope
Included if:
- Client-facing, generative-AI “copilot” features for investment/portfolio or financial-planning insights, simulations, recommendations, or actions.
- Launched into production (GA) between Jan 1, 2023 and Sep 16, 2025.
- Released by a U.S. or Canadian bank or registered broker-dealer (white labels included only if bank/BD brand and client availability are confirmed).

Excluded:
- Advisor-only or internal copilots.
- Generic support chatbots without portfolio/planning capabilities.
- Trading bots or longstanding robo-advisors absent a new 2023–2025 generative-AI copilot component.
- Pilots/betas that are not GA.

---

## Qualified GA Launches (U.S. Registered Broker-Dealers)

### Public Investing, Inc. (Public) — Alpha
1) Institution: Public Investing, Inc. (brand: Public); U.S. registered broker-dealer (FINRA/SIPC). Not a bank. [2]  
2) Assistant/product: Alpha (“AI for investors”; “investing co‑pilot”). Integrated across Public app; later also a standalone watchlist app. [2][3]  
3) Launch month/year: May 2023 (launch to Public members; “rolling out to all investors this summer”); standalone Alpha app launched Dec 2024. [3]  
4) Customer tiers & modality: Retail/self-directed Public brokerage members; text chat and AI summaries on asset and portfolio pages; proactive earnings/news summaries and email digests. [2][2]  
5) Geography & languages: U.S. brokerage; materials in English. Standalone Alpha app available globally; brokerage-linked features pertain to U.S. members. [2][4]  
6) Core AI stack (as disclosed): GPT‑4 from OpenAI; Public’s semantic indexing/context layer; further stack elements (hosting/orchestration, RAG, vector DB, evals) not disclosed. [2][3]  
7) Safety & compliance: Prominent “not investment advice” and “experimental” disclaimers; may be inaccurate; standard BD disclosures. No public detail on model risk governance, explainability, or certifications. [2][5]  
8) Integrations: Native across Public’s brokerage app (stocks/ETFs/crypto/treasuries, news, alerts); historic ChatGPT plugin announced in 2023 for GPT‑4 users. No third-party planning/CRM integrations disclosed. [2][3]  
9) Access & pricing: Included free for all Public members; standalone Alpha app available to any investor, with $1/week option for non‑Public brokerage users. Channels: web and mobile. GA (not beta). [2][4][3]  
10) KPIs: “Since debut in 2023, 90% of monthly active members have used [Alpha]” (as of Dec 12, 2024 PR). [3]  
11) Links:
- Official announcement: Public launches Alpha powered by OpenAI’s GPT‑4 (May 17, 2023). [3]
- Official product page: Public Alpha. [2]
- Official standalone app PR: Alpha watchlist app GA (Dec 12, 2024). [4]
- Independent trade press: MarketWatch feature on investor AIs (Jun 11, 2025). [1]
- Safety/FAQ: “What is Alpha?” (disclaimers/capabilities). [5]

### Stash Investments LLC / Stash Capital LLC — Money Coach AI
1) Institution: Stash (entities include Stash Investments LLC [RIA] and Stash Capital LLC [BD]); U.S. registered broker-dealer; not a bank. [6]  
2) Assistant/product: Money Coach AI — in‑app AI‑powered personalized financial guidance. [7]  
3) Launch month/year: May 2025 (PR: “launches Money Coach AI” alongside Series H funding). Prior beta noted in Oct 2024. [7][8]  
4) Customer tiers & modality: Retail/mass market Stash subscribers; text/chat guidance within the Stash app across money/investing flows. [7]  
5) Geography & languages: U.S.; English. [7]  
6) Core AI stack (as disclosed): Not naming model/vendor; claims to translate expert strategies into real‑time personalized recommendations with “end‑to‑end compliance supervision and training system.” No detail on hosting, vector DB, or evals. [7]  
7) Safety & compliance: PR mentions compliance supervision/training; standard BD/RIA disclosures apply. No public MRM/ISO/SOC specifics. [7]  
8) Integrations: Embedded in Stash platform (investing, Auto‑Stash, diversification nudges). No external CRM/planning integrations disclosed. [7]  
9) Access & pricing: Included with Stash subscription plans (tiers per general pricing pages); GA rollout to subscriber base. [7]  
10) KPIs: As of launch PR — “2.2M user interactions”; “1 in 4 customers who interact take a positive action within 10 minutes” (e.g., invest, deposit, diversify, turn on Auto‑Stash); 1.3M paying subscribers; $4.3B AUM (company-reported). [7][9]  
11) Links:
- Official PR (GA + Series H): Stash secures $146M; launches Money Coach AI (May 12, 2025). [7]
- Official PR (leadership/beta context): Co‑founders return as co‑CEOs (Oct 8, 2024). [8]
- Independent coverage: PYMNTS on Stash’s funding and AI plans (May 12, 2025). [9]

### Moomoo Financial Inc. — Moomoo AI
1) Institution: Moomoo Financial Inc.; U.S. registered broker-dealer (FINRA/SIPC). Not a bank. [10]  
2) Assistant/product: Moomoo AI — suite including AI Assistant (chat), AI Stock Analysis, Announcement Summaries, Trend/Pattern analysis, AI Stock Screener. [10]  
3) Launch month/year: 2025 GA on U.S. product site (exact U.S. GA month not disclosed). Corporate/APAC PR announced launches in Southeast Asia in June 2025; U.S. pages and learn articles show live availability for U.S. users in 2025. [10][11][12]  
4) Customer tiers & modality: Retail/self-directed moomoo users; in‑app text chat and context-aware assistants within asset/portfolio screens. [10]  
5) Geography & languages: U.S. product page (English); additional regional launches (e.g., Singapore/Malaysia) noted separately. [10][11]  
6) Core AI stack (as disclosed): “Based on the latest large language model,” integrated with moomoo market data/news/community; internet search supported. Privacy FAQ states prompts stored on moomoo on‑prem servers (Singapore); model providers prohibited from saving/using data; PII filtered before LLM. Model vendor/hosting not named. [10]  
7) Safety & compliance: “Not an investment advisor” and “no direct investment recommendations” disclaimers; prompt handling and PII filtering disclosed; U.S. BD disclosures and links to FINRA/SIPC present. [10]  
8) Integrations: Native to moomoo platform; context links to screeners, research, and trade tickets. No external planning/CRM integrations disclosed. [10]  
9) Access & pricing: Included as part of the moomoo platform; no incremental pricing indicated for AI assistant on U.S. page. [10]  
10) KPIs: None disclosed for U.S.; regional coverage highlights adoption but not U.S.-specific metrics. [11][13]  
11) Links:
- Official U.S. product page: Moomoo AI (features, FAQs, privacy). [10]
- Official learn article (U.S.) describing Moomoo AI: “Invest forward and easier.” [12]
- Official PR (APAC): Launch in Southeast Asia (June 13, 2025). [11]
- Independent coverage: Asian Banking & Finance article referencing AI assistant launch (regional). [13]

---

## Tentative/Partial (Flagged for U.S./Canada GA Ambiguity)

### Robinhood Financial LLC — Robinhood Cortex (and Digests)
- Status: UK GA for “Digests” (Aug 19, 2025). U.S. rollout described as “later in 2025” and “began rolling out this summer” by independent coverage; Robinhood’s U.S. Help Center still marked Cortex as “not available yet” as of snapshot. Treat U.S. GA as tentative/rolling as of Sep 16, 2025. [14][15][16][17]  
- Institution: U.S. registered broker-dealer (FINRA/SIPC).  
- Product: Robinhood Cortex (AI investing assistant); Digests summarizing stock movement and context. [14][16]  
- Launch timeline: Announced Mar 27, 2025; UK GA for Digests announced Aug 19, 2025; U.S. staged rollout indicated but not confirmed GA. [14][16][17]  
- Customer tiers & modality: Retail/self-directed; early access planned for Gold subscribers; text summaries/prompts in-app. [14][15]  
- Stack & safety: Not disclosed; standard disclosures apply. [14]  
- Access/pricing: Early access for Gold; broader rollout planned; UK Digests GA. [15][16]  
- Links: Official announcements and Help Center; independent PYMNTS and Benzinga coverage. [14][15][16][17]

### eToro USA Securities Inc. — Tori
- Status: Global announcement Aug 7, 2025; product page states “gradually rolling out” to select users/regions. U.S. production coverage not explicitly confirmed; treat as limited/unclear for U.S. as of Sep 16, 2025. [18][19]  
- Institution: U.S. registered broker-dealer (eToro USA) within global eToro Group. [20]  
- Product: Tori — AI investing companion for portfolio/asset Q&A and insights. [18]  
- Customer tiers & modality: Retail; in‑app chat. [18]  
- Stack & safety: In‑house AI with some rule-based flows; no vendor specifics. [18]  
- Access/pricing: Included, staged rollout. [18]  
- Links: Official product and press; independent secondary coverage. [18][19][20]

---

## Negative Findings (No Qualifying GA Client-Facing Generative-AI Copilot in Period)

### United States — Major Banks and Broker-Dealers
- Interactive Brokers: Added AI-generated news summaries (Dec 2024) and advisor-only AI commentary tools; no retail client GA portfolio/planning copilot in 2023–2025. [21]
- Charles Schwab: “Schwab Assistant” is a navigation/support agent, not a planning/investing copilot; no GA retail copilot disclosed. [22]
- Fidelity: Public updates emphasize internal/gen-AI associate tools; no retail client GA portfolio/planning copilot disclosed. [23]
- JPMorgan Chase/J.P. Morgan Wealth Management: Wealth Plan is a 2022 non-genAI digital coach; no retail client GA gen-AI copilot disclosed 2023–2025. [24]
- Wells Fargo: “Fargo” is a banking assistant; no GA investing copilot disclosed. [25]
- Goldman Sachs: Firmwide AI assistant for employees; no retail/institutional client GA copilot for Marquee/Marcus Invest disclosed. [26]
- Morgan Stanley/E*TRADE: Advisor-facing GPT-4 tools; no retail client GA copilot disclosed for E*TRADE. [27]

### Canada — Banks and Direct Brokerages
- RBC / RBC Direct Investing: Public reports on AI teams (capital markets), but no client-facing retail investing copilot launched. [28]
- TD / TD Direct Investing: TD Securities launched an employee-facing gen-AI assistant pilot; no retail client GA copilot. [29]
- Questrade: “Pilot Trading” app hub item offers AI-driven signals, not a conversational portfolio/planning copilot. [30]
- Others with no public GA copilot announcements found within the window: BMO InvestorLine, CIBC Investor’s Edge, Scotiabank (Scotia iTRADE), National Bank Direct Brokerage, Qtrade Direct Investing, Desjardins Online Brokerage, Wealthsimple, CI Direct Trading, Canaccord Genuity, HSBC Bank Canada (pre-RBC acquisition). If evidence of GA launches emerges, updates are warranted.

---

## Observations and Market Analysis
- Who launched: Fintech broker-dealers (Public, Stash, Moomoo) were first movers; no big U.S. universal banks or Canadian banks launched GA retail investing copilots in the period.
- Modality: Text chat plus embedded AI summaries/context on asset/portfolio pages; no prominent voice launches in this segment.
- Scope of advice: Systems emphasize research, education, and “insights,” with clear “not investment advice” language and no auto-trading by the assistant.
- AI stack disclosures: Sparse. Public cites GPT‑4; others avoid naming models/providers. Moomoo provides atypically detailed prompt-handling privacy notes (on‑prem storage and PII filtering).
- Governance posture: Public disclaimers and BD disclosures are standard; few public details on model risk management, evaluation, bias testing, or certifications. Stash cites an “end‑to‑end compliance supervision and training system,” but without technical specifics.
- Adoption signals: Public reports 90% of MAUs touching Alpha; Stash reports meaningful near-term “positive actions” post-interaction (1 in 4). These suggest strong engagement when copilots are integrated at point‑of‑need (holdings, watchlists, activity flows).
- Vendor approach: Mix of in-house and third-party LLMs; little transparency on orchestration/RAG layers. Expect heavier scrutiny from regulators (SEC/FINRA/CIRO/OSFI) as institutions move from research to recommendations and actions.
- Canadian gap: Despite active internal pilots and AI teams, no retail GA copilots surfaced in Canada in the period; likely due to regulatory conservatism and prioritization of internal efficiencies over direct-to-client advice features.

---

## What’s Unknown (Typical Gaps to Close in Diligence)
- Architecture details: hosting provider(s), RAG design, vector store choice, prompt governance/versioning, model routing/fine‑tuning, evals/observability, red‑team results.
- Data handling: precise data residency by client region, treatment of PII/transaction data, retention policies, and whether client data is excluded from LLM provider training.
- Risk controls: hallucination/bias mitigations, suitability layers, human‑in‑the‑loop designs for higher‑risk flows, auditability of prompts/responses, and coverage under Model Risk Management frameworks aligned to SEC/FINRA/OSFI/CIRO expectations.
- Performance: segment-level KPIs (e.g., AUM lift, trade quality, planning goal attainment), longitudinal engagement, and CSAT/NPS tied to assistant usage.

---

## Appendix — Due-Diligence Checklist (Applied to Each Entry)
- Entity scope and licenses (bank, BD, RIA)
- Copilot capabilities mapped to portfolio/planning use cases
- Launch timing and GA scope (segments, regions)
- Modalities, platforms, and authentication
- AI stack and data flows (providers, hosting, RAG, guardrails)
- Safety, MRM, and disclosures; advisor oversight where applicable
- Integrations with core systems and client channels
- Access, eligibility, pricing
- KPIs and measurement rigor
- Links to official announcements and independent confirmation

---

## Sources
[1] MarketWatch — What AI can—and can’t—do for investors right now (Jun 11, 2025): https://www.marketwatch.com/story/what-ai-can-and-cant-do-for-investors-right-now-b413497e  
[2] Public Alpha — Product page: https://public.com/alpha  
[3] PR Newswire — Public launches Alpha powered by OpenAI’s GPT‑4 (May 17, 2023): https://www.prnewswire.com/news-releases/public-launches-alpha-powered-by-openais-gpt-4-for-investment-research-301827312.html  
[4] PR Newswire — Alpha, a new AI watchlist app, launches today (Dec 12, 2024): https://www.prnewswire.com/news-releases/alpha-a-new-ai-watchlist-app-launches-to-give-individual-investors-real-time-insights-on-the-markets-302330059.html  
[5] Public Help Center — What is Alpha?: https://help.public.com/en/articles/9354354-what-is-alpha  
[6] Stash — About (company/entities): https://www.stash.com/about/team  
[7] Stash — Stash secures $146M Series H; launches Money Coach AI (May 12, 2025): https://lp.stash.com/news/stash-secures-146m-series-h/  
[8] Stash — Co‑founders return as co‑CEOs (beta/launch context) (Oct 8, 2024): https://lp.stash.com/news/co-founders-brandon-krieg-and-ed-robinson-return-as-stash-co-ceos/  
[9] PYMNTS — Stash secures $146M to add AI to financial guidance platform (May 12, 2025): https://www.pymnts.com/news/investment-tracker/2025/stash-secures-146-million-to-add-ai-to-financial-guidance-platform/  
[10] Moomoo U.S. — Moomoo AI product page (features, privacy FAQ, disclosures): https://invest.us.moomoo.com/ai  
[11] PR Newswire (APAC) — Moomoo launches Moomoo AI in Southeast Asia (Jun 13, 2025): https://www.prnewswire.com/apac/news-releases/moomoo-launches-moomoo-ai-setting-a-new-benchmark-for-retail-investing-in-southeast-asia-302480876.html  
[12] Moomoo U.S. Learn — Moomoo AI: Invest forward and easier: https://www.moomoo.com/us/learn/detail-moomoo-ai-invest-forward-and-easier-116907-250748015  
[13] Asian Banking & Finance — Moomoo Singapore surpasses 15 million app users (AI assistant mention): https://asianbankingandfinance.net/news/moomoo-singapore-surpasses-15-million-app-users  
[14] Robinhood Newsroom — Introducing Strategies, Banking, and Cortex (Mar 27, 2025): https://newsroom.aboutrobinhood.com/introducing-strategies-banking-and-cortex/  
[15] Robinhood Help Center — Robinhood Cortex (not available yet; rolling out later in 2025): https://robinhood.com/us/en/support/articles/robinhood-cortex/  
[16] Robinhood Newsroom — Introducing Digests by Robinhood Cortex for customers in the UK (Aug 19, 2025): https://newsroom.aboutrobinhood.com/crypto/home  
[17] PYMNTS — Robinhood adds AI-powered summaries to Cortex (Aug 19, 2025): https://www.pymnts.com/news/artificial-intelligence/2025/robinhood-adds-ai-powered-summaries-to-cortex-investing-assistant/  
[18] eToro — Tori product page: https://www.etoro.com/trading/platforms/tori/  
[19] eToro Press Release — eToro leverages AI to redefine social investing (Aug 7, 2025): https://www.etoro.com/en-us/news-and-analysis/latest-news/press-release/etoro-leverages-ai-to-redefine-social-investing/  
[20] GuruFocus — eToro leverages AI to redefine social investing (coverage): https://www.gurufocus.com/news/3044786/etoro-leverages-ai-to-redefine-social-investing  
[21] Interactive Brokers — Media Relations (Dec 11, 2024; AI news summaries; advisor-only AI commentary): https://www.interactivebrokers.com/en/general/about/mediaRelations/12-11-24.php  
[22] Charles Schwab — How to use Schwab Assistant on Schwab Mobile: https://www.schwab.com/content/how-to-use-schwab-assistant-on-schwab-mobile  
[23] Fidelity — 2024 Annual Report (digital assistant/GenAI context): https://www.fidelity.com/about-fidelity/2024-annual-report  
[24] JPMorgan Chase — Wealth Plan recognition (non-genAI, pre-window): https://www.jpmorganchase.com/newsroom/press-releases/2024/jp-morgan-wealth-plan-named-number-one-new-tool-among-online-brokers  
[25] Wells Fargo Newsroom — Fargo expands with Spanish (2023 banking VA): https://newsroom.wf.com/news-releases/news-details/2023/Wells-Fargos-Virtual-Assistant-Fargo-Expands-Capabilities-with-Spanish-Language-Feature/default.aspx  
[26] Reuters — Goldman Sachs launches AI assistant firmwide (employees) (Jun 23, 2025): https://www.reuters.com/business/goldman-sachs-launches-ai-assistant-firmwide-memo-shows-2025-06-23/  
[27] CNBC — Morgan Stanley rolls out OpenAI-powered chatbot for Wall Street division (advisor/internal) (Oct 23, 2024): https://www.cnbc.com/2024/10/23/morgan-stanley-rolls-out-openai-powered-chatbot-for-wall-street-division.html  
[28] Reuters — RBC sets up new AI team in capital markets (May 21, 2025): https://www.reuters.com/world/americas/rbc-sets-up-new-ai-team-capital-markets-unit-2025-05-21/  
[29] TD Stories — Virtual AI assistant to help power TD Securities (employee-facing) (Jul 8, 2025): https://stories.td.com/ca/en/news/2025-07-08-virtual-ai-assistant-to-help-power-td-securities  
[30] Questrade App Hub — Pilot Trading (signals; not a conversational copilot): https://www.questrade.com/partner-centre/app-hub/pilot