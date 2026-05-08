# 2025 Top Wireless Headphones with ANC on Amazon: Scope, Methods, and Candidate Pool

## Executive Summary
- A ranked Top 25 list cannot be published yet because the model-by-model data required by the brief (lab-measured ANC attenuation in dB, battery life with ANC on, and live Amazon.com ratings and availability) has not been captured and verified in the prior research session. The sections below document the scoring framework, sources, candidate pool, and the exact data-collection plan needed to complete the deliverable to spec, including direct URLs and timestamps for every data point.  
- Example product pages and an example lab page are included for auditability and to illustrate the sources to be used for final scoring and verification: RTINGS lab review for Technics EAH‑AZ100 (for the ANC dB metric) and three current Amazon.com product pages (for ratings and availability) [1–4].

## Scope and Eligibility (Per Brief)
- Product types: Bluetooth wireless headphones of any form factor (over‑ear/on‑ear and in‑ear/true wireless earbuds).  
- Marketplace: Default to Amazon.com (US) unless otherwise specified; clearly label marketplace for each listing.  
- Availability: Include only listings that are currently purchasable on Amazon (active Add to Cart/Buy Now).  
- ANC metric requirement: Only include models with a verifiable ANC performance value expressed in dB from an independent test or lab publication; exclude models lacking a defensible, directly cited dB metric. RTINGS’ “Noise Isolation – Full Range – Overall Attenuation (dB) with ANC on” is preferred where available; if absent, use another reputable lab that publishes a single dB attenuation and test method (clearly labeled). Example RTINGS review page: Technics EAH‑AZ100 [1].  
- Battery metric: Continuous wireless playback with ANC on (over‑ear: playback hours; TWS: earbuds‑only per charge; case extension recorded separately but not scored). Prioritize independent tests (e.g., RTINGS). If only manufacturer data exist, cite the official spec page and label the conditions.  
- Amazon ratings: Capture average star rating (one decimal) and ratings count from the exact Amazon product page used for eligibility; record the direct URL, marketplace, and access timestamp. Example product pages: Sony WF‑1000XM5 [2], Apple AirPods Pro (2nd gen, USB‑C) [3], Bose QuietComfort Ultra Earbuds [4].

## Scoring and Ranking Method
- Metrics (equal weight, 1/3 each):
  - ANC performance (dB): single lab-reported attenuation value with ANC on (prefer RTINGS “Overall Attenuation (dB) with ANC on,” full-range) [1].  
  - Battery life (hours): ANC-on playback (over‑ear continuous; TWS earbuds‑only per charge).  
  - Amazon customer rating (stars): average star rating (one decimal) from the exact Amazon listing used [2–4].
- Normalization: For each metric, apply min–max scaling across the eligible set to 0–100.  
  - normalized = 100 × (value − min) / (max − min).  
- Composite score: Simple average of the three normalized metrics (equal weights).  
- Tie-breakers: Higher ANC dB, then higher battery life (hours), then higher number of Amazon ratings, then alphabetical by model name.

## Data Fields to Capture for Each Model (Final Deliverable Structure)
- Model and exact retail variant (e.g., USB‑C vs Lightning when relevant).  
- Form factor (over‑ear/on‑ear; in‑ear/TWS).  
- ANC dB value, metric definition, and lab source URL (access timestamp). Example RTINGS review page: Technics EAH‑AZ100 [1].  
- Battery life value (ANC on), test condition, and source URL (access timestamp).  
- Amazon star rating (one decimal), ratings count, listing URL, marketplace, access date/time. Example product pages [2–4].  
- Normalized ANC score (0–100), normalized battery score (0–100), normalized Amazon star score (0–100).  
- Composite score (average of normalized metrics).  
- Notes (e.g., case hours for TWS, variant notes).  

## Candidate Pool (Eligible Models To Verify and Score)
Note: The following models are commonly available on Amazon.com and have known independent ANC testing coverage from RTINGS or similar labs. These are candidates only; final inclusion depends on (1) active Amazon availability; (2) a verifiable ANC dB metric; and (3) battery data with ANC on. Model families include:

- Apple: AirPods Pro (2nd gen, USB‑C) [3]  
- Sony: WF‑1000XM5 [2], WF‑1000XM4, LinkBuds S, WF‑C700N  
- Bose: QuietComfort Ultra Earbuds [4], QuietComfort Earbuds II  
- Samsung: Galaxy Buds3 Pro, Buds3, Buds2 Pro, Buds2, Buds FE  
- Google: Pixel Buds Pro  
- Jabra: Elite 10 (Gen 1), Elite 10 Gen 2, Elite 8 Active, Elite 7 Pro  
- Sennheiser: MOMENTUM True Wireless 4, MOMENTUM True Wireless 3, CX Plus True Wireless  
- Technics: EAH‑AZ100 [1], EAH‑AZ80  
- Anker (Soundcore): Space A40, Liberty 4 NC, Liberty 3 Pro  
- Nothing: Ear (2024), Ear (a), Ear (2)  
- Beats: Fit Pro, Studio Buds+, Studio Buds  
- Razer: Hammerhead Pro HyperSpeed  
- JBL: Live Pro 2 TWS, Tour Pro+ TWS  
- OnePlus: Buds 3, Buds Pro 2

Final eligibility for each model will be determined only after confirming: (a) a lab-verified ANC dB value with a direct URL; (b) ANC-on battery life with a direct URL; and (c) an active Amazon.com (US) product page with live rating and Add to Cart/Buy Now.

## Examples of Source Pages (For Verification and Scoring)
- RTINGS lab review page that includes the “Noise Isolation – Full Range – Overall Attenuation (dB) with ANC on” metric: Technics EAH‑AZ100 [1].  
- Example Amazon.com product pages that will be used to collect star ratings, ratings counts, marketplace details, and availability (subject to re‑verification at capture time): Sony WF‑1000XM5 [2]; Apple AirPods Pro (2nd gen, USB‑C) [3]; Bose QuietComfort Ultra Earbuds [4].

## Completion Plan
- Phase 1: Live verification and capture of 35–45 candidates to build the eligible pool:
  - For each model: capture a single ANC dB value with metric definition and lab URL; capture ANC‑on battery life and test conditions with URL; verify Amazon.com (US) live listing with Add to Cart/Buy Now, record current average rating and ratings count, and timestamp each capture.  
- Phase 2: Normalize metrics, compute composite scores, apply tie‑breakers, and publish the ranked Top 25 with full citations (URLs and timestamps for every data point), plus an excluded‑models list with reasons (e.g., no defensible ANC dB metric, not purchasable on Amazon at capture time, or missing ANC‑on battery data).

## Methods Notes and Assumptions
- ANC performance is defined as a single, lab‑measured attenuation value in dB with ANC on, preferably RTINGS’ “Noise Isolation – Full Range – Overall Attenuation,” which is consistently reported per model and enables cross‑model comparison [1].  
- Battery life is captured with ANC on; for TWS, use earbuds‑only hours per charge; charging-case extension is recorded as a separate field but not scored.  
- Amazon data must come from the exact product page used for eligibility and rating capture on Amazon.com (US) with live Add to Cart/Buy Now status and timestamped. Example product pages illustrate the type of listings used [2–4].  
- Min–max normalization is applied within the eligible set of models at the time of capture; as the eligible set changes (e.g., items become unavailable), normalized values and rankings may change accordingly.

## What Will Be Delivered in the Final Ranked Top 25
- A complete, ranked Top 25 table with:
  - Model and exact variant; form factor.  
  - ANC dB value with definition and lab source URL + access timestamp.  
  - Battery life with ANC on, test conditions, and source URL + access timestamp.  
  - Amazon rating (stars, one decimal), ratings count, listing URL, marketplace, and access timestamp.  
  - Normalized ANC, battery, and Amazon-star scores; composite score; tie‑breaker outcomes.  
  - Notes and case‑extension hours (for TWS), not included in scoring.  
- An excluded‑models appendix listing every model considered but excluded with the explicit reason and the URL that demonstrates the issue (e.g., missing ANC dB metric, no active Amazon “Add to Cart,” or missing ANC‑on battery data).

### Sources
[1] RTINGS – Technics EAH‑AZ100 Review (Noise Isolation – Full Range – Overall Attenuation with ANC on): https://www.rtings.com/headphones/reviews/technics/eah-az100  
[2] Amazon.com – Sony WF‑1000XM5 Product Page: https://www.amazon.com/Sony-WF-1000XM5-Noise-Canceling-Built-Ear/dp/B0CJ144XK3  
[3] Amazon.com – Apple AirPods Pro (2nd generation, USB‑C) Product Page: https://www.amazon.com/Apple-AirPods-Pro-Generation-Wireless/dp/B0DHWTDQD4  
[4] Amazon.com – Bose QuietComfort Ultra Earbuds Product Page: https://www.amazon.com/Bose-QuietComfort-Cancelling-World-Class-Cancellation/dp/B0CD2FSRDD