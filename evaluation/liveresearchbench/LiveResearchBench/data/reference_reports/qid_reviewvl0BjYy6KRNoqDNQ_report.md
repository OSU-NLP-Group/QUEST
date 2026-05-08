# Evaluating LLM-Based Single- and Multi-Agent Systems (Jan 2023–Sep 16, 2025): Evolution, Benchmarks, Metrics, and Unified Pipelines

## Executive Summary
- From 2023–2025, evaluation shifted from static NLP benchmarks to interactive, tool-using agents, with concurrent growth in LLM-as-judge protocols, web/OS/code sandboxes, and multi-agent social/debate arenas [1][3][23][30][39][40]. Meta-evaluation studies show judge reliability is task- and template-sensitive, with systematic biases and adversarial vulnerabilities; hybrid pipelines (programmatic checks + calibrated judges + targeted human adjudication) are increasingly preferred [1][11][12][13][14][15][27].  
- Single-agent evaluations emphasize programmatic correctness and end-to-end success (e.g., unit tests in SWE-bench; web/OS goal completion), while multi-agent evaluations add coordination, communication, negotiation, and competition metrics, often using LLM-judges or human ratings for process quality [30][31][23][24][46][47].  
- Practical harnesses and platforms (lm-eval-harness, HELM, OpenAI Evals, SWE-bench harness, BrowserGym, HAL) standardize seeds, logging, versions, and reproducibility; dynamic benchmarking (Dynabench) and online evaluation (Online-Mind2Web) address distribution shift and freshness [33][34][36][31][38][41][37][26].  
- Actionable design principles: adopt standardized schemas/APIs for trajectories; prioritize deterministic programmatic checks where possible; use rubric-guided, bias-controlled LLM judges with multi-judge/debate and selective escalation to humans; refresh tasks dynamically; and publish prompt/judge/version/seed metadata for reproducibility [20][21][1][8][9][10][11][12][13][14][15][37][26][41][34].

## 1) Background and Evolution (2023–2025)
- 2023 introduced scalable LLM-as-judge protocols and human-preference arenas (MT-Bench, Chatbot Arena), reporting GPT‑4 judge–human agreement over 80% (comparable to inter-human) and documenting position/verbosity biases [1]. The FastChat judge pipeline operationalized swap/tie rules and rubric prompting [2].  
- Agent evaluation diversified into realistic, interactive environments: web (WebArena, VisualWebArena, Mind2Web), OS (OSWorld, WindowsAgentArena), and code (SWE-bench), shifting emphasis to end-to-end success under reproducible harnesses [23][24][25][26][39][40][30].  
- By 2024–2025, meta-evaluations critiqued LLM judges for template sensitivity, adversarial susceptibility, family/self-preference, and uneven performance on knowledge/math/code pairs (JudgeBench), motivating calibrated judges, multi-judge ensembles/debate, and selective human escalation [11][12][13][14][10][15][27].  
- Multi-agent research matured with frameworks/benchmarks for collaboration, negotiation, debate, and social intelligence (AgentBench, SOTOPIA, STSS), and large-scale MAS infrastructure demonstrated 10k-agent runs with distributed scaling [30][46][47][45].

## 2) Benchmarks and Sandboxes: Single- vs Multi-Agent

### 2.1 Single-Agent Environments

- Code (SWE-bench family)
  - Task diversity: 2,294 real GitHub issues across 12 Python repos; evaluate by running project test suites inside Docker (deterministic pass/fail) [30]. Robustness work (SWE‑bench+) strengthens tests and shows drops for some agents; harness docs expose logs, cache levels, and timeouts [32][31].  
  - Scalability: execution-based, fully automated; parallelizable via containers; token cost dominated by agent runs.  
  - Realism: real-world repos and CI-like execution; high fidelity with determinism via Docker [30][31].

- Web agents
  - WebArena: fully functional sites (e-commerce, forums, Git) with programmatic success checks; early GPT‑4 agent success 14.41% vs human 78.24% [23]. VisualWebArena adds multimodal grounding [24]. Mind2Web captures real-site tasks; Online‑Mind2Web introduces online evaluations and reports ~85% agreement between LLM-judge and human trajectory ratings, highlighting both the need and limits of LLM-based automatic scoring [25][26].  
  - Scalability: replayable snapshots and scripted evaluators support high-throughput runs; token/compute depends on agent loops; Online-Mind2Web enables continual refresh [26].  
  - Realism: dynamic web tasks; high-fidelity, open-world-like interactions (online variant) [26].

- OS agents
  - OSWorld: 369 real-computer tasks across apps and OS workflows; humans ~72.36% vs best model ~12.24%, underscoring large gaps [39]. WindowsAgentArena runs 150+ Windows tasks and parallelizes in Azure, enabling completion of the full benchmark in ~20 minutes (wall-clock) [40].  
  - Scalability: VM/container isolation, cloud parallelization; deterministic per-task checkers [39][40].  
  - Realism: high-fidelity OS interactions (apps, files, multi-app workflows) [39][40].

Trade-offs (single-agent):
- Task diversity: code (unit-testable, narrow but rigorous), web/OS (broad skills, multi-step tool use) [23][30][39].  
- Scalability: programmatic checks scale; throughput constrained by agent steps and environment (browser/VM latency) [23][39][40].  
- Realism: web/OS more open-world; code is static but faithful to real repos; online web eval addresses staleness [26][30].

### 2.2 Multi-Agent Arenas and Frameworks

- AgentBench: eight interactive environments for decision-making, tool use, and planning; used to evaluate 27–29 LLMs; focuses on agent reasoning beyond static QA [30]. Harness integrates datasets/environments for batch evaluation [31].  
- Social intelligence: SOTOPIA provides open-ended role-play with cooperative/competitive social goals and standardized evaluator rubrics; includes human baselines and curated “hard” subset [46]. STSS introduces objective, action-level evaluation of social goals in a dynamic multi-agent sandbox and an associated language-level benchmark for lower-cost screening [47].  
- Debate/judging: ChatEval demonstrates that multi-agent debate and referee agents can improve evaluation robustness over single judges in open-ended tasks [44].  
- Standardization: BrowserGym unifies observation/action spaces and experiment management across web agent benchmarks, enabling multi-benchmark comparisons under consistent evaluation interfaces [38].

Trade-offs (multi-agent):
- Task diversity: adds coordination, communication, negotiation, competition; usually text-based, sometimes multimodal (web/GUI) [30][38][46][47].  
- Scalability: more tokens and turns per episode; evaluation frequently uses LLM judges for process quality; automation possible but costlier than single-episode grading [44][27].  
- Realism: social/debate/negotiation and role-specialized workflows approximate real collaborative settings; fidelity depends on environment (closed-world sims vs real web/OS) [46][47][38].

## 3) Metric Families, Meta-Evaluation, and Trade-Offs

### 3.1 Programmatic/string/structure metrics
- Instruction following: deterministic checks for exact/regex/format/word count (IFEval defines 25 verifiable instruction types, with strict/loose metrics) [20][21].  
- Code: execution-based unit tests and pass@k (HumanEval formalized pass@k; widely adopted for code LMs/agents) [22].  
- Agents: end-to-end success and cumulative reward in web/OS (e.g., WebArena, OSWorld) via rule-based checkers [23][39].

Strengths: high precision, transparent, scalable, and reproducible [20][22][23][39].  
Limitations: under-counts success when tasks have multiple valid solutions; limited coverage for qualitative aspects (e.g., side-effects, reasoning quality); sometimes weak tests (SWE-bench+) [32][27].

### 3.2 LLM-as-judge (pointwise/pairwise; rubric-guided)
- Protocols: MT‑Bench/Arena introduced scalable pairwise and pointwise LLM judging with swap/tie rules; GPT‑4 judge–human agreement over 80% on chat tasks; verbosity and position biases identified [1][2][3].  
- Rubric-guided judges: G‑Eval (form-filling, chain-of-thought) improved correlations to human ratings; Prometheus (open-weight evaluator) reported Pearson r≈0.897 with human scores across diverse rubrics; Auto‑J supports pairwise/pointwise with rationales; JudgeLM fine-tunes 7B–33B judges (e.g., throughput: “7B judges 5k samples in ~3 minutes on 8×A100”) [4][5][6][7].  
- Meta-evaluation: JudgeBench shows many judges struggle on objectively verifiable pairs (knowledge/math/code); template sensitivity and family/self-preference biases documented; adversarial prompts can flip outcomes; diverse judge ensembles/debate can help but are not foolproof [14][11][12][13][10][44].  
- Agent trajectories: AgentRewardBench compares 12 LLM judges to rule-based checks and human experts on 1,302 web-agent trajectories, finding rule-based metrics often under-report success and no single LLM judge dominates across criteria (success, side-effects, repetition) [27].

Strengths: scalable, expressive (can assess coherence, safety, helpfulness) [1][3][4][5].  
Limitations: bias (position, verbosity, family/self-preference), prompt/template sensitivity, adversarial vulnerability; reliability depends on judge competence; needs calibration and human backstops [8][9][10][11][12][13][14][15][27].

### 3.3 Agent-as-judge/self-critique
- Reflexion adds self-reflection guided by environment feedback to improve decision-making and tool use; Self‑RAG introduces “reflection tokens” to trigger retrieval and self-critique; Re‑ReST leverages environment feedback (e.g., tests) for reflection-enhanced self-training [16][17][18].  
- Risks: sycophancy—models agreeing with prompts/user beliefs even if false—can compromise self-critique; observed to increase with scaling/instruction-tuning; mitigated via synthetic data and reward shaping in some studies [19].  
- Trade-offs: low-latency programmatic signals (tests/checkers) are strong drivers of reliability; iterative critique increases token/latency and can propagate errors if unchecked [18][20][22].

### 3.4 Human ratings/annotations
- Chatbot Arena: live pairwise human preferences aggregated via Elo; paper reports ≥240k votes (at publication) and expert alignment analyses [3].  
- HELM Instruct details recruitment, screening tests, tolerant accuracy quality control, absolute rubrics (e.g., helpfulness), costs (~$0.76 per response for two annotations), and publishes raw runs for reproducibility [34].  
- Dataset cards document licensing and data provenance (e.g., Arena Conversations: prompts CC‑BY‑4.0; model outputs CC‑BY‑NC‑4.0) [42]; MT‑Bench human judgment set (CC‑BY‑4.0) enables judge–human agreement analysis [43].

Strengths: highest validity and nuanced judgments on hard cases [3][34].  
Limitations: slower/costly; needs IAA/adjudication; subject to noise/manipulation in open leaderboards; useful as an escalation layer and for periodic calibration [3][34].

## 4) Implementation Details, Cost, and Reproducibility

### 4.1 Evaluation harnesses and infrastructure
- lm-evaluation-harness: 60+ tasks; supports Hugging Face/vLLM/API backends; seed control and bootstrap standard errors; logging to W&B/Zeno; MIT-licensed [33].  
- HELM: versioned leaderboards with prompt-level transparency; HELM Lite documents seed policy (1 seed vs HELM Classic’s 3 seeds) and format strictness [34][35].  
- OpenAI Evals: framework for dataset- and model-graded evals; MIT-licensed; widely used reference patterns [36].  
- SWE-bench harness: Dockerized reproducible runs with logs, cache, and timeouts; deterministic pass/fail via test suites [31].  
- BrowserGym: standardized web agent interface and experiment management across benchmarks [38].  
- HAL harness: unified CLI and logging across agent benchmarks (e.g., SWE-bench, USACO, GAIA), with encrypted trace uploads and cost tracking for reproducibility [41].  
- AutoGenBench: containerized, repeatable multi-agent experiments with metrics for cost, turns, and completion time; logs conversations/artifacts per run [48].

### 4.2 Environment/tool integrations and logging
- Web/OS agents: Selenium/Playwright and VM backends, DOM/action/screenshot traces, and per-task checkers (WebArena, OSWorld, WindowsAgentArena) [23][39][40].  
- Code agents: compilers/test runners inside containers with full stdout/stderr and diffs; patched code artifacts persisted (SWE-bench) [31].  
- Multi-agent debate/judging: stores all turns, roles, and judge rationales (ChatEval protocols) [44].

### 4.3 Data provenance, licensing, and versioning
- Arena Conversations dataset: prompts CC‑BY‑4.0; model outputs CC‑BY‑NC‑4.0; PII filtering and moderation flags documented [42].  
- MT‑Bench human judgments: CC‑BY‑4.0; judge labels, conversations, and agreement computation provided [43].  
- HELM and lm-eval-harness publish versioned configs, seeds, and prompts/templates for transparency [33][35].

### 4.4 Reproducibility controls and seed management
- Seed control and bootstrap CIs available in lm-eval-harness; HELM records seed policy; SWE-bench harness specifies deterministic execution parameters [33][35][31].  
- For judges: record model/version, prompt template, order randomization, swap/tie rules, and rubric definitions; apply majority voting/multi-judge checks to reduce variance [2][1][4][5][6][14][44].

### 4.5 Human annotation workflows and reliability
- HELM Instruct: crowd recruitment with screening tests and hidden gold checks; absolute rubric scoring; cost accounting; comparison with LLM evaluators; public release of raw runs [34].  
- AgentRewardBench: expert annotations provide ground truth against which LLM judges are compared, surfacing where rules/LLMs diverge [27].  
- SOTOPIA and OSWorld report human baselines to calibrate difficulty and evaluate alignment with automated metrics [46][39].

### 4.6 Empirical evidence on scalability/cost
- Human cost: HELM Instruct ~$0.76 per response for two annotations in their setup [34].  
- Judge throughput: JudgeLM reports “7B judges 5k samples in ~3 minutes on 8×A100,” demonstrating scalable judge inference [7].  
- Benchmark throughput: WindowsAgentArena reports cloud-parallel execution completing the full suite in ~20 minutes [40].

## 5) Cross-Cutting Failure Modes and Risks
- Biases in LLM-judging: position/order effects and verbosity/length bias systematically influence pairwise outcomes; family/self-preference (“preference leakage”) can favor related models [8][9][10].  
- Template sensitivity and adversarial vulnerability: prompt template changes affect reliability; attack prompts can flip judge outcomes [11][12].  
- Judge capability limits: judges underperform on objective knowledge/math/code pairs (JudgeBench) and can be overconfident [14][13].  
- Rule-based under-/over-counting: strict programmatic checks may miss valid solutions or mis-score nuanced successes/side-effects (AgentRewardBench) [27].  
- Reproducibility drift: online environments change; mitigate with snapshots/versioned sites (or online benchmarks with documented state at eval time) [26][23].  
- Sycophancy: can compromise self-critique and agent-as-judge loops without explicit mitigations [19].

## 6) Toward Unified, Generalizable, and Trustworthy Evaluation Pipelines

### 6.1 Standardized schemas and APIs
- Define a shared, benchmark-agnostic schema for episodes: prompt/template hash, model/judge IDs and versions, seeds, tool/action traces with timestamps, environment snapshot IDs, cost tokens, and evaluation outcomes (programmatic and judge scores). Use BrowserGym-like standardized interfaces for environments to enable cross-benchmark comparability [38].  
- Adopt harness conventions (lm-eval-harness/HELM/OpenAI Evals) for config/versioning, seeds, bootstrap CIs, and structured results, with public JSON manifests and signed artifacts [33][35][36].

### 6.2 Evaluation layering and judge calibration
- Primary: deterministic, environment-grounded checks (unit tests, success rules) wherever feasible (code/web/OS) [22][23][39].  
- Secondary: rubric-guided LLM judges (G‑Eval/Prometheus/Auto‑J/JudgeLM) with bias mitigations (randomize order, swap/tie rules, anonymize models, require cites) and template/version logging; consider multi-judge/debate (ChatEval) and majority voting [4][5][6][7][1][2][44].  
- Tertiary: selective human adjudication (“Trust or Escalate”) on judge disagreements, high-risk items, or out-of-distribution cases, targeting measured human-agreement rates under coverage constraints [15][34].

### 6.3 Dynamic and continual evaluation
- Use Dynabench-style adversarial data collection to refresh tasks and stress-test failure modes; adopt online evaluation where safe/practical (Online-Mind2Web) to monitor regressions under live environments [37][26].  
- Maintain longitudinal tracks and adversarial tracks (prompt-injection/red teaming) for judges and agents; publish robustness dashboards [12][13][11].

### 6.4 Cross-environment comparability and leaderboards
- Normalize core metrics across environments (e.g., success@k, steps-to-success, cost-to-success, side-effect rate), and publish uncertainty via bootstrap CIs; HAL-style unified harnesses and BrowserGym interfaces facilitate consistent comparison [41][38].  
- Report human baselines and calibration curves (judge score vs human score) with periodic re-alignment [3][34][27].

### 6.5 Reproducibility and transparency by default
- Snapshot and version environments (Docker/VMs, site snapshots), publish seeds and prompt templates, log judge prompts/rubrics, and release traces with PII/misuse filtering consistent with dataset licenses (e.g., Arena CC‑BY vs CC‑BY‑NC for outputs) [31][39][40][42][35].  
- Provide cost/throughput telemetry and compute profiles to support fair budget comparisons; leverage cloud-parallelization safely (e.g., WindowsAgentArena) [40].

### 6.6 Open questions
- How to calibrate judges reliably on hard reasoning/math/code without heavy human ground truth (JudgeBench highlights the gap) [14]?  
- How to eliminate family/self-preference and verbosity/position biases at scale while remaining cost-effective [8][9][10]?  
- How to standardize multi-agent “process” metrics (e.g., coordination, negotiation quality) with high validity across domains [44][46][47]?  
- How to make online evaluations safe and reproducible while capturing real-world dynamics [26]?  
- How to design meta-evaluations and guarantees for judge reliability under adversarial conditions (selective judging + formal guarantees) [12][15]?

## 7) Comparative Snapshot: Single vs Multi-Agent Evaluation

- Task diversity:
  - Single-agent: strong on programmatically verifiable tasks (code/web/OS) with clear outcomes [22][23][39].  
  - Multi-agent: adds coordination/communication/negotiation/debate and process quality metrics (often judge/human-scored) [30][44][46][47].

- Scalability:
  - Single-agent: high throughput with deterministic checkers; cost dominated by agent steps [30][23].  
  - Multi-agent: higher token/latency due to interactions; scalable with containerization and parallelism but judge/human scoring may bottleneck; selective judging reduces cost [44][15][48].

- Realism:
  - Single-agent web/OS provide high-fidelity tool use and open-world dynamics (especially online web); code tasks are realistic but closed-world [23][26][39].  
  - Multi-agent simulations approximate social/organizational settings; realism depends on environment fidelity and human-in-the-loop components [46][47][44].

## 8) Practical Templates (Ready-to-Implement)

- For code agents (SWE-bench):
  - Use official harness with Docker; evaluate pass/fail via tests; log diffs and runtime; augment with LLM judge for patch quality and side-effects; escalate disagreements to human reviewers on a sampled subset; publish seeds and harness version [30][31][27].

- For web/OS agents (WebArena/OSWorld/WindowsAgentArena):
  - Primary metric: programmatic success; collect steps-to-success and cost-to-success; add LLM-judge scoring for qualitative aspects; randomize judge order and anonymize models; calibrate vs periodic human annotations; parallelize runs with environment snapshots or cloud VMs; report CIs [23][39][40][27].

- For multi-agent social/debate (SOTOPIA/ChatEval):
  - Define rubric dimensions; run multi-judge or debate-based evaluation; aggregate via majority/weighted voting; log all turns and judge rationales; periodically validate with human ratings; monitor template sensitivity and adversarial susceptibility [46][44][11][12][13].

- Cross-benchmark harness baseline:
  - Use lm-eval-harness/HELM/OpenAI Evals conventions for seeds, bootstrap, config versioning; adopt BrowserGym interface for web tasks; HAL-like logging for cost and encrypted traces; open-source configs and summarize compute budgets for reproducibility [33][35][36][38][41].

---

### Sources
[1] MT‑Bench and Chatbot Arena (original paper): https://arxiv.org/abs/2306.05685  
[2] FastChat LLM Judge (prompts/pipeline): https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge  
[3] Chatbot Arena (2024): https://arxiv.org/abs/2403.04132  
[4] G‑Eval: https://arxiv.org/abs/2303.16634  
[5] Prometheus: https://arxiv.org/abs/2310.08491  
[6] Auto‑J: https://arxiv.org/abs/2310.05470  
[7] JudgeLM: https://arxiv.org/abs/2310.17631  
[8] Position Bias in LLM Pairwise Evaluation: https://arxiv.org/abs/2406.07791  
[9] Length/Verbosity Bias in LLM Judging: https://arxiv.org/abs/2407.01085  
[10] Preference Leakage in LLM‑as‑Judge: https://arxiv.org/abs/2502.01534  
[11] LLM‑as‑Judge Under Scrutiny (template sensitivity): https://arxiv.org/abs/2408.13006  
[12] Adversarial Attacks on LLM‑as‑Judge: https://arxiv.org/abs/2505.13348  
[13] LLMs Cannot Reliably Judge (Yet?): https://arxiv.org/abs/2506.09443  
[14] JudgeBench: https://arxiv.org/abs/2410.12784  
[15] Trust or Escalate (Selective Judging): https://arxiv.org/abs/2407.18370  
[16] Reflexion (verbal RL for agents): https://arxiv.org/abs/2303.11366  
[17] Self‑RAG: https://arxiv.org/abs/2310.11511  
[18] Re‑ReST: https://arxiv.org/abs/2406.01495  
[19] Sycophancy in LLMs: https://arxiv.org/abs/2310.13548  
[20] IFEval: https://arxiv.org/abs/2311.07911  
[21] IFEval checker code: https://github.com/google-research/google-research/tree/master/instruction_following_eval  
[22] HumanEval (pass@k): https://arxiv.org/abs/2107.03374  
[23] WebArena: https://arxiv.org/abs/2307.13854  
[24] VisualWebArena: https://arxiv.org/abs/2401.13649  
[25] Mind2Web: https://arxiv.org/abs/2306.06070  
[26] Online‑Mind2Web: https://arxiv.org/abs/2504.01382  
[27] AgentRewardBench: https://arxiv.org/abs/2504.08942  
[28] AgentBench (paper): https://arxiv.org/abs/2308.03688  
[29] AgentBench (GitHub): https://github.com/THUDM/AgentBench  
[30] SWE‑bench (paper): https://arxiv.org/abs/2310.06770  
[31] SWE‑bench Harness Reference: https://www.swebench.com/SWE-bench/reference/harness/  
[32] SWE‑bench+ (strengthened tests): https://arxiv.org/html/2410.06992v1  
[33] EleutherAI lm‑evaluation‑harness: https://github.com/EleutherAI/lm-evaluation-harness  
[34] HELM Instruct (crowd workflows/costs): https://crfm.stanford.edu/2024/02/18/helm-instruct.html  
[35] HELM Lite (seed policy/transparency): https://crfm.stanford.edu/2023/12/19/helm-lite.html  
[36] OpenAI Evals (framework): https://github.com/openai/evals  
[37] Dynabench: https://arxiv.org/abs/2104.14337  
[38] BrowserGym: https://arxiv.org/abs/2412.05467  
[39] OSWorld: https://arxiv.org/abs/2404.07972  
[40] WindowsAgentArena: https://arxiv.org/abs/2409.08264  
[41] HAL harness (GitHub): https://github.com/princeton-pli/hal-harness  
[42] Chatbot Arena Conversations (dataset card/licensing): https://huggingface.co/datasets/lmsys/chatbot_arena_conversations  
[43] MT‑Bench Human Judgments (dataset card): https://huggingface.co/datasets/lmsys/mt_bench_human_judgments  
[44] ChatEval (Agents as Judges via Debate): https://arxiv.org/abs/2308.07201  
[45] Very Large‑Scale MAS (10k+ agents with AgentScope): https://arxiv.org/abs/2407.17789  
[46] SOTOPIA: https://arxiv.org/abs/2310.11667  
[47] STSS (ACL 2024 Findings): https://aclanthology.org/2024.findings-acl.526/  
[48] AutoGenBench (blog): https://autogenhub.github.io/autogen/blog/2024/01/25/AutoGenBench/