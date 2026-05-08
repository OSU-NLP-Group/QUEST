# LLM-Based Generative Agents for Social Interaction and Societal Simulation (2023–2025): Architectures, Emergent Behaviors, and Evaluation

## Executive Summary

LLM-driven generative agents matured rapidly from 2023 to 2025 into a distinct research area spanning social simulation, cooperative/competitive games, markets, and governance. Across single- and multi-agent settings, four design axes consistently shaped emergent behavior and evaluation: (1) mind and memory architectures (episodic/semantic/working memory, reflection/RAG, planning hierarchies, explicit beliefs and theory-of-mind), (2) social and norm-governance mechanisms (roles, norms, institutions, sanctioning/reputation, voting and coordination protocols, deception mitigation), (3) environment sandboxes (text-only, simulated/virtual worlds, embodied and web settings), and (4) scale and systems (population size, time horizon, parallelism, context windows, cost).

Key patterns:
- Memory and reflection enable stable identity and multi-episode coherence, supporting believable social routines and coordination, as in the classic “Smallville” study of generative agents [Generative Agents](https://arxiv.org/abs/2304.03442)[1]. Explicit belief states and ToM scaffolds further improve coordination, credit assignment, and long-horizon performance [Theory of Mind for Multi‑Agent Collaboration](https://aclanthology.org/2023.emnlp-main.13/)[13].
- Institutions matter. Introducing explicit norm creation/evaluation/compliance modules or authoritative institutions stabilizes cooperation, reduces conflict, and improves welfare compared with purely role-prompted societies [Emergence of Social Norms — CRSEC](https://arxiv.org/abs/2403.08251)[18], [Normative Modules for Cooperation](https://arxiv.org/abs/2405.19328)[19].
- Multi-agent conversation patterns and orchestration profoundly affect outcomes and cost; combining facilitators/critics/workers (AutoGen-style) increases reliability but expands attack surface at the communication layer [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)[4], [Agent‑in‑the‑Middle](https://aclanthology.org/2025.findings-acl.349/)[27].
- Evaluation moved from static QA to interactive benchmarks (e.g., Sotopia), social games (Avalon/Werewolf), process- and welfare-oriented metrics (efficiency, fairness, sustainability), ablations, and human studies [SOTOPIA](https://arxiv.org/abs/2310.11667)[5], [GRAIL](https://camp-lab-purdue.github.io/bayesian-social-deduction/)[12], [GLEE](https://arxiv.org/abs/2410.05254)[25], [GovSim](https://arxiv.org/abs/2404.16698)[20]. Utterance-level multidimensional rewards address credit assignment in long dialogues [Sotopia‑RL](https://arxiv.org/abs/2508.03905)[7].
- At scale (>10k agents), synthesizing polarization, echo chambers, and realistic policy responses became feasible, but results underscore that LLMs are not drop-in human proxies in all domains (e.g., asset markets show muted bubbles) [AgentSociety](https://arxiv.org/abs/2502.08691)[16], [Human‑like polarization](https://arxiv.org/abs/2501.05171)[22], [LLM Trading in Experimental Asset Markets](https://arxiv.org/abs/2502.15800)[34].

Safety concerns and ethical considerations remain central: deception, collusion, communication-level attacks, evaluator overestimation, and reproducibility gaps require stronger controls, threat models, and open infrastructure [Evil Geniuses](https://arxiv.org/abs/2311.11855)[28], [Strategic Collusion of LLM Agents](https://arxiv.org/abs/2410.00031)[24], [MART](https://aclanthology.org/2024.naacl-long.107/)[39].

## Scope and Inclusion Criteria

- Focus: LLM-driven agents engaged in social interaction or societal/organizational simulation (text, speech, vision), including single- and multi-agent systems.
- Period: 2023–2025.
- Venues prioritized: NeurIPS, ICML, ICLR, CHI/CSCW, AAAI, ACL/EMNLP, TMLR/TPAMI, Science/Nature HCI; supplemented with 2025 arXiv.
- Exclusions: Purely non-LLM ABMs; single-agent automation without a social component.

## Taxonomy of Design Choices

### Mind and Memory Architectures

- Episodic/semantic/working memory and reflection:
  - A memory stream of observations/speech plus reflection to “insights” supports consistent, believable routines and planning in open-ended worlds [Generative Agents](https://arxiv.org/abs/2304.03442)[1]. Similar memory tiers (MemGPT-style context management) recur across frameworks for long-horizon interactions [MemGPT](https://www.emergentmind.com/papers/2310.08560)[30].
- RAG and planning hierarchies:
  - Tool-augmented retrieval and hierarchical planners (slow “mind” for intent, fast “mind” for macro decisions, executor for atomic actions) improve real-time coordination and latency in embodied teamwork [HLA](https://www.emergentmind.com/papers/2312.15224)[42].
- Theory-of-Mind and explicit belief states:
  - Explicit belief tracking and ToM reasoning enhance coordination and higher-order inference in cooperative games [Theory of Mind for Multi‑Agent Collaboration](https://aclanthology.org/2023.emnlp-main.13/)[13]. Bayesian factor graphs integrated with LMs enable robust social deduction, even surpassing humans in controlled settings [GRAIL](https://camp-lab-purdue.github.io/bayesian-social-deduction/)[12].
- Lifelong and history-dependent interaction:
  - Multi-episode settings reveal performance declines without robust memory; advanced memory helps but still trails human baselines on history-heavy scenarios [LIFELONG SOTOPIA](https://arxiv.org/abs/2506.12666)[8].

### Social and Norm-Governance Mechanisms

- Roles and collaboration protocols:
  - Well-specified roles and multi-agent chat patterns boost task performance but must be complemented by coordination protocols to avoid drift [CAMEL](https://arxiv.org/abs/2303.17760)[2], [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)[4].
- Norm creation and enforcement:
  - Architectures that explicitly support norm creation, evaluation, and sanctioning (e.g., CRSEC) reduce conflict and induce compliance; learning which institutions are authoritative yields more stable cooperation [Emergence of Social Norms — CRSEC](https://arxiv.org/abs/2403.08251)[18], [Normative Modules for Cooperation](https://arxiv.org/abs/2405.19328)[19].
- Deception mitigation and belief inference:
  - RL selectors that choose among diverse LLM-generated utterances reduce bias and improve deceptive-play performance in social deduction [LLM Werewolf with RL selector](https://arxiv.org/abs/2310.18940)[10].
  - External “Thinkers” for structured reasoning plus RL boost deduction and online play [Thinker‑augmented Werewolf](https://arxiv.org/abs/2402.02330)[11].

### Environment Sandboxes

- Text-only role-play and social benchmarks:
  - Open-ended social goal role-play with mixed cooperative/competitive objectives (Sotopia) shifted evaluation to interaction quality and goal completion [SOTOPIA](https://arxiv.org/abs/2310.11667)[5].
- Virtual towns and synthetic worlds:
  - Sims-like 2D towns (“Smallville”) enable daily routines, relationships, and spontaneous social events [Generative Agents](https://arxiv.org/abs/2304.03442)[1].
- Social games and negotiation:
  - Avalon/Werewolf test deception, belief inference, and coalition dynamics; multi-party negotiation benchmarks stress strategy and safety [AvalonBench](https://arxiv.org/abs/2310.05036)[9], [Werewolf Arena](https://arxiv.org/abs/2407.13943)[32], [LLM‑Deliberation](https://arxiv.org/abs/2309.17234)[15].
- Markets and economic environments:
  - Auctions, collusion, and asset markets probe specialization, welfare, and strategic manipulation [Auction Arena](https://arxiv.org/abs/2310.05746)[23], [Strategic Collusion of LLM Agents](https://arxiv.org/abs/2410.00031)[24], [GLEE](https://arxiv.org/abs/2410.05254)[25], [LLM Trading in Experimental Asset Markets](https://arxiv.org/abs/2502.15800)[34].
- Embodied and real-time settings:
  - Overcooked-style teamwork isolates collaboration, adaptation, and communication under time pressure [Collab‑Overcooked](https://arxiv.org/abs/2502.20073)[26].
- Platforms and frameworks:
  - Multi-agent orchestration and language-game APIs (AutoGen, AgentVerse, ChatArena) standardize agent composition and experiments while making protocols a crucial surface for safety [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)[4], [AgentVerse](https://arxiv.org/abs/2308.10848)[3], [ChatArena](https://github.com/Farama-Foundation/chatarena)[31].

### Scale and Systems

- Population size and time horizon:
  - Societal-scale simulations now exceed 10,000 agents and millions of interactions, enabling policy experiments and polarization studies [AgentSociety](https://arxiv.org/abs/2502.08691)[16].
- Context windows and memory:
  - Larger windows reduce short-term forgetting but do not replace structured long-term memory or belief tracking; virtual context managers (MemGPT) and retrieval remain common [MemGPT](https://www.emergentmind.com/papers/2310.08560)[30].
- Compute and cost:
  - Orchestration with multiple roles and debates increases reliability but amplifies inference cost; selective generation (RL selectors, speech-act policies) and caching help [LLM Werewolf with RL selector](https://arxiv.org/abs/2310.18940)[10], [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)[4].

## How Design Choices Shape Emergent Behaviors

### Cooperation, Coordination, and Collective Problem Solving

- Episodic memory + reflection + day planning produce believable routines and spontaneously coordinated events (e.g., party planning) in open worlds [Generative Agents](https://arxiv.org/abs/2304.03442)[1].
- Role prompts and structured dialogues improve division of labor and task success, but without explicit norm/governance modules, societies remain fragile to drift [CAMEL](https://arxiv.org/abs/2303.17760)[2], [AgentVerse](https://arxiv.org/abs/2308.10848)[3].
- Explicit belief states and ToM scaffolds enable higher-order coordination and improved success on cooperative tasks [Theory of Mind for Multi‑Agent Collaboration](https://aclanthology.org/2023.emnlp-main.13/)[13].
- In real-time tasks, hierarchical planners with separate slow/fast minds achieve better coordination and lower latency [HLA](https://www.emergentmind.com/papers/2312.15224)[42]; Collab‑Overcooked shows current gaps in adaptation and active collaboration across 10 LLMs [Collab‑Overcooked](https://arxiv.org/abs/2502.20073)[26].

### Norm Formation, Enforcement, and Institutional Dynamics

- Modular norm pipelines (creation/representation, dissemination, evaluation, compliance) reduce conflicts; integrating sanctioning/reputation mechanisms stabilizes cooperation and improves welfare [Emergence of Social Norms — CRSEC](https://arxiv.org/abs/2403.08251)[18], [Normative Modules for Cooperation](https://arxiv.org/abs/2405.19328)[19].
- Populations can spontaneously form conventions and exhibit tipping points due to committed minorities; the design of communication and incentives determines whether convergence or fragmentation occurs [Dynamics of Social Conventions](https://arxiv.org/abs/2410.08948)[21].

### Polarization, Misinformation, and Social Influence

- Large-scale agent societies reproduce homophily and echo-chamber dynamics; interventions align with social-science patterns [AgentSociety](https://arxiv.org/abs/2502.08691)[16], [Human‑like polarization](https://arxiv.org/abs/2501.05171)[22].
- Without engineered confirmation bias, agents may converge on factual consensus; injecting bias or roles can reintroduce fragmentation [Simulating opinion dynamics with networks of LLM agents](https://paperswithcode.com/paper/simulating-opinion-dynamics-with-networks-of)[41].
- Social-deduction games (Avalon/Werewolf) and hybrid frameworks (GRAIL) isolate deception and belief dynamics; Bayesian structures with LLMs enable strong deductive performance, even surpassing humans in controlled experiments [AvalonBench](https://arxiv.org/abs/2310.05036)[9], [GRAIL](https://camp-lab-purdue.github.io/bayesian-social-deduction/)[12].

### Markets, Negotiation, and Institutional/Policy Outcomes

- Auction agents show specialization and resource management, yet simple heuristics can remain competitive; multi-issue negotiations highlight arithmetic and strategic failures in weaker models [Auction Arena](https://arxiv.org/abs/2310.05746)[23], [LLM‑Deliberation](https://arxiv.org/abs/2309.17234)[15].
- LLM agents can tacitly collude by dividing markets—raising regulatory concerns [Strategic Collusion of LLM Agents](https://arxiv.org/abs/2410.00031)[24].
- LLM traders display “textbook rationality” and muted bubbles relative to humans, cautioning against naive use of LLMs as proxies for human behavior in financial domains [LLM Trading in Experimental Asset Markets](https://arxiv.org/abs/2502.15800)[34].
- Language-based economic benchmarks unify efficiency/fairness metrics and enable human vs. agent comparisons [GLEE](https://arxiv.org/abs/2410.05254)[25].

## Evaluation Practices and Metrics

- Interactive social benchmarks:
  - Sotopia family evaluates social-goal completion and safety; RL with utterance-level multidimensional rewards improves credit assignment in dialogue [SOTOPIA](https://arxiv.org/abs/2310.11667)[5], [Sotopia‑RL](https://arxiv.org/abs/2508.03905)[7]. Interactive learning yields compact models that rival GPT‑4-based agents, but LLM-based evaluators can overestimate performance [SOTOPIA‑π](https://aclanthology.org/2024.acl-long.698/)[6].
- Social games and human studies:
  - Avalon/Werewolf enable rigorous measurement of deception, belief updates, and team outcomes; hybrid Bayesian-LM systems have begun to outperform humans in controlled conditions [AvalonBench](https://arxiv.org/abs/2310.05036)[9], [GRAIL](https://camp-lab-purdue.github.io/bayesian-social-deduction/)[12].
- Welfare, sustainability, and fairness:
  - Commons management and governance simulations measure sustainability and cooperative stability; language economic environments report fairness/efficiency [GovSim](https://arxiv.org/abs/2404.16698)[20], [GLEE](https://arxiv.org/abs/2410.05254)[25].
- Process-oriented metrics:
  - Negotiation success, social goal completion, adaptability in embodied teamwork, time-to-coordination, and sanction/compliance rates are increasingly standard [LLM‑Deliberation](https://arxiv.org/abs/2309.17234)[15], [Collab‑Overcooked](https://arxiv.org/abs/2502.20073)[26].
- Ablations and robustness:
  - Memory/planning/reflection ablations show each component’s necessity for coherent agent behavior [Generative Agents](https://arxiv.org/abs/2304.03442)[1]. Communication-layer robustness and red-teaming reveal vulnerabilities [Agent‑in‑the‑Middle](https://aclanthology.org/2025.findings-acl.349/)[27], [Evil Geniuses](https://arxiv.org/abs/2311.11855)[28], [MART](https://aclanthology.org/2024.naacl-long.107/)[39].
- Reproducibility:
  - Widespread release of platforms and datasets (Sotopia, AutoGen, AgentVerse, GRAIL, Overcooked) improves comparability, yet reporting of costs, parallelism, and infrastructure remains inconsistent [SOTOPIA](https://arxiv.org/abs/2310.11667)[5], [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)[4], [AgentVerse](https://arxiv.org/abs/2308.10848)[3], [GRAIL](https://camp-lab-purdue.github.io/bayesian-social-deduction/)[12], [Collab‑Overcooked](https://arxiv.org/abs/2502.20073)[26].

## Safety, Security, and Ethics

- Communication-layer attacks compromise multi-agent systems; message interception and role-conditioned attacks expose systemic vulnerabilities [Agent‑in‑the‑Middle](https://aclanthology.org/2025.findings-acl.349/)[27], [Evil Geniuses](https://arxiv.org/abs/2311.11855)[28].
- Multi-round automated red-teaming plus safety fine-tuning raises robustness, but defense-in-depth and protocol-hardening are required, especially with orchestration-heavy frameworks [MART](https://aclanthology.org/2024.naacl-long.107/)[39].
- Tacit collusion, misinformation spread, and institutional brittleness need explicit governance modules calibrated with human oversight and transparent evaluation [Strategic Collusion of LLM Agents](https://arxiv.org/abs/2410.00031)[24], [GovSim](https://arxiv.org/abs/2404.16698)[20].
- Human proxy validity varies by domain; e.g., LLM traders’ muted bubbles vs. human exuberance highlight ecological limits [LLM Trading in Experimental Asset Markets](https://arxiv.org/abs/2502.15800)[34].

## Methodological Recommendations

- Use layered memory (episodic + semantic/reflection + belief state) with explicit ToM for multi-episode, multi-agent settings.
- Introduce minimal institutional scaffolds (authority, sanctioning, voting) to stabilize cooperation and align welfare metrics with outcomes.
- Prefer utterance-level, multidimensional rewards or credit assignment when optimizing dialogue agents; validate with human studies.
- Harden communication layers (signing, mediation, audits) and run automated red-teaming before deployment.
- Report scale and cost (agents, time horizon, tokens, context, parallelism) for reproducibility.
- Avoid assuming LLMs are accurate human proxies; triangulate with human-subject experiments.

## Verification-Ready Comparison Table (Selected Works)

| Paper (venue, year) | Agent type | LLM backend | Mind/memory | Social/norms | Communication | Environment/tools | Scale/cost | Emergent behaviors | Evaluation/metrics | Key design→outcome finding | Limits/ethics | Code/data | URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Generative Agents (UIST’23) | Multi (25+) | GPT‑3.5/4-era | Episodic stream, reflection to “insights,” day planning | Roles/relationships | Turn-based NL | 2D town sandbox | ~25 agents, multi-day | Routines, info diffusion, spontaneous coordination | Human believability, ablations | Memory+reflection+planning → believable social routines, coordination | Hallucinations, small scale | Partial | [1](https://arxiv.org/abs/2304.03442) |
| CAMEL (arXiv’23) | Multi | GPT-family, pluggable | Role inception; no long-term store | Roles/goals | Structured dialogues | Text-only tasks | Pairs/groups | Cooperation, division of labor | Task success vs single | Role prompting improves collaboration | No institutions; drift risk | Yes | [2](https://arxiv.org/abs/2303.17760) |
| AgentVerse (arXiv’23) | Multi | Pluggable | Configurable buffers | Roles, strategies | Messaging across topologies | Text + plugins | Varied | Cooperation, specialization | Task performance, qualitative | Orchestration patterns shape outcomes | Reporting inconsistency | Announced | [3](https://arxiv.org/abs/2308.10848) |
| AutoGen (COLM’24) | Multi/hybrid | Multiple | Conversation history + tools | Roles (facilitator/critic) | Multi-agent chat graphs | Code/math/QA tools | Pattern-dependent | Reliability gains via roles | Case studies | Orchestration ↑ reliability, ↑ cost | Attack surface ↑ | Yes | [4](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/) |
| SOTOPIA (NeurIPS’23) | Single/Multi | GPT‑4, others | Scenario conditioning | Scenario roles | Turn-based NL | Open-ended social goals | Broad | Coordination/competition | Social-goal completion, human vs AI | LLMs < humans on hard social tasks | Evaluator bias risks | Yes | [5](https://arxiv.org/abs/2310.11667) |
| SOTOPIA‑π (ACL’24) | Single/Multi | 7B finetuned | Interactive learning | Safety tuning | Turn-based NL | Sotopia tasks | N/A | Better social goal completion | Automatic + human eval | Smaller models match GPT‑4-based agent post-training | LLM evaluator overestimation | Yes | [6](https://aclanthology.org/2024.acl-long.698/) |
| Sotopia‑RL (arXiv’25) | Single/Multi | Various | RL credit assignment | N/A | Utterance-level rewards | Sotopia-hard/full | N/A | Improved dialogue policy | Multi-dim rewards, SOTA | Utterance-level rewards solve partial observability | Generality pending | Yes | [7](https://arxiv.org/abs/2508.03905) |
| LIFELONG SOTOPIA (arXiv’25) | Single/Multi | Various | Lifelong memory variants | N/A | Multi-episode | Long-horizon social | Multi-episode | Performance decay over time | History-dependent tasks | Advanced memory helps but < humans | Drift, memory brittleness | Yes | [8](https://arxiv.org/abs/2506.12666) |
| AvalonBench (arXiv’23) | Multi | GPT-family | ReAct-style scratchpad | Implicit norms | Game turns | Resistance: Avalon | Tournaments | Deception/deduction behaviors | Win rates, belief updates | LLMs struggle vs strong bots | Sparse supervision | Yes | [9](https://arxiv.org/abs/2310.05036) |
| Werewolf RL selector (arXiv’23) | Multi | LLM + RL selector | Candidate generation+selection | Social roles | Turn-based | Werewolf | Human sessions | Human-level strategic play | Win rates, human tests | RL selecting utterances mitigates bias | Safety trade-offs | Yes | [10](https://arxiv.org/abs/2310.18940) |
| Thinker‑Werewolf (arXiv’24) | Multi | LLM + “Thinker” | External reasoning + RL | Social roles | Turn-based | Werewolf; 18.8k human sess. | Large dataset | Better deduction | Online play metrics | System‑2 aug + RL improves play | Generalization | Yes | [11](https://arxiv.org/abs/2402.02330) |
| GRAIL (project, 2025) | Multi | LM + Bayesian factor graph | Explicit beliefs | Deception modeling | Structured + NL | Avalon-like | Controlled lab | Defeats humans (67% win) | Human-vs-agent | Factor-graph beliefs + LM → strong deduction | Ecological validity | Yes | [12](https://camp-lab-purdue.github.io/bayesian-social-deduction/) |
| ToM for Collaboration (EMNLP’23) | Multi | GPT-family | Explicit belief state | Cooperative roles | Turn-based | Text game | N/A | Higher-order ToM | Task success | Belief-state scaffolds ↑ coordination | Scope of tasks | Yes | [13](https://aclanthology.org/2023.emnlp-main.13/) |
| LLM‑Coordination (arXiv’23) | Multi | GPT‑4‑turbo, others | Prompted reasoning | None | Messaging | Pure coordination games | N/A | Mixed coordination | Success rates | Commonsense helps; ToM remains weak | Limited domains | Yes | [14](https://arxiv.org/abs/2310.03903) |
| LLM‑Deliberation (NeurIPS D&B’24) | Multi | GPT‑4 > others | Deliberative planning | Negotiation norms | Multi-party | Multi-issue negotiation | N/A | Failures in arithmetic/BATNA | Payoff metrics | Model capability affects safe negotiation | Adversarial players | Yes | [15](https://arxiv.org/abs/2309.17234) |
| AgentSociety (arXiv’25) | Massive multi | Various | Population-scale memory | Institutions/policies | Broadcast/networks | Realistic societal sim | >10k agents; ~5M inter. | Polarization, policy responses | Social-science metrics | Aligns with empirical patterns | Ethics at scale | Planned | [16](https://arxiv.org/abs/2502.08691) |
| EconAgent (arXiv’23/ACL’24 page) | Multi | LLMs for firms/households | Trend memory | Market rules | Market comms | Macro economy sim | Populations | Macro phenomena | Econ metrics | LLM decisions yield realistic macro patterns | Baseline sensitivity | Yes | [17](https://arxiv.org/html/2310.10436v3) |
| CRSEC (arXiv’24) | Multi | LLMs | Creation/Spread/Eval/Compliance | Sanctions/compliance | NL + rule hooks | Smallville-like | Town-scale | Reduced conflicts | Human eval, conflict rates | Norm modules stabilize behavior | Generality | Yes | [18](https://arxiv.org/abs/2403.08251) |
| Normative Modules (arXiv’24) | Multi | LLMs | Institutional learning | Coordinated sanctioning | NL protocols | New env w/ institutions | N/A | Stable cooperation, welfare ↑ | Welfare metrics | Learning authoritative institutions helps generalize | Transfer limits | Yes | [19](https://arxiv.org/abs/2405.19328) |
| GovSim (arXiv’24) | Multi | LLMs | Moral reasoning prompts | Governance rules | Group deliberation | Commons mgmt sim | Multi-episode | Frequent collapse; universalization helps | Sustainability, welfare | Moral reasoning improves sustainability | Fragile comms | Yes | [20](https://arxiv.org/abs/2404.16698) |
| Social Conventions (arXiv’24) | Multi | LLMs | N/A | Convention dynamics | Dialogue | Populations | Varying | Spontaneous conventions; tipping points | Convergence measures | Communication/incentives set dynamics | Oversimplification | Yes | [21](https://arxiv.org/abs/2410.08948) |
| Human-like polarization (arXiv’25) | Multi | LLMs | N/A | Network homophily | Social network | Network sim | Large | Echo chambers, polarization | Polarization indices | Networked LLMs reproduce human-like polarization | External validity | Yes | [22](https://arxiv.org/abs/2501.05171) |
| Auction Arena (arXiv’23) | Multi | LLMs | Budget/goal tracking | Auction rules | Bidding messages | Auction games | Tournaments | Specialization; resource mgmt | Revenue, efficiency | LLMs exhibit strategy but not universally SOTA | Heuristics strong | Yes | [23](https://arxiv.org/abs/2310.05746) |
| Strategic Collusion (arXiv’24) | Multi | LLMs | N/A | Tacit collusion | Market messages | Cournot-like | Multi-commodity | Market division, monopoly | Prices/quantities | Agents collude without explicit instruction | Policy risk | Yes | [24](https://arxiv.org/abs/2410.00031) |
| GLEE (arXiv’24) | 2-player | LLMs | N/A | Fairness/efficiency norms | Sequential dialogue | Econ language games | Benchmark | Cooperation/competition | Efficiency/fairness, human vs AI | Standardizes econ-social eval | Narrow tasks | Yes | [25](https://arxiv.org/abs/2410.05254) |
| Collab‑Overcooked (arXiv’25) | 2-agent | 10 LLMs | Planning variants | Team norms | Real-time comms | Embodied teamwork | Benchmark | Active collaboration gaps | Process metrics | Benchmarks expose adaptation limits | Latency/cost | Yes | [26](https://arxiv.org/abs/2502.20073) |
| Agent‑in‑the‑Middle (Findings ACL’25) | Multi | Framework-agnostic | N/A | Security focus | Message interception/mutation | AutoGen/AgentVerse/others | Attack cost modest | Systemic compromise | Attack success rates | Comms layer is critical attack surface | Need defenses | Yes | [27](https://aclanthology.org/2025.findings-acl.349/) |
| Evil Geniuses (arXiv’23) | Multi | CAMEL/MetaGPT/ChatDev | N/A | Adversarial roles | Red-blue exercises | Multiple | N/A | Harmful behaviors under role attacks | Safety metrics | Multi-agent ↑ risk via roles | Governance needed | Yes | [28](https://arxiv.org/abs/2311.11855) |
| Machiavelli (ICML’23) | Single/Multi | LLMs | N/A | Ethical trade-offs | NL trajectories | 134 text worlds | Large | Reward vs ethics conflicts | Reward/ethics scores | Reveals power-seeking/deception pressures | Misuse risk | Yes | [29](https://aypan17.github.io/machiavelli/) |
| MemGPT (arXiv’23) | Single/Multi | LLMs | Virtual memory tiers | N/A | Conversational | Long-term agents | N/A | Better long-term coherence | Retention/recall | Memory mgmt crucial for long-horizon | Not social per se | Yes | [30](https://www.emergentmind.com/papers/2310.08560) |
| ChatArena (GitHub’23–’25) | Multi | Many | Configurable | None by default | Turn-based | Language games | Varied | Coordination, deception | Game-specific | Standardizes MAS experiments | Deprecated (2025) | Yes | [31](https://github.com/Farama-Foundation/chatarena) |
| Werewolf Arena (arXiv’24) | Multi | Gemini/GPT | Strategy prompts | Role dynamics | Bidding + turns | Werewolf tournament | Large | Comparative agent strengths | Win rates | Platform for deception study | Eval variance | Yes | [32](https://arxiv.org/abs/2407.13943) |

Notes: LLM backend and exact costs are often underreported; many works are framework-agnostic or evaluate multiple model families.

## Cross-Cutting Findings Linking Design to Outcomes

- Memory/reflection/explicit beliefs:
  - Enable stable identity and continuity, higher-order inference, and better coordination; improve deductive play when paired with structured inference (e.g., Bayesian graphs) [1][12][13].
- Institutions, sanctioning, and reputation:
  - Reduce conflict, stabilize cooperation, and improve welfare and sustainability; governance becomes essential as populations scale [18][19][20][16].
- Orchestration patterns:
  - Facilitate division of labor and reliability but increase cost and security risk; communication integrity is essential [4][27][28].
- Reward design and credit assignment:
  - Utterance-level, multidimensional rewards improve dialogical credit assignment; interactive learning can close gaps to large closed LMs, but human studies remain vital [6][7].
- Social structure and network effects:
  - Homophily, echo chambers, and convention dynamics emerge in LLM societies; interventions mirror social science but require caution about external validity [16][21][22][41].
- Economic behavior and safety:
  - Agents can collude or misprice strategically; they do not always mimic humans (e.g., muted bubbles) and may demand distinct policy/safety controls [23][24][25][34].

## Open Problems and Future Directions

- Robust lifelong memory and identity under distribution shift; combining memory and ToM with scalable belief inference.
- Institution design for LLM societies: composable governance modules, audits, and constitutional constraints with verifiable compliance.
- Secure multi-agent communication: message signing, mediation, provenance, and formal protocol verification.
- Standardized reporting of scale/cost, seeds, and infra; open large-scale simulators with reproducible pipelines.
- Human proxy validity: deeper human-subject comparisons and hybrid human-in-the-loop experiments, especially in high-stakes domains (policy, markets).
- Safety alignment under adversarial multi-agent dynamics: red-team/blue-team tournaments, dynamic risk assessment, and policy-aware agents.

### Sources
[1] Generative Agents: https://arxiv.org/abs/2304.03442  
[2] CAMEL: https://arxiv.org/abs/2303.17760  
[3] AgentVerse: https://arxiv.org/abs/2308.10848  
[4] AutoGen: https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/  
[5] SOTOPIA: https://arxiv.org/abs/2310.11667  
[6] SOTOPIA‑π: https://aclanthology.org/2024.acl-long.698/  
[7] Sotopia‑RL: https://arxiv.org/abs/2508.03905  
[8] LIFELONG SOTOPIA: https://arxiv.org/abs/2506.12666  
[9] AvalonBench: https://arxiv.org/abs/2310.05036  
[10] LLM Werewolf with RL selector: https://arxiv.org/abs/2310.18940  
[11] Thinker‑augmented Werewolf: https://arxiv.org/abs/2402.02330  
[12] GRAIL — Bayesian Social Deduction: https://camp-lab-purdue.github.io/bayesian-social-deduction/  
[13] Theory of Mind for Multi‑Agent Collaboration: https://aclanthology.org/2023.emnlp-main.13/  
[14] LLM‑Coordination: https://arxiv.org/abs/2310.03903  
[15] LLM‑Deliberation: https://arxiv.org/abs/2309.17234  
[16] AgentSociety: https://arxiv.org/abs/2502.08691  
[17] EconAgent: https://arxiv.org/html/2310.10436v3  
[18] Emergence of Social Norms — CRSEC: https://arxiv.org/abs/2403.08251  
[19] Normative Modules for Cooperation: https://arxiv.org/abs/2405.19328  
[20] GovSim: Cooperate or Collapse: https://arxiv.org/abs/2404.16698  
[21] Dynamics of Social Conventions in LLM Populations: https://arxiv.org/abs/2410.08948  
[22] Human‑like polarization among LLM agents: https://arxiv.org/abs/2501.05171  
[23] Auction Arena: https://arxiv.org/abs/2310.05746  
[24] Strategic Collusion of LLM Agents: https://arxiv.org/abs/2410.00031  
[25] GLEE: Language‑based Economic Environments: https://arxiv.org/abs/2410.05254  
[26] Collab‑Overcooked: https://arxiv.org/abs/2502.20073  
[27] Agent‑in‑the‑Middle — Findings of ACL 2025: https://aclanthology.org/2025.findings-acl.349/  
[28] Evil Geniuses: https://arxiv.org/abs/2311.11855  
[29] Machiavelli: https://aypan17.github.io/machiavelli/  
[30] MemGPT: https://www.emergentmind.com/papers/2310.08560  
[31] ChatArena: https://github.com/Farama-Foundation/chatarena  
[32] Werewolf Arena: https://arxiv.org/abs/2407.13943  
[33] LLM Trading in Experimental Asset Markets: https://arxiv.org/abs/2502.15800  
[34] InfoBid: https://arxiv.org/abs/2503.22726  
[35] Simulating the U.S. Senate: https://arxiv.org/abs/2406.18702  
[36] Humanoid Agents: https://arxiv.org/abs/2310.05418  
[37] MetaAgents: https://arxiv.org/abs/2310.06500  
[38] MART — NAACL 2024: https://aclanthology.org/2024.naacl-long.107/  
[39] RedAgent: https://arxiv.org/abs/2407.16667  
[40] HLA: Hierarchical Language Agent: https://www.emergentmind.com/papers/2312.15224  
[41] Simulating opinion dynamics with networks of LLM agents: https://paperswithcode.com/paper/simulating-opinion-dynamics-with-networks-of