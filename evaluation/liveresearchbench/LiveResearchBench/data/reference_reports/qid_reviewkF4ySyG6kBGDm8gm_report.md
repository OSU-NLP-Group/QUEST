# Routing for Large Language Models (2022–2025): Mixture-of-Experts and Cross-Model Dispatch

## Overview

From 2022–2025, routing for large language models (LLMs) matured along two intertwined fronts: (i) intra-model expert routing within Mixture-of-Experts (MoE) architectures, where token- or layer-level gate functions conditionally activate subsets of experts under capacity and load-balancing constraints; and (ii) inter-model routing, which allocates queries across model families, sizes, adaptation strategies, tools, and decoding policies to optimize utility under cost/latency budgets. Across both fronts, the research coalesces around a unifying decision-theoretic principle: learn an input- (and context-) conditioned dispatch policy that maximizes expected task utility subject to compute, memory, communication, and latency constraints, with explicit mechanisms for stability, calibration, and post-training optimization. This review synthesizes algorithms, systems, training paradigms, empirical trade-offs, deployment patterns, and emerging unifying principles as reported in top-venue publications from 2022–2025. [1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18]

## Intra-Model Expert Routing in MoE LMs

### Routing rules, capacity, and load balancing

Work in this period largely prioritized sparse, hard routing at the token level with per-layer gates, building on Top-k/Top-2 and Switch-style dispatch, but introducing new capacity control and balancing mechanisms. GLaM’s large-scale MoE demonstrated that sparse activation can match or outperform dense LMs at lower training/inference FLOPs, establishing the central cost–quality motivation for conditional computation in language modeling. [2] Expert-Choice routing reframed dispatch from token→expert to expert→token selection with fixed per-expert quotas, improving convergence and fine-tuning stability while reducing pathological load imbalance relative to Top-k/Switch baselines. [1] Systems and serving studies identified persistent inference-time inefficiencies—hotspots, communication overhead, and memory residency—and proposed dynamic gating, expert buffering across CPU/GPU, and improved batching/load balancing to increase throughput under strict latency constraints. [6] DeepSpeed-MoE complemented these findings with an end-to-end training/serving stack that materially reduced cost and latency for large-scale MoEs, underscoring the necessity of co-designing routing with communication and placement. [3]

### Stability, regularization, and parameterization of routers

Routing stability emerged as a first-order challenge, particularly expert collapse and brittle training dynamics. HyperRouter proposed generating router parameters via a fixed hypernetwork and trainable embeddings, striking a balance between frozen routers (stable but suboptimal) and fully trained routers (unstable), yielding better efficiency and task performance across language modeling and downstream tasks. [4] Expert-Choice’s fixed capacity buckets likewise acted as an implicit regularizer that mitigates overload and collapse, stabilizing training and improving fine-tuning. [1]

### Training- vs. inference-time policies and post-training optimization

Beyond training-time routing, MoE work incorporated inference-time optimization and post-training model reduction. NeurIPS 2024 characterized inference-time routing pathologies and presented practical policies (dynamic gating, buffering) to maintain utilization and latency under variable loads. [6] A theory-guided pruning method for fine-tuned MoEs provided accuracy-preserving expert pruning rules—reducing active capacity while maintaining quality—to improve deployment efficiency. [5]

## Cross-Model Routing: Portfolios, Cascades, Tools, and Controllers

### Cost- and quality-aware query dispatch across models

Quality- and cost-aware query routers learned to assign requests to smaller vs. larger LLMs, exposing tunable operating points that reduce expensive model usage without sacrificing accuracy. Hybrid LLM trained a router to meet a user-specified target quality, cutting calls to the large model by up to 40% on benchmark suites, and enabling explicit trade-off control. [8] Industry evidence from the TensorOpera Router reported multi-model routing improvements in efficiency (up to 40% query efficiency gains) and cost (up to 30% reductions) while maintaining or improving accuracy in production-like settings. [11] These results instantiate classical cascades/deferral under modern LLM portfolios, using data-driven difficulty/quality prediction to trigger escalation. [8][11][13]

### Reward- and task-informed routing

Moving beyond difficulty heuristics, reward- and task-informed routing aligned dispatch with task utility. ZOOTER distilled reward model signals into a routing function to select the “expert” LLM per query, beating the best single model on average and winning on 44% of tasks across 26 subsets, while reducing online evaluation cost relative to response-ranking approaches. [9] Task-specialized controllers also emerged: OrchestraLLM orchestrated small and large models for dialogue state tracking, achieving both higher task accuracy and >50% compute savings, illustrating domain-aware controllers in a structured application. [10]

### Tool and retrieval-related routing

Routing ideas extended to RAG/tool pipelines where multiple homogeneous or heterogeneous tools are available. Query Routing for Homogeneous Tools jointly predicted tool performance and cost to assign queries cost-effectively, offering a formalized assignment objective and empirical evidence in tool-rich environments typical of retrieval pipelines. [15]

### Token-level inter-model synergy and controller granularity

At the finer granularity of token-level dispatch across models, “An Expert is Worth One Token” introduced expert token routing that integrates external expert LLMs to strengthen a generalist model—an inter-model analogue of token-to-expert MoE that demonstrates synergy by routing subproblems at token resolution. [14]

## Adaptive Computation in Decoding and Prompting Policies

Routing at decode time complements model- or expert-level dispatch. Confident Adaptive Language Modeling (CALM) formalized early-exit decoding via confidence thresholds with performance guarantees, delivering up to 3× speedups while maintaining task quality under explicit confidence criteria, and highlighting uncertainty estimation as a compute-allocation signal. [7] Prompt-level and token-level control, as in expert token routing, provided an additional lever to direct computation selectively across external experts, connecting prompting to conditional computation. [14]

## How Routers Are Trained: Supervision, Rewards, and Consistent Surrogates

Supervised difficulty/quality prediction underpins many cross-model routers (e.g., Hybrid LLM), while reward-guided distillation offers a scalable alternative to online evaluation (ZOOTER). [8][9] The learning-to-defer literature furnished consistent, cost-sensitive objectives for single- and multi-expert deferral: post-hoc estimators addressed underfitting and weak dependency on base predictions in cost-aware deferral, and two-stage learning yielded H-consistent/Bayes-consistent surrogate losses for multi-expert dispatch—principles directly applicable to multi-LLM routing. [13][12] In MoEs, router parameterization innovations such as hypernetworks (HyperRouter) stabilized training, and adapter mixtures (Mixture-of-LoRA-Experts) expanded routing to PEFT modules with hierarchical gating across tasks in NLP and vision-language settings. [4][18]

## Empirical and Theoretical Trade-offs

- Utility vs. compute/latency/cost. Sparse MoEs (GLaM, Expert-Choice) and improved MoE serving (DeepSpeed-MoE; NeurIPS 2024 systems) established that conditional computation can preserve or improve quality with reduced FLOPs and better throughput, provided load-balancing and communication are engineered jointly with the routing policy. [2][3][6] Cross-model routers reported substantial reductions in expensive model usage and overall cost without quality loss, with explicit control over operating points. [8][11]
- Robustness and stability. Router collapse and imbalance in MoEs were mitigated by capacity quotas and hypernetwork parameterizations; inference-time strategies addressed burstiness and hotspots. [1][4][6] Reward-informed inter-model routing improved robustness across diverse task subsets by aligning dispatch with utility estimates rather than static heuristics. [9]
- Calibration and abstention. CALM offered compute savings with confidence-based early exit, while calibration surveys and methods showed how to produce more trustworthy uncertainty for routing, including in black-box settings where only generations are available. [7][16][17]
- Post-training optimization. Theory-guided expert pruning preserved fine-tuned accuracy while shrinking active capacity for inference. [5]

## Data Characteristics, Distribution Shift, and Calibration

Routing decisions depend on uncertainty under distribution shift. CALM’s confidence thresholds, the NAACL 2024 survey on confidence estimation and calibration, and practical black-box calibration with “generations only” provide tools to diagnose and improve uncertainty estimates that gate computation and escalation, a prerequisite for robust cascades and token-level early exit. [7][17][16] Reward-model supervision for routing (ZOOTER) leverages signals correlated with downstream utility, which can be more stable across datasets than raw accuracy proxies, particularly when routing across heterogeneous model families. [9]

## Benchmarks, Workloads, and Domains

- Language modeling and few-shot/general NLP: GLaM evaluated sparse MoE scaling against dense baselines across standard language modeling and few-shot tasks, emphasizing efficiency at scale. [2]
- Multi-domain portfolio routing: Hybrid LLM assessed routing across diverse evaluation sets to quantify reductions in large-model usage under quality constraints. [8]
- Dialogue state tracking: OrchestraLLM targeted a structured dialogue domain, reporting improved accuracy with >50% compute savings through SLM/LLM orchestration. [10]
- Tool- and RAG-like settings: Query Routing for Homogeneous Tools formalized cost-aware assignment where tools (retrievers/processors) differ in cost and performance. [15]
- Production-style multi-model services: Industry track results documented cost and latency impacts under real workload conditions for multi-model routing. [11]
- MoE LM training/inference at scale: DeepSpeed-MoE and NeurIPS 2024 MoE inference work used large-scale language modeling and serving scenarios to quantify throughput, memory, and communication trade-offs. [3][6]

## Deployment Patterns

Three deployment patterns recurred:

- Cascades with escalation: Start with a small or cheap model and escalate when predicted error exceeds a cost threshold, formalized via cost-sensitive deferral or target-quality routing. [8][13][12]
- Portfolios with specialization: Route to the most competent model/expert for the input, trained via reward distillation or task-aware supervision, sometimes at token granularity. [9][14]
- Tool-using modular agents: Assign queries to tools or submodules based on predicted performance and cost; integrate routing with retrieval/tool orchestration. [15][11]

Across all patterns, systems co-design—pre-placing hot models/experts, batching under capacity, buffering across CPU/GPU, and monitoring load balance—is critical for realizing theoretical savings in practice. [3][6][3]

## Unifying Principles Connecting MoE and Cross-Model Routing

A common constrained optimization view unifies MoE expert routing and cross-model dispatch. Both solve: choose a subset of compute “experts” (experts within a layer, or external models/tools/decoders) to maximize expected utility under per-step budgets (expert capacity, device memory/bandwidth, money/latency), with learned policies trained from supervision, uncertainty, or reward surrogates. Expert-Choice’s fixed buckets mirror service quotas in multi-model routing; quality-target routers expose operating points akin to capacity knobs; reward-distilled dispatch is the inter-model analogue of gating optimized for downstream utility. [1][8][9] Stability and calibration are shared concerns: load balancing and regularization in MoEs correspond to portfolio balance and confidence-aware deferral in cross-model systems. [1][4][7][13][12] Finally, token-level inter-model routing (“expert tokens”) demonstrates that granularity of specialization—not the boundary of a single network—defines where conditional computation is most effective. [14]

## Open Questions and Research Gaps

- Learning-to-route with regret guarantees: While deferral work provides consistency and principled objectives, regret-optimal bandit/RL formulations tailored to multi-LLM routing (with nonstationary costs and QoS constraints) are rare in 2022–2025 top-venue publications, suggesting an opportunity for theory and practice to converge. [12][13][8]
- Jointly routing adaptation methods: Beyond MoE over LoRA adapters (MoLE), few works explicitly compare routing among adaptation regimes (ICL vs. PEFT vs. full fine-tuning vs. RLHF) as first-class arms under shared budgets; extending portfolio routing to these choices is a promising direction. [18]
- Retrieval policy depth and cost: Tool routing formalizes cost-aware assignment, but adaptive retrieval depth/latency-quality control in RAG with guarantees remains comparatively underexplored in the surveyed venues. [15]
- Robustness under persistent shift: Calibrated uncertainty helps, but end-to-end routers that remain reliable under domain drift and adversarial prompt changes warrant deeper study, especially in production. [16][17][11]

## Recommendations and Best Practices

- Treat the router as a first-class, learnable component with explicit, tunable SLOs (target quality/latency), trained with cost-sensitive objectives (deferral) or reward supervision where appropriate. [8][13][12]
- Co-design routing with systems: pre-place hot experts/models, use buffering across memory tiers, monitor utilization entropy and token drop, and batch within capacity constraints to avoid hotspots. [6][3]
- Report full cost–quality frontiers, per-domain breakdowns, and load-balance diagnostics; include burstiness/shift stress tests and failover behavior. [8][6][1][4][11]
- After fine-tuning, consider theory-guided expert pruning to reduce inference costs while preserving quality. [5]
- Improve and monitor calibration; consider black-box calibration when model internals are unavailable; align compute allocation with confidence. [7][16][17]

### Sources

[1] Mixture-of-Experts with Expert-Choice Routing (NeurIPS 2022): https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html  
[2] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (ICML 2022, PMLR): https://proceedings.mlr.press/v162/du22c.html  
[3] DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale (ICML 2022, PMLR): https://proceedings.mlr.press/v162/rajbhandari22a.html  
[4] HyperRouter: Parameter-Efficient MoE Routing with a Hypernetwork (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.351/  
[5] A Provably Effective Method for Pruning Experts in Fine-tuned Sparse MoE (ICML 2024, PMLR): https://proceedings.mlr.press/v235/chowdhury24a.html  
[6] Toward Efficient Inference for Mixture of Experts (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html  
[7] Confident Adaptive Language Modeling (NeurIPS 2022): https://proceedings.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html  
[8] Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/b47d93c99fa22ac0b377578af0a1f63a-Abstract-Conference.html  
[9] Routing to the Expert: How to Make Large Language Models Work as a Team? (NAACL 2024): https://aclanthology.org/2024.naacl-long.109/  
[10] OrchestraLLM: Orchestrating Small and Large Language Models for Dialogue State Tracking (NAACL 2024): https://aclanthology.org/2024.naacl-long.79/  
[11] TensorOpera Router: Achieving Cost-Effective LLM Serving via Query Routing (EMNLP 2024 Industry): https://aclanthology.org/2024.emnlp-industry.34/  
[12] Two-Stage Learning to Defer with Multiple Experts (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html  
[13] Post-hoc Estimators for Learning to Defer to an Expert (NeurIPS 2022): https://proceedings.neurips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html  
[14] An Expert is Worth One Token: Synergizing Expert LLMs into a Generalist via Expert Token Routing (ACL 2024): https://aclanthology.org/2024.acl-long.614/  
[15] Query Routing for Homogeneous Tools (Findings of EMNLP 2024): https://aclanthology.org/2024.findings-emnlp.598/  
[16] Calibrating Large Language Models Using Their Generations Only (ACL 2024): https://aclanthology.org/2024.acl-long.824/  
[17] Confidence Estimation and Calibration in Large Language Models: A Survey (NAACL 2024): https://aclanthology.org/2024.naacl-long.366/  
[18] Mixture-of-LoRA-Experts: Adaptive MoE for Parameter-Efficient Fine-Tuning (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/ce806d8b4bf38cd92d483e5a0490d983-Abstract-Conference.html