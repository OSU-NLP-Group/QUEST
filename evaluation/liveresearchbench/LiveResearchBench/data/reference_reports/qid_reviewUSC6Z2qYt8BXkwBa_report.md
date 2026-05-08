# Reinforcement Learning Algorithms for LLM Reasoning: How DAPO, GFPO, GMPO, GPPO, GPG, CPO, RPO, PPO, COPO, and GSPO Compare to GRPO

## Executive Summary
Group Relative Policy Optimization (GRPO) has become a widely used baseline for training LLMs to reason through verifiable tasks by sampling groups of responses per prompt, normalizing group rewards, and optimizing a PPO-style clipped objective—typically without a value model and often with rule-based rewards. Recent works propose modifications aimed at stability on long chain-of-thought (CoT), sample efficiency, length control, and handling degenerate groups (all-correct/all-wrong). Among the methods with public papers, DAPO adds decoupled clipping, dynamic sampling, token-level losses, and overlength handling; GFPO filters group samples to curb length inflation; GPPO alters clipping to preserve gradients beyond clip boundaries; and COPO augments GRPO with a global consistency signal to provide gradients when group variance vanishes. Other entries (GMPO, GPG, GSPO) lack primary sources in the provided materials. Comparative and reparameterization lines of work (CPO for role-playing dialogue; RPO in general RL) highlight alternative formulations, but are not directly evaluated on verifiable reasoning tasks.

The clearest cross-cutting findings supported by primary sources are:
- Group-based sampling and normalization (GRPO family) remain the dominant pattern for verifiable reasoning [1,2,3].
- Decoupled or modified clipping and sample filtering can improve stability and efficiency on long-CoT, with DAPO and GFPO reporting concrete benefits [3,4].
- Length inflation is a real and measurable issue in GRPO-style training; GFPO reports large length reductions with limited accuracy loss by filtering groups [4].
- Degenerate groups (all-correct or all-wrong) cause vanishing advantages; COPO proposes a global consistency reward to keep gradients informative [9].

All other statements below are directly linked to primary sources.

## Baseline: What GRPO Is and Why It Matters
- Core idea. GRPO removes the value model (critic), samples multiple responses per prompt, normalizes rewards within each group to compute relative advantages, and applies a PPO-style clipped objective (optionally with a KL penalty to a reference). This is designed for verifier-based (rule-based) rewards in reasoning tasks, avoiding a learned reward model [1,2,3].
- Origin and usage. GRPO was introduced and analyzed in the context of mathematical reasoning (DeepSeekMath), then used at scale in DeepSeek-R1’s reinforcement learning stage for reasoning [1,2].

Key references:
- DeepSeekMath introduces GRPO and its group-normalized advantage and clipped objective for math reasoning [1].
- DeepSeek-R1 describes using GRPO at scale with verifiable rewards for accuracy/formatting [2].
- DAPO’s preliminaries restate the GRPO objective and advantage normalization (equations) and position DAPO as an extension [3].

## Algorithm Mini-Reviews (each versus GRPO)

### DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization [3]
1) Motivation and core idea
- Addressed issues include entropy collapse, vanishing gradients when groups are all-correct/all-wrong, instability from overlong truncation, and scale imbalance in sample-level losses on long-CoT. DAPO proposes four techniques: Clip-Higher (decoupled upper/lower clip), Dynamic Sampling (oversample-and-filter to avoid degenerate groups), Token-level policy-gradient loss (instead of sample-level reduction), and Overlong reward shaping [3].

2) Key takeaways and claimed contributions
- Reports a large-scale open RL system that reaches 50 on AIME-2024 with Qwen2.5-32B, surpassing DeepSeek-R1-Zero-Qwen-32B’s 47 with half the RL steps; presents ablations for each component and releases code/data pointers [3].

3) Differences vs. GRPO
- Objective: Keeps GRPO’s clipped surrogate but switches to token-level loss aggregation [3].
- Clipping: Decouples upper/lower clipping bounds; “Clip-Higher” uses a higher upper-clip ε than lower-clip ε [3].
- Group sampling: Dynamic sampling filters out all-correct/all-wrong groups to avoid zero-variance/vanishing-advantage batches [3].
- Preference formulation: Uses rule-based/verifiable rewards, not pairwise preferences [3].
- Gradient bias/variance: Mitigates zero-variance groups; stabilizes gradients via dynamic sampling [3].
- On-/off-policy: Same on-policy framework (old-policy rollouts) as GRPO; no replay [3].
- Reward model: Rule-based verifiers for accuracy/formatting; no learned RM [3].
- KL/entropy control: Argues that an explicit KL penalty is unnecessary for long-CoT reasoning; removes KL penalty in their setup [3].
- Decoupled clipping: Yes (central innovation) [3].
- Reparameterization: No [3].

4) Empirical setup
- Models: Qwen2.5-32B base [3].
- Training: verl framework; token-level loss; group sampling with filtering; reported clip ε values and use of asymmetric clip (e.g., εlow ≈ 0.2, εhigh ≈ 0.28); maximum generation lengths set to handle long-CoT; other training hyperparameters are enumerated in Section 4.1 of the paper [3].
- Compute: GPU types/counts/hours not explicitly quantified; not reported [3].
- Data/rewards: DAPO-Math-17K with integerized answers for reliable checking; verifier-based rewards [3].
- Evaluation: AIME-2024 (avg@32); ablations showing incremental gains from each technique [3].
- Ablations: Contributions of Overlong handling, Clip-Higher, Token-level loss, and Dynamic Sampling are reported [3].

5) Limitations/failure modes
- Increased sampling cost from filtering groups; potential latency impacts in synchronized systems [3].
- Removing KL may risk distribution drift in non-reasoning domains; no safety constraints studied [3].

### GFPO — Group Filtered Policy Optimization [4]
1) Motivation and core idea
- GRPO-trained models tend to inflate response length. GFPO increases group size per prompt (e.g., G=16–24) and filters responses, retaining top-k by shortest length or token efficiency (reward per token), so optimization emphasizes efficient chains without length bloat [4].

2) Key takeaways and claimed contributions
- Reports substantial length reductions while preserving accuracy on the Phi-4-reasoning family; introduces an Adaptive-Difficulty variant that adjusts k based on problem difficulty [4].

3) Differences vs. GRPO
- Objective: Uses GRPO/DAPO-style clipped objective; applies a selection mask so unretained samples contribute zero advantage [4].
- Clipping: Inherits the parent GRPO/DAPO clipping; no new math for clipping [4].
- Group sampling: Larger G; retain top-k by shortest length or token efficiency; adaptive k by difficulty [4].
- Preference: Verifier-based rewards; no pairwise preferences [4].
- Gradient bias/variance: Reduces contributions from inefficient/overlong chains; can increase variance under token-efficiency selection [4].
- On-/off-policy: On-policy as in GRPO; no replay [4].
- Reward model: Length-aware accuracy with repetition penalty (mirrors baseline) [4].
- KL/entropy control: Uses KL and entropy regularization (coefficients reported in training config) [4].
- Decoupled clipping: Not introduced; implementation follows the underlying GRPO/DAPO stack [4].
- Reparameterization: No [4].

4) Empirical setup
- Models: Phi-4-reasoning (14B) as base; compares to GRPO baseline “Phi-4-reasoning-plus” [4].
- Training: verl; 100 steps; global batch size 64 on 32×H100; Adam learning rate 1e-7 (baseline 5e-8); cosine schedule with 10-step warmup; KL and entropy coefficients given in Section 4; GRPO baseline G=8, GFPO runs with G=16 or 24 and different k settings (e.g., 6/16, 4/16; 6/24, 4/24) [4].
- Compute: GPU count/type given (32×H100); wall-clock/hours not reported [4].
- Data/rewards: Subset of Phi-4-reasoning-plus math corpus; length-aware accuracy + 5-gram repetition penalty [4].
- Evaluation: AIME-2024/2025 (32 samples), GPQA (5), Omni-MATH (1), LiveCodeBench (3); reports 46–71% length reductions with shortest-k and 70.9–84.6% with token-efficiency, with no statistically significant accuracy drop per Wilcoxon tests in their setups [4].
- Ablations: Vary G and k; shortest-k vs token-efficiency; adaptive difficulty; Pareto front analyses [4].

5) Limitations/failure modes
- Higher training-time compute due to larger G; potential accuracy trade-offs in some tasks; variance concerns under token-efficiency selection [4].

### GMPO — Geometric-Mean Policy Optimization
- Primary source: not reported in the provided materials. No arXiv or official project link was provided.
- Motivation/core idea: not reported.
- Claimed contributions: not reported.
- Differences vs. GRPO: not reported.
- Empirical setup: not reported.
- Limitations: not reported.

### GPPO — Gradient-Preserving clipping Policy Optimization [5]
1) Motivation and core idea
- Standard hard clipping discards gradients when the importance ratio exceeds the clip range; GPPO proposes “gently backpropagating” signals from clipped tokens to preserve gradients beyond the clip boundary, aiming to improve exploration and learning from negative samples, within a PPO/GRPO-like framework [5].

2) Key takeaways and claimed contributions
- Introduces GPPO and a long-CoT training pipeline (“Klear-Reasoner”), reporting AIME-2024 90.5, AIME-2025 83.2, LiveCodeBench v5 66.0, v6 58.1 as headline scores in the abstract [5].

3) Differences vs. GRPO
- Objective: A modified clipped surrogate that preserves gradients beyond the clip band; still an importance-weighted policy-gradient surrogate [5].
- Clipping: Gradient-preserving variant vs. GRPO’s hard clamp [5].
- Group sampling: Not the core innovation; can be combined with group/outcome rewards [5].
- Preference: Outcome/verifier-style reasoning rewards (as implied by the pipeline); not pairwise preferences [5].
- Gradient bias/variance: Seeks to reduce the bias introduced by hard clipping and expand gradient coverage [5].
- On-/off-policy: On-policy as in PPO/GRPO [5].
- Reward model: Reasoning/verifier tasks; specific RM details not reported in abstract [5].
- KL/entropy control: Not central; compatible with standard regularization [5].
- Decoupled clipping: Different emphasis (gradient preservation) from DAPO’s asymmetric clip [5].
- Reparameterization: No [5].

4) Empirical setup
- Models/training: Pipeline includes long-CoT SFT followed by RL; detailed architectures, hyperparameters (batch size, learning rate, clip ε, KL/entropy coefficients, group size) not reported in the abstract [5].
- Compute: Not reported [5].
- Data/rewards: Not reported in the abstract [5].
- Evaluation: Headline results include AIME-2024 90.5, AIME-2025 83.2, LiveCodeBench v5 66.0, v6 58.1; error bars and protocols not reported in the abstract [5].
- Ablations: Not reported in the abstract [5].

5) Limitations/failure modes
- Stability and variance under different clip bands or group imbalance are not characterized in the abstract; not reported [5].

### GPG — Group Policy Gradient
- Primary source: not reported in the provided materials. No arXiv or official project link was provided.
- Motivation/core idea: not reported.
- Claimed contributions: not reported.
- Differences vs. GRPO: not reported.
- Empirical setup: not reported.
- Limitations: not reported.

### CPO — Comparative Policy Optimization (for role-playing dialogue) [6]
1) Motivation and core idea
- For subjective, open-ended tasks like role-playing dialogue, scalar sample-wise rewards are noisy/ambiguous. CPO proposes comparative, trajectory-level, group-wise evaluation to derive learning signals for policy optimization [6].

2) Key takeaways and claimed contributions
- Introduces CPO and CharacterArena (contextualized multi-turn simulation with trajectory-level comparative evaluation); reports improvements on CharacterEval, CharacterBench, and CharacterArena in the abstract [6].

3) Differences vs. GRPO
- Objective: Explicitly comparative and trajectory-level rather than per-response scalar rewards; distinct from GRPO’s group-normalized scalar outcomes [6].
- Clipping: Exact clipping mechanics not specified in the abstract; not reported [6].
- Group sampling: Yes—group-wise comparative evaluation [6].
- Preference: Yes—comparative formulation [6].
- Gradient bias/variance: Aims to reduce ambiguity/noise; variance analysis not reported [6].
- On-/off-policy: Not reported [6].
- Reward model: Comparative evaluator for trajectories rather than scalar RM or verifiers [6].
- KL/entropy control: Not reported [6].
- Decoupled clipping / Reparameterization: Not reported [6].

4) Empirical setup
- Models/training: Not reported in the abstract [6].
- Compute: Not reported [6].
- Data/evaluation: Benchmarks mentioned (CharacterEval, CharacterBench, CharacterArena); numerical results not reported in the abstract [6].
- Ablations: Not reported [6].

5) Limitations/failure modes
- Domain-specific (role-playing dialogue); applicability to verifiable math/code reasoning (the GRPO setting) is not established in the abstract [6].

Related note: Pairwise/trajectory-wise PPO variants exist (e.g., P3O), arguing invariance to reward transforms for preference learning; included here as related comparative PPO literature, not as one of the 10 algorithms under review [10].

### RPO — Reparameterization Proximal Policy Optimization (general RL) [7]
1) Motivation and core idea
- Bridges reparameterization policy gradients (RPG) with PPO: a PPO-like clipped surrogate trained via reparameterized gradients, enabling efficient sample reuse, lower variance, and KL regularization [7].

2) Key takeaways and claimed contributions
- Shows improved sample efficiency on continuous control tasks by connecting PPO with RPG through a reparameterized objective [7].

3) Differences vs. GRPO
- Objective: Reparameterized PPO surrogate; not group-relative and not critic-free; distinct from GRPO’s group advantage and verifier-driven setup [7].
- Clipping: PPO-like clipped surrogate with KL regularization [7].
- Group sampling: No (standard trajectory RL), unlike GRPO’s per-prompt groups [7].
- Preference: No [7].
- Gradient bias/variance: RPG reduces gradient variance; different trade-offs from GRPO’s variance via group normalization [7].
- On-/off-policy: On-policy with sample reuse [7].
- Reward model: Standard environment rewards; not verifier-based [7].
- KL/entropy: Uses KL regularization [7].
- Decoupled clipping / Reparameterization: Reparameterization is the core technique [7].

4) Empirical setup (LLM reasoning)
- Not evaluated on LLM reasoning; all LLM-specific configurations are not reported [7].

5) Limitations/failure modes
- Applicability to sequence modeling/LLMs not tested in the paper [7].

### PPO — Proximal Policy Optimization (as used in RLHF) [8,3]
1) Motivation and core idea
- Canonical on-policy policy gradient with a clipped surrogate and KL regularization that underpins RLHF pipelines like InstructGPT: learn a reward model from human preferences, then optimize the policy against the RM with a KL penalty to a reference [8]. DAPO preliminaries restate PPO in contrast to GRPO [3].

2) Key takeaways and claimed contributions (for alignment)
- InstructGPT reported that small PPO-trained models were preferred by humans over GPT-3 175B on instruction-following tasks [8].

3) Differences vs. GRPO
- Objective: PPO uses a critic/value function and GAE; GRPO removes the critic and uses group-normalized advantages [3,8].
- Clipping: PPO’s importance-ratio clipping; GRPO uses a similar form but with group advantages and sometimes without KL [3,8].
- Group sampling: PPO typically samples one response per prompt; GRPO samples multiple per prompt [3].
- Preference: PPO uses learned reward models; GRPO often uses verifiable (rule-based) outcomes [3,8].
- Gradient bias/variance: PPO reduces variance with a value function; GRPO avoids critic bias but increases sampling [3].
- KL/entropy: PPO uses KL penalties; GRPO may use or omit KL depending on setting (DAPO omits) [3,8].
- Decoupled clipping / Reparameterization: Not in standard PPO [8].

4) Empirical setup
- InstructGPT reports full RM training and PPO hyperparameters (see paper); we do not restate them here [8].

5) Limitations/failure modes
- PPO-based RLHF is sensitive to hyperparameters and may be compute/memory-intensive due to the critic and token-level updates (discussed across alignment literature; InstructGPT provides the baseline PPO+RM pipeline) [8].

### COPO — Consistency-Aware Policy Optimization [9]
1) Motivation and core idea
- In GRPO-like RLVR, when all responses in a group have the same outcome (all-correct or all-wrong), normalized advantages vanish. COPO augments the objective with a global, within-group consistency reward and a global loss term, and blends signals via entropy-based scheduling to maintain informative gradients [9].

2) Key takeaways and claimed contributions
- Proposes consistency-aware reward/optimization; validates on math reasoning benchmarks; promises released code (per abstract) [9].

3) Differences vs. GRPO
- Objective: Adds a global consistency reward/loss in addition to GRPO’s local group-relative objective [9].
- Clipping: Not the focus; builds atop GRPO’s machinery [9].
- Group sampling: Same group sampling, but avoids vanishing gradients via the global term [9].
- Preference: Verifier outcomes augmented with a consistency signal; no pairwise preferences [9].
- Gradient bias/variance: Addresses zero-variance groups; encourages exploration/convergence via entropy blending [9].
- On-/off-policy, KL/entropy, decoupled clipping, reparameterization: Not specifically detailed in abstract [9].

4) Empirical setup
- Models/training/config/compute: Not reported in the abstract [9].
- Data/evaluation: Multiple math benchmarks mentioned; exact protocols and numbers not reported in the abstract [9].
- Ablations: Not reported in the abstract [9].

5) Limitations/failure modes
- Potential to overweight consistency in edge cases; sensitivity analyses not reported in the abstract [9].

### GSPO — Group Sequence Policy Optimization
- Primary source: not reported in the provided materials. No arXiv or official project link was provided.
- Motivation/core idea: not reported.
- Claimed contributions: not reported.
- Differences vs. GRPO: not reported.
- Empirical setup: not reported.
- Limitations: not reported.

## Synthesis: Cross-Cutting Trends, Trade-offs, and Gaps (versus GRPO)

- Grouped rollouts and verifier rewards remain central to reasoning RL. GRPO’s critic-free, group-normalized approach is widely used for verifiable domains (math/code) and scales in practice [1,2,3].
- Stabilizing long-CoT gradients and batches is a shared focus.
  - Degenerate groups lead to vanishing advantages; DAPO’s dynamic sampling filters such groups, while COPO adds a global consistency reward, both aiming to restore useful gradients [3,9].
  - Modifying clipping can help: DAPO’s asymmetric “Clip-Higher” and GPPO’s gradient-preserving clipping both target lost signal at the clip boundary, albeit differently [3,5].
- Length control is an important practical objective. GFPO shows that selecting top-k responses by shortest length or token efficiency within groups can cut response length dramatically without significant accuracy loss in its settings, addressing a known GRPO side effect (length inflation) [4].
- KL penalty usage is mixed in reasoning RL. DAPO argues that explicit KL constraints are unnecessary or even counterproductive for long-CoT reasoning and removes them; GFPO retains KL/entropy regularization; DeepSeek-R1 uses KL/formatting penalties with verifiers. The decision appears task- and setup-dependent [2,3,4].
- Comparative preferences vs. verifiable outcomes. CPO’s comparative, trajectory-level approach is tailored to subjective domains (role-playing dialogue), contrasting with the verifier-based setting in GRPO/DAPO/GFPO/COPO. This highlights a split: comparative methods for subjective alignment vs. verifier-based methods for factual/verifiable reasoning [6].
- Reparameterization is active in general RL, but not yet established for LLM reasoning. RPO shows benefits in continuous control by connecting PPO and RPG, suggesting a potential but untested path for sequence modeling [7].
- Reporting gaps remain. Several 2025-named methods (GMPO, GPG, GSPO) lack primary sources in the provided materials, and even for available papers, many crucial configuration and compute details are not reported in abstracts. Releasing full hyperparameters, compute budgets, and error bars would improve reproducibility and fair comparison.

Open problems suggested by the cited works:
- Robust, principled handling of degenerate groups without excessive sampling overhead (DAPO vs. COPO trade-offs) [3,9].
- Systematic guidelines for KL control in reasoning RL: when to remove, relax, or retain it (DAPO vs. GFPO vs. DeepSeek-R1) [2,3,4].
- Balancing length efficiency and accuracy across tasks and model sizes (GFPO’s selection strategies), and understanding variance implications of token-efficiency selection [4].
- Understanding gradient bias/variance trade-offs among asymmetric clipping (DAPO), gradient-preserving clipping (GPPO), and standard clipping (GRPO/PPO) with rigorous ablations [3,5].
- Transferability of comparative formulations (CPO) to verifiable reasoning, and vice versa [6].
- Applicability of reparameterization gradients (RPO) to sequence-level RL for LLMs, including token-level credit assignment and long-context stability [7].

## Appendix: What “vs. GRPO” Means in This Review
When contrasting algorithms to GRPO, the baseline is:
- Objective: critic-free, group-normalized advantage computed from rule-based/verifiable rewards; PPO-style clipped ratio objective; optional KL penalty to a reference model [1,2,3].
- Sampling: multiple responses per prompt (grouped rollouts) [1,2].
- Preference/reward: verifier-based, not pairwise preferences (in typical math/code settings) [1,2].
- Regularization: KL/entropy used variably; formatting rewards often included [2,3].

## Sources
[1] DeepSeekMath: https://arxiv.org/abs/2402.03300  
[2] DeepSeek-R1 Technical Report (ar5iv mirror): https://ar5iv.labs.arxiv.org/html/2501.12948  
[3] DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization (arXiv HTML v2): https://arxiv.org/html/2503.14476v2  
[4] GFPO: Group Filtered Policy Optimization (ar5iv PDF mirror): https://ar5iv.org/pdf/2508.09726  
[5] GPPO: Gradient-Preserving clipping Policy Optimization (arXiv abstract): https://arxiv.org/abs/2508.07629  
[6] Comparative Policy Optimization for Role-Playing Dialogue (arXiv abstract): https://arxiv.org/abs/2508.09074  
[7] RPO: Reparameterization Proximal Policy Optimization (arXiv abstract): https://arxiv.org/abs/2508.06214  
[8] InstructGPT (PPO-based RLHF): https://arxiv.org/abs/2203.02155  
[9] COPO: Consistency-Aware Policy Optimization (arXiv abstract): https://arxiv.org/abs/2508.04138  
[10] P3O: Pairwise Proximal Policy Optimization (related comparative PPO): https://web3.arxiv.org/abs/2310.00212