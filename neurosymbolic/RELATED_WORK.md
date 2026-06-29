# Related work: interactive / exploratory reasoning & perceiving state from the environment

Positioning our **source-free, modality-agnostic neuro-symbolic agent** (discovers a game's hidden
rules from raw pixels + win/lose feedback, then plans + verifies) against prior work. The throughline
the user cares about: **reasoning that requires *acting on* the environment to gather the information
needed to reason** — and recovering state from raw observations rather than a privileged API.

---

## A. Interactive / exploratory reasoning — act to discover, then infer (closest to us)
The core loop we share: *form a hypothesis → act to gather evidence → test against feedback → refine.*
- **Active Reasoning in an Open-World Environment** (arXiv:2311.02018). Frames *active* abductive
  reasoning: the agent must **explore to gather data** before it can answer, vs. classic reasoning over
  given premises. This is exactly our stance — ls20/ARC-AGI-3 give no premises; you must act to learn them.
- **VisEscape** (arXiv:2503.14427). Exploration-driven decision-making in visual escape rooms as a
  **hypothesis formulate-and-test cycle**: hypothesize from memory → act → analyze outcome → keep/reject.
  Mirrors our probe→observe→classify loop (action basis, object roles, hidden energy).
- **ADAM: An Embodied Causal Agent in Open-World Environments** (arXiv:2410.22194). Couples **causal
  discovery with embodied exploration** in Minecraft, interpretable and lifelong. Closest "discover the
  rules by intervening" precedent; ours specializes it to source-free pixel games + symbolic planning.
- **EXPLORER: Exploration-guided Reasoning for Textual RL** (arXiv:2403.10692). Learns symbolic rules
  (with exceptions, non-monotonic) **online** in partially observable worlds — same "induce rules while
  exploring" idea, in text rather than pixels.
- **Virtuous Machines: Towards Artificial General Science** (arXiv:2508.13421). Autonomous
  hypothesis→experiment→update loops (the science-as-search framing behind active reasoning).

## B. Learning *action models* / one's own body by acting (our probe operators)
What our `learn_action_basis`, `find_agent` (agent-by-motion), and `interaction_role` operators do has
deep roots:
- **Learning Partially Observable Deterministic Action Models** (arXiv:1401.3437). Classic: an agent in
  a new domain learns *how its actions affect the world* by acting, then plans with the learned model —
  the planning-community ancestor of our "probe each action, learn its effect."
- **Body Discovery of Embodied AI** (arXiv:2503.19941). Discovering which signals are *one's own
  effectors* — the same problem as our "find the agent = the blob that moves under a known action."
- **Agents Explore Beyond Good Actions to Improve Their Model** (arXiv:2306.03408). Epistemic
  exploration to *improve the model*, not just maximize reward — our exploration is purely model-building.

## C. Exploration as belief refinement (POMDP / active inference)
The unobserved-state half (our "energy is a hidden variable, inferred from consequences"):
- **Align While Search: Belief-Guided Exploratory Inference for World-Grounded Embodied Agents**
  (arXiv:2512.24461). In partially-observable settings the agent must *act to collect information that
  refines its belief*, not only to reach the goal — directly our hidden-resource oracle.
- **Active inference for navigation / structure learning** (arXiv:2408.05982) and **AXIOM** (below):
  epistemic, curiosity-driven exploration of hidden state.

## D. Learning to play **from pixels**: world models & active inference (the perception axis)
- **AXIOM** (arXiv:2505.24784). Bayesian **object-centric** agent that learns arcade games **from raw
  pixels in minutes** via **active inference + structure learning**, with object priors and epistemic
  exploration. The single closest prior work on *perceive-from-pixels + explore + infer dynamics* — but
  it learns a Bayesian model and is goal-prior-driven; ours is training-free and symbolic.
- **Dreamer / DreamerV2 "Mastering Atari with Discrete World Models"** (arXiv:2010.02193); **object-
  centric world models** (arXiv:2501.16443) and **OC-STORM** (arXiv:2511.06136). Learn latent/object
  world models from pixels for model-based RL — black-box and gradient-trained, the opposite of our
  interpretable, training-free route.

## E. Inducing a world model **as code** from interaction / video (our "dynamics as code")
- **Finite Automata Extraction** (arXiv:2508.11836). Learns a world model **as programs** from gameplay
  video via a symbolic grid + DSL search — closest to "dynamics as executable code."
- **Neuro-Symbolic Synergy for Interactive World Modeling** (arXiv:2602.10480); **WALL-E 2.0: World
  Alignment by NeuroSymbolic Learning** (arXiv:2504.15785, code rules align an LLM world model to the
  env, gradient-free); **World Programs** (arXiv:1912.13007).
- **Voyager** (arXiv:2305.16291). LLM writes an **executable-code skill library**, refined by **env
  feedback + self-verification** — the precedent for our "LLM induces code → run → fix" loop, but on
  textual state with a strong closed model. Our `LLMAdapter` is the hook for this per-game inducer.

## F. ARC-specific & the open/closed-model gap (why our cheap angle matters)
- **ARC-AGI-3 Tech Report** (arXiv:2603.24621): interactive ARC; frontier AI <1%, humans ~100%.
  **Symbolica** harness reaches **36.08%** on a *frontier closed* model. **ARC Prize 2025/2026** reports
  (arXiv:2601.10904) and **ARC-AGI-2** (arXiv:2505.11831).
- Open vs closed: best open weights (GLM-5.2, 77% v1 / 22.8% v2) trail closed (o3 ~87.5% v1 at ~$17/task;
  Opus 4.5 37.6% v2). **Program-synthesis + test-time search** (GridCoder/RSPC, arXiv:2411.17708;
  execution-guided synthesis vs test-time FT, arXiv:2507.15877; NSA, arXiv:2501.04424; ABPR,
  arXiv:2603.20334) close the gap by spending compute on *search/verification*, not model size.
- Surveys for context: **Scaling Environments for LLM Agents — Learning from Interaction**
  (arXiv:2511.09586); **The Landscape of Agentic RL for LLMs** (arXiv:2509.02547); **Self-Evolving
  Agents** (arXiv:2507.21046).

## G. Where our work sits (the gap)
| Axis | Prior art | Ours |
|---|---|---|
| Reasoning mode | active/abductive reasoning, mostly **text** or symbolic API (2311.02018, ADAM, VisEscape) | active reasoning on **raw pixels**, win/lose only |
| Model learning | learned action models / belief (1401.3437, AXIOM, Dreamer) — Bayesian or gradient | **interaction-discovered roles + cycle-counted code**, training-free |
| Control | learned policy (RL) or LLM acting (Voyager) | **classical planner**, environment-verified |
| Model size | strong closed LLM (Symbolica/Voyager) or trained net | **free local model + symbolic search** |
| Hidden state | belief over latent (POMDP) | a specific **unrendered variable inferred by act-until-failure** |
| Generality | per-domain | **modality-agnostic** operators (movement + click), auto-detected |

**One-line positioning.** We occupy a point few others do: **training-free, source-free (pixels only),
modality-agnostic** active reasoning — *discover everything observable by interaction (position, action
basis, object roles, goals) and infer the unobservable (a hidden resource) from consequences, then plan
+ verify with a free model.* Nearest neighbors: **AXIOM** (pixels + active exploration, but
Bayesian/learned), **ADAM/Active-Reasoning** (discover rules by acting, but text/Minecraft & strong
models), **Voyager** (code + verify loop, but textual state + closed model).

### Reading shortlist (interactive-reasoning first)
1. Active Reasoning in an Open-World Environment — arXiv:2311.02018
2. ADAM: Embodied Causal Agent — arXiv:2410.22194
3. VisEscape — arXiv:2503.14427
4. AXIOM (pixels + active inference) — arXiv:2505.24784
5. Learning Partially Observable Deterministic Action Models — arXiv:1401.3437
6. Voyager — arXiv:2305.16291
7. Finite Automata Extraction (world model as programs) — arXiv:2508.11836
8. ARC-AGI-3 Tech Report — arXiv:2603.24621
