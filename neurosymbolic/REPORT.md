# Neuro-symbolic ARC-AGI-3: a proof-of-concept on `ls20`

A self-contained study: instead of an LLM directly choosing actions (the Arcgentica
harness) or training a policy end-to-end, the LLM **induces the game's dynamics as
executable code**, a **classical planner** does the search, and the induced model is
**verified symbolically against the real engine** (à la ABPR's program-as-hypothesis +
algorithmic debugging, arXiv:2603.20334, here adapted to an *interactive* ARC game).

Everything runs **locally and for free** (engine offline; LLM = local `gpt-oss-120b` via Ollama).

## TL;DR results
- **Symbolic planning solves 7/7 `ls20` levels** with the (verified) dynamics model
  — i.e. `ls20` is *fully* solvable this way.
- A **free local LLM induces a correct model for level 0** from observed transitions
  (auto-discovering even the win condition), refined by a plan-based verify loop.
- **Planning is trivial once the model is right** (L0: 90 search nodes). The hard parts are
  (1) inducing the dynamics-as-code and (2) perceiving symbolic state from pixels.

## What `ls20` actually is (reverse-engineered from the obfuscated env)
A **configuration-delivery puzzle**, *not* maze navigation:
- Player moves U/D/L/R, **5 px per step**; walls block.
- Player carries an object with **(shape, color, rotation)**.
- **Stations** cycle one attribute on contact: `gsu`→shape, `gic`→color, `bgt`→rotation.
- **Goal slots** each require a specific `(shape,color,rotation)`; deliver the matching object
  onto a slot to solve it; solve all slots → next level; the **last level → `GameState.WIN`**.
- **Energy** (yellow bar) −1 per action, refills (`iri`) reset to max; 0 energy costs a **life**
  (red squares, 3) and respawns (resetting position/config/energy **and unsolving slots**).
- Deterministic. The 7 levels form a curriculum (L0 rotation-only … L5 has 2 slots).

> The pixels are a **non-trivial render** of this small state (5px steps + a multi-cell carried
> sprite redraw), which is why naive pixel diffs are misleading — and why perception is the real wall.

## The pipeline (`ls20_solver.py`, `llm_induce.py`)
1. **State/layout** read from the engine (`extract_layout`, `init_state`) — bypasses pixel perception for local dev.
2. **Model as code** (`model_step`): ~40 lines predicting the next symbolic state.
3. **Planner** (`plan`): sequential sub-goal BFS (solve one slot at a time), pruning states that run out of energy.
4. **Verify**: replay the plan on the real engine; a win = level advanced or `GameState.WIN`.
5. **Induce** (`llm_induce.py`): the LLM writes `model_step` from observed transitions + a win
   observation; we verify by planning + engine-replay and feed back the **first diverging step**.

## Bugs/insights found by debugging (each confirmed vs the engine)
| Symptom | Root cause | Fix |
|---|---|---|
| Energy model regressed 4→3/7 | engine tests a **5×5 box** at the target, not the exact cell (refills sit at offsets) | box-containment |
| L6 "no win" despite correct replay | **last level win = `GameState.WIN`**, not a level-index increment | check WIN state |
| Deliver fired wrongly | delivery needs player **exactly** on the slot (`nje`), while *reject* is box-based | exact-position deliver |
| L5 unsolvable even w/o energy | a **solved slot still "rejected"** the player, blocking the 2nd slot | solved slots become inert |
| L5 too big for BFS | 2 slots blow up joint search | **sub-goal decomposition** (one slot at a time) |

## Honest limitations / what's next
- The LLM induction is robust on level 0; **multi-level / energy / multi-slot induction by the
  free model is harder** (see `llm_induce.py` output) — stronger feedback or a stronger model helps.
- **Perception** (`perception.py`): ls20's **camera is static** (stays at (0,0)) → **screen coords ==
  world coords**, so there is *no* camera-tracking problem. From pixels alone we recover a
  planning-valid obstacle map (impassable = color 4, covering 103/107 collision sprites), the
  player (orange, ~3px), and goal-slot markers (blue). What pixels do NOT give: a station's
  *type* and the carried/required *configs* — those are **semantics learned by INTERACTION**
  (press a station, watch what changes). Perception(structure) + interaction(semantics) + planner
  = a source-free agent (the concrete next milestone).
- The free-LLM induction reaches 3/7 (see `llm_induce.py`); stronger model/feedback closes the gap.
- Energy/lives respawn is modeled enough to plan no-death solutions; full lives modeling is future work.

## Source-free agent (`source_free_agent.py`) — ✅ solves L0–L4 from pixels only (5/7)
Solves `ls20` levels 0 **through 4** **end-to-end from pixels + actions only**, *chained* (you reach
each level only by winning the previous — `RESET` returns to L0, exactly as a real agent progresses).
Every decision uses only the rendered grid + the environment's win/lose feedback; no engine internals
are read for planning. With engine state the planner does 7/7 (`ls20_solver.py`); **5/7 is how far
pixels-only reaches** (L5 = two slots, L6 = last level — see below). The per-level difficulty climbs:
L0 = 1 rotation cycle, L1 = 3 rotation cycles, L2 = colour + rotation, L3, L4 = shape + colour +
rotation — each delivered through energy-budgeted routing. Insights that got here, after naive
color-based perception failed:

- **Color overloading defeats naive perception.** One render color plays several roles — color 4 =
  collision-walls *and* non-colliding void; color 5 = borders *and* the deliver cell; color 9 = the
  goal marker *and* decorations. So you cannot label walkability/goals by color.
- **Dead-reckon position by the *learned action direction*, not the orange centroid.** The player is
  the orange carried sprite, whose shape *rotates at stations*, so its centroid jumps on a config
  cycle. Instead we learn each action's direction once (probe in free space) and track position by
  "did the blob change at all?" (moved 5 px in that known direction) vs "identical" (blocked). This
  is immune to the redraw — the exact wall the earlier prototype hit.
- **A station can be pixel-invisible; the win is the oracle.** L0's rotation station cycles rot 3→0,
  but for this shape rot-3 and rot-0 render to the *same* pixels — the config change is real (the
  slot rejects rot 3, accepts rot 0) yet invisible. And an unsolved slot *blocks* movement (5×5 box
  reject) until the config matches, so it reads as a wall. So we record **blocked frontiers**, then
  brute a pass-through "station candidate" over the reachable cells and retry each blocked frontier;
  the combination that **wins** (environment feedback) reveals both the station and the slot.

Generalizing past L0 added four mechanisms, each forced by a real obstacle found by running it:
- **Prefix-chaining.** `RESET` always returns to L0, so to work on level *k* every replay is
  `RESET + (known winning solutions for L0…k-1) + path` (cached as a deepcopy checkpoint). Levels
  chain by *winning*, never by jumping — `set_level` (engine-internal) is never used.
- **Direction learning by elimination.** The four actions map bijectively to the four cardinals;
  per-level probing mis-infers when a move is wall-blocked at that level's start (caused a
  401-phantom-cell bug). Learn the map once where unobstructed; fill any single unknown by the one
  missing cardinal.
- **Color-agnostic, multi-part player tracking.** The player sprite is an orange **body** plus a
  separate **carried-object sub-blob**; the body never recolors but the carried blob does (at a
  color station). Track one part by color-continuity (prefer the last color; fall back only when it
  vanishes) and classify move-vs-block by *which lattice point the centroid is nearer* — robust to
  blob size, reshape, and recolor.
- **Station cycling.** A station cycles its attribute once per entry; to cycle *k* times, enter,
  then step off-and-back (k−1) times. L0 needs 1 rotation cycle, **L1 needs 3** — found by the win
  oracle over (candidate cell × k).

Pipeline: learn dirs once → per level, checkpoint at start via prefix → explore the freely-reachable
graph by deterministic replay (dead-reckoned, color-agnostic) → search (station candidate × cycles ×
blocked frontier), blue-marker-first, confirmed by the win signal.

Two more mechanisms unlocked L2–L4:

- **Stations read from a fixed HUD panel.** The carried config is rendered in a **corner inventory**,
  not on the player sprite. So a station = a cell whose *on-and-back* touch **persistently** changes
  the far-from-body player-palette pixels (the HUD), compared **at the same position** so the
  color-9-overloaded slot markers cancel. This finds the color/shape/rotation stations cleanly. The
  search then tries D=1/2/3 station-cycle combinations (one/two/three attributes) and the **win signal
  confirms** which delivers — L2 = color + rotation, L4 = shape + color + rotation.

- **Energy is a *hidden* variable — inferred, not perceived.** Energy is **not in the 64×64 render at
  all** (it drains with *zero* pixel change); a source-free agent cannot read it. We measure it with an
  **oracle**: spam moves until the player dies and teleports back to start — the step count *is* the
  energy (−1/step), giving EMAX. Refills (`iri`) *are* visible (a distinct **color-11** marker) and
  reset energy to EMAX. Routing is then **energy-aware Dijkstra** over `(cell, energy)` with
  **proactive refuelling** (min-steps alone arrives drained and strands the long final leg), avoiding
  other stations so the carried config isn't perturbed. Death resets the config, so staying alive is
  part of the plan. This cracked the ~68-step L2 route and the multi-station L3/L4.

**Multi-slot (L5) — infrastructure built, search not yet converging.** L5 has **two slots** with
different configs, so a single delivery doesn't win. We detect a solved slot **robustly by a drop in
the marker-component count** (solving turns a slot's requirement marker inert; player/sub-blob drift
can only *add* a component, never remove one → no false positives), and **chain sub-goals** (solve
slot 1, re-explore from that state, solve slot 2). Tractability fixes were added — index-ordered
station *combinations* (not permutations), dedup to one cyclable rep per station, `MAXK=6` (shape needs
0→5). L5 **engages correctly** (detects 2 slots, no false solves) but the per-slot D=3 + energy search
doesn't find slot 1's config within budget. The clear next step: **compute exact cycle counts instead
of brute-searching** — the slot marker *encodes the required color* (color-9→tmx 1, color-8→tmx 3) and
the rotation station is the one with no HUD change, so the color/shape/rotation cycles can be derived,
collapsing the search. L6 is single-slot D=3 (win = `GameState.WIN`) — solvable by the current agent
once it can chain past L5.

Reported honestly: **5/7 source-free end-to-end** (7/7 with engine state). The progression is itself
the result: the agent self-discovers position, stations, slots, multi-step configuration, *and* a
hidden resource — exactly the "infer the hidden rules from consequences" skill ARC-AGI-3 targets.

## Files
- `ls20_solver.py` — model + planner + engine verification; `python ls20_solver.py` → solves 7/7.
- `llm_induce.py` — local-LLM induction with plan-based APD refine loop.
- `perception.py` — pixel→symbol structure extractor (camera static; recovers obstacle map/player/slot).
- `source_free_agent.py` — pixels-only agent; **solves L0–L4 (5/7), chained, from render + win feedback**.
- `SLIDES_zh.md` — Chinese presentation outline.
