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

## Source-free agent (`source_free_agent.py`, prototype)
First cut at solving from **pixels + actions only** (no engine internals). Works: camera-static
so screen==world; player centroid; **action→direction learned by probing** (ACTION1‑4 = U/D/L/R);
connected-component candidates for slot/station (the true ones are among them). Open frontier —
**color overloading**: one render color plays several roles, breaking naive color classification:
color 4 = collision-walls *and* non-colliding void (→ "all color-4 = obstacle" over-blocks, BFS
finds no path); color 5 = borders *and* the slot's exact deliver cell; color 9 = goal marker *and*
decorations. Next fix: learn walkability + the exact deliver cell by **interaction** (move and
watch if the player actually moved; wiggle around the slot until delivery fires) — build the map
from experience, not from fragile color labels.

## Files
- `ls20_solver.py` — model + planner + engine verification; `python ls20_solver.py` → solves 7/7.
- `llm_induce.py` — local-LLM induction with plan-based APD refine loop.
- `perception.py` — pixel→symbol structure extractor (camera static; recovers obstacle map/player/slot).
- `source_free_agent.py` — pixels-only agent prototype; characterizes the color-overloading frontier.
- `SLIDES_zh.md` — Chinese presentation outline.
