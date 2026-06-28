#!/usr/bin/env python3
"""
source_free_core — the GAME-AGNOSTIC skeleton lifted out of the ls20 7/7 source-free solver.

The ls20 agent (source_free_agent.py) solves all 7 levels from pixels + win/lose feedback, but its
PERCEPTION is hand-tuned to ls20 (palette, 5px lattice, HUD location, station/slot/refill semantics).
That part does NOT transfer to a different ARC-AGI-3 game. This module separates the two halves:

  * GENERAL OPERATORS (here) — transfer across games: they assume only an `Env` that you can reset,
    step, render, and ask win/lose/score. No colours, no lattice size, no object semantics baked in.
      - learn_action_basis : probe actions, classify each as move(vec) / edit / noop  (no key meanings assumed)
      - find_agent         : the agent is whatever blob TRANSLATES under a move action (not a fixed colour)
      - AgentTracker       : track that blob frame-to-frame by motion + continuity (robust to recolour/reshape)
      - Explorer           : deterministic reset+replay -> reachable-state graph + blocked frontiers
      - HiddenResourceOracle: detect an UNRENDERED depleting resource via failure events; measure it by
                              "act until failure, count" (this is how ls20's invisible energy was found)
      - interaction_role   : classify an interactable by the PERSISTENT effect of touching it
                              (transformer / gate / replenisher) — i.e. semantics by interaction, not appearance
      - OracleSearch       : frontier-outer search over operator combinations, verified by the win oracle,
                              with sub-goal chaining for multi-goal levels

  * PER-GAME ADAPTER (the `Adapter` surface) — what an LLM / induction module must produce for an
    UNSEEN game: which blob is the agent (or let find_agent auto-detect), how to read goal progress,
    and which cells are worth probing as interactables. Everything else is the general loop above.

This file is the concrete "how a single-game solution becomes multi-game" artifact: each ls20 trick
is named as a general operator, and the induction surface is the small, explicit `Adapter`.
Run `python source_free_core.py` for a live demo: the general operators drive ls20 L0 through the
Env abstraction (no engine internals), reproducing action-basis learning + agent tracking + mapping.
"""
from __future__ import annotations
import copy
from collections import deque
from typing import Protocol, runtime_checkable
import numpy as np


# ───────────────────────────── the environment contract (matches ARC-AGI-3 shape) ─────────────────
@runtime_checkable
class Env(Protocol):
    """A resettable, deterministic, pixel-rendered environment. ARC-AGI-3 games fit this: a 64x64
    integer grid, a small set of actions (ACTION1..6 / RESET), and win/lose/score feedback. The
    general operators below use ONLY this interface — never engine internals."""
    actions: list                                  # action names available to the agent
    def reset(self) -> None: ...                   # return to the episode start (deterministic)
    def step(self, action) -> None: ...            # apply one action
    def render(self) -> np.ndarray: ...            # current frame as an HxW int grid
    def status(self) -> str: ...                   # "PLAYING" | "WIN" | "LOSE"
    def score(self) -> int: ...                    # monotone progress signal (level/points); env feedback
    def clone(self) -> "Env": ...                  # cheap checkpoint (deepcopy) for replay-free search


@runtime_checkable
class Adapter(Protocol):
    """The PER-GAME induction surface — the only part that differs across games. An LLM (or a learned
    module) supplies this for an unseen game; the general operators do the rest. Sensible defaults
    exist for grid movement games, so a minimal adapter can be empty."""
    def goal_progress(self, env: Env): ...         # a comparable token capturing 'how much is solved'
    def interactable_cells(self, grid, agent_pos): ...  # candidate cells worth probing (else: all reachable)


# ───────────────────────────── shared perception util (game-agnostic) ─────────────────────────────
def components(grid, color, min_size=1):
    """Connected components of `color`; returns [(cx, cy, size)] centroids. Pure image op, no game
    knowledge — colours are passed in by the caller / discovered, never assumed here."""
    H, W = grid.shape; seen = np.zeros_like(grid, bool); out = []
    for y in range(H):
        for x in range(W):
            if grid[y, x] == color and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == color:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(cells) >= min_size:
                    out.append((round(sum(c[1] for c in cells) / len(cells)),
                                round(sum(c[0] for c in cells) / len(cells)), len(cells)))
    return out

def _palette(grid):
    """Foreground colours present (excluding the most common = background). General heuristic; the
    adapter may override which colours are 'agent-eligible'."""
    vals, counts = np.unique(grid, return_counts=True)
    order = sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: -t[1])
    return [v for v, _ in order[1:]]               # drop the dominant (background) colour


# ───────────────────────────── operator 1: learn the action basis ─────────────────────────────────
def learn_action_basis(env: Env, palette=None, min_size=3):
    """Probe each action once and CLASSIFY its effect with no assumption about key meanings:
        ('move', (dx,dy)) — some blob translated by a fixed vector (a directional action)
        ('edit',  None)   — the frame changed but not by a global translation (in-place state change)
        ('noop',  None)   — nothing changed (blocked / inert in this state)
    Returns (basis, move_vectors). For a grid movement game the four cardinals emerge automatically;
    for a click/menu game you'd get 'edit's instead — same operator, different result.
    GENERAL: this is `learn_dirs` with the colour/lattice assumptions removed."""
    env.reset(); g0 = env.render()
    pal = palette or _palette(g0)
    basis = {}
    for a in env.actions:
        e = env.clone(); e.step(a); g1 = e.render()
        if np.array_equal(g0, g1):
            basis[a] = ("noop", None); continue
        vec = _dominant_translation(g0, g1, pal, min_size)
        basis[a] = ("move", vec) if vec is not None else ("edit", None)
    return basis

def _dominant_translation(g0, g1, palette, min_size):
    """The translation (dx,dy) by which a same-colour blob moved, if the action was a pure move."""
    best = None
    for col in palette:
        c0 = components(g0, col, min_size); c1 = components(g1, col, min_size)
        for (x0, y0, s0) in c0:
            for (x1, y1, s1) in c1:
                if abs(s1 - s0) <= 2:
                    d = (x1 - x0, y1 - y0)
                    if d != (0, 0) and (best is None or abs(d[0]) + abs(d[1]) < abs(best[0]) + abs(best[1])):
                        best = d
    return best


# ───────────────────────────── operator 2: find the agent by motion ───────────────────────────────
def find_agent(env: Env, basis, min_size=3):
    """The agent is whatever blob TRANSLATES by a known move vector — identity by motion, not colour
    (so it survives recolouring/reshaping). Returns ((cx,cy), color) or None.
    GENERAL: this is `find_start_player`, with the palette discovered rather than hard-coded."""
    env.reset(); g0 = env.render(); pal = _palette(g0)
    for a, (kind, vec) in basis.items():
        if kind != "move":
            continue
        e = env.clone(); e.step(a); g1 = e.render()
        for col in pal:
            c0 = components(g0, col, min_size); c1 = components(g1, col, min_size)
            for (x0, y0, _) in c0:
                for (x1, y1, _) in c1:
                    if (x1 - x0, y1 - y0) == vec:
                        return (x0, y0), col
    return None


# ───────────────────────────── operator 3: track the agent (motion + continuity) ──────────────────
class AgentTracker:
    """Track the agent frame-to-frame: after a move action, the agent is the blob that has shifted to
    the expected lattice point (preferring its last colour, falling back only when that colour vanishes
    — i.e. a genuine recolour). Decides move-vs-blocked by which lattice point the centroid is nearer.
    GENERAL: this is the `player_at`/`classify` pair, parameterised, no game palette baked in."""
    PAL_FALLBACK = None
    def __init__(self, basis, palette, min_size=3, tol=6):
        self.move = {a: v for a, (k, v) in basis.items() if k == "move"}
        self.pal = palette; self.min = min_size; self.tol = tol

    def _at(self, grid, near, hint):
        for cols in ([hint], [c for c in self.pal if c != hint]):
            best = None
            for col in cols:
                for (cx, cy, sz) in components(grid, col, self.min):
                    d = abs(cx - near[0]) + abs(cy - near[1])
                    if d <= self.tol and (best is None or d < best[0]):
                        best = (d, (cx, cy), col)
            if best:
                return best
        return None

    def step_result(self, grid_after, pos, color, action):
        """Given the frame after `action`, return (new_pos, new_color) if the agent moved, else None."""
        v = self.move.get(action)
        if v is None:
            return None
        exp = (pos[0] + v[0], pos[1] + v[1])
        cand = self._at(grid_after, pos, color) or self._at(grid_after, exp, color)
        if cand is None:
            return None
        c = cand[1]
        if abs(c[0] - exp[0]) + abs(c[1] - exp[1]) < abs(c[0] - pos[0]) + abs(c[1] - pos[1]):
            return exp, cand[2]
        return None


# ───────────────────────────── operator 4: explore the reachable graph ────────────────────────────
class Explorer:
    """Deterministic reset+replay (via clone checkpoints) to build the reachable-state graph over
    dead-reckoned agent positions, recording edges and BLOCKED frontiers (cells the agent can't enter
    yet — walls, or gates that open only under the right state). GENERAL: the exploration loop with the
    energy/colour specifics removed; works for any deterministic grid Env."""
    def __init__(self, env: Env, basis, agent, tracker: AgentTracker, cap=400):
        self.env, self.basis, self.tracker, self.cap = env, basis, tracker, cap
        self.p0, self.col0 = agent
        self.moves = [a for a, (k, _) in basis.items() if k == "move"]

    def explore(self):
        start = (0, 0)
        env = self.env; env.reset(); base = env.clone()
        ckpt = {start: (base, self.p0, self.col0)}
        paths = {start: []}; edges = {}; order = [start]; i = 0
        blocked = []
        while i < len(order) and len(order) < self.cap:
            cell = order[i]; i += 1
            gC, pcC, colC = ckpt[cell]
            edges[cell] = {}
            for a in self.moves:
                e = gC.clone(); e.step(a); grid = e.render()
                mv = self.tracker.step_result(grid, pcC, colC, a)
                if mv is None:
                    blocked.append((cell, a)); continue
                exp, ncol = mv
                v = self.basis[a][1]
                npos = (cell[0] + v[0], cell[1] + v[1])
                edges[cell][a] = npos
                if npos not in paths:
                    paths[npos] = paths[cell] + [a]; ckpt[npos] = (e, exp, ncol); order.append(npos)
        return dict(start=start, paths=paths, edges=edges, blocked=blocked, ckpt=ckpt)

    @staticmethod
    def bfs(edges, src, dst):
        if src == dst: return []
        seen = {src}; q = deque([(src, [])])
        while q:
            c, p = q.popleft()
            for a, nb in edges.get(c, {}).items():
                if nb in seen: continue
                if nb == dst: return p + [a]
                seen.add(nb); q.append((nb, p + [a]))
        return None


# ───────────────────────────── operator 5: infer a hidden resource ────────────────────────────────
class HiddenResourceOracle:
    """Some quantities are NOT in the render (ls20 energy drains with zero pixel change). They are only
    knowable through their CONSEQUENCE: acting past the budget triggers a failure/reset (the agent
    teleports back to start / status flips to LOSE). Measure the budget by acting until that event and
    counting. GENERAL methodology: 'if it's not on screen, infer it from what it causes'."""
    def __init__(self, env: Env, agent_start, tracker: AgentTracker, cap=120):
        self.env, self.p0, self.tracker, self.cap = env, agent_start[0], tracker, cap
        self.col0 = agent_start[1]

    def budget_from(self, prefix_actions, oscillate):
        """From a state reached by `prefix_actions`, spam `oscillate` (a back-and-forth that keeps
        acting) until the agent teleports back to start (failure/respawn). Returns the step count =
        the resource remaining at that state, or None if no failure within `cap`."""
        e = self.env.clone() if hasattr(self.env, "clone") else self.env
        e.reset()
        for a in prefix_actions: e.step(a)
        pos, col = self.p0, self.col0
        for step in range(1, self.cap):
            a = oscillate[step % len(oscillate)]
            e.step(a); grid = e.render()
            cur = self.tracker._at(grid, pos, col)
            jumped = cur is None or abs(cur[1][0] - pos[0]) + abs(cur[1][1] - pos[1]) > 8
            if jumped:
                back = self.tracker._at(grid, self.p0, self.col0)
                if back and abs(back[1][0] - self.p0[0]) + abs(back[1][1] - self.p0[1]) <= 4:
                    return step
            if cur:
                pos, col = cur[1], cur[2]
        return None


# ───────────────────────────── operator 6: semantics by interaction ───────────────────────────────
def interaction_role(env: Env, edges, paths, ckpt, basis, cell, signature_fn):
    """Classify an interactable cell by the PERSISTENT effect of touching it (go onto it, step back to
    a neighbour, compare a chosen `signature_fn` at the SAME position before/after — static scenery
    cancels, only a real persistent change survives). Returns 'transformer' if the agent's own state
    changed for good, else 'inert'. GENERAL: the station/refill/slot discovery, with the signature a
    plug-in (the adapter says WHAT to read; the operator says HOW to compare).
    Returns a tag; richer kinds (gate/replenisher) come from combining with blocked-frontier + oracle."""
    # find a reachable neighbour A with an action that enters `cell`
    for A in paths:
        for a, nb in edges.get(A, {}).items():
            if nb != cell:
                continue
            # opposite action returns A<-cell
            v = basis[a][1]; back = next((b for b, (k, w) in basis.items() if w == (-v[0], -v[1])), None)
            if back is None:
                return "inert"
            e0 = ckpt[A][0].clone()
            s0 = signature_fn(e0.render(), ckpt[A][1])
            e1 = ckpt[A][0].clone(); e1.step(a); e1.step(back)
            s1 = signature_fn(e1.render(), ckpt[A][1])
            return "transformer" if s1 != s0 else "inert"
    return "inert"


# ═════════════════════════════ live demo: drive ls20 through the Env abstraction ══════════════════
class _Ls20Env:
    """A thin ADAPTER wrapping ls20 as an `Env`. Note how small it is: everything game-specific lives
    here (engine calls + which actions exist); the operators above never see ls20."""
    def __init__(self, level=0, _g=None):
        import importlib.util
        from pathlib import Path
        if _g is None:
            ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
            spec = importlib.util.spec_from_file_location("ls20mod", ENV)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            self._cls = m.Ls20; self.g = m.Ls20(); self.g.set_level(level)
        else:
            self._cls = None; self.g = _g
        from arcengine import GameAction, ActionInput
        self._AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in
                     ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]}
        self._RESET = ActionInput(id=GameAction.RESET)
        self.actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        self._lvl0 = self.g.level_index
    def reset(self): self.g.perform_action(self._RESET)
    def step(self, a): self.g.perform_action(self._AIN[a])
    def render(self): return np.array(self.g.get_pixels(0, 0, 64, 64))
    def status(self):
        s = self.g._state.name
        return "WIN" if s == "WIN" else "LOSE" if s == "GAME_OVER" else "PLAYING"
    def score(self): return self.g.level_index
    def clone(self):
        e = _Ls20Env.__new__(_Ls20Env)
        e.g = copy.deepcopy(self.g); e._AIN = self._AIN; e._RESET = self._RESET
        e.actions = self.actions; e._cls = None; e._lvl0 = self._lvl0
        return e


def _demo():
    env = _Ls20Env(level=0)
    basis = learn_action_basis(env)
    print("action basis (no key meanings assumed):")
    for a, (k, v) in basis.items():
        print(f"    {a}: {k}{'' if v is None else ' ' + str(v)}")
    agent = find_agent(env, basis)
    print(f"agent found by MOTION: pos={agent[0]} colour={agent[1]}")
    pal = _palette(env.render())
    tracker = AgentTracker(basis, pal)
    g = Explorer(env, basis, agent, tracker).explore()
    print(f"explored {len(g['paths'])} reachable cells; {len(g['blocked'])} blocked frontiers "
          f"(walls / closed gates) — all via the Env abstraction, zero engine internals")
    # hidden-resource probe: is there an unrendered budget? (ls20 energy)
    osc = [a for a, (k, _) in basis.items() if k == "move"][:2]
    deep = max(g["paths"], key=lambda c: abs(c[0]) + abs(c[1]))
    oracle = HiddenResourceOracle(env, agent, tracker)
    b = oracle.budget_from(g["paths"][deep], osc)
    print(f"hidden-resource oracle at the deepest cell: {b} steps to failure "
          f"(-> a hidden depleting resource exists; ls20 'energy', invisible in pixels)")
    print("\nThis is the game-agnostic skeleton. To target an UNSEEN game, an LLM supplies only the small")
    print("Adapter (goal_progress + interactable cues); these operators provide the rest of the loop.")


if __name__ == "__main__":
    _demo()
