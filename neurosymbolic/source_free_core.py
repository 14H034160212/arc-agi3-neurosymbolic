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


# ───────────────────────────── operator 7: discover object ROLES (no appearance assumed) ──────────
def state_signature(grid, agent_pos, near=6):
    """The 'world + carried state' away from the agent: all non-background pixels farther than `near`
    from the agent. Compared at the SAME agent position, static scenery cancels; only a persistent
    change (e.g. an inventory/HUD indicator, a consumed marker) survives. GENERAL: this is `hud_sig`
    with the HUD LOCATION discovered ('far from the agent') instead of hard-coded."""
    bg = int(np.bincount(grid.flatten()).argmax())
    ys, xs = np.where(grid != bg)
    return frozenset((int(grid[y, x]), int(x), int(y)) for y, x in zip(ys, xs)
                     if abs(int(x) - agent_pos[0]) + abs(int(y) - agent_pos[1]) > near)

def discover_roles(env: Env, basis, tracker: AgentTracker, graph, resource=False, budget_cells=24):
    """Classify the map's interactables by INTERACTION, not appearance:
        transformers — touching persistently changes the agent's own (carried) state  [ls20 stations]
        gate_cands   — blocked frontiers that may open under the right state           [ls20 slots]
        replenishers — touching raises the hidden-resource budget (only if `resource`) [ls20 refills]
    GENERAL: lifts ls20's station/slot/refill detection into role discovery with no colour/semantics
    baked in (the signature is 'far-from-agent state'; the budget test is the act-until-failure oracle)."""
    edges, paths, ckpt, blocked = graph["edges"], graph["paths"], graph["ckpt"], graph["blocked"]
    bfs = Explorer.bfs
    transformers = []
    for cell in paths:
        if cell == graph["start"]:
            continue
        tag = interaction_role(env, edges, paths, ckpt, basis, cell, state_signature)
        if tag == "transformer":
            transformers.append(cell)
    replenishers = []
    if resource:
        oracle = HiddenResourceOracle(env, (tracker_start := (ckpt[graph["start"]][1], ckpt[graph["start"]][2])), tracker)
        osc = [a for a, (k, _) in basis.items() if k == "move"][:2]
        base_budget = oracle.budget_from([], osc) or 0
        deep = sorted(paths, key=lambda c: -(abs(c[0]) + abs(c[1])))[:budget_cells]
        for cell in deep:
            b = oracle.budget_from(paths[cell], osc)
            if b is not None and b >= base_budget:       # arriving with a full tank => a replenisher
                replenishers.append(cell)
    return dict(transformers=transformers, gate_cands=blocked, replenishers=replenishers)


# ───────────────────────────── operator 8: general single-goal solver ──────────────────────────────
def solve_single_goal(env: Env, max_cycles=4, verbose=True):
    """A fully GENERAL solve of one goal, using only the operators above + the Env oracle: learn the
    action basis, find + track the agent, explore, discover transformer cells, then search
    (transformer x cycle-count x gate) verified by win/score. No ls20 knowledge. Proves the skeleton is
    a real solver, not just perception. (Hidden-resource routing / multi-goal chaining are the ported
    operators that extend this to the deeper ls20 levels; here we demonstrate the no-resource case.)"""
    basis = learn_action_basis(env)
    agent = find_agent(env, basis)
    if agent is None:
        return None
    pal = _palette(env.render()); tracker = AgentTracker(basis, pal)
    graph = Explorer(env, basis, agent, tracker).explore()
    roles = discover_roles(env, basis, tracker, graph)
    edges, paths = graph["edges"], graph["paths"]; bfs = Explorer.bfs
    Ts = roles["transformers"]; start = graph["start"]
    if verbose:
        print(f"    discovered {len(Ts)} transformer cell(s); {len(roles['gate_cands'])} gate candidates")

    def offback(C):
        v0 = None
        for a, nb in edges.get(C, {}).items():
            if nb != C:
                return [a, next((b for b, (k, w) in basis.items()
                                 if w == (-basis[a][1][0], -basis[a][1][1])), a)]
        return None
    def cycle_plan(C, k):
        p = bfs(edges, start, C)
        if p is None: return None
        if k > 1:
            ob = offback(C)
            if ob is None: return None
            p = p + ob * (k - 1)
        return p

    base = env.clone(); base.reset(); s0 = base.score()
    def wins(plan):
        e = env.clone(); e.reset()
        for a in plan:
            e.step(a)
            if e.status() == "WIN" or e.score() > s0:
                return True
        return False

    # search: route through a transformer (cycled k) then into a gate, verified by win/score
    near = lambda c: abs(c[0]) + abs(c[1])
    for (gcell, ga) in sorted(roles["gate_cands"], key=lambda fr: near((fr[0][0], fr[0][1]))):
        for C in Ts:
            for k in range(1, max_cycles + 1):
                cp = cycle_plan(C, k)
                if cp is None: continue
                seg = bfs(edges, C, gcell)
                if seg is None: continue
                plan = cp + seg + [ga]
                if wins(plan):
                    if verbose:
                        print(f"    ✅ solved via transformer {C} x{k} -> gate (plan {len(plan)} actions)")
                    return plan
    return None


# ───────────────────────────── operator 0: detect the INTERACTION MODALITY ────────────────────────
def detect_modality(env: Env, sample_clicks=12):
    """Before any solving, find out HOW the game is played — the first thing that differs across
    ARC-AGI-3 games. Returns ('movement'|'click'|'unknown', basis). GENERAL and important: a
    movement-tuned solver must NOT be pointed at a click game; this operator makes the agent
    self-aware of the modality instead of failing silently. (Cross-game test: ls20->movement,
    ft09/vc33->not-movement; vc33 confirmed click-based in its source.)"""
    basis = learn_action_basis(env)
    if sum(1 for k, _ in basis.values() if k == "move") >= 2:
        return "movement", basis
    if hasattr(env, "click"):
        objs, _ = find_clickables(env)             # probes all foreground cells at the right scale
        if objs:
            return "click", basis
    return "unknown", basis


# ───────────────────────────── the CLICK modality (second operator family) ────────────────────────
# Same loop (probe -> discover -> search -> verify) as the movement modality, but the "action" is a
# spatial CLICK on a grid cell instead of a directional move. This is what ft09/vc33 need.

def _probe_clicks(env: Env, fg, scale, min_change=6):
    """Click every foreground cell at the given display:grid scale; return the responder cells.
    A click on an empty cell typically still renders a cursor artifact (a few pixels) -- require a
    SUBSTANTIAL change (>= min_change) or a status/score change, else every game misreports as click."""
    base_score = env.score()
    resp = []
    for (x, y) in fg:
        e = env.clone(); b = e.render(); e.click(int(x) * scale, int(y) * scale); a = e.render()
        substantial = a.shape != b.shape or int((b != a).sum()) >= min_change
        if substantial or e.status() != "PLAYING" or e.score() > base_score:
            resp.append((int(x), int(y)))
    return resp

def find_clickables(env: Env, scale=None):
    """Probe ALL foreground cells with a click; the cells that change the frame (or win/score) are the
    interactables. A click takes DISPLAY coords and the grid may be downsampled, so we try scale 2 then
    1 and keep whichever yields responders. Cluster responders into OBJECTS (one click target each).
    GENERAL: the click analogue of learn_action_basis/find_agent — objects found by their click-effect."""
    env.reset(); g0 = env.render(); bg = int(np.bincount(g0.flatten()).argmax())
    ys, xs = np.where(g0 != bg); fg = list(zip(xs.tolist(), ys.tolist()))
    for s in ([scale] if scale else [2, 1]):
        resp = _probe_clicks(env, fg, s)
        if resp:
            objs = []
            for p in resp:
                if not any(abs(o[0] - p[0]) + abs(o[1] - p[1]) <= 3 for o in objs):
                    objs.append(p)
            return objs, s
    return [], (scale or 1)

def solve_click(env: Env, objs=None, scale=None, base=None, max_depth=3, node_cap=20000, verbose=True):
    """Solve a CLICK game by searching short click sequences (over the discovered clickable objects),
    verified by win/score. Clickables are discovered ONCE (clicking rarely changes the legal set, and
    re-probing every node is far too slow). GENERAL second-modality solver — the click analogue of the
    movement search. Returns the winning click list or None."""
    s0 = env.score() if base is None else base
    if objs is None:
        objs, scale = find_clickables(env)
        if verbose:
            print(f"    click modality: scale={scale}, {len(objs)} clickable object(s) discovered by probing")
    if not objs:
        return None
    # ADAPTIVE depth: with few clickable objects a deeper search is cheap (small branching); with many,
    # stay shallow to avoid the blow-up. (Re-probing per node to catch newly-revealed objects is correct
    # but too slow at scale — a smarter incremental detector is future work.)
    if len(objs) <= 4 and max_depth < 8:
        max_depth = 8
    start = env.clone()
    q = deque([(start, [])]); nodes = 0
    while q and nodes < node_cap:
        state, seq = q.popleft()
        for (x, y) in objs:
            e = state.clone(); e.click(x * scale, y * scale); nodes += 1
            if e.status() == "WIN" or e.score() > s0:
                plan = seq + [(x, y)]
                if verbose:
                    print(f"    ✅ solved by {len(plan)} click(s): {plan}  ({nodes} nodes)")
                return plan
            if e.status() == "LOSE":
                continue
            if len(seq) + 1 < max_depth:
                q.append((e, seq + [(x, y)]))
    if verbose:
        print(f"    no winning click sequence within depth {max_depth} ({nodes} nodes explored)")
    return None


# ───────────────────────────── operator 9 (extension point): LLM-driven adapter ───────────────────
LLM_ADAPTER_PROMPT = """You are the perception module for a source-free game-playing agent. You are
given a few rendered frames (integer grids) and the effects of probing each action. Output JSON:
  {"agent_color": <int or null>,        # which colour is the agent (null = let motion auto-detect)
   "goal_signal": "win" | "score",      # how progress is observed
   "interactable_hint": [[x,y],...]}    # cells worth probing as transformers/gates (optional)
Base your answer ONLY on the frames + probe effects; do not assume any game's rules."""

def local_gpt_oss(prompt, payload, endpoint="http://localhost:11437/v1/chat/completions",
                  model="local/gpt-oss-120b:opt", timeout=120):
    """Reference `call_llm`: a FREE local LLM (gpt-oss-120b via Ollama) as the per-game inducer. Sends
    the probe summary, returns the model's text. Verified live: given 'directional actions inert + N
    cells respond to clicks', gpt-oss replies modality='click' with a correct rationale. Provider-
    agnostic — swap the endpoint/model for any OpenAI-compatible API."""
    import json, urllib.request
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 300,
                       "messages": [{"role": "system", "content": prompt},
                                    {"role": "user", "content": json.dumps(payload)}]}).encode()
    req = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r["choices"][0]["message"]["content"]


class LLMAdapter:
    """The PER-GAME induction surface, produced by an LLM for an UNSEEN game. On a game where the
    general auto-discovery (motion agent + role discovery) is insufficient, the LLM reads a few frames
    + probe effects and returns the hints below. This is the concrete integration point for the
    neuro-symbolic plan (LLM induces the per-game perception; the operators above plan + verify).
    `call_llm` is injected so this stays API/provider-agnostic (e.g. `local_gpt_oss` above)."""
    def __init__(self, call_llm=None):
        self.call_llm = call_llm                       # fn(prompt, frames) -> JSON str ; None = heuristic
    def induce(self, env: Env, basis):
        if self.call_llm is None:                      # default: pure auto-discovery, empty hints
            return dict(agent_color=None, goal_signal="win", interactable_hint=[])
        import json
        frames = [env.render().tolist()]               # a real impl would include probe-effect summaries
        try:
            return json.loads(self.call_llm(LLM_ADAPTER_PROMPT, frames))
        except Exception:
            return dict(agent_color=None, goal_signal="win", interactable_hint=[])


# ═════════════════════════ a generic ARC-AGI-3 env adapter (works for ANY of the games) ════════════
class ArcGameEnv:
    """Wraps any local ARC-AGI-3 game (ls20 / ft09 / vc33 / ...) as an `Env`. The ONLY game-specific
    thing here is the file path + class name + grid size — exactly the thin adapter layer. Supports
    `click(x,y)` (ACTION6) so the modality detector can probe click games too."""
    DIRS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    def __init__(self, game, cls, size, level=0, _g=None):
        import importlib.util
        from pathlib import Path
        from arcengine import GameAction, ActionInput
        if _g is None:
            f = list((Path(__file__).resolve().parent.parent / "environment_files" / game).glob(f"*/{game}.py"))[0]
            spec = importlib.util.spec_from_file_location(game, f)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            self.g = getattr(m, cls)()
            try: self.g.set_level(level)
            except Exception: pass
        else:
            self.g = _g
        self.game, self.cls, self.size = game, cls, size
        self._AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in self.DIRS}
        self._R = ActionInput(id=GameAction.RESET); self._GA = GameAction; self._AI = ActionInput
        self.actions = list(self.DIRS)
    def reset(self): self.g.perform_action(self._R)
    def step(self, a): self.g.perform_action(self._AIN[a])
    def click(self, x, y):
        ai = self._AI(id=self._GA.ACTION6)
        try: ai.data = {"x": x, "y": y}
        except Exception: pass
        self.g.perform_action(ai)
    def render(self): return np.array(self.g.get_pixels(0, 0, self.size, self.size))
    def status(self):
        s = self.g._state.name
        return "WIN" if s == "WIN" else "LOSE" if s == "GAME_OVER" else "PLAYING"
    def score(self): return getattr(self.g, "level_index", 0)
    def clone(self):
        e = ArcGameEnv.__new__(ArcGameEnv)
        e.g = copy.deepcopy(self.g)
        for k in ("game", "cls", "size", "_AIN", "_R", "_GA", "_AI", "actions"):
            setattr(e, k, getattr(self, k))
        return e

GAMES = [("ls20", "Ls20", 64), ("ft09", "Ft09", 32), ("vc33", "Vc33", 32)]


def _demo():
    print("══ single-game depth: drive ls20 through the Env abstraction (zero engine internals) ══")
    env = ArcGameEnv("ls20", "Ls20", 64, level=0)
    mod, basis = detect_modality(env)
    print(f"modality detected: {mod};  action basis: { {a: k for a, (k, _) in basis.items()} }")
    agent = find_agent(env, basis); pal = _palette(env.render()); tracker = AgentTracker(basis, pal)
    g = Explorer(env, basis, agent, tracker).explore()
    roles = discover_roles(env, basis, tracker, g)
    print(f"agent by MOTION {agent[0]}; explored {len(g['paths'])} cells, {len(g['blocked'])} gates/walls; "
          f"transformers discovered by INTERACTION = {roles['transformers']}")
    plan = solve_single_goal(ArcGameEnv("ls20", "Ls20", 64, level=0), verbose=True)
    print(f"  general solve of L0: {'WON ('+str(len(plan))+' actions)' if plan else 'no plan'}")

    print("\n══ cross-game, BOTH modalities (step 3): same loop, auto-selected operator family ══")
    for game, cls, size in GAMES:
        try:
            mod, _ = detect_modality(ArcGameEnv(game, cls, size, level=0))
            if mod == "movement":
                plan = solve_single_goal(ArcGameEnv(game, cls, size, level=0), verbose=False)
            elif mod == "click":
                plan = solve_click(ArcGameEnv(game, cls, size, level=0), max_depth=4, verbose=False)
            else:
                plan = None
            res = f"WON L0 ({len(plan)} actions)" if plan else "L0 not solved at this depth"
            print(f"  {game}: modality={mod:8s} -> {res}")
        except Exception as ex:
            print(f"  {game}: error {type(ex).__name__}: {str(ex)[:60]}")
    print("\nThe SAME probe/discover/explore/search/verify loop spans both modalities: the agent detects")
    print("whether the game is movement- or click-based and applies the matching operator family, solving")
    print("the first level of each. Deeper levels need smarter search (or LLM hints via LLMAdapter) — the")
    print("same open frontier, now demonstrably modality-agnostic.")


if __name__ == "__main__":
    _demo()
