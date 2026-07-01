#!/usr/bin/env python3
"""
online_play — run our source-free PERCEPTION on the REAL ARC-AGI-3 games via the official API.

The public 25 games ship NO source (only ls20/ft09/vc33 are bundled) — they are played blind through
https://three.arcprize.org, which is exactly our source-free setting. The online API supports reset()
and step() but NOT clone()/deepcopy, so the heavy clone-based search can't run online as-is; here we run
the cheap PERCEPTION layer (modality detection + agent/clickable discovery from the start state) on the
real games and check it against the official modality tags — i.e. does our agent correctly perceive
*unseen* real games. Set ARC_API_KEY env var (or pass --key).
"""
from __future__ import annotations
import importlib.util as _ilu
import os, sys, time
from pathlib import Path
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

# Reuse the validated AgentTracker (move-vs-blocked classification by tracked position, not raw
# frame-equality -- raw equality misclassifies wall-hits whenever an unrelated animation/decoration
# also changes the frame) from the offline general-operator module instead of reimplementing it.
_sfc_spec = _ilu.spec_from_file_location("source_free_core", Path(__file__).resolve().parent / "source_free_core.py")
_sfc = _ilu.module_from_spec(_sfc_spec); _sfc_spec.loader.exec_module(_sfc)
AgentTracker = _sfc.AgentTracker

DIRS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class OnlineGame:
    """Thin wrapper over one real game session (reset/step/click/render/status/score).

    The remote SDK (arc_agi's RemoteEnvironmentWrapper) returns None -- it never raises -- on ANY
    transient network/HTTP/parse failure, for reset() AND step(). A stale/None self._last would crash
    every grid()/status()/score() call downstream, so every mutator here retries a few times on None
    and only ever assigns self._last from a non-None response."""
    def __init__(self, arcade, game_id, card):
        self.env = arcade.make(game_id, scorecard_id=card); self.game_id = game_id
        if self.env is None:
            raise RuntimeError(f"arcade.make({game_id!r}) returned None (bad API key, or metadata "
                               f"fetch failed) -- cannot play this game")
        self._last = None
        self.reset()                                     # establish an initial frame up front
    def reset(self, retries=3):
        for _ in range(retries):
            r = self.env.reset()
            if r is not None:
                self._last = r; return self
        if self._last is None:
            raise RuntimeError(f"{self.game_id}: reset() returned None {retries}x in a row "
                               f"(network/API failure) and there is no prior frame to fall back to")
        return self                                       # keep the last-good frame; caller can retry
    def step(self, name, retries=2):
        for _ in range(retries):
            r = self.env.step(getattr(GameAction, name))
            if r is not None: self._last = r; break
        return self
    def click(self, x, y, retries=2):
        for _ in range(retries):
            r = self.env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            if r is not None: self._last = r; break
        return self
    def grid(self):
        fr = np.array(self._last.frame)
        return fr[-1] if fr.ndim == 3 else fr
    def status(self):
        s = str(self._last.state)
        return "WIN" if "WIN" in s else "LOSE" if "OVER" in s or "LOSE" in s else "PLAYING"
    def score(self):
        return self._last.levels_completed or 0
    def avail(self):
        return self._last.available_actions or []


def _comps(grid, color, min_size=3):
    from collections import deque
    H, W = grid.shape; seen = np.zeros_like(grid, bool); out = []
    for y in range(H):
        for x in range(W):
            if grid[y, x] == color and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == color:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(cells) >= min_size:
                    out.append((round(sum(c[1] for c in cells)/len(cells)),
                                round(sum(c[0] for c in cells)/len(cells)), len(cells)))
    return out


def probe_modality(g: OnlineGame, click_samples=14, min_change=6):
    """Source-free modality detection on a real game, using only reset()+step() (no clone):
    movement if >=2 directional actions translate a blob; else click if clicking a foreground cell
    changes the frame SUBSTANTIALLY; else unknown. Returns (modality, n_move_dirs, n_click_responders).
    A click on an empty cell only renders a 1px cursor -- min_change filters that artifact out (the
    same threshold online_find_clickables uses), so pure-movement games aren't misclassified as click."""
    g.reset(); g0 = g.grid()
    pal = [int(c) for c in np.unique(g0)]
    bg = int(np.bincount(g0.flatten()).argmax())
    pal = [c for c in pal if c != bg]
    # directional probe
    moves = 0
    for a in DIRS:
        g.reset(); g.step(a); g1 = g.grid()
        if g1.shape != g0.shape or np.array_equal(g0, g1):
            continue
        # a translation of some same-colour blob => movement
        for col in pal:
            c0 = _comps(g0, col); c1 = _comps(g1, col)
            if any(abs(s0-s1) <= 2 and (x1-x0, y1-y0) != (0, 0) and abs(x1-x0)+abs(y1-y0) <= 6
                   for (x0, y0, s0) in c0 for (x1, y1, s1) in c1):
                moves += 1; break
    # click probe too (don't early-return on movement: the majority class is keyboard_click = BOTH)
    H, W = g0.shape
    ys, xs = np.where(g0 != bg); fg = list(zip(xs.tolist(), ys.tolist()))
    resp = 0
    for (x, y) in fg[:: max(1, len(fg)//click_samples)][:click_samples]:
        if not (0 <= x < W and 0 <= y < H):
            continue
        g.reset(); g.click(x, y); gg = g.grid()
        substantial = gg.shape != g0.shape or int((g0 != gg).sum()) >= min_change
        if substantial or g.status() != "PLAYING" or g.score() > 0:
            resp += 1
    if moves >= 2 and resp:
        return "movement+click", moves, resp
    if moves >= 2:
        return "movement", moves, resp
    if resp:
        return "click", moves, resp
    return "unknown", moves, resp


def online_find_clickables(g: OnlineGame, max_probes=90, min_change=6):
    """Discover clickable objects on a real game using reset()+click() only (no clone). A click on an
    empty cell only renders the 1-px CURSOR; a real interactable changes many pixels — so we require a
    SUBSTANTIAL change (>= min_change) or a status/score change. Sample foreground cells (strided to
    cap API calls), keep substantial responders, cluster into objects. Returns (objects, base_grid)."""
    g.reset(); g0 = g.grid(); H, W = g0.shape
    bg = int(np.bincount(g0.flatten()).argmax())
    ys, xs = np.where(g0 != bg); fg = list(zip(xs.tolist(), ys.tolist()))
    step = max(1, len(fg)//max_probes)
    objs = []
    for (x, y) in fg[::step]:
        if not (0 <= x < W and 0 <= y < H):
            continue
        g.reset(); g.click(x, y); gg = g.grid()
        substantial = gg.shape != g0.shape or int((g0 != gg).sum()) >= min_change
        if not (substantial or g.status() != "PLAYING" or g.score() > 0):
            continue
        if not any(abs(o[0]-x) + abs(o[1]-y) <= 3 for o in objs):
            objs.append((x, y))
    return objs, g0


def online_solve_click(g: OnlineGame, max_depth=4, node_cap=400, verbose=True):
    """Solve a click level by RESET+replay candidate search (no clone): BFS over click-target
    sequences, each evaluated by reset()+replay+win-check. Returns the winning click list or None."""
    from collections import deque
    s0 = g.reset().score()
    objs, _ = online_find_clickables(g)
    if verbose:
        print(f"    {g.game_id}: {len(objs)} clickable object(s) found")
    if not objs:
        return None
    q = deque([[]]); nodes = 0
    while q and nodes < node_cap:
        seq = q.popleft()
        for (x, y) in objs:
            cand = seq + [(x, y)]; nodes += 1
            g.reset()
            won = False
            for (cx, cy) in cand:
                g.click(cx, cy)
                if g.status() == "WIN" or g.score() > s0:
                    won = True; break
                if g.status() == "LOSE":
                    break
            if won:
                if verbose: print(f"    ✅ {g.game_id} WON L{s0} in {len(cand)} clicks ({nodes} candidates)")
                return cand
            if g.status() == "PLAYING" and len(cand) < max_depth:
                q.append(cand)
    if verbose: print(f"    {g.game_id}: no win within depth {max_depth} ({nodes} candidates)")
    return None


def _scene_objects(grid, k=24):
    """Compact structured scene: the largest non-background connected components as
    (color, cx, cy, size) — a representation an LLM can reason over (vs a raw 64x64 int grid)."""
    bg = int(np.bincount(grid.flatten()).argmax())
    objs = []
    for col in [int(c) for c in np.unique(grid) if int(c) != bg]:
        objs += [(col, cx, cy, sz) for (cx, cy, sz) in _comps(grid, col, 1)]
    objs.sort(key=lambda o: -o[3])
    return objs[:k]


def llm_click_plan(scene, clickables, call_llm, n=6):
    """Ask a FREE local LLM to propose promising click sequences (orderings of clickable indices),
    given the scene + clickable objects. LLM proposes -> environment verifies (correctness kept)."""
    import json
    prompt = (
        "You are solving a hidden-goal ARC-AGI-3 puzzle. You click objects to make progress; the goal "
        "is unknown but clicking the right objects in the right order wins the level. Given the scene "
        "objects (color,x,y,size) and the CLICKABLE objects (indexed), output ONLY JSON: a list of up "
        f"to {n} candidate click-sequences to try first, best guess first, each a list of clickable "
        "indices (length 1-4). Favor clicking distinctive/odd-one-out objects, matching pairs, or all "
        'clickables in a sensible order. Example: {"sequences": [[0],[1,0],[0,1,2]]}'
    )
    payload = {"scene_objects": scene[:24],
               "clickable_objects": [{"i": i, "x": x, "y": y} for i, (x, y) in enumerate(clickables)]}
    try:
        txt = call_llm(prompt, payload)
        seqs = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("sequences", [])
        out = []
        for s in seqs:
            cells = [clickables[i] for i in s if isinstance(i, int) and 0 <= i < len(clickables)]
            if cells:
                out.append(cells)
        return out
    except Exception as e:
        return []


def online_solve_click_llm(g: OnlineGame, call_llm, verbose=True):
    """LLM-guided online click solve: discover clickables -> describe scene -> LLM proposes click
    sequences -> verify each by reset()+replay (env-verified). Falls back to blind BFS if LLM misses."""
    s0 = g.reset().score(); base = g.grid()
    objs, _ = online_find_clickables(g)
    if verbose: print(f"    {g.game_id}: {len(objs)} clickable object(s)")
    if not objs:
        return None
    scene = _scene_objects(base)
    cands = llm_click_plan(scene, objs, call_llm)
    if verbose: print(f"    LLM proposed {len(cands)} click-sequence(s) to try first")
    for cand in cands:                                   # verify LLM proposals first (cheap, few)
        g.reset(); won = False
        for (x, y) in cand:
            g.click(x, y)
            if g.status() == "WIN" or g.score() > s0: won = True; break
            if g.status() == "LOSE": break
        if won:
            if verbose: print(f"    ✅ {g.game_id} WON L{s0} via LLM plan {cand}")
            return cand
    if verbose: print(f"    LLM plans didn't win; falling back to blind BFS")
    return online_solve_click(g, max_depth=4, verbose=verbose)


PALETTE = {0: (0, 0, 0), 1: (70, 70, 70), 2: (200, 60, 60), 3: (35, 35, 40), 4: (120, 120, 120),
           5: (210, 80, 80), 6: (250, 120, 40), 7: (250, 220, 60), 8: (70, 90, 230), 9: (60, 190, 230),
           10: (240, 130, 200), 11: (245, 225, 70), 12: (245, 150, 50), 13: (120, 220, 120),
           14: (170, 100, 220), 15: (255, 255, 255)}

def render_png_b64(grid, clickables=None, up=16):
    """Render the integer grid to an upscaled PNG (distinct palette); label clickables 0..N so the
    VLM can refer to them by index. Returns base64 string."""
    import io, base64
    from PIL import Image, ImageDraw
    H, W = grid.shape
    img = Image.new("RGB", (W*up, H*up)); px = img.load()
    for y in range(H):
        for x in range(W):
            v = int(grid[y, x]); c = PALETTE.get(v, (v*17 % 256, v*53 % 256, v*97 % 256))
            for dy in range(up):
                for dx in range(up):
                    px[x*up+dx, y*up+dy] = c
    if clickables:
        d = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(clickables):
            d.text((x*up, y*up), str(i), fill=(255, 255, 0))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def call_vlm(b64, prompt, model=None, endpoint=None):
    """Call a local multimodal model (qwen3-vl) with an image + prompt; return its text.
    Model/endpoint overridable via VLM_MODEL / VLM_ENDPOINT env vars (to swap in a stronger VLM)."""
    import json, urllib.request
    model = model or os.environ.get("VLM_MODEL", "qwen3-vl:8b")
    endpoint = endpoint or os.environ.get("VLM_ENDPOINT", "http://localhost:11439/api/chat")
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt, "images": [b64]}]}).encode()
    req = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["message"]["content"]

def vlm_click_plan(grid, clickables, n=6):
    """Render the frame with numbered clickables, ask the VLM for click-order(s) to solve. Returns a
    list of candidate click-sequences (each a list of (x,y))."""
    import json
    b64 = render_png_b64(grid, clickables)
    prompt = (f"This is a puzzle game screenshot. The {len(clickables)} CLICKABLE objects are labelled "
              f"0..{len(clickables)-1} in yellow. Infer the likely goal, then output ONLY JSON: "
              '{"sequences": [[i,j],...]} = up to ' + str(n) + " candidate click-orders (lists of the "
              "labelled indices, length 1-4) most likely to solve the level, best first.")
    try:
        txt = call_vlm(b64, prompt)
        seqs = json.loads(txt[txt.index("{"):txt.rindex("}")+1]).get("sequences", [])
        out = []
        for s in seqs:
            cells = [clickables[i] for i in s if isinstance(i, int) and 0 <= i < len(clickables)]
            if cells:
                out.append(cells)
        return out
    except Exception:
        return []

def online_solve_click_vlm(g: OnlineGame, verbose=True):
    """Multimodal-guided online click solve: discover clickables -> render frame -> qwen3-vl picks the
    click order (it SEES the puzzle) -> env verifies. Falls back to blind BFS."""
    s0 = g.reset().score(); base = g.grid()
    objs, _ = online_find_clickables(g)
    if verbose: print(f"    {g.game_id}: {len(objs)} clickable object(s)")
    if not objs:
        return None
    cands = vlm_click_plan(base, objs)
    if verbose: print(f"    qwen3-vl proposed {len(cands)} click-order(s)")
    for cand in cands:
        g.reset(); won = False
        for (x, y) in cand:
            g.click(x, y)
            if g.status() == "WIN" or g.score() > s0: won = True; break
            if g.status() == "LOSE": break
        if won:
            if verbose: print(f"    ✅ {g.game_id} WON via qwen3-vl plan {cand}")
            return cand
    if verbose: print(f"    VLM plans didn't win; blind BFS fallback")
    return online_solve_click(g, max_depth=4, verbose=verbose)


def vlm_next_click(grid, clickables, tried, call_vlm):
    """One closed-loop step: show the CURRENT frame (numbered clickables), tell the VLM what was just
    clicked, ask for the single best NEXT click index toward winning. Returns an index or None."""
    import json
    b64 = render_png_b64(grid, clickables)
    hist = ", ".join(str(i) for i in tried[-6:]) or "none yet"
    prompt = (f"Puzzle game, hidden goal — you win by clicking the right objects in order. The "
              f"{len(clickables)} clickable objects are labelled 0..{len(clickables)-1} (yellow). You "
              f"have already clicked: [{hist}]. Look at the CURRENT state and pick the SINGLE next "
              f'object to click to make progress toward winning. Output ONLY JSON: {{"click": <index>, '
              f'"reason": "<short>"}}.')
    try:
        txt = call_vlm(b64, prompt)
        obj = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
        i = int(obj.get("click"))
        return i if 0 <= i < len(clickables) else None
    except Exception:
        return None


def online_solve_click_loop(g: OnlineGame, call_vlm, max_steps=14, verbose=True):
    """Closed-loop multimodal solve (the hypothesis-test loop): at each step the VLM SEES the current
    frame and picks the next click; we apply it, re-observe, and repeat — verified by win/score.
    More robust than one-shot for puzzles needing several visually-guided clicks."""
    s0 = g.reset().score()
    objs, _ = online_find_clickables(g)
    if verbose: print(f"    {g.game_id}: {len(objs)} clickable object(s); closed-loop (max {max_steps})")
    if not objs:
        return None
    tried = []; last = None; stuck = 0
    for step in range(max_steps):
        grid = g.grid()
        i = vlm_next_click(grid, objs, tried, call_vlm)
        if i is None:
            i = (max(tried[-1], -1) + 1) % len(objs) if tried else 0   # fallback: cycle objects
        before = grid
        g.click(*objs[i]); tried.append(i)
        if g.status() == "WIN" or g.score() > s0:
            if verbose: print(f"    ✅ {g.game_id} WON in {len(tried)} closed-loop clicks: {tried}")
            return tried
        if g.status() == "LOSE":
            if verbose: print(f"    {g.game_id} LOSE at step {step}")
            return None
        stuck = stuck + 1 if np.array_equal(before, g.grid()) else 0
        if stuck >= 3:                                  # frame frozen -> bail
            break
    if verbose: print(f"    {g.game_id}: closed-loop no win in {max_steps} steps (tried {tried})")
    return None


# ───────────────────────────── MOVEMENT solver (online, no clone) ──────────────────────────────────
# Code review finding: main()'s dispatch previously NEVER tried directional actions when solving --
# probe_modality() detected movement correctly, but no online_solve_movement existed and nothing ever
# called it, so movement / movement+click games (17/25 = 68% of the public set) structurally scored 0
# regardless of model/search quality. This section fixes that: the same reset()+replay approach as the
# click solvers, adapted from source_free_core.py's clone()-based operators to the no-clone online API.

def _dominant_translation(g0, g1, palette, min_size=8):
    """The translation (dx,dy) by which a same-colour blob moved -- picks the SMALLEST-magnitude match
    across all colour/component pairs, not the first one found (the first-match approach is fragile:
    it can consistently latch onto a same-sized decorative blob instead of the real single-step move).
    min_size=8 (raised from the offline default of 3): verified against ls20 that several small STATIC
    decorative icons (e.g. a row of same-size 'lives' pips, size ~4, a few px apart) get falsely cross-
    matched to EACH OTHER as a small 'translation' even though neither actually moved -- confirmed by
    direct inspection (repeated reset() is exactly deterministic; the false match is a same-color/same-
    size ambiguity, not state drift). Real player/carried blobs observed at size >=10, so min_size=8
    filters the false positives while keeping true matches."""
    best = None
    for col in palette:
        c0 = _comps(g0, col, min_size); c1 = _comps(g1, col, min_size)
        for (x0, y0, s0) in c0:
            for (x1, y1, s1) in c1:
                if abs(s1 - s0) <= 2:
                    d = (x1 - x0, y1 - y0)
                    if d != (0, 0) and (best is None or abs(d[0]) + abs(d[1]) < abs(best[0]) + abs(best[1])):
                        best = d
    return best

def online_learn_action_basis(g: OnlineGame, dirs=DIRS):
    """Learn ACTION->(dx,dy) for directional actions via reset()+step() (no clone). Fills a single
    unresolved direction by elimination (the 4 directions are a bijection onto the cardinals)."""
    g.reset(); g0 = g.grid()
    bg = int(np.bincount(g0.flatten()).argmax())
    pal = [int(c) for c in np.unique(g0) if int(c) != bg]
    basis = {}
    for a in dirs:
        g.reset(); g.step(a); g1 = g.grid()
        vec = None
        if g1.shape == g0.shape and not np.array_equal(g0, g1):
            vec = _dominant_translation(g0, g1, pal)
        basis[a] = vec
    CARD = {(0, -5), (0, 5), (-5, 0), (5, 0)}
    known = {v for v in basis.values() if v}
    unknown_acts = [a for a in dirs if basis[a] is None]
    missing = CARD - known
    if len(unknown_acts) == 1 and len(missing) == 1:
        basis[unknown_acts[0]] = missing.pop()
    return {a: v for a, v in basis.items() if v}

def online_find_agent_start(g: OnlineGame, basis, min_size=8):
    """Find the agent's start (pixel_pos, colour) via reset()+step() (no clone): the blob that
    translates by a KNOWN move vector -- identity by motion, not colour. Mirrors source_free_core's
    find_agent, adapted off env.clone() since the online API doesn't support it."""
    g.reset(); g0 = g.grid()
    bg = int(np.bincount(g0.flatten()).argmax())
    pal = [int(c) for c in np.unique(g0) if int(c) != bg]
    for a, vec in basis.items():
        g.reset(); g.step(a); g1 = g.grid()
        for col in pal:
            c0 = _comps(g0, col, min_size); c1 = _comps(g1, col, min_size)
            for (x0, y0, _) in c0:
                for (x1, y1, _) in c1:
                    if (x1 - x0, y1 - y0) == vec:
                        return (x0, y0), col
    return None

def online_explore_movement(g: OnlineGame, basis, cap=120, min_size=8):
    """Explore the reachable grid via reset()+replay (no clone): dead-reckoned BFS over positions,
    move-vs-blocked classified by TRACKED PLAYER POSITION (AgentTracker), not raw frame-equality --
    raw equality misclassifies a real wall-hit as 'moved' whenever an unrelated pixel (e.g. a HUD/
    animation tick) also differs, and can under-count real walls. Checks win/score after every
    replayed action (so a game winnable by pure movement is caught immediately). Returns a dict with
    start/paths/edges/blocked/won_path."""
    s0 = g.reset().score()
    agent = online_find_agent_start(g, basis, min_size)
    if agent is None:
        return dict(start=(0, 0), paths={(0, 0): []}, edges={}, blocked=[], won_path=None)
    p0, col0 = agent
    pal = [int(c) for c in np.unique(g.grid())]
    tracker = AgentTracker({a: ("move", v) for a, v in basis.items()}, pal, min_size=min_size)

    start = (0, 0)
    # cell -> (dead-reckoned pos, tracked pixel pos, tracked colour)
    paths = {start: []}; pix = {start: (p0, col0)}; edges = {}; order = [start]; i = 0
    blocked = []; won_path = None
    while i < len(order) and len(order) < cap and won_path is None:
        cell = order[i]; i += 1
        edges[cell] = {}
        for a, v in basis.items():
            path = paths[cell]
            g.reset(); ok = True
            for step_a in path:
                g.step(step_a)
                if g.status() != "PLAYING":
                    ok = False; break
            if not ok:
                continue
            pc, col = pix[cell]
            g.step(a)
            if g.status() == "WIN" or g.score() > s0:
                won_path = path + [a]; break
            if g.status() == "LOSE":
                continue
            mv = tracker.step_result(g.grid(), pc, col, a)
            if mv is None:
                blocked.append((cell, a))            # tracked position didn't advance -> wall / gated
                continue
            npos = (cell[0] + v[0], cell[1] + v[1])
            edges[cell][a] = npos
            if npos not in paths:
                paths[npos] = path + [a]; pix[npos] = mv; order.append(npos)
    return dict(start=start, paths=paths, edges=edges, blocked=blocked, won_path=won_path, pix=pix)

def online_discover_transformers(g: OnlineGame, basis, graph, cap=30, min_diff=6):
    """Detect cells whose touch-then-return PERSISTENTLY changes carried/world state (station-like).
    Compares state_signature (pixels FAR from the tracked agent position) before vs after an out-and-
    back, not the raw full frame -- a raw full-frame compare is confounded by the agent's own sprite
    changing appearance with facing/last-move-direction (a cosmetic effect near the agent). Requires a
    SUBSTANTIAL signature diff (>= min_diff pixels): verified against ls20 that a trivial round-trip
    from the very start (nowhere near any real station) already flips a tiny ~4-pixel patch in the HUD
    corner every time -- likely a per-action cosmetic tick, not a real config change -- so a small-size
    threshold (consistent with the min_change used for click-detection elsewhere in this file) is
    needed the same way it was for clicks."""
    edges, paths, pix = graph["edges"], graph["paths"], graph["pix"]
    opp = {a: next((b for b, v2 in basis.items() if v2 == (-v[0], -v[1])), None) for a, v in basis.items()}
    transformers = []; checked = 0
    for A, nbrs in edges.items():
        if checked >= cap: break
        pA, _ = pix[A]
        for a, X in nbrs.items():
            back = opp.get(a)
            if back is None or X == A:
                continue
            checked += 1
            g.reset()
            for step_a in paths[A]: g.step(step_a)
            before = _sfc.state_signature(g.grid(), pA)
            g.step(a); g.step(back)
            after = _sfc.state_signature(g.grid(), pA)
            if len(before ^ after) >= min_diff:
                transformers.append(X)
    return sorted(set(transformers))

def online_solve_movement(g: OnlineGame, max_cycles=3, cap=120, verbose=True):
    """Solve a MOVEMENT game online: learn the action basis, explore the reachable grid (win-checked
    inline), discover transformer cells by persistent effect, then search (transformer x cycle-count x
    blocked-frontier), verified by win/score. Adapts source_free_core.solve_single_goal's approach to
    the no-clone online API. Honest scope: no hidden-resource (energy) inference or multi-goal chaining
    here yet -- this fixes the 'movement is never even attempted' structural gap, it does not claim to
    replicate the full local ls20 pipeline's depth."""
    from collections import deque
    basis = online_learn_action_basis(g)
    if len(basis) < 2:
        if verbose: print(f"    {g.game_id}: action basis too weak {basis} -- not a movement game")
        return None
    if verbose: print(f"    {g.game_id}: action basis {basis}")
    graph = online_explore_movement(g, basis, cap=cap)
    if graph["won_path"]:
        if verbose: print(f"    ✅ {g.game_id} WON by exploration alone: {graph['won_path']}")
        return graph["won_path"]
    if verbose:
        print(f"    {g.game_id}: explored {len(graph['paths'])} cells, "
              f"{len(graph['blocked'])} blocked frontiers")
    if not graph["blocked"]:
        if verbose: print(f"    {g.game_id}: no blocked frontier to target, no win during exploration")
        return None
    transformers = online_discover_transformers(g, basis, graph)
    if verbose: print(f"    {g.game_id}: {len(transformers)} transformer cell(s): {transformers}")

    edges, paths, start = graph["edges"], graph["paths"], graph["start"]
    def bfs_path(src, dst):
        if src == dst: return []
        seen = {src}; q = deque([(src, [])])
        while q:
            c, p = q.popleft()
            for a, nb in edges.get(c, {}).items():
                if nb in seen: continue
                if nb == dst: return p + [a]
                seen.add(nb); q.append((nb, p + [a]))
        return None
    def offback(C):
        for a, nb in edges.get(C, {}).items():
            if nb == C: continue
            back = next((b for b, nb2 in edges.get(nb, {}).items() if nb2 == C), None)
            if back: return [a, back]
        return None
    def cycle_path(C, k):
        bp = bfs_path(start, C)
        if bp is None or k <= 1: return bp
        ob = offback(C)
        return None if ob is None else bp + ob * (k - 1)

    s0 = g.reset().score()
    def try_plan(plan):
        g.reset()
        for a in plan:
            g.step(a)
            if g.status() == "WIN" or g.score() > s0: return True
            if g.status() == "LOSE": return False
        return False

    depth = lambda p: abs(p[0]) + abs(p[1])
    ftgt = lambda fr: (fr[0][0] + basis[fr[1]][0], fr[0][1] + basis[fr[1]][1])
    frontiers = sorted(graph["blocked"], key=lambda fr: -depth(ftgt(fr)))
    cands = transformers or sorted(paths.keys(), key=lambda c: -depth(c))[:12]

    # Honest caveat: transformer discovery can false-positive (e.g. a minimap/secondary position
    # indicator elsewhere on screen can look like a persistent state change even with no real
    # station touched -- observed on ls20). Bound the search so a bad transformer list can't burn
    # thousands of network round-trips chasing a false lead; each trial here is a real API replay.
    trial_cap = 300; t0 = time.time(); time_cap = 90.0
    trials = 0
    for (cell, a) in frontiers:
        for C in cands:
            for k in range(1, max_cycles + 1):
                if trials >= trial_cap or time.time() - t0 > time_cap:
                    if verbose: print(f"    {g.game_id}: search capped ({trials} trials, {time.time()-t0:.0f}s)")
                    return None
                cp = cycle_path(C, k)
                if cp is None: continue
                seg = bfs_path(C, cell)
                if seg is None: continue
                plan = cp + seg + [a]; trials += 1
                if try_plan(plan):
                    if verbose:
                        print(f"    ✅ {g.game_id} WON via transformer {C}x{k} -> frontier "
                              f"({trials} trials): {plan}")
                    return plan
    if verbose: print(f"    {g.game_id}: no winning (transformer x cycle x frontier) combo ({trials} trials)")
    return None


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except Exception:
        pass
    key = os.environ.get("ARC_API_KEY", "")
    args = sys.argv[1:]
    mode = ("loop" if "--loop" in args else "vlm" if "--vlm" in args else "llm" if "--llm" in args
            else "solve" if "--solve" in args else "probe")
    games = [a for a in args if not a.startswith("--")]

    def call_llm(prompt, payload):
        import json, urllib.request
        body = json.dumps({"model": "local/gpt-oss-120b:opt", "temperature": 0, "max_tokens": 2000,
                           "messages": [{"role": "system", "content": prompt},
                                        {"role": "user", "content": json.dumps(payload)}]}).encode()
        req = urllib.request.Request("http://localhost:11437/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    arcade = Arcade(arc_api_key=key, operation_mode=OperationMode.ONLINE)
    card = arcade.open_scorecard(tags=[f"sf-{mode}"])
    click_solvers = {
        "loop": lambda g: online_solve_click_loop(g, lambda b, p: call_vlm(b, p)),
        "vlm": online_solve_click_vlm,
        "llm": lambda g: online_solve_click_llm(g, call_llm),
        "solve": lambda g: online_solve_click(g, max_depth=4),
    }
    try:
        for gid in games:
            g = OnlineGame(arcade, gid, card)
            t = time.time()
            if mode == "probe":
                mod, mv, ck = probe_modality(g)
                print(f"{gid:18s} modality={mod:8s} (move-dirs={mv}, click-responders={ck})  {time.time()-t:.0f}s")
                continue
            # Route by DETECTED modality instead of blindly click-solving everything: movement and
            # movement+click games (17/25 = 68% of the public set) previously always scored 0 here
            # because no solver ever tried a directional action.
            mod, mv, ck = probe_modality(g)
            print(f"{gid}: modality={mod} (move-dirs={mv}, click-responders={ck})")
            plan = None
            if mod in ("movement", "movement+click"):
                plan = online_solve_movement(g)
            if plan is None and mod in ("click", "movement+click", "unknown"):
                plan = click_solvers[mode](g)
            print(f"      {'WON' if plan else 'no win'} ({time.time()-t:.0f}s)")
    finally:
        sc = arcade.close_scorecard(card)
        try:
            print("SCORECARD:", sc.model_dump() if hasattr(sc, "model_dump") else sc)
        except Exception:
            pass


if __name__ == "__main__":
    main()
