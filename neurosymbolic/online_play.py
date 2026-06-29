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
import os, sys, time
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

DIRS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class OnlineGame:
    """Thin wrapper over one real game session (reset/step/click/render/status/score)."""
    def __init__(self, arcade, game_id, card):
        self.env = arcade.make(game_id, scorecard_id=card); self.game_id = game_id
        self._last = None
    def reset(self):
        self._last = self.env.reset(); return self
    def step(self, name):
        r = self.env.step(getattr(GameAction, name))
        if r is not None: self._last = r
        return self
    def click(self, x, y):
        r = self.env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        if r is not None: self._last = r
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


def probe_modality(g: OnlineGame, click_samples=14):
    """Source-free modality detection on a real game, using only reset()+step() (no clone):
    movement if >=2 directional actions translate a blob; else click if clicking a foreground cell
    changes the frame; else unknown. Returns (modality, n_move_dirs, n_click_responders)."""
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
        g.reset(); g.click(x, y)
        if g.grid().shape != g0.shape or not np.array_equal(g0, g.grid()) or g.status() != "PLAYING":
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


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except Exception:
        pass
    key = os.environ.get("ARC_API_KEY", "")
    args = sys.argv[1:]
    mode = "llm" if "--llm" in args else "solve" if "--solve" in args else "probe"
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
    try:
        for gid in games:
            g = OnlineGame(arcade, gid, card)
            t = time.time()
            if mode == "llm":
                online_solve_click_llm(g, call_llm)
                print(f"      ({time.time()-t:.0f}s)")
            elif mode == "solve":
                online_solve_click(g, max_depth=4)
                print(f"      ({time.time()-t:.0f}s)")
            else:
                mod, mv, ck = probe_modality(g)
                print(f"{gid:18s} modality={mod:8s} (move-dirs={mv}, click-responders={ck})  {time.time()-t:.0f}s")
    finally:
        sc = arcade.close_scorecard(card)
        try:
            print("SCORECARD:", sc.model_dump() if hasattr(sc, "model_dump") else sc)
        except Exception:
            pass


if __name__ == "__main__":
    main()
