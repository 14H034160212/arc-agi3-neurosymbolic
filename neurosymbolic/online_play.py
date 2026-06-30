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
    try:
        for gid in games:
            g = OnlineGame(arcade, gid, card)
            t = time.time()
            if mode == "loop":
                vlm_call = lambda b, p: call_vlm(b, p)
                online_solve_click_loop(g, vlm_call)
                print(f"      ({time.time()-t:.0f}s)")
            elif mode == "vlm":
                online_solve_click_vlm(g)
                print(f"      ({time.time()-t:.0f}s)")
            elif mode == "llm":
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
