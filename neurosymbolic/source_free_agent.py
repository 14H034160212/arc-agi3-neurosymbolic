#!/usr/bin/env python3
"""
source_free_agent — solve ls20 from PIXELS + ACTIONS only (no engine internals for decisions).
Every decision uses ONLY the rendered grid + win/lose feedback from the environment.

THE INSIGHTS THAT MAKE A SOURCE-FREE SOLVE WORK:

1) DEAD-RECKON BY THE KNOWN ACTION DIRECTION, NOT BY A FIXED COLOUR.
   The player is the carried sprite; its appearance changes at stations (rotation reshapes it,
   a COLOUR station even recolours it: orange->colour-9). So we (a) learn each action's direction
   once by elimination (the four actions map bijectively to the four cardinals), and (b) track the
   player COLOUR-AGNOSTICALLY: after an action, the player is whatever player-palette blob now sits
   at the dead-reckoned position (and has left the old one). Robust to reshape AND recolour.

2) AN UNSOLVED SLOT *BLOCKS* MOVEMENT (5x5 box reject) UNTIL THE CARRIED CONFIG MATCHES.
   So a slot reads as a wall at first -> we record BLOCKED FRONTIERS, prioritised by proximity to
   the blue (colour-9) slot-marker components; the frontier that becomes enterable + wins is it.

3) STATIONS: VISIBLE (shape/colour change the player's pixels) vs INVISIBLE (a rotation that renders
   identically). Visible stations are detected during exploration (the player recolours/reshapes on
   entry). The invisible rotation station is found by the WIN ORACLE (brute pass-through). Stations
   are cycled k times by an enter / off-and-back / re-enter maneuver.

4) RESET RETURNS TO L0 — LEVELS ARE REACHED BY WINNING, NOT BY JUMPING.
   To work on level k we PREFIX every replay with the known winning solutions for L0..L(k-1)
   (cached as a deepcopy checkpoint == RESET + replay). Solutions chain level by level.

RESULT (pixels + win/lose feedback only): solves ls20 L0 and L1 END-TO-END, chained (win L0 to reach
L1). L0 = one rotation cycle; L1 = three rotation cycles via the off-and-back cycling maneuver. The
full perception stack works: action->direction by elimination, colour-agnostic multi-part player
tracking (hint-colour continuity + centroid-closeness, robust to the carried-object sub-blob), and
blue-component slot prioritisation. With engine state (ls20_solver.py) all 7 solve; this file
measures how far PIXELS-ONLY reaches: 2/7.

L2 SOLVED IN PARTS — the remaining blocker is ENERGY, precisely isolated:
  * Station detection WORKS. The carried config is shown in a FIXED HUD inventory panel (a corner),
    not on the player sprite — so we detect a station as a cell whose on-and-back touch PERSISTENTLY
    changes the far-from-body player-palette pixels (the HUD), compared at the SAME position so the
    colour-9-overloaded slot markers cancel. L2's colour station and rotation station are both found.
  * Config logic WORKS. Cycling colour x1 + rotation x3 reaches exactly the slot requirement (5,1,3),
    verified on the engine.
  * BLOCKER = ENERGY. The full traversal (start -> colour stn -> rotation stn -> 3 cycles -> slot) is
    ~68 steps but energy is 42 and -1/step; the agent dies en route, and DEATH RESETS THE CARRIED
    CONFIG, so it never delivers. L2 needs ENERGY-AWARE ROUTING through the refill (iri) stations
    (detectable source-free: the yellow energy bar grows). That is the next milestone.
Multi-slot levels (e.g. L5) additionally need an intermediate-slot-solved detector. Honest status:
2/7 pixels-only end-to-end; L2's perception/logic solved, only energy-budgeted planning remains.
"""
from __future__ import annotations
import copy
import importlib.util
import time
from collections import deque
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
SLOT, LAST_LEVEL = 9, 6
PCOLORS = [12, 9, 14, 8]          # carried-object palette (hul): the player is always one of these
ACTS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

def _ls20():
    s = importlib.util.spec_from_file_location("ls20mod", ENV)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.Ls20

def render(g): return np.array(g.get_pixels(0, 0, 64, 64))

def comps_centroids(grid, color, min_size=2):
    """Connected components of `color`; return [(cx, cy, size)] (screen coords)."""
    H, W = grid.shape; seen = np.zeros_like(grid, bool); out = []
    for y in range(H):
        for x in range(W):
            if grid[y, x] == color and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == color:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(cells) >= min_size:
                    out.append((round(sum(c[1] for c in cells)/len(cells)),
                                round(sum(c[0] for c in cells)/len(cells)), len(cells)))
    return out

PLAYER_MIN = 5   # the carried sprite is a chunky blob; ignore small same-palette decorations

def player_at(grid, near, hint, tol):
    """Find the player blob nearest `near` within `tol`. The player sprite is multi-part (e.g. an
    orange body plus a colour-9 sub-element), so we track ONE part by colour-continuity: prefer the
    `hint` colour; only fall back to other palette colours when the hint colour has vanished within
    range (a genuine recolour at a colour station). Returns (dist, (cx,cy), color, size) or None."""
    for cols in ([hint], [c for c in PCOLORS if c != hint]):
        best = None
        for col in cols:
            for (cx, cy, sz) in comps_centroids(grid, col, PLAYER_MIN):
                d = abs(cx - near[0]) + abs(cy - near[1])
                if d <= tol and (best is None or d < best[0]):
                    best = (d, (cx, cy), col, sz)
        if best is not None:
            return best
    return None

def hud_sig(grid, body_center):
    """Signature of the carried-object state, read from the FIXED HUD inventory panel (ls20 renders
    the carried object's colour/shape in a corner, NOT on the player sprite). We take all player-
    palette pixels FAR from the body: across two frames at the SAME player position, static walls/
    slots/decor cancel, the moving body is excluded, and only a persistent carried-config change (the
    HUD indicator recolouring/reshaping) survives. Robust to the colour-9 overload."""
    sig = []
    for col in PCOLORS:
        ys, xs = np.where(grid == col)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if abs(x - body_center[0]) + abs(y - body_center[1]) > 12:
                sig.append((col, x, y))
    return frozenset(sig)

def find_start_player(base, DIR, AIN):
    """Identify the player at a level start as the blob that TRANSLATES by a known action's vector
    (robust identity — not just 'first same-colour component'). Returns ((cx,cy), color) or None."""
    from copy import deepcopy
    gb = render(base)
    for a, d in DIR.items():
        gg = deepcopy(base); gg.perform_action(AIN[a]); ga = render(gg)
        for col in PCOLORS:
            cb = comps_centroids(gb, col, PLAYER_MIN); ca = comps_centroids(ga, col, PLAYER_MIN)
            for (bx, by, _) in cb:
                for (ax, ay, _) in ca:
                    if (round((ax-bx)/5)*5, round((ay-by)/5)*5) == d:
                        return (bx, by), col
    return None

def learn_dirs(verbose=True):
    """Learn ACTION->(dx,dy) ONCE (a fixed game property). Probe from several positions and fill a
    single remaining unknown by ELIMINATION (the one missing cardinal)."""
    Ls20 = _ls20(); from arcengine import GameAction, ActionInput
    AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in ACTS}
    RESET = ActionInput(id=GameAction.RESET)
    CARD = {(0, -5), (0, 5), (-5, 0), (5, 0)}
    DIR = {a: None for a in ACTS}

    def pcent(g):
        for col in PCOLORS:
            cs = comps_centroids(render(g), col, 3)
            if cs: return (cs[0][0], cs[0][1])
        return None
    def probe_from(g):
        for a in ACTS:
            if DIR[a] is not None: continue
            gg = copy.deepcopy(g); c0 = pcent(gg); gg.perform_action(AIN[a]); c1 = pcent(gg)
            if c0 and c1 and c1 != c0:
                d = (round((c1[0]-c0[0])/5)*5, round((c1[1]-c0[1])/5)*5)
                if d in CARD: DIR[a] = d

    g = Ls20(); g.perform_action(RESET); probe_from(g)
    for _ in range(12):
        if all(v is not None for v in DIR.values()): break
        moved = False
        for a in ACTS:
            if DIR[a] is not None:
                c0 = pcent(g); g.perform_action(AIN[a])
                if pcent(g) != c0: moved = True; break
        probe_from(g)
        if not moved: g = Ls20(); g.perform_action(RESET)
    unknown = [a for a in ACTS if DIR[a] is None]
    missing = CARD - {v for v in DIR.values() if v is not None}
    if len(unknown) == 1 and len(missing) == 1:
        DIR[unknown[0]] = missing.pop()
    if verbose: print("learned dirs (once):", DIR)
    assert sorted(DIR.values()) == sorted(CARD), f"failed to learn all four directions: {DIR}"
    return DIR

def solve_level(level, prefix=(), dirs=None, trial_cap=8000, time_cap=160.0, verbose=True):
    """Solve one ls20 level from pixels + win/lose feedback only.
    `prefix` = known winning actions for earlier levels; `dirs` = learned action->direction map."""
    Ls20 = _ls20(); from arcengine import GameAction, ActionInput
    AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in ACTS}
    RESET = ActionInput(id=GameAction.RESET)
    DIR = dirs or learn_dirs(verbose=False)
    t0 = time.time()

    base = Ls20(); base.perform_action(RESET)
    for a in prefix:
        base.perform_action(AIN[a])
    def won(gg): return gg.level_index > level or gg._state.name == "WIN"
    if won(base):
        return False, dict(level=level, error="prefix overshot")

    # player anchor + colour at the level-start checkpoint (identified by motion under a known action)
    g0 = render(base)
    sp = find_start_player(base, DIR, AIN)
    if sp is None:
        return False, dict(level=level, error="could not locate player at start")
    P0, COL0 = sp

    def classify(grid, pc, col, a):
        """Did action `a` move the player? Decide by whether the player centroid is now closer to the
        new lattice point (exp) than the old (pc) — robust to blob size, reshape, and recolour.
        Returns (new_pc, new_col) if moved, else None (blocked)."""
        exp = (pc[0] + DIR[a][0], pc[1] + DIR[a][1])
        cand = player_at(grid, pc, col, tol=6) or player_at(grid, exp, col, tol=6)
        if cand is None: return None
        c = cand[1]
        dexp = abs(c[0]-exp[0]) + abs(c[1]-exp[1]); dpc = abs(c[0]-pc[0]) + abs(c[1]-pc[1])
        return (exp, cand[2]) if dexp < dpc else None

    def trace(path):
        """Execute RESET+prefix+path (via the cached checkpoint); track the player colour-agnostically.
        Return (dead_reckon_pos, body_centre_screen, colour, won, final_grid)."""
        gg = copy.deepcopy(base); pc, col, pos = P0, COL0, (0, 0); grid = render(gg)
        for a in path:
            gg.perform_action(AIN[a]); grid = render(gg)
            mv = classify(grid, pc, col, a)
            if mv:
                pc, col = mv; pos = (pos[0]+DIR[a][0], pos[1]+DIR[a][1])
            if won(gg):
                return pos, pc, col, True, grid
        return pos, pc, col, False, grid

    def run(path):
        pos, _, _, w, _ = trace(path); return pos, w

    def run_fast(path):
        """Execute a planned action sequence and report only WIN (environment feedback). No pixel
        tracking — the sequence was already planned from the pixel-derived graph; this just asks the
        environment 'did it win?'. Much faster than trace (no get_pixels), used for the search."""
        gg = copy.deepcopy(base)
        for a in path:
            gg.perform_action(AIN[a])
            if won(gg): return True
        return False

    # --- explore the freely-reachable graph; record cells, blocked frontiers, VISIBLE stations
    start = (0, 0)
    ckpt = {start: (copy.deepcopy(base), P0, COL0)}
    paths = {start: []}; edges = {}; order = [start]; i = 0
    blocked = []
    while i < len(order) and len(order) < 200:
        cell = order[i]; i += 1
        gC, pcC, colC = ckpt[cell]
        edges[cell] = {}
        for a in ACTS:
            gg = copy.deepcopy(gC); gg.perform_action(AIN[a]); grid = render(gg)
            mv = classify(grid, pcC, colC, a)
            if mv is None:
                blocked.append((cell, a)); continue       # wall or unsolved-slot reject
            exp, ncol = mv
            npos = (cell[0] + DIR[a][0], cell[1] + DIR[a][1])
            edges[cell][a] = npos
            if npos not in paths:
                paths[npos] = paths[cell] + [a]; ckpt[npos] = (gg, exp, ncol); order.append(npos)
    if verbose:
        print(f"L{level}: explored {len(paths)} cells; blocked frontiers={len(blocked)}")

    def bfs(src, dst):
        if src == dst: return []
        seen = {src}; q = deque([(src, [])])
        while q:
            c, p = q.popleft()
            for a, nb in edges.get(c, {}).items():
                if nb in seen: continue
                if nb == dst: return p + [a]
                seen.add(nb); q.append((nb, p + [a]))
        return None

    depth = lambda p: abs(p[0]) + abs(p[1])
    ftgt = lambda fr: (fr[0][0] + DIR[fr[1]][0], fr[0][1] + DIR[fr[1]][1])
    blue_dr = [(round((cx - P0[0]) / 5) * 5, round((cy - P0[1]) / 5) * 5)
               for (cx, cy, _) in comps_centroids(g0, SLOT)]
    nearblue = lambda p: min((abs(p[0]-b[0])+abs(p[1]-b[1]) for b in blue_dr), default=depth(p))
    frontiers = sorted(blocked, key=lambda fr: (nearblue(ftgt(fr)), -depth(ftgt(fr))))
    cells = sorted(paths.keys(), key=lambda c: -depth(c))
    if verbose: print(f"L{level}: blue markers (dead-reckon) {blue_dr}")

    def offback(C):
        for a, nb in edges.get(C, {}).items():
            if nb == C: continue
            for b, m in edges.get(nb, {}).items():
                if m == C: return [a, b]
        return None

    # --- disambiguate STATIONS from slots/refills/walls: a station PERSISTENTLY changes the carried
    # config. Probe each reachable cell X by stepping onto it and back to a neighbour A, comparing the
    # player signature AT A before vs after. Static slot markers (colour-9 overload!) sit at the same
    # screen position both times -> they cancel; only a real carried-config change survives.
    def find_stations():
        sts = []
        nbr = {}                                      # X -> (A, action A->X)
        for A in paths:
            for a, X in edges.get(A, {}).items():
                if X != A and X not in nbr:
                    nbr[X] = (A, a)
        for X, (A, a) in nbr.items():
            if time.time() - t0 > time_cap * 0.5: break
            b = next((x for x in ACTS if DIR[x] == (-DIR[a][0], -DIR[a][1])), None)
            if b is None: continue
            _, cenA, _, _, gA = trace(paths[A])
            _, cenA2, _, w2, gA2 = trace(paths[A] + [a, b])
            if w2 or abs(cenA2[0]-cenA[0]) + abs(cenA2[1]-cenA[1]) > 2:
                continue                              # didn't cleanly return to A
            if hud_sig(gA2, cenA2) != hud_sig(gA, cenA):
                sts.append(X)                         # carried config changed for good -> a station
        return sts
    detected_stations = sorted(set(find_stations()), key=lambda c: -depth(c))
    if verbose: print(f"L{level}: detected stations (persistent config change) {detected_stations}")

    def cycle_path(C, k):
        bp = bfs(start, C)
        if bp is None: return None
        if k <= 1: return bp
        ob = offback(C)
        return None if ob is None else bp + ob * (k - 1)
    MAXK = 4

    trials = 0
    def try_combos(combo_iter, dval):
        nonlocal trials
        for build, (cell, a) in combo_iter:
            if trials >= trial_cap or time.time() - t0 > time_cap:
                return ("capped", None)
            pre = build()
            if pre is None: continue
            leg = bfs(pre[1], cell)
            if leg is None: continue
            sol = pre[0] + leg + [a]; trials += 1
            if run_fast(sol):
                return ("won", (sol, pre[2], ftgt((cell, a)), dval))
        return ("exhausted", None)

    # D=1: one station cycled k times (L0: rotation k=1; L1: rotation k=3)
    def one(C, k):
        return lambda: ((cp, C, [(C, k)]) if (cp := cycle_path(C, k)) is not None else None)
    g1 = ((one(C, k), fr) for fr in frontiers for C in cells for k in range(1, MAXK + 1))
    status, res = try_combos(g1, 1)

    # D=2: two stations, each cycled. Both L2 stations (colour + rotation) are detected, so search
    # detected x detected FIRST (small); fall back to detected x all-cells for any undetected station.
    def two(C1, k1, C2, k2):
        def b():
            cp = cycle_path(C1, k1)
            if cp is None: return None
            leg = bfs(C1, C2)
            if leg is None: return None
            cp = cp + leg
            if k2 > 1:
                ob = offback(C2)
                if ob is None: return None
                cp = cp + ob * (k2 - 1)
            return (cp, C2, [(C1, k1), (C2, k2)])
        return b
    if status != "won":
        c1set = detected_stations or cells[:12]
        for c2set in ([detected_stations] if detected_stations else []) + [cells]:
            g2 = ((two(c1, k1, c2, k2), fr) for fr in frontiers
                  for c1 in c1set for k1 in range(1, MAXK + 1)
                  for c2 in c2set for k2 in range(1, MAXK + 1) if c2 != c1)
            status, res = try_combos(g2, 2)
            if status == "won": break

    dt = time.time() - t0
    if res:
        sol, passlist, slot_at, dval = res
        if verbose:
            print(f"  ✅ SOLVED L{level} FROM PIXELS ONLY — {len(sol)} actions, {trials} trials, "
                  f"D={dval}, {dt:.0f}s; stations {passlist}, slot @ {slot_at}")
        return True, dict(level=level, actions=len(sol), trials=trials, D=dval, secs=round(dt, 1), sol=sol)
    if verbose:
        print(f"  ❌ L{level}: no winning combination ({trials} trials, {status}, {dt:.0f}s)")
    return False, dict(level=level, trials=trials, status=status, secs=round(dt, 1))

def main():
    import sys
    levels = [int(x) for x in sys.argv[1:]] or list(range(LAST_LEVEL + 1))
    DIR = learn_dirs()
    results = []; prefix = []
    for lv in levels:
        ok, info = solve_level(lv, prefix=tuple(prefix), dirs=DIR)
        results.append((lv, ok, info)); print()
        if ok:
            prefix += info["sol"]
        else:
            print(f"(stopping chain: cannot reach levels beyond L{lv} without its solution)"); break
    print("=== SOURCE-FREE SUMMARY (pixels + win/lose feedback only) ===")
    nsolved = sum(1 for _, ok, _ in results if ok)
    for lv, ok, info in results:
        tag = "✅" if ok else "❌"
        extra = (f"{info['actions']} acts, {info['trials']} trials, D={info['D']}, {info['secs']}s"
                 if ok else f"{info.get('trials','?')} trials, {info.get('status','')}, {info.get('secs','?')}s")
        print(f"  {tag} L{lv}: {extra}")
    print(f"  TOTAL: {nsolved} consecutive levels solved source-free (chained from L0)")

if __name__ == "__main__":
    main()
