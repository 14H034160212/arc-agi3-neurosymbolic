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

5) STATIONS ARE READ FROM A FIXED HUD PANEL; the carried config is rendered in a corner inventory,
   not on the player sprite. A station = a cell whose on-and-back touch PERSISTENTLY changes the
   far-from-body player-palette pixels (the HUD), compared at the SAME position so the colour-9-
   overloaded slot markers cancel. The win signal then confirms which stations/cycles deliver.

6) ENERGY IS A HIDDEN VARIABLE — it is NOT in the 64x64 render at all (drains with zero pixel change).
   We measure it source-free with an ORACLE: spam moves until the player dies and teleports back to
   start; the step count IS the energy (-1/step) -> EMAX. Refills (iri) ARE visible (colour-11) and
   reset energy to EMAX. Routing is then energy-aware Dijkstra over (cell, energy) with PROACTIVE
   refuelling (min-steps alone would arrive drained and strand the long final leg), avoiding other
   stations so the carried config isn't perturbed. Death resets the config, so staying alive matters.

RESULT (pixels + win/lose feedback only): solves ls20 L0-L4 END-TO-END, chained (win each to reach the
next). L0 = 1 rotation cycle (D=1); L1 = 3 rotation cycles (D=1); L2 = colour + rotation, energy-routed
through refills (D=2); L3 (D=2); L4 = shape + colour + rotation (D=3). The search is FRONTIER-OUTER,
interleaving D=1/2/3 station-cycle combos over the detected stations, delivery cells ordered by the
blue slot-marker. With engine state (ls20_solver.py) all 7 solve; PIXELS-ONLY now reaches 5/7.

MULTI-SLOT (L5) — infrastructure built, search not yet converging: L5 has TWO slots with different
configs, so a single delivery doesn't win. We detect a solved slot ROBUSTLY by a DROP in the
marker-component count (solving turns a slot's requirement marker inert; player/sub-blob drift can
only ADD a component, never remove one -> no false positives), and chain sub-goals (solve slot1,
re-explore from that state, solve slot2). Tractability fixes: index-ordered station COMBINATIONS (not
permutations), dedup stations to one cyclable rep, MAXK=6 (shape needs 0->5). L5 engages correctly
(detects 2 slots) but the per-slot D=3 + energy search doesn't find slot1's config within budget.
NEXT: compute exact cycle counts instead of brute-searching k — the slot marker ENCODES the required
colour (colour-9 -> tmx 1, colour-8 -> tmx 3), and the rotation station is the one with no HUD change,
so colour/shape/rotation cycles can be derived, collapsing the search. L6 = single-slot D=3 (win =
GameState.WIN) — solvable by the current agent once it can chain past L5.
"""
from __future__ import annotations
import copy
import importlib.util
import time
from collections import deque
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
SLOT, REFILL, LAST_LEVEL = 9, 11, 6
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

MARKER_COLORS = [9, 14, 8]    # carried-palette colours used as slot-requirement markers (not the body-12)

def count_markers(grid, body_center):
    """Count the slots' requirement-marker COMPONENTS, excluding the player neighbourhood and the
    bottom-left carried HUD corner. A delivery that SOLVES a slot turns its marker inert -> the count
    DROPS by one. We detect intermediate slot solves by a DECREASE (robust: player/sub-blob drift can
    only add a component, never remove one — so it can't cause a false 'solved')."""
    n = 0
    for col in MARKER_COLORS:
        for (cx, cy, _) in comps_centroids(grid, col, 3):
            if abs(cx-body_center[0]) + abs(cy-body_center[1]) > 12 and not (cx < 18 and cy > 50):
                n += 1
    return n

def solve_level(level, prefix=(), dirs=None, trial_cap=8000, time_cap=160.0, verbose=True,
                partial_ok=False, maxk=4):
    """Solve one ls20 level from pixels + win/lose feedback only.
    `prefix` = known winning actions for earlier levels; `dirs` = learned action->direction map.
    `partial_ok` (multi-slot): accept a move that solves ONE slot (detected by slot_state change),
    not just a full win; the caller then chains sub-goals. `maxk` = max station cycles to try."""
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

    # --- ENERGY (a HIDDEN variable — not in the render). Refills ARE visible (colour-11 markers) and
    # reset energy to EMAX. We measure EMAX source-free with an oracle: spam moves until the player
    # dies and teleports back to start; the step count IS the energy at that cell (−1/step).
    refills = set()
    for (cx, cy, _) in comps_centroids(g0, REFILL):
        rc = (round((cx - P0[0]) / 5) * 5, round((cy - P0[1]) / 5) * 5)
        near = min(paths, key=lambda c: abs(c[0]-rc[0]) + abs(c[1]-rc[1]), default=None)
        if near is not None and abs(near[0]-rc[0]) + abs(near[1]-rc[1]) <= 5:
            refills.add(near)
    station_set = set(detected_stations)

    opp = {a: next((b for b in ACTS if DIR[b] == (-DIR[a][0], -DIR[a][1])), a) for a in ACTS}
    def energy_at(path_to):
        gg = copy.deepcopy(base); pc, col = P0, COL0
        for a in path_to:
            gg.perform_action(AIN[a]); mv = classify(render(gg), pc, col, a)
            if mv: pc, col = mv
        osc = [ACTS[0], opp[ACTS[0]]]
        for step in range(1, 60):
            gg.perform_action(AIN[osc[step % 2]]); grid = render(gg)
            f = player_at(grid, pc, col, 7)
            if f is None or abs(f[1][0]-pc[0]) + abs(f[1][1]-pc[1]) > 8:
                z = player_at(grid, P0, COL0, 5)
                if z and abs(z[1][0]-P0[0]) + abs(z[1][1]-P0[1]) <= 4:
                    return step                      # died here -> energy was `step`
            if f: pc, col = f[1], f[2]
        return None
    # EMAX = min over a few deep cells of (energy_there + steps_to_there) — a refill-free path gives EMAX
    EMAX = 42
    ests = []
    for C in [c for c in cells if c not in refills][:4]:
        e = energy_at(paths[C])
        if e: ests.append(e + len(paths[C]))
    if ests: EMAX = min(ests) - 1            # the oracle lags death detection by ~1 frame; be conservative
    if verbose: print(f"L{level}: refills(dead-reckon)={sorted(refills)} EMAX={EMAX}")

    import heapq
    def eroute(src, dst, e_in, avoid):
        """Min-step path src->dst keeping energy>0; refills reset energy to EMAX; never pass `avoid`
        cells (other stations) so the carried config isn't perturbed. Returns (path, e_out) or None."""
        pq = [(0, src, e_in, [])]; best = {(src, e_in): 0}
        while pq:
            steps, cell, e, path = heapq.heappop(pq)
            if cell == dst: return path, e
            if steps > best.get((cell, e), 1 << 30): continue
            for a, nb in edges.get(cell, {}).items():
                if nb != dst and nb in avoid: continue
                if e < 1: continue
                ne = e - 1
                if ne == 0 and nb not in refills: continue        # would die
                if nb in refills: ne = EMAX
                k = (nb, ne)
                if steps + 1 < best.get(k, 1 << 30):
                    best[k] = steps + 1; heapq.heappush(pq, (steps + 1, nb, ne, path + [a]))
        return None

    def cyc_wps(C, k):
        wps = [C]
        if k > 1:
            nbr = next((nb for a, nb in edges.get(C, {}).items()
                        if nb != C and nb not in station_set and nb not in refills), None)
            if nbr is None: return None
            wps += [nbr, C] * (k - 1)
        return wps
    def nearest_refill(pos, e):
        best = None
        for rf in refills:
            r = eroute(pos, rf, e, station_set)
            if r and (best is None or len(r[0]) < len(best[1])):
                best = (rf, r[0], r[1])
        return best                              # (refill_cell, path, EMAX) or None
    def realize(wps, deliver):
        """Energy-aware action sequence visiting waypoints in order then delivering. Refuels
        PROACTIVELY (Dijkstra minimises steps, so it would otherwise arrive drained and strand the
        long final leg). Whenever energy is below half and a refill is reachable, top up first."""
        e = EMAX; pos = start; full = []
        for wp in wps:
            if e < EMAX * 0.55 and refills:
                tp = nearest_refill(pos, e)
                if tp: _, pth, e = tp; full += pth; pos = tp[0]
            r = eroute(pos, wp, e, station_set)
            if r is None:                        # can't reach directly -> force a refill detour
                tp = nearest_refill(pos, e)
                if tp is None: return None
                _, pth, e = tp; full += pth; pos = tp[0]
                r = eroute(pos, wp, e, station_set)
                if r is None: return None
            p, e = r; full += p; pos = wp
        return full + [deliver] if e >= 2 else None
    MAXK = maxk

    # success check: full win (fast), or (multi-slot only) a newly-solved slot (slower, pixel-based).
    # Only enable the partial path when the level ACTUALLY has >=2 slot markers, else single-slot
    # levels mis-fire on spurious slot_state changes (a real single-slot solve is already a full win).
    n_slots = count_markers(g0, P0)
    eff_partial = partial_ok and n_slots >= 2
    if verbose and partial_ok: print(f"L{level}: slot markers detected = {n_slots} (partial={'on' if eff_partial else 'off'})")
    def success(sol):
        if run_fast(sol): return (True, True)        # fast path: full win
        if not eff_partial: return (False, True)
        _, pc, _, _, grid = trace(sol)               # slow path: did the marker count DROP (slot solved)?
        return (count_markers(grid, pc) < n_slots, False)

    import os
    if os.environ.get("SF_DEBUG_COMBO"):         # "c1x,c1y,k1,c2x,c2y,k2" -> diagnose this combo
        vals = [int(v) for v in os.environ["SF_DEBUG_COMBO"].split(",")]
        C1, k1, C2, k2 = (vals[0], vals[1]), vals[2], (vals[3], vals[4]), vals[5]
        wps = (cyc_wps(C1, k1) or []) + (cyc_wps(C2, k2) or [])
        print(f"  DEBUG combo C1{C1}x{k1} C2{C2}x{k2}; wps={wps}")
        for fr in frontiers[:4]:
            cell, a = fr
            sol = realize(wps + [cell], a)
            won_it = run_fast(sol) if sol else None
            print(f"    frontier {ftgt(fr)}: sol_len={len(sol) if sol else None} won={won_it}")
        return False, dict(level=level, debug=True)

    # Search FRONTIER-OUTER, interleaving D=1 (one station) and D=2 (two stations) so the most
    # promising (blue-marker-nearest) delivery cell gets both passes early. Station candidates are the
    # DETECTED stations (small), which is why D=1 no longer drowns the search (the earlier bug: 65
    # cells x 86 frontiers exhausted the budget before D=2 even started).
    Cs = detected_stations or cells[:14]
    if maxk > 4 and len(Cs) > 3:                  # multi-slot D=3: dedup to one cyclable rep per station
        reps = []
        for p in sorted(Cs, key=lambda c: -depth(c)):
            if all(abs(p[0]-r[0]) + abs(p[1]-r[1]) > 5 for r in reps) and cyc_wps(p, 2) is not None:
                reps.append(p)
        if reps: Cs = reps
        if verbose: print(f"L{level}: station reps (deduped) {Cs}")
    res = None; status = "exhausted"
    # Stations control INDEPENDENT attributes, so cycling order doesn't change the final config ->
    # use index-ordered COMBINATIONS of stations (not permutations); only the cycle counts are searched.
    krange = range(1, MAXK + 1)
    def combos_for(cell, a):
        n = len(Cs)
        for i in range(n):                             # D=1
            for k in krange:
                w = cyc_wps(Cs[i], k)
                if w is not None: yield (w + [cell], a, 1)
        for i in range(n):                             # D=2 (i<j)
            for j in range(i + 1, n):
                for k1 in krange:
                    w1 = cyc_wps(Cs[i], k1)
                    if w1 is None: continue
                    for k2 in krange:
                        w2 = cyc_wps(Cs[j], k2)
                        if w2 is not None: yield (w1 + w2 + [cell], a, 2)
        for i in range(n):                             # D=3 (i<j<l) — e.g. L4: shape + colour + rotation
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    for k1 in krange:
                        w1 = cyc_wps(Cs[i], k1)
                        if w1 is None: continue
                        for k2 in krange:
                            w2 = cyc_wps(Cs[j], k2)
                            if w2 is None: continue
                            for k3 in krange:
                                w3 = cyc_wps(Cs[l], k3)
                                if w3 is not None: yield (w1 + w2 + w3 + [cell], a, 3)
    trials = 0; fully = True
    for fr in frontiers:
        if res is not None: break
        cell, a = fr
        for wps, da, dval in combos_for(cell, a):
            if trials >= trial_cap or time.time() - t0 > time_cap:
                status = "capped"; break
            sol = realize(wps, da)
            if sol is None: continue
            trials += 1
            ok, won_full = success(sol)
            if ok:
                res = (sol, ftgt(fr), dval); status = "won"; fully = won_full; break
        if status == "capped": break

    dt = time.time() - t0
    if res:
        sol, slot_at, dval = res
        if verbose:
            tag = "SOLVED" if fully else "solved one slot of"
            print(f"  ✅ {tag} L{level} FROM PIXELS ONLY — {len(sol)} actions, {trials} trials, "
                  f"D={dval}, {dt:.0f}s; slot @ {slot_at}")
        return True, dict(level=level, actions=len(sol), trials=trials, D=dval, secs=round(dt, 1),
                          sol=sol, fully_won=fully)
    if verbose:
        print(f"  ❌ L{level}: no winning combination ({trials} trials, {status}, {dt:.0f}s)")
    return False, dict(level=level, trials=trials, status=status, secs=round(dt, 1))

def solve_level_full(level, prefix, DIR, maxk=4, max_subgoals=4, time_cap=160.0):
    """Solve a level fully, chaining SUB-GOALS for multi-slot levels: each solve_level(partial_ok)
    call solves one more slot (or wins); we extend a within-level prefix and re-run until won."""
    sub = []; total_trials = 0; t0 = time.time()
    for _ in range(max_subgoals):
        ok, info = solve_level(level, prefix=tuple(prefix) + tuple(sub), dirs=DIR,
                               partial_ok=True, maxk=maxk, time_cap=time_cap, verbose=True)
        if not ok:
            return False, dict(level=level, trials=total_trials, secs=round(time.time()-t0, 1))
        sub += info["sol"]; total_trials += info["trials"]
        if info.get("fully_won"):
            return True, dict(level=level, actions=len(sub), trials=total_trials,
                              secs=round(time.time()-t0, 1), sol=sub)
    return False, dict(level=level, trials=total_trials, status="max_subgoals")

def main():
    import sys
    levels = [int(x) for x in sys.argv[1:]] or list(range(LAST_LEVEL + 1))
    DIR = learn_dirs()
    results = []; prefix = []
    for lv in levels:
        ok, info = solve_level_full(lv, prefix, DIR, maxk=6 if lv >= 5 else 4,
                                    time_cap=420.0 if lv >= 5 else 160.0)
        results.append((lv, ok, info)); print()
        if ok:
            prefix += info["sol"]
        else:
            print(f"(stopping chain: cannot reach levels beyond L{lv} without its solution)"); break
    print("=== SOURCE-FREE SUMMARY (pixels + win/lose feedback only) ===")
    nsolved = sum(1 for _, ok, _ in results if ok)
    for lv, ok, info in results:
        tag = "✅" if ok else "❌"
        extra = (f"{info['actions']} acts, {info['trials']} trials, {info['secs']}s"
                 if ok else f"{info.get('trials','?')} trials, {info.get('status','')}, {info.get('secs','?')}s")
        print(f"  {tag} L{lv}: {extra}")
    print(f"  TOTAL: {nsolved} consecutive levels solved source-free (chained from L0)")

if __name__ == "__main__":
    main()
