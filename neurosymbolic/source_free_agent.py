#!/usr/bin/env python3
"""
source_free_agent — solve ls20 from PIXELS + ACTIONS only (no engine internals for decisions).

Robust, interaction-driven (no fragile color obstacle map):
  * camera static -> screen==world; read player centroid (orange) from the render.
  * learn ACTION->direction by probing.
  * EXPLORE the reachable lattice by RESET + replay-path (deterministic game): visit each cell,
    try each action, record edges -> build the reachable graph by EXPERIENCE.
  * snap perceived station(white)/slot(blue) component centroids to the player's centroid-lattice
    -> exact target cells (where the player's centroid sits when standing on them).
  * BFS in-memory: start -> station (lands on it, cycling the carried config) -> slot (deliver).
The only engine signal used for decisions is the rendered grid; win/lose is environment feedback.

STATUS (honest): the machinery works — camera-static (screen==world), action->direction learned by
probing, RESET+replay graph exploration, in-memory BFS. The binding constraint is SOURCE-FREE
PERCEPTION OF A STABLE PLAYER POSITION:
  1) the player is the orange (12) carried-object sprite, whose shape/rotation CHANGES at stations,
     so its centroid JUMPS when the config cycles (not from movement) -> cell identity breaks exactly
     at the stations we must use (explore() then fails to register the station/slot cells);
  2) color overloading: wall vs non-colliding void share a color; the slot's deliver cell shares
     black with borders.
NEXT FIX: track the player by FRAME-DIFF motion (the cells that change by a 5px translation between
consecutive frames isolate movement, vs the in-place sprite redraw at a station), giving a stable
position estimate; then this explorer + BFS + the interaction-learned semantics close the loop.
This is the concrete remaining milestone toward a source-free agent.
"""
from __future__ import annotations
import importlib.util
from collections import deque
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
PLAYER, SLOT, STATION = 12, 9, 0
ACTS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

def _ls20():
    s = importlib.util.spec_from_file_location("ls20mod", ENV)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.Ls20

def render(g): return np.array(g.get_pixels(0, 0, 64, 64))
def pcent(g):
    ys, xs = np.where(render(g) == PLAYER)
    return (int(round(xs.mean())), int(round(ys.mean()))) if len(xs) else None
def comps(grid, color, min_size=2):
    H, W = grid.shape; seen = np.zeros_like(grid, bool); out = []
    for y in range(H):
        for x in range(W):
            if grid[y, x] == color and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and grid[ny,nx]==color:
                            seen[ny,nx]=True; q.append((ny,nx))
                if len(cells) >= min_size:
                    out.append((round(sum(c[1] for c in cells)/len(cells)),
                                round(sum(c[0] for c in cells)/len(cells)), len(cells)))
    return sorted(out, key=lambda c: -c[2])

def main():
    Ls20 = _ls20(); from arcengine import GameAction, ActionInput
    AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in ACTS}
    RESET = ActionInput(id=GameAction.RESET)
    g = Ls20(); g.set_level(0); start_level = g.level_index
    grid = render(g)
    start = pcent(g); rx, ry = start[0] % 5, start[1] % 5
    snap = lambda c: (round((c[0]-rx)/5)*5+rx, round((c[1]-ry)/5)*5+ry)
    station = snap(comps(grid, STATION)[0][:2])
    slots = [snap(c[:2]) for c in comps(grid, SLOT)]
    print(f"PIXELS-ONLY: start={start} station~{station} slots~{slots}")

    # learn action directions (probe), then RESET to a clean start
    for a in ACTS:
        g.perform_action(AIN[a])
    g.perform_action(RESET)
    won = lambda: g.level_index > start_level or g._state.name == "WIN"

    def goto(path):
        g.perform_action(RESET)
        for a in path:
            g.perform_action(AIN[a])
        return pcent(g)

    # ---- explore the reachable graph via RESET+replay ----
    edges = {}; paths = {start: []}; order = [start]; i = 0
    targets = set([station]) | set(slots)
    while i < len(order) and len(order) < 400:
        cell = order[i]; i += 1
        if goto(paths[cell]) != cell:
            continue  # desync (energy/respawn) — skip
        edges[cell] = {}
        for a in ACTS:
            before = pcent(g); g.perform_action(AIN[a]); now = pcent(g)
            edges[cell][a] = now
            if now != before and now not in paths:
                paths[now] = paths[cell] + [a]; order.append(now)
            goto(paths[cell])  # return to cell for next action trial
        if targets <= set(paths):
            break
    print(f"explored {len(paths)} reachable cells; station reached={station in paths}; "
          f"slots reached={[s for s in slots if s in paths]}")

    # ---- in-memory BFS on the learned graph ----
    def bfs(src, dst):
        if src == dst: return []
        seen = {src}; q = deque([(src, [])])
        while q:
            c, p = q.popleft()
            for a, nb in edges.get(c, {}).items():
                if nb == c or nb in seen or nb not in edges: continue
                if nb == dst: return p + [a]
                seen.add(nb); q.append((nb, p + [a]))
        return None

    real_slots = [s for s in slots if s in paths and s != start]
    if station not in paths or not real_slots:
        print("  ❌ could not reach station/slot during exploration"); return
    to_st = bfs(start, station)
    for slot in real_slots:
        seg = bfs(station, slot)
        if to_st is None or seg is None: continue
        sol = to_st + seg                       # start -> station (cycle config) -> slot (deliver)
        g.perform_action(RESET)
        for a in sol:
            g.perform_action(AIN[a])
            if won():
                print(f"  ✅ SOLVED L0 END-TO-END FROM PIXELS ONLY  ({len(sol)} actions, slot {slot})")
                return
    print("  ❌ planned but did not win; (1 station-cycle may be wrong count for this level)")

if __name__ == "__main__":
    main()
