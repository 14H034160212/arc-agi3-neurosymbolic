#!/usr/bin/env python3
"""
source_free_agent — solve ls20 using ONLY pixels + actions (no engine internals).

Demonstrates the source-free loop on level 0:
  PERCEIVE structure from the render (player, obstacle map, station, slot)
  + LEARN semantics by INTERACTION (which action moves where; what a station does)
  + PLAN navigation on the perceived obstacle map
  + DELIVER by interaction (try the slot; if rejected, cycle at the station and retry).
The only engine signal used for *decisions* is the rendered grid; win/lose is read as
environment feedback (legitimate). No reading of mgu/snw/gfy/qqv/etc.

STATUS / FINDINGS (honest):
  WORKS from pixels: camera is static (screen==world); player centroid; action->direction
  learned by probing (ACTION1..4 = U/D/L/R); connected-component candidates for slot/station
  (the true slot/station ARE among them).
  OPEN (the real perception frontier — COLOR OVERLOADING): a single render color serves
  multiple roles, so naive color classification fails navigation:
    * color 4 = BOTH collision-walls AND non-colliding void  -> "all color-4 = obstacle"
      over-blocks every lattice move (5x5 box always touches void) -> BFS finds no path;
    * color 5 = BOTH black borders AND the slot's exact deliver cell;
    * color 9 = BOTH the goal-slot marker AND decorative brackets.
  Fix direction (next): learn walkability + the exact deliver cell by INTERACTION (try a move,
  see if the player actually moved; try stepping around the slot marker until delivery fires),
  i.e. build the obstacle map from experience rather than from fragile color labels.
"""
from __future__ import annotations
import importlib.util
from collections import deque
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
WALL, FLOOR, PLAYER, SLOT, STATION = 4, 3, 12, 9, 0   # render-color roles (learned in perception.py)
ACTS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

def _ls20():
    s = importlib.util.spec_from_file_location("ls20mod", ENV)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.Ls20

def render(g):
    return np.array(g.get_pixels(0, 0, 64, 64))

def centroid(grid, color):
    ys, xs = np.where(grid == color)
    if len(xs) == 0: return None
    return (int(round(xs.mean())), int(round(ys.mean())))

def obstacles(grid):
    ys, xs = np.where(grid == WALL)
    return {(int(x), int(y)) for x, y in zip(xs, ys)}

def comps(grid, color, min_size=2):
    """Connected components (4-conn) of a color -> list of centroids (largest first)."""
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

def blocked(obs, tx, ty):
    return any((x, y) in obs for x in range(tx, tx+5) for y in range(ty, ty+5)) \
        or tx < 0 or ty < 0 or tx > 60 or ty > 60

def learn_action_dirs(g, AIN):
    """Probe each action once; measure the player's pixel displacement -> direction."""
    dirs = {}
    for a in ACTS:
        p0 = centroid(render(g), PLAYER)
        g.perform_action(AIN[a])
        p1 = centroid(render(g), PLAYER)
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        if (dx, dy) != (0, 0):
            dirs[a] = (1 if dx > 0 else -1 if dx < 0 else 0,
                       1 if dy > 0 else -1 if dy < 0 else 0)
    return dirs  # action -> unit (dx,dy)

def plan_to(obs, start, goal, dirs, step=5, cap=4000):
    """BFS over the 5-lattice (perceived obstacles) from start to the lattice cell whose
    5x5 box contains `goal`. Returns a list of actions."""
    def near(p):  # does the 5x5 box at p contain the goal?
        return p[0] <= goal[0] < p[0]+5 and p[1] <= goal[1] < p[1]+5
    seen = {start}; q = deque([(start, [])]); n = 0
    while q and n < cap:
        (px, py), path = q.popleft(); n += 1
        if near((px, py)): return path
        for a, (dx, dy) in dirs.items():
            tx, ty = px+dx*step, py+dy*step
            if blocked(obs, tx, ty) or (tx, ty) in seen: continue
            seen.add((tx, ty)); q.append(((tx, ty), path+[a]))
    return None

def main():
    Ls20 = _ls20(); from arcengine import GameAction, ActionInput
    AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in ACTS}
    g = Ls20(); g.set_level(0)
    start_level = g.level_index

    grid = render(g)
    slot_cands = [(x, y) for (x, y, n) in comps(grid, SLOT)]      # blue components
    stn_cands = [(x, y) for (x, y, n) in comps(grid, STATION)]    # white components
    print(f"PERCEIVED (pixels only): player={centroid(grid,PLAYER)} obstacles={len(obstacles(grid))}")
    print(f"  slot candidates (blue comps): {slot_cands}")
    print(f"  station candidates (white comps): {stn_cands}")

    dirs = learn_action_dirs(g, AIN)
    print(f"LEARNED action->direction: { {a:dirs[a] for a in dirs} }")

    def player(): return centroid(render(g), PLAYER)
    def won(): return g.level_index > start_level or g._state.name == "WIN"
    def go(goal):
        for _ in range(8):
            p = player()
            if p[0] <= goal[0] < p[0]+5 and p[1] <= goal[1] < p[1]+5: return True
            path = plan_to(obstacles(render(g)), p, goal, dirs)
            if not path: return False
            for a in path:
                g.perform_action(AIN[a])
                if won(): return True
        return True

    # Interaction-driven: cycle config at a station, then try delivering at each slot candidate.
    # We don't know which white=station or which blue=slot, so we try combinations until a win.
    for stn in (stn_cands or [None]):
        for cyc in range(5):                 # how many times to cycle the config
            if stn is not None:
                go(stn)                      # entering the station cycles one attribute
            for slot in slot_cands:
                go(slot)                     # move onto the slot = deliver if config matches
                if won():
                    print(f"  ✅ SOLVED L0 from PIXELS-ONLY perception + interaction "
                          f"(station~{stn}, slot~{slot}, cycles={cyc})")
                    return
    print("  ❌ not solved within attempts")

if __name__ == "__main__":
    main()
