#!/usr/bin/env python3
"""
perception (prototype) — extract a symbolic state from ls20's RENDERED grid, toward
source-free transfer (real test games give no engine internals, only pixels).

What's hard: the grid is a camera-rendered view; world coords = screen coords + camera
offset. At a level's start the camera is at (0,0) (screen==world), which we use here to
measure how much of the symbolic layout is recoverable from pixels alone. Across moves the
camera follows the player, so a real agent must TRACK the camera (frame registration) — the
key open problem, sketched at the bottom.
"""
from __future__ import annotations
import importlib.util
from collections import deque, Counter
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"

def _ls20():
    spec = importlib.util.spec_from_file_location("ls20mod", ENV)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.Ls20

FLOOR = {3, 4}     # greys = walkable floor
WALL_C = {5}       # black = wall/void (structural)

def components(grid, colors):
    """4-connected components of cells whose color is in `colors`."""
    H, W = grid.shape; seen = np.zeros_like(grid, bool); comps = []
    for y in range(H):
        for x in range(W):
            if grid[y, x] in colors and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and grid[ny,nx] in colors:
                            seen[ny,nx]=True; q.append((ny,nx))
                cy=sum(c[0] for c in cells)/len(cells); cx=sum(c[1] for c in cells)/len(cells)
                comps.append(dict(color=int(grid[cells[0][0],cells[0][1]]), n=len(cells),
                                  cy=round(cy,1), cx=round(cx,1)))
    return comps

def perceive(grid):
    """Return a rough symbolic reading of the rendered grid."""
    interesting = set(int(v) for v in np.unique(grid)) - FLOOR - WALL_C
    comps = components(grid, interesting)
    comps.sort(key=lambda c: -c["n"])
    return comps

def main():
    Ls20 = _ls20(); g = Ls20(); g.set_level(0)
    grid = np.array(g.get_pixels(g.camera.x, g.camera.y, 64, 64))
    print("=== ls20 L0: pixel perception vs engine ground truth ===")
    print(f"camera=({g.camera.x},{g.camera.y})  -> screen coords == world coords here")
    print(f"colors present: {sorted(set(int(v) for v in np.unique(grid)))}")
    print(f"\nENGINE TRUTH (world):")
    print(f"  player(mgu) = ({g.mgu.x},{g.mgu.y}); carried=(shape{g.snw},color{g.tmx},rot{g.tuv})")
    for i, s in enumerate(g.qqv):
        print(f"  slot{i} @ ({s.x},{s.y}) needs (shape{g.gfy[i]},color{g.vxy[i]},rot{g.cjl[i]})")
    stations = {(s.x, s.y): t for s in g.current_level._sprites for t in (s.tags or [])
                if t in ("gsu", "gic", "bgt")}
    print(f"  stations(world) = {stations}")

    print(f"\nPERCEIVED (colored components, note (x,y)=(col,row)):")
    for c in perceive(grid)[:8]:
        print(f"  color={c['color']:2d} size={c['n']:3d} centroid=(x{c['cx']},y{c['cy']})")

    print("\nREADING:")
    print("  - colored (non-grey) components recover the player / stations / slot markers as")
    print("    distinct sprites; their on-screen centroids == world positions at this start frame.")
    print("  - carried (shape,color,rotation) is in principle decodable from the carried sprite's")
    print("    pixels (color directly; shape/rotation from its pixel pattern) — TODO classifier.")
    print("\nOPEN PROBLEM (camera): after moves the camera follows the player, so")
    print("  world = screen + camera_offset. A source-free agent must TRACK the camera by")
    print("  registering consecutive frames (estimate the dominant shift), then everything above")
    print("  works in world coords. This frame-registration is the main perception milestone.")

if __name__ == "__main__":
    main()
