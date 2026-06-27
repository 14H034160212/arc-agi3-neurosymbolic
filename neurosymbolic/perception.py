#!/usr/bin/env python3
"""
perception — extract a symbolic state from ls20's RENDERED grid (toward source-free transfer:
real test games give only pixels). Measures recovery vs engine ground truth.

KEY FINDING: ls20's camera is STATIC (stays at (0,0)); the player moves within a fixed 64x64
view. So **screen coords == world coords** — there is NO camera-tracking problem for ls20.
That makes structural perception directly solvable:
    walls  = render color 4 (impassable);  floor = color 3
    player = orange (color 12) sprite;     goal-slot markers = blue (color 9)
What pixels DON'T give: a station's *type* and the carried/required *configs* — those are
SEMANTICS that require interaction (the exploration step), not perception.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

ENV = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
WALL_COLOR, FLOOR_COLOR, PLAYER_COLOR, SLOT_COLOR = 4, 3, 12, 9

def _ls20():
    spec = importlib.util.spec_from_file_location("ls20mod", ENV)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.Ls20

def camera_is_static(g, AIN, probe=("ACTION1","ACTION3","ACTION2","ACTION4")):
    c0 = (g.camera.x, g.camera.y)
    for a in probe:
        g.perform_action(AIN[a])
        if (g.camera.x, g.camera.y) != c0:
            return False
    return True

def perceive(grid: np.ndarray) -> dict:
    """Recover the navigation structure from pixels (screen==world for ls20)."""
    walls = {(int(x), int(y)) for y, x in zip(*np.where(grid == WALL_COLOR))}
    player_pts = [(int(x), int(y)) for y, x in zip(*np.where(grid == PLAYER_COLOR))]
    slot_pts = [(int(x), int(y)) for y, x in zip(*np.where(grid == SLOT_COLOR))]
    player = None
    if player_pts:
        px = round(sum(p[0] for p in player_pts) / len(player_pts))
        py = round(sum(p[1] for p in player_pts) / len(player_pts))
        player = (px, py)
    return dict(walls=walls, player=player, slot_pixels=slot_pts)

def main():
    Ls20 = _ls20(); from arcengine import GameAction, ActionInput
    AIN = {a: ActionInput(id=getattr(GameAction, a)) for a in ("ACTION1","ACTION2","ACTION3","ACTION4")}

    # camera check on a fresh game
    gchk = Ls20(); gchk.set_level(0)
    print("camera static across moves:", camera_is_static(gchk, AIN), "(=> screen coords == world coords)")

    g = Ls20(); g.set_level(0)
    grid = np.array(g.get_pixels(0, 0, 64, 64))
    per = perceive(grid)

    # ground truth
    true_jdd = {(s.x, s.y) for s in g.current_level._sprites for t in (s.tags or []) if t == "jdd"}
    true_player = (g.mgu.x, g.mgu.y)
    floor = {(int(x), int(y)) for y, x in zip(*np.where(grid == FLOOR_COLOR))}

    print("\n=== pixel -> symbol recovery (ls20 L0) vs engine truth ===")
    # color-4 is the full IMPASSABLE region; the jdd sprite list is a subset of it.
    print(f"  impassable (color 4): {len(per['walls'])} cells  (a planning-valid obstacle map;")
    print(f"      the {len(true_jdd)} jdd collision-sprites are a subset). walkable floor (color 3): {len(floor)} cells.")
    covered = len(true_jdd & per['walls'])
    print(f"      jdd sprites covered by perceived-impassable: {covered}/{len(true_jdd)}")
    pp = per["player"]; dist = abs(pp[0]-true_player[0]) + abs(pp[1]-true_player[1]) if pp else None
    print(f"  player: perceived {pp}, true {true_player}  (L1 dist = {dist}px)")
    print(f"  slot markers (blue): {len(per['slot_pixels'])} px near true slot(s) {[(s.x,s.y) for s in g.qqv]}")
    print("\n  -> obstacle map / player / slot RECOVERED from pixels at true world coords (camera static).")
    print("  -> NOT from pixels: station TYPE (shape/color/rot) and carried/required CONFIGS")
    print("     = game SEMANTICS, obtained by INTERACTION (press a station, watch what changes).")
    print("     That exploration step + this perception + the planner = a source-free agent (next).")

if __name__ == "__main__":
    main()
