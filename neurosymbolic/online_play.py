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


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except Exception:
        pass
    key = os.environ.get("ARC_API_KEY", "")
    games = sys.argv[1:] or []
    arcade = Arcade(arc_api_key=key, operation_mode=OperationMode.ONLINE)
    card = arcade.open_scorecard(tags=["sf-modality-probe"])
    try:
        for gid in games:
            g = OnlineGame(arcade, gid, card)
            t = time.time()
            mod, mv, ck = probe_modality(g)
            print(f"{gid:18s} modality={mod:8s} (move-dirs={mv}, click-responders={ck})  {time.time()-t:.0f}s")
    finally:
        arcade.close_scorecard(card)


if __name__ == "__main__":
    main()
