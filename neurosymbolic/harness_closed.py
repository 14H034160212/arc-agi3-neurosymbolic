#!/usr/bin/env python3
"""
harness_closed — validate the harness with a strong CLOSED multimodal model (Claude via `claude -p`).

Closed-loop: render the current online frame -> Claude sees it + the click history -> picks the next
click (grid x,y) -> we apply it and re-observe -> repeat, verified by the environment's win/score.
This tests whether a STRONG model + our harness clears games the weak local VL could not. `claude -p`
uses the Claude Code subscription (no API key), and Claude is multimodal. Set ARC_API_KEY in .env.
"""
from __future__ import annotations
import os, sys, json, base64, subprocess, importlib.util, time
from pathlib import Path
import numpy as np

_op = Path(__file__).resolve().parent / "online_play.py"
spec = importlib.util.spec_from_file_location("online_play", _op)
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)

FRAME_PNG = Path(__file__).resolve().parent / "_hframe.png"

def call_claude(png_path, prompt, timeout=180):
    """Invoke Claude (multimodal) headless on an image; return its text."""
    r = subprocess.run(
        ["claude", "-p", f"Read the image {png_path}. {prompt}", "--allowedTools", "Read"],
        capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def parse_click(txt, W, H):
    try:
        obj = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
        c = obj.get("click")
        if isinstance(c, list) and len(c) == 2:
            x, y = int(c[0]), int(c[1])
            if 0 <= x < W and 0 <= y < H:
                return (x, y), obj.get("reason", "")
    except Exception:
        pass
    return None, txt[:80]

def solve_closed(g, max_steps=8, verbose=True):
    s0 = g.reset().score()
    hist = []
    for step in range(max_steps):
        grid = g.grid(); H, W = grid.shape
        b64 = op.render_png_b64(grid)                  # upscaled PNG of the current state
        FRAME_PNG.write_bytes(base64.b64decode(b64))
        prompt = (f"It is the CURRENT screen of a {W}x{H}-cell puzzle game (upscaled 16x), hidden goal. "
                  f"You win by clicking the right cells in order. Clicks so far (grid x,y): {hist[-6:]}. "
                  f"Pick the SINGLE next cell to CLICK to make progress. Reply ONLY JSON: "
                  f'{{"click":[x,y],"reason":"<short>"}} with 0<=x<{W}, 0<=y<{H}.')
        t = time.time()
        txt = call_claude(FRAME_PNG, prompt)
        click, reason = parse_click(txt, W, H)
        if verbose:
            print(f"    step {step}: claude -> {click} ({reason[:60]}) [{time.time()-t:.0f}s]")
        if click is None:
            break
        g.click(*click); hist.append(list(click))
        if g.status() == "WIN" or g.score() > s0:
            if verbose: print(f"    ✅ {g.game_id} WON in {len(hist)} closed-loop clicks: {hist}")
            return hist
        if g.status() == "LOSE":
            if verbose: print(f"    {g.game_id} LOSE");
            return None
    if verbose: print(f"    {g.game_id}: no win in {max_steps} steps")
    return None

def solve_hybrid(g, max_depth=4, verbose=True):
    """The proposed HARNESS: strong model (Claude) provides SEMANTICS — which objects matter + candidate
    orders — and our SYMBOLIC search does systematic sequences (incl. repeats) + env verification.
    Tests whether model-semantics + symbolic-search clears games neither alone could."""
    from collections import deque
    s0 = g.reset().score()
    objs, _ = op.online_find_clickables(g)
    if verbose: print(f"    {g.game_id}: {len(objs)} clickable object(s)")
    if not objs:
        return None
    grid = g.grid(); b64 = op.render_png_b64(grid, objs)
    FRAME_PNG.write_bytes(base64.b64decode(b64))
    prompt = (f"Puzzle game screen; the {len(objs)} CLICKABLE objects are labelled 0..{len(objs)-1} in "
              f"yellow. Infer the goal, then output ONLY JSON: {{\"key_objects\":[indices that matter "
              f"most],\"sequences\":[[i,j,...],...]}} — key_objects (<=5) and up to 6 candidate click "
              f"orders (repeats ALLOWED, length 1-5), best first.")
    txt = call_claude(FRAME_PNG, prompt)
    try:
        obj = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
        key = [i for i in obj.get("key_objects", []) if isinstance(i, int) and 0 <= i < len(objs)]
        seqs = [[objs[i] for i in s if isinstance(i, int) and 0 <= i < len(objs)] for s in obj.get("sequences", [])]
    except Exception:
        key, seqs = list(range(min(4, len(objs)))), []
    if verbose: print(f"    claude key_objects={key}, {len(seqs)} candidate sequence(s)")
    # 1) verify Claude's proposed sequences
    for cand in seqs:
        g.reset(); won = False
        for (x, y) in cand:
            g.click(x, y)
            if g.status() == "WIN" or g.score() > s0: won = True; break
            if g.status() == "LOSE": break
        if won:
            if verbose: print(f"    ✅ {g.game_id} WON via Claude sequence {cand}")
            return cand
    # 2) symbolic BFS over Claude's KEY objects (small set, repeats allowed)
    kobjs = [objs[i] for i in key] or objs[:4]
    q = deque([[]]); nodes = 0
    while q and nodes < 600:
        seq = q.popleft()
        for cell in kobjs:
            cand = seq + [cell]; nodes += 1
            g.reset(); won = False
            for (x, y) in cand:
                g.click(x, y)
                if g.status() == "WIN" or g.score() > s0: won = True; break
                if g.status() == "LOSE": break
            if won:
                if verbose: print(f"    ✅ {g.game_id} WON via symbolic-over-key {cand} ({nodes} nodes)")
                return cand
            if g.status() == "PLAYING" and len(cand) < max_depth:
                q.append(cand)
    if verbose: print(f"    {g.game_id}: hybrid no win ({nodes} nodes)")
    return None


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from arc_agi import Arcade, OperationMode
    arcade = Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""), operation_mode=OperationMode.ONLINE)
    card = arcade.open_scorecard(tags=["harness-closed"])
    try:
        hybrid = os.environ.get("HYBRID", "") == "1"
        for gid in sys.argv[1:]:
            g = op.OnlineGame(arcade, gid, card)
            print(f"{gid}:")
            if hybrid: solve_hybrid(g)
            else: solve_closed(g, max_steps=int(os.environ.get("MAX_STEPS", "8")))
    finally:
        sc = arcade.close_scorecard(card)
        try: print("SCORECARD levels:", sc.model_dump().get("total_levels_completed"))
        except Exception: pass

if __name__ == "__main__":
    main()
