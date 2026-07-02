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
    """Invoke Claude (multimodal) headless on an image; return its text. Raises on a CLI failure
    instead of silently returning '' -- an auth/permission/crash failure must not be indistinguishable
    from 'the model proposed nothing' (the caller's own except-blocks already handle degrading from a
    genuine call failure; masking that here would just hide it one layer earlier)."""
    r = subprocess.run(
        ["claude", "-p", f"Read the image {png_path}. {prompt}", "--allowedTools", "Read"],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"`claude -p` exited {r.returncode}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()

def call_openai_vision(png_path, prompt, timeout=180):
    """Teacher = a strong multimodal OpenAI model (gpt-5.5). Sends the frame as an inline image.

    reasoning_effort matters a lot here: verified directly against the API that gpt-5.5's DEFAULT
    reasoning effort burns the ENTIRE token budget on invisible reasoning tokens and returns an EMPTY
    message (finish_reason='length', reasoning_tokens=8000/8000, content_len=0) even at
    max_completion_tokens=8000 for this single-click-pick task -- this looked exactly like 'the model
    has no answer' but is actually 'the model was never allowed to finish reasoning'. Setting
    reasoning_effort='low' fixed it immediately (finish_reason='stop', real JSON content). Without
    this, every closed-loop run would silently degrade to 0 real model turns and look like a
    capability failure when it was a token-budget misconfiguration."""
    import urllib.request
    key = os.environ.get("OPENAI_SECRET_KEY")
    if not key:
        raise RuntimeError("OPENAI_SECRET_KEY is not set (needed for the default OpenAI teacher backend; "
                           "set TEACHER_BACKEND=claude to use the Claude CLI backend instead)")
    model = os.environ.get("TEACHER_MODEL", "gpt-5.5")
    effort = os.environ.get("TEACHER_REASONING_EFFORT", "low")
    b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    payload = {"model": model, "max_completion_tokens": 4000,
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
    if effort:
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", body,
                                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    choice = resp["choices"][0]
    content = choice["message"]["content"]
    if not content and choice.get("finish_reason") == "length":
        usage = resp.get("usage", {})
        raise RuntimeError(f"gpt vision call returned EMPTY content (finish_reason=length, "
                           f"reasoning_tokens={usage.get('completion_tokens_details', {}).get('reasoning_tokens')}"
                           f"/{usage.get('completion_tokens')}) -- the model burned its whole budget on "
                           f"reasoning and never answered; this is a config issue, not 'no answer'")
    return content

def call_model(png_path, prompt):
    """Teacher dispatcher: OpenAI gpt-5.5 (default) or Claude CLI (TEACHER_BACKEND=claude)."""
    return call_claude(png_path, prompt) if os.environ.get("TEACHER_BACKEND") == "claude" \
        else call_openai_vision(png_path, prompt)

def parse_click(txt, W, H, upscale=16):
    """Parse the model's {"click":[x,y]} reply. Despite the prompt stating the expected grid range
    (0<=x<W), a vision model shown the UPSCALED render (render_png_b64's default up=16) sometimes
    answers in the PIXEL space it actually sees (e.g. returned [504,168] on a 64x64 grid, which is
    exactly (504,168)/16 = (31.5,10.5) in grid space) rather than the stated grid coordinates -- so an
    out-of-range answer is auto-rescaled by the known upscale factor before being rejected."""
    try:
        obj = op.extract_json(txt)
        c = obj.get("click") if obj else None
        if isinstance(c, list) and len(c) == 2:
            x, y = int(c[0]), int(c[1])
            if 0 <= x < W and 0 <= y < H:
                return (x, y), obj.get("reason", "")
            xs, ys = round(x / upscale), round(y / upscale)
            if 0 <= xs < W and 0 <= ys < H:
                return (xs, ys), obj.get("reason", "") + " [auto-rescaled from pixel space]"
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
        try:
            txt = call_model(FRAME_PNG, prompt)
        except Exception as e:
            if verbose: print(f"    step {step}: model call FAILED ({type(e).__name__}: {e}) -- stopping this game, not a real 'no click'")
            break
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
    try:
        txt = call_model(FRAME_PNG, prompt)
    except Exception as e:
        if verbose: print(f"    model call FAILED ({type(e).__name__}: {e}) -- falling back to a blind key-object guess, not a real 'no ideas'")
        txt = ""
    try:
        obj = op.extract_json(txt)
        key = [i for i in (obj.get("key_objects", []) if obj else []) if isinstance(i, int) and 0 <= i < len(objs)]
        seqs = [[objs[i] for i in s if isinstance(i, int) and 0 <= i < len(objs)] for s in (obj.get("sequences", []) if obj else [])]
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
            print(f"{gid}:")
            try:
                g = op.OnlineGame(arcade, gid, card)
                if hybrid: solve_hybrid(g)
                else: solve_closed(g, max_steps=int(os.environ.get("MAX_STEPS", "8")))
            except Exception as e:
                print(f"    {gid}: FAILED ({type(e).__name__}: {e}) -- skipping to the next game")
    finally:
        sc = arcade.close_scorecard(card)
        try: print("SCORECARD levels:", sc.model_dump().get("total_levels_completed"))
        except Exception: pass

if __name__ == "__main__":
    main()
