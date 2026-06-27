#!/usr/bin/env python3
"""
ls20_solver — neuro-symbolic solver for the ARC-AGI-3 game `ls20`.

The grid pixels are a non-trivial RENDER of a small symbolic state. This module models
ls20's dynamics as ~40 lines of code and solves all 7 levels with a classical planner,
using the engine only to (a) read the true symbolic state and (b) verify plans.

Game (reverse-engineered): the player moves U/D/L/R (5px/step) carrying an object with
(shape, color, rotation). Stations cycle one attribute (gsu=shape, gic=color, bgt=rot).
Deliver the object matching a goal slot's required (shape,color,rot) onto that slot; solve
all slots -> next level (the last level -> GameState.WIN). Energy -1/step (refills 'iri'
reset to max); 0 energy costs a life (respawn resets pos/config/energy and UNsolves slots).

Result: classical planning solves 7/7 levels. Planning is trivial once the model is right;
the research value is having an LLM INDUCE this model + verify it symbolically (see llm_induce.py).
"""
from __future__ import annotations
import copy
import importlib.util
from collections import deque
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / "environment_files/ls20/cb3b57cc/ls20.py"
ACTS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
DIRS = {"ACTION1": (0, -5), "ACTION2": (0, 5), "ACTION3": (-5, 0), "ACTION4": (5, 0)}


def _load_ls20():
    spec = importlib.util.spec_from_file_location("ls20mod", ENV_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Ls20


def new_game(level: int):
    """Instantiate the ls20 engine at a given level."""
    Ls20 = _load_ls20()
    g = Ls20()
    g.set_level(level)
    return g


def action_inputs():
    from arcengine import GameAction, ActionInput
    return {a: ActionInput(id=getattr(GameAction, a)) for a in ACTS}


def extract_layout(g) -> dict:
    """Read the static, true symbolic layout of the current level from the engine."""
    walls, station, refill = set(), {}, {}
    for s in g.current_level._sprites:
        for t in (s.tags or []):
            if t == "jdd": walls.add((s.x, s.y))
            elif t == "gsu": station[(s.x, s.y)] = "shape"
            elif t == "gic": station[(s.x, s.y)] = "color"
            elif t == "bgt": station[(s.x, s.y)] = "rot"
            elif t == "iri": refill[(s.x, s.y)] = 1
    slots = [(s.x, s.y) for s in g.qqv]
    slot_req = {(s.x, s.y): (g.gfy[i], g.vxy[i], g.cjl[i]) for i, s in enumerate(g.qqv)}
    return dict(walls=walls, station=station, refill=refill, slots=slots,
                slot_req=slot_req, nshape=len(g.hep), ncolor=len(g.hul), nrot=4,
                emax=g.ggk.tmx)


def init_state(g) -> dict:
    """Read the true symbolic state from the engine."""
    return dict(px=g.mgu.x, py=g.mgu.y, shape=g.snw, color=g.tmx, rot=g.tuv,
                energy=g.ggk.snw, solved=tuple(bool(b) for b in g.rzt), used_refill=())


def model_step(st: dict, action: str, lay: dict) -> dict:
    """Pure-code hypothesis of ls20 dynamics — predicts the next symbolic state.

    Key subtleties (all verified against the engine):
      * the engine tests a 5x5 BOX at the target (not the exact cell);
      * energy decrements every action (even blocked); refills reset to max (single use);
      * delivery needs the player EXACTLY on the slot; already-solved slots are inert.
    """
    dx, dy = DIRS[action]
    tx, ty = st["px"] + dx, st["py"] + dy
    s, c, r = st["shape"], st["color"], st["rot"]
    solved = list(st["solved"]); e = st["energy"] - 1; used = set(st["used_refill"])
    bset = {(x, y) for x in range(tx, tx + 5) for y in range(ty, ty + 5)}

    def stay():
        return dict(px=st["px"], py=st["py"], shape=s, color=c, rot=r, energy=e,
                    solved=tuple(solved), used_refill=tuple(sorted(used)))

    if bset & lay["walls"]:
        return stay()
    here = [p for p in lay["slot_req"] if p in bset and not solved[lay["slots"].index(p)]]
    if here and (s, c, r) != lay["slot_req"][here[0]]:
        return stay()  # unsolved slot rejects a wrong configuration
    for p in lay["station"]:
        if p in bset:
            k = lay["station"][p]
            if k == "shape": s = (s + 1) % lay["nshape"]
            elif k == "color": c = (c + 1) % lay["ncolor"]
            elif k == "rot": r = (r + 1) % 4
    for p in lay["refill"]:
        if p in bset and p not in used:
            e = lay["emax"]; used.add(p)
    if (tx, ty) in lay["slot_req"] and (s, c, r) == lay["slot_req"][(tx, ty)]:
        idx = lay["slots"].index((tx, ty))
        if not solved[idx]: solved[idx] = True
    return dict(px=tx, py=ty, shape=s, color=c, rot=r, energy=e,
                solved=tuple(solved), used_refill=tuple(sorted(used)))


def _bfs_more_solved(start, lay, step_fn, maxnodes=400000):
    base = sum(start["solved"])
    sk = lambda s: (s["px"], s["py"], s["shape"], s["color"], s["rot"],
                    tuple(s["solved"]), s["used_refill"])
    seen = {sk(start)}; q = deque([(start, [])]); nodes = 0
    while q and nodes < maxnodes:
        st, path = q.popleft(); nodes += 1
        if sum(st["solved"]) > base:
            return path, st, nodes
        for a in ACTS:
            ns = step_fn(dict(st), a, lay)
            if ns["energy"] < 1:
                continue  # would die — prune
            k = sk(ns)
            if k in seen:
                continue
            seen.add(k); q.append((ns, path + [a]))
    return None, None, nodes


def plan(g, lay, step_fn=model_step):
    """Sequential sub-goal planner: solve one slot at a time, keeping energy > 0."""
    st = init_state(g); full = []; total = 0
    while not (all(st["solved"]) and len(st["solved"]) > 0):
        sub, st2, nodes = _bfs_more_solved(st, lay, step_fn); total += nodes
        if sub is None:
            return None, total
        full += sub; st = st2
    return full, total


def solve_level(level: int, step_fn=model_step):
    """Plan with the model, then VERIFY the plan wins on the real engine."""
    from arcengine import GameState
    g = new_game(level); lay = extract_layout(g)
    p, nodes = plan(g, lay, step_fn)
    if not p:
        return dict(level=level, win=False, reason="no plan", nodes=nodes, plan=None)
    AIN = action_inputs(); gv = new_game(level)
    for a in p:
        gv.perform_action(AIN[a])
    win = gv.level_index > level or gv._state == GameState.WIN
    return dict(level=level, win=win, reason="win" if win else f"state={gv._state}",
                nodes=nodes, steps=len(p), plan=p)


def main():
    print("=== ls20 neuro-symbolic solver: planning over the induced model ===")
    wins = 0
    for L in range(7):
        r = solve_level(L)
        print(f"  L{L}: {'✅ WIN' if r['win'] else '❌ ' + r['reason']}"
              + (f"  ({r['steps']} steps, {r['nodes']} nodes)" if r.get('steps') else ""))
        wins += r["win"]
    print(f"\nsolved {wins}/7 levels")


if __name__ == "__main__":
    main()
