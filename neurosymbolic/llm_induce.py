#!/usr/bin/env python3
"""
llm_induce — have a (free, local) LLM INDUCE the ls20 dynamics model as code, verified
symbolically against the real engine with a plan-based, APD-style refine loop.

Pipeline:  observed transitions (+ a win observation)  ->  LLM writes candidate_model_step
       ->  plan with it and replay on the engine  ->  feed back the FIRST diverging step
       ->  repeat until the induced model plans real wins on the training levels.

Uses the solver's planner/verifier (ls20_solver). Model: gpt-oss via local Ollama (free).
"""
from __future__ import annotations
import copy, re, sys
from collections import deque
import requests

import ls20_solver as S   # run from inside neurosymbolic/, or add it to sys.path

OLLAMA = "http://localhost:11437/v1/chat/completions"
MODEL = "gpt-oss:120b-ctx32k"
TRAIN = [0, 3]            # rotation-only + shape&color (both 1-slot, energy-OK)
ACTS = S.ACTS


def fmt(s):
    return (f"(px={s['px']},py={s['py']},shape={s['shape']},color={s['color']},"
            f"rot={s['rot']},energy={s['energy']},solved={list(s['solved'])})")


def collect(level, cap=160):
    g = S.new_game(level); AIN = S.action_inputs()
    out = []; seen = {(g.mgu.x, g.mgu.y, g.snw, g.tmx, g.tuv, g.level_index)}
    q = deque([g])
    n = 0
    while q and n < cap:
        gg = q.popleft(); n += 1
        if gg.level_index > level:
            continue
        for a in ACTS:
            g2 = copy.deepcopy(gg); g2.perform_action(AIN[a])
            if g2.level_index == level:
                out.append((S.init_state(gg), a, S.init_state(g2)))
            k = (g2.mgu.x, g2.mgu.y, g2.snw, g2.tmx, g2.tuv, g2.level_index)
            if k not in seen and g2.level_index == level and g2.lbq == gg.lbq:
                seen.add(k); q.append(g2)
    return out


def build_prompt():
    pool = []
    for L in (0, 2, 3, 4):
        pool += collect(L)

    def cls(s, s2):
        if (s2["px"], s2["py"]) == (s["px"], s["py"]): return "block"
        if s2["shape"] != s["shape"]: return "shape"
        if s2["color"] != s["color"]: return "color"
        if s2["rot"] != s["rot"]: return "rot"
        if s2["energy"] > s["energy"]: return "refill"
        return "move"
    ex = []; cap = {"move": 5, "block": 3, "rot": 3, "color": 3, "shape": 3, "refill": 3}
    for (s, a, s2) in pool:
        t = cls(s, s2)
        if len([e for e in ex if e[3] == t]) < cap[t]:
            ex.append((s, a, s2, t))
    examples = "\n".join(f"  st={fmt(s)} action={a} -> next={fmt(s2)}" for (s, a, s2, _) in ex)

    # win observation from L0's known solution
    AIN = S.action_inputs()
    SOL = S.solve_level(0)["plan"]
    gw = S.new_game(0)
    for a in SOL[:-1]: gw.perform_action(AIN[a])
    wb = S.init_state(gw); lay0 = S.extract_layout(S.new_game(0))
    sp = lay0["slots"][0]; need = lay0["slot_req"][sp]
    win_obs = (f"WIN OBSERVATION: from st={fmt(wb)}, action {SOL[-1]} put the player exactly on slot "
               f"{sp} (lay['slot_req'][{sp}]={need}) while carrying that exact (shape,color,rot); "
               f"the level was WON (slot became solved).")
    return f"""Reverse-engineer the transition dynamics of a grid game as a PURE Python function.
Infer ALL rules from the observations: movement, walls, attribute stations, energy/refills, and the WIN/deliver condition.

State `st`: dict px,py,shape,color,rot,energy (ints), solved (tuple of bool per slot), used_refill (tuple).
`action` in "ACTION1".."ACTION4". `lay`: walls(set (x,y)), station(dict (x,y)->"shape"|"color"|"rot"),
refill(dict keyed by (x,y)), slots(list of (x,y)), slot_req(dict (x,y)->(shape,color,rot)),
nshape, ncolor, nrot(=4), emax. The engine tests a 5x5 BOX [tx,tx+5)x[ty,ty+5) at the target for these elements.

OBSERVED TRANSITIONS:
{examples}

{win_obs}

Write `def candidate_model_step(st, action, lay):` returning the next-state dict (update energy, solved, used_refill).
Be GENERAL: handle shape/color/rot stations uniformly and any number of slots; a solved slot becomes inert.
Output ONLY one ```python code block."""


def call_llm(msgs):
    r = requests.post(OLLAMA, json={"model": MODEL, "messages": msgs,
                                    "temperature": 0.2, "max_tokens": 4000}, timeout=600)
    return r.json()["choices"][0]["message"]["content"]


def code_of(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    return m.group(1) if m else t


def divergence_feedback(fn, level):
    AIN = S.action_inputs(); g = S.new_game(level); lay = S.extract_layout(g)
    p, _ = S.plan(g, lay, step_fn=fn)
    if not p:
        return f"On level {level} your model cannot plan a win. Re-check station/deliver/energy rules."
    gg = S.new_game(level); st = S.init_state(gg)
    for i, a in enumerate(p):
        pr = fn(dict(st), a, lay); gg.perform_action(AIN[a]); en = S.init_state(gg)
        keys = ("px", "py", "shape", "color", "rot")
        if tuple(pr.get(k) for k in keys) != tuple(en[k] for k in keys):
            return (f"On level {level}, plan fails on engine at step {i} ({a}) from {fmt(st)}: "
                    f"model->{tuple((k,pr.get(k)) for k in keys)} engine->{tuple((k,en[k]) for k in keys)}. Fix it.")
        st = en
    return f"On level {level} the plan ran but did not win — your deliver/goal rule is wrong."


def induce(rounds=6):
    msgs = [{"role": "user", "content": build_prompt()}]
    for rnd in range(rounds):
        print(f"\n[induce] round {rnd}", flush=True)
        out = call_llm(msgs); ns = {}
        try:
            exec(code_of(out), ns)
        except Exception as e:
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": f"exec failed: {e}. Output one corrected python block."}]
            print("  exec error:", str(e)[:100]); continue
        fn = ns.get("candidate_model_step")
        if not fn:
            print("  no function"); break
        # robust verification: a buggy model must not crash the loop — feed the error back
        try:
            results = {L: S.solve_level(L, step_fn=fn)["win"] for L in TRAIN}
        except Exception as e:
            print("  model raised:", str(e)[:100], flush=True)
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": f"Your function raised {type(e).__name__}: {e}. "
                      f"`lay` keys are exactly: walls, station, refill, slots, slot_req, nshape, ncolor, nrot, emax. "
                      f"Fix and output one corrected python block."}]
            continue
        print("  train:", " ".join(f"L{L}={'WIN' if results[L] else 'fail'}" for L in TRAIN), flush=True)
        if all(results.values()):
            print(f"  ✅ induced model solves training levels {TRAIN}")
            return fn
        failL = next(L for L in TRAIN if not results[L])
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": divergence_feedback(fn, failL) + "\nOutput the corrected full function, one python block."}]
    return None


if __name__ == "__main__":
    fn = induce()
    if fn:
        print("\n=== generalization of the LLM-induced model on all 7 levels ===")
        wins = sum(S.solve_level(L, step_fn=fn)["win"] for L in range(7))
        print(f"LLM-induced model solves {wins}/7 levels")
    else:
        print("\nno working model induced within round limit")
