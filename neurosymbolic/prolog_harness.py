#!/usr/bin/env python3
"""
prolog_harness — LLM writes a Prolog RULE (not a raw click list); REAL SWI-Prolog derives the full
plan; the REAL game verifies it; on failure, the first divergence is localized and fed back for the
model to REVISE its rule. Implements the "program-as-hypothesis + algorithmic debugging" idea (ABPR,
arXiv:2603.20334, already used in this project for ls20's dynamics) applied to click-puzzle STRUCTURE,
directly following the design sketched in "LLM + Prolog：用逻辑程序进行可解释规则推理" (FLARE /
Prolog-MATH / LogicLease) — but using REAL Prolog execution (swipl via subprocess) rather than an
LLM-simulated trace (FLARE's weaker mode), since local compute makes real execution free and reliable.

Why this over the old "ask the model for a JSON list of clicks" hybrid design (harness_closed.py):
  - a RULE is reusable/general ("click the nearest unclicked object each time"), not a one-off guess;
  - swipl's real derivation can't hallucinate an invalid plan the way free-form JSON generation can;
  - a failed plan's first divergence from the environment localizes WHICH PART of the rule is wrong
    (mirrors FLARE's F_code vs F_search comparison), giving the model a much more specific revision
    signal than "try again" -- closer to how a human debugs a wrong hypothesis about game mechanics.

Usage: python prolog_harness.py <game_id> [--rounds N]
"""
from __future__ import annotations
import importlib.util as _ilu, os, re, subprocess, sys, tempfile, time
from pathlib import Path

_op = Path(__file__).resolve().parent / "online_play.py"
_spec = _ilu.spec_from_file_location("online_play", _op)
op = _ilu.module_from_spec(_spec); _spec.loader.exec_module(op)

# A small standard library of helper predicates the model can call, mirroring Prolog-MATH's
# "canonical predicate set" idea -- reduces how much Prolog syntax the model has to invent from
# scratch, the same way Prolog-MATH's Stage-1 predicate suggestion lowers the burden on Stage 2.
PROLOG_STDLIB = """
manhattan(X1,Y1,X2,Y2,D) :- D is abs(X1-X2) + abs(Y1-Y2).
nearest(Cur, Cands, Best) :-
    object(Cur, CX, CY),
    findall(Dist-Id, (member(Id, Cands), Id \\= Cur, object(Id, IX, IY), manhattan(CX,CY,IX,IY,Dist)), Pairs),
    sort(Pairs, [_-Best|_]).
"""

def call_llm_text(prompt, payload_text, model=None, endpoint=None, timeout=150):
    """Text-only call to a local Ollama model (no image -- the scene is already given as facts in
    the prompt, so a vision step isn't required here; keeps this fast and model-agnostic)."""
    import json, urllib.request
    model = model or os.environ.get("TEACHER_MODEL", "qwen3-vl:8b")
    endpoint = endpoint or os.environ.get("TEACHER_ENDPOINT", "http://localhost:11439/api/chat")
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt + "\n\n" + payload_text}]}).encode()
    req = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]


def extract_prolog_block(text):
    """Pull the Prolog source out of a ```prolog ... ``` fence, or the whole text if unfenced."""
    m = re.search(r"```(?:prolog)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def write_rule_prompt(objects, feedback=None):
    obj_lines = "\n".join(f"object({i}, {x}, {y})." for i, (x, y) in enumerate(objects))
    fb = f"\n\nPREVIOUS ATTEMPT FAILED. Divergence: {feedback}\nRevise the rule to fix this specific issue." if feedback else ""
    return (
        "You are solving a hidden-goal ARC-AGI-3 click puzzle. You are given the CLICKABLE objects as "
        "Prolog facts object(Id, X, Y). You do NOT see the image -- you must hypothesize the click "
        "ORDER RULE from the object layout (e.g. nearest-neighbour path, sorted by position, a "
        "geometric pattern, symmetry, grouping by proximity).\n\n"
        f"Facts:\n{obj_lines}\n\n"
        "You may use these helper predicates (already defined, do not redefine them):\n"
        f"{PROLOG_STDLIB}\n"
        "Write ONLY a Prolog rule defining plan(Plan) that binds Plan to the ordered list of object Ids "
        "to click, e.g.:\n"
        "plan(Plan) :- findall(Id, object(Id,_,_), All), plan_from(0, All, [0], Plan).\n"
        "IMPORTANT: the plan is NOT required to be a simple one-visit-per-object tour -- clicking the "
        "SAME object multiple times in a row (e.g. Plan=[0,0,0]) is a legitimate hypothesis too (many "
        "of these puzzles need repeated interaction with one object to cycle through a hidden state), "
        "so consider that shape of rule as well, especially with very few objects.\n"
        "(define any helper clauses you need). Wrap your code in a ```prolog``` block. No explanation."
        f"{fb}"
    )


def run_prolog_plan(rule_code, objects, timeout=10):
    """Execute rule_code + PROLOG_STDLIB + the object facts in REAL SWI-Prolog, querying plan(Plan).
    Returns the list of object ids (ints) in the derived order, or None if the query failed/errored."""
    facts = "\n".join(f"object({i}, {x}, {y})." for i, (x, y) in enumerate(objects))
    goal = ':- catch((plan(P), format("PLAN:~w~n", [P])), E, format("ERROR:~w~n", [E])), halt.'
    src = f"{PROLOG_STDLIB}\n{facts}\n{rule_code}\n{goal}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run(["swipl", "-q", "-g", "true", "-t", "halt", path],
                           capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        m = re.search(r"PLAN:\[([^\]]*)\]", out)
        if m:
            ids = [int(x) for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()]
            return ids, out
        return None, out
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    finally:
        os.unlink(path)


def solve_prolog(g, max_rounds=3, verbose=True):
    """The full loop: discover clickables -> LLM writes a Prolog rule -> swipl derives plan(Plan) ->
    replay against the REAL game -> on failure, localize the first divergence and feed it back for a
    revision, up to max_rounds."""
    s0 = g.reset().score()
    objs, _ = op.online_find_clickables(g)
    if verbose: print(f"    {g.game_id}: {len(objs)} clickable object(s)")
    if not objs:
        return None
    feedback = None
    for round_i in range(max_rounds):
        prompt = write_rule_prompt(objs, feedback)
        try:
            txt = call_llm_text("", prompt)
        except Exception as e:
            if verbose: print(f"    round {round_i}: model call FAILED ({type(e).__name__}: {e})")
            break
        rule_code = extract_prolog_block(txt)
        if verbose: print(f"    round {round_i}: rule =\n{rule_code}\n")
        ids, raw = run_prolog_plan(rule_code, objs)
        if ids is None:
            feedback = f"the Prolog query for plan(Plan) failed to produce a list. Raw output: {raw[:300]}"
            if verbose: print(f"    round {round_i}: Prolog derivation FAILED -- {feedback}")
            continue
        plan = [objs[i] for i in ids if 0 <= i < len(objs)]
        if verbose: print(f"    round {round_i}: derived plan (object ids) = {ids}")
        # replay against the REAL environment, tracking the first divergence
        g.reset(); divergence = None
        for step, (x, y) in enumerate(plan):
            before = g.grid()
            g.click(x, y)
            if g.status() == "WIN" or g.score() > s0:
                if verbose: print(f"    ✅ {g.game_id} WON via Prolog-derived plan (round {round_i}): {ids}")
                return plan[:step + 1]
            if g.status() == "LOSE":
                divergence = f"step {step} (clicking object {ids[step]} at {(x,y)}) caused a LOSE"
                break
            if (before == g.grid()).all() if hasattr(before, "all") else before == g.grid():
                divergence = (f"step {step}: clicking object {ids[step]} at {(x,y)} caused NO visible "
                             f"change -- this object may not be clickable at this point, or is not next")
                break
        if divergence is None:
            divergence = f"the full plan {ids} ran without error but did not win"
        feedback = divergence
        if verbose: print(f"    round {round_i}: plan failed -> {divergence}")
    if verbose: print(f"    {g.game_id}: no win after {max_rounds} rule-revision rounds")
    return None


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from arc_agi import Arcade, OperationMode
    arcade = Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""), operation_mode=OperationMode.ONLINE)
    card = arcade.open_scorecard(tags=["prolog-harness"])
    rounds = 3
    raw = sys.argv[1:]
    if "--rounds" in raw:
        i = raw.index("--rounds")
        rounds = int(raw[i + 1])
        raw = raw[:i] + raw[i + 2:]           # drop the flag AND its value, not just the flag
    args = [a for a in raw if not a.startswith("--")]
    try:
        for gid in args:
            print(f"{gid}:")
            try:
                g = op.OnlineGame(arcade, gid, card)
                solve_prolog(g, max_rounds=rounds)
            except Exception as e:
                print(f"    {gid}: FAILED ({type(e).__name__}: {e})")
    finally:
        sc = arcade.close_scorecard(card)
        try: print("SCORECARD levels:", sc.model_dump().get("total_levels_completed"))
        except Exception: pass


if __name__ == "__main__":
    main()
