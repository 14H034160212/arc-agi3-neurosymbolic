# Neuro-symbolic ARC-AGI-3 (proof-of-concept on `ls20`)

The LLM **induces the game's dynamics as executable code**, a **classical planner** does the
search, and the induced model is **verified against the real engine** (à la ABPR's
program-as-hypothesis + algorithmic debugging, arXiv:2603.20334, adapted to an *interactive*
ARC-AGI-3 game). Everything runs **locally and free**.

## Headline results
- **Symbolic planning solves 7/7 `ls20` levels** with the verified dynamics model.
- A **free local LLM (gpt-oss)** induces a correct model for level 0 (auto-discovering the win
  condition) and, with a plan-based verify/refine loop, learns shape/color/rotation stations
  (generalizes to ~3/7; stronger model/feedback closes the rest).
- **Planning is trivial once the model is right** — the research value is induction + verification.

See [`neurosymbolic/REPORT.md`](neurosymbolic/REPORT.md) for the full writeup.

## Run it
These scripts use the `ls20` environment + `arcengine`/`arc_agi` from the ARC-AGI-3-Agents project.

```bash
# 1) clone the ARC-AGI-3-Agents project (provides environment_files/ls20 + deps) and `uv sync`
git clone https://github.com/symbolica-ai/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents && uv sync && git lfs install && git lfs pull

# 2) drop this repo's `neurosymbolic/` folder inside that checkout, then:
uv run python neurosymbolic/ls20_solver.py     # -> solves 7/7 levels
uv run python neurosymbolic/perception.py       # pixel->symbol prototype
# (optional) LLM induction needs a local Ollama serving gpt-oss on :11437
uv run python neurosymbolic/llm_induce.py
```

## Files
| File | What |
|---|---|
| `neurosymbolic/ls20_solver.py` | verified dynamics model + sub-goal planner → 7/7 |
| `neurosymbolic/llm_induce.py` | local-LLM dynamics induction + plan-based APD refine loop |
| `neurosymbolic/perception.py` | prototype pixel→symbol extractor (camera tracking = open problem) |
| `neurosymbolic/REPORT.md` | mechanics, results, bugs/insights, limitations, next steps |
